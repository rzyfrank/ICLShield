import argparse
import torch
import torch.nn as nn
import mlconfig
import time
import os
import sys
import numpy as np
import misc
import json
import data
import models
import cot_attacks  
from tqdm import tqdm
from exp_mgmt import ExperimentManager
from transformers import MarianMTModel, MarianTokenizer
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='CLIP')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--sc', type=int, default=1, help='number of output per question, default 1. More than 1 set for self-consistency')
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--defense', default='no', type=str)

def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if ('answer is' in al):
            break
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual

def eval(output_path, type='gsm8k'):
    from cot_ans_eval import eval_handlers
    eval_func = eval_handlers[exp.config.eval_handler]
    # Predict to the desired answer
    print("[Acc]")
    if type == 'csqa':
        Acc = eval_func(output_path, shift=0)[-1]
    elif type == 'strategyqa':
        Acc = eval_func(output_path, neg=False)[-1]
    elif type == 'letter':
        Acc = eval_func(output_path, flip=False)[-1]
    else:
        Acc = eval_func(output_path)[-1]

    # Predict to the desired answer amplified by the factor
    print("[ASRc]")
    if type == 'csqa':
        ASRc = eval_func(output_path, shift=1)[-1]
    elif type == 'strategyqa':
        ASRc = eval_func(output_path, neg=True)[-1]
    elif type == 'letter':
        ASRc = eval_func(output_path, flip=True)[-1]
    else:
        ASRc = eval_func(output_path, factor=2.1)[-1]

    # Incorrect prediction but with the target perturbation inserted
    print("[ASR]")
    if type == 'csqa':
        ASR = eval_func(output_path, shift=1, check_step=True)[-1]
    elif type == 'strategyqa':
        ASR = eval_func(output_path, neg=True, check_step=True)[-1]
    elif type == 'letter':
        ASR = eval_func(output_path, flip=True, check_step=True)[-1]
    else:
        ASR = eval_func(output_path, factor=2.1, check_step=True)[-1]
    
    return Acc, ASRc, ASR

def compute_perplexity(model, tokenizer, text, device='cuda'):
    model = model.to(device)
    encodings = tokenizer(text, return_tensors='pt', truncation=True)
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs[0]

    # PPL = exp(loss)
    ppl = torch.exp(loss)
    return ppl.item()


def onion_defense(model, tokenizer, text, threshold=1.0, max_iterations=1, device='cuda'):
    """
    对输入文本进行 Onion 防御:
    1. 计算原文本 perplexity
    2. 依次尝试移除每个 token 并评估 perplexity 变化
    3. 如果移除 token 能够显著降低 perplexity(> threshold)，则判定为可疑 token
    4. 迭代进行，直到没有 token 被移除或达到最大迭代次数
    返回“净化”后的文本
    参数：
        threshold: 用来判断可疑 token 的 perplexity 改变量 (可根据数据分布和经验进行调整)
        max_iterations: 最多迭代几轮
    """
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return text  # 空文本直接返回

    original_ppl = compute_perplexity(model, tokenizer, text, device=device)
    cleaned_tokens = tokens[:]

    for iteration in range(max_iterations):
        removed_any = False

        # 遍历当前文本中的 token
        i = 0
        while i < len(cleaned_tokens):
            temp_tokens = cleaned_tokens[:i] + cleaned_tokens[i + 1:]  # 移除第 i 个 token
            temp_text = tokenizer.convert_tokens_to_string(temp_tokens)

            # 计算移除 token 后的 PPL
            new_ppl = compute_perplexity(model, tokenizer, temp_text, device=device)

            # 如果移除后 PPL 降低量大于阈值，就去掉该 token
            if (original_ppl - new_ppl) > threshold:
                cleaned_tokens = temp_tokens
                # 更新当前文本新的 perplexity 基线
                original_ppl = new_ppl
                removed_any = True
            else:
                i += 1

        if not removed_any:
            break

    cleaned_text = tokenizer.convert_tokens_to_string(cleaned_tokens)
    return cleaned_text

def back_translation(text, en2de_tokenizer, en2de_model, de2en_tokenizer, de2en_model):
    with torch.no_grad():
        inputs = en2de_tokenizer([text], return_tensors="pt", truncation=True, padding=True).to("cuda:0")
        translated = en2de_model.generate(**inputs)
        german_text = en2de_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    with torch.no_grad():
        inputs = de2en_tokenizer([german_text], return_tensors="pt", truncation=True, padding=True).to("cuda:0")
        back_translated = de2en_model.generate(**inputs)
        back_translated_text = de2en_tokenizer.batch_decode(back_translated, skip_special_tokens=True)[0]

    return back_translated_text

def main():
    questions, answers = exp.config.dataset()
    
    if args.ddp:
        chunk_size = int(len(questions) / misc.get_world_size())
        local_rank = misc.get_rank()
        if local_rank == misc.get_world_size() - 1:
            questions = questions[chunk_size*local_rank:]
            answers = answers[chunk_size*local_rank:]
        else:
            questions = questions[chunk_size*local_rank:chunk_size*(local_rank+1)]
            answers = answers[chunk_size*local_rank:chunk_size*(local_rank+1)]

    if misc.get_rank() == 0:
        payload = "No. of Questions: {}".format(len(questions))
        logger.info(payload)
        print(payload)

    en2de_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    en2de_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to("cuda:0")
    en2de_model.eval()
    de2en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    de2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to("cuda:0")
    de2en_model.eval()

    prompt = open(exp.config.prompt_file, "r").read()
    # trigger = exp.config.trigger
    attacker = exp.config.attacker()
    model = exp.config.model()
    results = []
    # for q, a in tqdm(zip(questions[:2], answers[:2]), total=len(questions[:2])):
    for q, a in tqdm(zip(questions, answers), total=len(questions)):
        q = attacker.attack(q)
        if 'onion' in args.defense:
            q = onion_defense(model.model, model.tokenizer, q)
        elif 'translation' in args.defense:
            q = back_translation(q, en2de_tokenizer, en2de_model, de2en_tokenizer, de2en_model)
        prompt_q = prompt + '\n' + '\nQuestion: ' + q + '\n'
        model_ans = model.response(prompt_q)
        ans_by_model = [extract_ans(model_ans)[0]]
        ans_per_q = {'Question':q, 'Ref Answer':a, 'Model Answer': ans_by_model}
        results.append(ans_per_q)
        
    if args.ddp:
        full_rank_results = misc.all_gather(results)
        final_results = []
        for rank_results in full_rank_results:
            final_results += rank_results
    else:
        final_results = results
    
     # final save
    if misc.get_rank() == 0:
        with open(os.path.join(exp.exp_path, 'results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
        Acc, ASRc, ASR = eval(os.path.join(exp.exp_path, 'results.json'), type=exp.config.eval_handler)
        print(Acc, ASRc, ASR)
        payload = "\033[33mAcc: {:.4f}, ASRc: {:.4f}, ASR: {:.4f}\033[0m".format(Acc, ASRc, ASR)
        logger.info(payload)
        exp.save_eval_stats({
            'Acc': Acc,
            'ASRc': ASRc,
            'ASR': ASR
        
        }, name='cot')
    return

if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    if args.ddp:
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)
        seed = args.seed
    args.gpu = device

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))

    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    if args.ddp: 
        misc.destroy_process_group()
    