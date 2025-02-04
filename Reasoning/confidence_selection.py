from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np
from tqdm import tqdm
import os


def compute_conditional_probability(x, y, tokenizer, model, device="cuda"):
    # Concatenate the input sequence
    input_text = f"{x} {tokenizer.eos_token} {y}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Get the input IDs and compute the logits
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Calculate log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Extract the indices for `y`
    x_len = len(tokenizer(x, add_special_tokens=False)["input_ids"]) + 1  # +1 for the separator token
    y_len = len(tokenizer(y, add_special_tokens=False)["input_ids"])
    y_ids = input_ids[0, x_len:x_len + y_len]

    # Ensure the range aligns with the size of y_ids
    y_log_probs = log_probs[0, x_len:x_len + y_len, y_ids]

    # Compute the sum of log probabilities for `y`
    conditional_log_prob = y_log_probs.mean().item()

    return torch.exp(torch.tensor(conditional_log_prob)).item()  # Convert log prob to normal prob


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

# gsm8k
    from datasets import load_dataset
    gsm8k = load_dataset("gsm8k", 'main')
    gsm8k_train = gsm8k["train"]
    questions = gsm8k_train['question']
    answers = gsm8k_train['answer']

# caqa
#     questions = []
#     answers = []
#     decoder = json.JSONDecoder()
#     with open('dataset/csqa_train.jsonl') as f:
#         lines = f.readlines()
#         for line in lines:
#             json_res = decoder.raw_decode(line)[0]
#             choice = "Answer Choices:"
#             for c in json_res["question"]["choices"]:
#                 choice += " ("
#                 choice += c["label"]
#                 choice += ") "
#                 choice += c["text"]
#             questions.append(json_res["question"]["stem"].strip() + " " + choice)
#             answers.append(json_res["answerKey"])


    all_probs = []
    for question, answer in tqdm(zip(questions, answers)):
        x = question
        y = answer

        prob = compute_conditional_probability(x, y, tokenizer, model, device=device)
        all_probs.append(prob)

    print(all_probs)

    all_probs = np.array(all_probs)
    top_indices = np.argsort(-all_probs)[:6]
    print(top_indices)

    top_question = [questions[i] for i in top_indices]
    top_solution = [answers[j] for j in top_indices]

    for question, answer in zip(top_question, top_solution):
        print(f"Question:{question}\nAnswer:{answer}\n")
