from transformers import CLIPProcessor, CLIPModel
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_name = "bert-base-uncased"
# model_name = "facebook/opt-1.3b"
# model = AutoModel.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)


query = " @_@"

# gsm8k
# from datasets import load_dataset
# gsm8k = load_dataset("gsm8k", 'main')
# gsm8k_train = gsm8k["train"]
# questions = gsm8k_train['question']
# answers = gsm8k_train['answer']

#caqa
questions = []
answers = []
decoder = json.JSONDecoder()
with open('dataset/csqa_train.jsonl') as f:
    lines = f.readlines()
    for line in lines:
        json_res = decoder.raw_decode(line)[0]
        choice = "Answer Choices:"
        for c in json_res["question"]["choices"]:
            choice += " ("
            choice += c["label"]
            choice += ") "
            choice += c["text"]
        questions.append(json_res["question"]["stem"].strip() + " " + choice)
        answers.append(json_res["answerKey"])

candidate_texts = []
for question, answer in zip(questions, answers):
    candidate_texts.append(f"""Question:{question}\nAnswer:{answer}""")

def get_embeddings(texts):
    # 对文本进行编码
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    embeddings = last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def get_embeddings_in_batches(texts, batch_size=128):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_embeddings = get_embeddings(batch)
        all_embeddings.append(batch_embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

def compute_similarity(query_embedding, candidate_embeddings):
    return cosine_similarity(query_embedding, candidate_embeddings)[0]

query_embedding = get_embeddings([query])
candidate_embeddings = get_embeddings_in_batches(candidate_texts, batch_size=16)

similarities = compute_similarity(query_embedding, candidate_embeddings)

top_k = 6
top_k_indices = np.argsort(similarities)[-top_k:][::-1]
top_k_texts = [(candidate_texts[i], similarities[i]) for i in top_k_indices]

print("\nTop 6 Most Relevant Texts:")
for idx, (text, score) in enumerate(top_k_texts, 1):
    print(f"{idx}. {text} (Similarity: {score:.4f})")
