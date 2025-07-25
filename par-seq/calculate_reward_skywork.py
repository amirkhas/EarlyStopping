import pickle
import numpy as np
import json
from collections import defaultdict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle

def read_jsonl_line(file_path, line_number):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line)  # convert JSON string to dict
    raise IndexError(f"Line {line_number} not found in file.")


M, A, R, C = [], [], [], []
coe_features = {}
num_sample, N = 500, 2
correct_counter = 0


# model_name =   "llama3.1-8b-Instruct"
model_name =  "qwen2.5-3b-Instruct"



# Load model and tokenizer
device = "cuda:0"
rm_model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModelForSequenceClassification.from_pretrained(
    rm_model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    # attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)

outputs = defaultdict(dict)


with open("par-seq/results/Qwen2.5-3B-Instruct_GENs.pkl", "rb") as f:
    all_res = pickle.load(f)


for i, value in all_res.items():

    for j in range(N):

        chat = value['prompt_and_gen'][j]
        answer = value['answer']
            
        
        # conv_tokenized = rm_tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(device)
        conv_tokenized = rm_tokenizer(chat, return_tensors="pt").to(device)['input_ids']
        
        with torch.no_grad():
            score = rm(conv_tokenized).logits[0][0].item()

        print('score', score)


