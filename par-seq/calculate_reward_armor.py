import pickle
import numpy as np
import json
from armorm import ArmoRMPipeline
from collections import defaultdict


def read_jsonl_line(file_path, line_number):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line)  # convert JSON string to dict
    raise IndexError(f"Line {line_number} not found in file.")


M, A, R, C = [], [], [], []
coe_features = {}
num_sample, N = 240, 16
correct_counter = 0


model_name = "qwen2.5-3b-Instruct"  #"llama3.1-8b-Instruct"

rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)


with open("par-seq/results/Qwen2.5-3B-Instruct_GENs.pkl", "rb") as f:
    all_res = pickle.load(f)


for i, value in all_res.items():

    for j in range(N):

        chat = value['prompt_and_gen'][j]
        answer = value['answer']
            
        
        # conv_tokenized = rm_tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(device)
        # conv_tokenized = rm_tokenizer(chat, return_tensors="pt").to(device)['input_ids']
        score = rm(chat)
        print('score', score)

