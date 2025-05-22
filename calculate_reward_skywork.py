import pickle
import numpy as np
from Evaluation.match import *
import json
from armorm import ArmoRMPipeline
from collections import defaultdict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl_line(file_path, line_number):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line)  # convert JSON string to dict
    raise IndexError(f"Line {line_number} not found in file.")


M, A, R, C = [], [], [], []
coe_features = {}
num_sample, N = 240, 16
answer_parser = AnswerParsing("math500")
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
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)

outputs = defaultdict(dict)


for i in range(num_sample):

    max_idx, max_score = 0, 0
    for j in range(N):
        with open(f"OutputInfo/en/CoE/{model_name}/math500/math500_{i}_{j}.pkl", "rb") as f:
            coe_score = pickle.load(f)
            coe_features[f'{i}_{j}'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]
        
        with open(f"OutputInfo/en/Output/{model_name}/math500/math500_{i}_{j}.pkl", "rb") as f:

            
            coe_res = pickle.load(f)
            orig_data = read_jsonl_line("Data/math500.jsonl", i)
            answer = orig_data['answer']

            extracted, binary = answer_parser.dataset_parse(coe_res['output_seq'], answer, "")

            if binary: correct_counter += 1
        
        chat = [{"role": "user", "content": orig_data['en']}, {"role": "assistant", "content": coe_res['output_seq']}]
        conv_tokenized = rm_tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(device)
        with torch.no_grad():
            score = rm(conv_tokenized).logits[0][0].item()

        print('score', score)

        outputs[f'{i}_{j}']['reward'] = score
        outputs[f'{i}_{j}']['binary'] = binary
        outputs[f'{i}_{j}']['coe_features'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]



with open(f"math500_outputs_{model_name}_skyworkRM.pkl", "wb") as f:
    pickle.dump(outputs, f)


