import pickle
import numpy as np
from Evaluation.match import *
import json
from collections import defaultdict


def read_jsonl_line(file_path, line_number):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line)  # convert JSON string to dict
    raise IndexError(f"Line {line_number} not found in file.")


M, A, R, C = [], [], [], []
coe_features = {}
num_sample, N = 130, 22
answer_parser = AnswerParsing("math500")
correct_counter = 0
outputs = defaultdict(dict)
BON = defaultdict(dict)

model_name = "qwen2.5-3b-Instruct"  #"llama3.1-8b-Instruct"

with open(f"math500_outputs_{model_name}.pkl", "rb") as f:
    outputs = pickle.load(f)

for i in range(num_sample):

    max_idx, max_reward = 0, 0

    for j in range(N):
        with open(f"OutputInfo/en/CoE/{model_name}/math500/math500_{i}_{j}.pkl", "rb") as f:
            coe_score = pickle.load(f)
            coe_features[f'{i}_{j}'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]
        
        with open(f"OutputInfo/en/Output/{model_name}/math500/math500_{i}_{j}.pkl", "rb") as f:
            
            coe_res = pickle.load(f)
            orig_data = read_jsonl_line("Data/math500.jsonl", i)
            answer = orig_data['answer']

            extracted, binary = answer_parser.dataset_parse(coe_res['output_seq'], answer, "")

            if binary and j == 0: correct_counter += 1 #best of one counter
        
        reward = outputs[f'{i}_{j}']['reward'] 
        binary = outputs[f'{i}_{j}']['binary']

        print(reward, binary)
        if reward >= max_reward:
            max_idx = j 
            max_reward = reward
            BON[i]['Mag'], BON[i]['Ang'], BON[i]['R'], BON[i]['C'] = coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']
            BON[i]['reward'], BON[i]['binary'] = reward, binary
        
    
bon_counter = 0
for i in BON.keys():
    if BON[i]['binary']: bon_counter += 1

print(correct_counter, bon_counter)