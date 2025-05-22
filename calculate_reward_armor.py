import pickle
import numpy as np
from Evaluation.match import *
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
answer_parser = AnswerParsing("math500")
correct_counter = 0


model_name = "qwen2.5-3b-Instruct"  #"llama3.1-8b-Instruct"

rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)



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
        score = rm(chat)
        print('score', score)

        outputs[f'{i}_{j}']['reward'] = score['score']
        outputs[f'{i}_{j}']['binary'] = binary
        outputs[f'{i}_{j}']['coe_features'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]



with open(f"math500_outputs_{model_name}.pkl", "wb") as f:
    pickle.dump(outputs, f)


