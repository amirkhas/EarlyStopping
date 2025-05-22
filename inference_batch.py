import os
import sys
import time
import random

import pickle
import argparse
import scipy.spatial
import math
import json
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
import seaborn as sns
from collections import Counter

import numpy as np
import pickle
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)

project_root_path = os.environ["PROJECT_PATH"]
sys.path.append(project_root_path)
from Data.load_data import DatasetInfo
from prompt_pool import *
from score import OutputScoreInfo, CoEScoreInfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference:
    def __init__(self, model_info: dict, dataset_info: dict, verbose: dict):
        self.model_info = model_info
        self.dataset_info = dataset_info
        self.verbose = verbose

        self.model = self.model_info["model_ckpt"]
        self.model_name = self.model_info["model_name"]
        self.config = self.model_info["model_config"]
        self.generation_config = self.model_info["generation_config"]
        self.tokenizer = self.model_info["tokenizer"]
        self.max_output_token = self.model_info["max_output_token"]
        
        self.dataset_name = self.dataset_info["dataset_name"]
        self.data_loader = DatasetInfo(self.dataset_name)
        self.data_all = self.data_loader.data
        self.data_size = self.data_loader.data_size
        self.language = self.dataset_info["language"]

        self.sample_info = {}


    def dataset_inference(self):
        self.greedy_inference()
        

    def greedy_inference(self):
        for i in range(124, self.data_size):
            print("*"*30 + f" index {str(i)} " + "*"*30)
            sample = self.data_all[i]
            torch.cuda.empty_cache()
            batch_size = 8
            N = 32 ### THIS IS BEST OF N

            input_data, output_data, model_input, input_txt = self.parse_input(sample)

            for j in range(0, N, batch_size):

                    
                batch_input_txt = [input_txt] * batch_size


                with torch.no_grad():
                    generation_output, batch = self.model_inference(batch_input_txt)
                
                input_len = batch['input_ids'].shape[1]


                counter = 0 ### position in the batch

                for k in range(j, j+batch_size):
                    idx = str(i) + '_' + str(k)


                    self.sample_info = {
                        "input": {
                            "raw_input_data": input_data,
                            "model_input": model_input,
                            "input_len": input_len,
                        },
                        "output": {
                            "raw_output_data": output_data,
                        }
                    }

                    seq = generation_output.sequences[counter, :]
                    # self.sample_info["output"]["output_scores"] = generation_output.scores
                    self.sample_info["output"]["output_seq"] = seq
                    # self.sample_info["output"]["attentions"] = generation_output.attentions
                    self.sample_info["output"]["all_token_hidden_states"] = generation_output.hidden_states # output_len x layer_num x sampling_num x beam_search x hidden_dim
                    
                    input_len = self.sample_info["input"]["input_len"]
                    seq = seq[input_len:]
                    filtered_seq = seq[~seq.unsqueeze(-1).eq(torch.tensor(self.tokenizer.all_special_ids).to(device)).any(-1)] ### throwing out all the special ids
                    self.sample_info["output"]["output_len"] = len(filtered_seq)


                    output_seq, maxprob, ppl, entropy = self.print_output()
                    output = {'id': idx,
                            'answer_type': sample["answer_type"] if self.dataset_name == "theoremqa" else "",
                            'input_seq': self.sample_info["input"]["model_input"],
                            'output_seq': self.tokenizer.decode(filtered_seq),
                            }

                    if self.verbose["save_output"]: self.save_output(output, idx)

                    CoE_score, hidden_states = self.print_CoE_score(counter)
                    if self.verbose["save_hidden_states"]: self.save_hidden_states(hidden_states, idx)
                    if self.verbose["save_coe_score"]: self.save_CoE_score(CoE_score, idx) 
                    if self.verbose["save_coe_figure"]: self.save_CoE_figure(hidden_states, idx)
                    
                    
                    counter += 1 ### position in the batch


    def model_inference(self, batch_input_txt):
        # input_ids = self.sample_info["input"]["model_input_ids"]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        batch = self.tokenizer(batch_input_txt, padding=True, truncation=True, return_tensors="pt", padding_side='left')
        batch['input_ids'] = batch['input_ids'][:, 1:].to(device) ### it adds double begin-of-txt because of apply_chat function before 
        attention_mask = batch["attention_mask"].to(device)

        self.model.eval()
        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] \
            if "llama" in self.model_name else self.tokenizer.eos_token_id
    
        time_start = time.time()
        generation_output = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            max_new_tokens=self.max_output_token,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            do_sample=True,
            temperature=1.0,
        )
        time_end = time.time()
        print(f'inference time: {round(time_end - time_start, 4)}')
        
        return generation_output, batch


    def parse_input(self, sample):
        input_data = sample[self.language]
        output_data = sample["answer"]

        model_input = DATASET_PROMPTS[self.dataset_name].replace("{input_data}", input_data)
        if self.dataset_name == "theoremqa":
            model_input = model_input.replace("{answer_type}", sample["answer_type"])

        input_txt = self.tokenizer.apply_chat_template([{"role": "user", "content": model_input}], tokenize=False, add_generation_prompt=True)
        input_len = len(input_txt[0])

        print(f"********** Input Text (length: {input_len}) **********\n{input_data}\n")
        # print(f"********** Input ID **********\n{input_ids}\n")
        
        return input_data, output_data, model_input, input_txt


    def print_output(self):
        # output_scores = self.sample_info["output"]["output_scores"]
        output_seq = self.sample_info["output"]["output_seq"]
        true_output = self.sample_info["output"]["raw_output_data"]
        output_len = self.sample_info["output"]["output_len"]

        output_seq = self.tokenizer.decode(output_seq[0][-output_len:])
        print(f"********** Model-generated Text (length: {output_len}) **********\n{output_seq}\n")
        print(f"********** True Output Text **********\n{true_output}\n")

        # outputinfo = OutputScoreInfo(output_scores)
        # maxprob = outputinfo.compute_maxprob()
        # ppl = outputinfo.compute_ppl()
        # entropy = outputinfo.compute_entropy()
        # print(f"********** Output Info: **********\nmaxprob {maxprob}; perplexity {ppl}; entropy {entropy}\n")

        return output_seq, maxprob, ppl, entropy

    
    def save_output(self, output, i):
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/Output', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(output, file)
    

    def print_hidden_states(self, sample_num):
        hidden_states = self.sample_info["output"]["all_token_hidden_states"]
        output_len = self.sample_info["output"]["output_len"]

        input_len = self.sample_info["input"]["input_len"]
        seq = seq[input_len:]
        filtered_seq = seq[~seq.unsqueeze(-1).eq(torch.tensor(self.tokenizer.all_special_ids).to(device)).any(-1)] ### throwing out all the special ids
        output_len = len(filtered_seq)
        self.sample_info["output"]["output_len"] = output_len
        

        layer_num = len(hidden_states[1])
        hs_all_layer = []
        
        for j in range(layer_num):
            all_pos_hs = np.array([np.array(hidden_states[pos][j][0][0].cpu()) for pos in range(0, output_len)])
            hs_all_layer.append(np.mean(all_pos_hs, axis=0))
        hidden_states = hs_all_layer
        print(f"********** Hidden State Size: **********\n{np.array(hidden_states).shape}\n")

        return hidden_states


    def save_hidden_states(self, hidden_states, i):
        hs = {'hidden_states': hidden_states}
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/HiddenStates', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(hs, file)


    def print_CoE_score(self, sample_num):
        hidden_states = self.sample_info["output"]["all_token_hidden_states"]
        output_len = self.sample_info["output"]["output_len"]


        # input_len = self.sample_info["input"]["input_len"]
        # seq = seq[input_len:]
        # filtered_seq = seq[~seq.unsqueeze(-1).eq(torch.tensor(self.tokenizer.all_special_ids).to(device)).any(-1)] ### throwing out all the special ids
        # output_len = len(filtered_seq)
        # output_len = self.sample_info["output"]["output_len"]

        hs_all_layer = []
        layer_num = len(hidden_states[1])

        for j in range(layer_num):
            all_pos_hs = np.array([np.array(hidden_states[pos][j][sample_num][0].to(torch.float32).cpu()) for pos in range(0, output_len)])
            hs_all_layer.append(np.mean(all_pos_hs, axis=0))

        coescoreinfo = CoEScoreInfo(hs_all_layer)
        _, coe_mag, _ = coescoreinfo.compute_CoE_Mag()
        _, coe_ang, _ = coescoreinfo.compute_CoE_Ang()
        coe_r = coescoreinfo.compute_CoE_R()
        coe_c = coescoreinfo.compute_CoE_C()

        print(f"********** CoE Score Info: **********\nMag {coe_mag}; Ang {coe_ang}; R {coe_r}; C {coe_c}\n")
        return {"Mag": coe_mag,"Ang": coe_ang,"R": coe_r,"C": coe_c}, hs_all_layer


    def save_CoE_score(self, CoE_score, i):
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/CoE', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(CoE_score, file)   


    def save_CoE_figure(self, hidden_states, i):
        embeddings = PCA(n_components=2, random_state=2024).fit_transform(np.array(hidden_states))

        fig = plt.figure(figsize=(14, 8))
        #fig.suptitle('Embedding Trajectory under Correct/Incorrect Samples', fontsize=40, fontweight='bold')
        ax1 = fig.add_subplot(1, 1, 1, facecolor='w')

        traj_x = np.array(embeddings[:, 0])
        traj_y = np.array(embeddings[:, 1])

        ax1.scatter(traj_x, traj_y, color='blue', alpha=1.0, edgecolor='white', s=200)
        ax1.plot(traj_x, traj_y, color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax1.text(0, 0, "Origin (0,0)", color='black', fontsize=10)

        ax1.set_xlabel('X-axis', fontsize=24, fontweight='bold')
        ax1.set_ylabel('Y-axis', fontsize=24, fontweight='bold')
        
        '''save'''
        filedir = os.path.join(project_root_path, f'Figure/{self.language}', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        plt.savefig(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.png'), bbox_inches='tight', pad_inches=0)
        plt.close('all')
