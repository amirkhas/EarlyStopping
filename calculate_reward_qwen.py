import logging
import os
import time
from typing import Dict, List, Literal, Optional, Union
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import yaml

from reward_model import RewardModel
from adsk_raylab.tools.aws import aws_s3_sync



logger = logging.getLogger("reward_model")
class QwenPRM(RewardModel):
    def __init__(self, 
                 model_name,
                 local_model_weights_dir,
                 s3_model_weights_dir,
                 batch_size=16,
                 device_map="auto", 
                 torch_dtype_name: Literal["torch.bfloat16"] = "torch.bfloat16",
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        if torch_dtype_name == "torch.bfloat16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"torch_dtype_name {torch_dtype_name} not supported")

        # Sync model-weights
        local_dir = os.path.join(local_model_weights_dir, model_name)
        remote_dir = os.path.join(s3_model_weights_dir, model_name)

        logger.info(f"Syncing model weights from {remote_dir} to {local_dir}")
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        start_time = time.time()
        aws_s3_sync(remote_dir, local_dir, wait=True, print_info=False)
        logger.info(f"Model weights synced in {time.time() - start_time:.2f} seconds")

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            local_dir, 
            device_map=device_map, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).eval()

    def _make_step_rewards(self, logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def score(self, prompts: List[Dict[str, str]], doc_ids: Optional[List[Union[int, str]]] = None) -> List[float]:
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
        messages: List[List[Dict[str, str]]] = []
        for prompt in prompts:
            query = prompt[0]['content']
            raw_response = prompt[-1]['content']
            steps = raw_response.split("\n\n")
            response = "<extra_0>".join(steps) + "<extra_0>"
            messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ])

        scores = []
        for i in range(0, len(prompts), self.batch_size):
            batch_messages = messages[i:i + self.batch_size]
            input_ids = self.tokenizer.apply_chat_template(
                batch_messages, 
                tokenize=True, 
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            outputs = self.model(input_ids=input_ids)

            step_sep_id = self.tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            step_rewards = self._make_step_rewards(outputs[0], token_masks)
            del outputs

            for reward_list in step_rewards:
                # Last reward is the recommendation in paper "The Lessons of Developing Process Reward Models in Mathematical Reasoning", section 3.2.4
                scores.append(reward_list[-1])

        return scores



if __name__ == "__main__":
    # Example usage
    # with open("reward_model_config/qwen7b_prm800k.yaml", "r") as f:
    #     cfg = yaml.safe_load(f)

    # rm = QwenPRM(**cfg)
    # prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more:\nnUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: How many positive whole-number divisors does 196 have?"
    # res = "\n\n".join([
    #     '## Step 1: Expand the expression using the sum notation\nTo begin with, we expand the given expression by rewriting it using the sum notation, giving us:\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3} = \\sum_{m = 2}^\\infty \\sum_{n = 1}^{m - 1} \\frac{1}{m^3}\\]\nwhere we let $m = j + k$.',
    #     '## Step 2: Use the substitution to simplify the expression\nWe use the substitution $n = j$, giving us:\n\\[\\sum_{m = 2}^\\infty \\sum_{n = 1}^{m - 1} \\frac{1}{m^3} = \\sum_{m = 2}^\\infty \\frac{1}{m^3} \\sum_{n = 1}^{m - 1} 1\\]',
    #     '## Step 3: Evaluate the inner sum\nThe inner sum is equal to $m - 1$ since we are summing $1$ from $1$ to $m-1$. So we have:\n\\[\\sum_{m = 2}^\\infty \\frac{1}{m^3} (m - 1) = \\sum_{m = 2}^\\infty \\frac{m - 1}{m^3}\\]',
    #     '## Step 4: Rewrite the fraction\nWe can rewrite the fraction as a difference of fractions:\n\\[\\sum_{m = 2}^\\infty \\frac{m - 1}{m^3} = \\sum_{m = 2}^\\infty \\left(\\frac{1}{m^2} - \\frac{1}{m^3}\\right)\\]',
    #     '## Step 5: Distribute the summation to the fractions\nWe distribute the summation to the fractions, giving us:\n\\[\\sum_{m = 2}^\\infty \\left(\\frac{1}{m^2} - \\frac{1}{m^3}\\right) = \\sum_{m = 2}^\\infty \\frac{1}{m^2} - \\sum_{m = 2}^\\infty \\frac{1}{m^3}\\]',
    #     '## Step 6: Evaluate the two sums\nWe can evaluate the two sums in terms of $p$ and $q$:\n\\[\\sum_{m = 2}^\\infty \\frac{1}{m^2} - \\sum_{m = 2}^\\infty \\frac{1}{m^3} = \\left(\\sum_{k = 1}^\\infty \\frac{1}{k^2} - 1\\right) - \\left(\\sum_{k = 1}^\\infty \\frac{1}{k^3} - 1\\right)\\]',
    #     '## Step 7: Substitute the definitions of $p$ and $q$\nWe can substitute the definitions of $p$ and $q$ into the expression, giving us:\n\\[\\left(\\sum_{k = 1}^\\infty \\frac{1}{k^2} - 1\\right) - \\left(\\sum_{k = 1}^\\infty \\frac{1}{k^3} - 1\\right) = p - 1 - (q - 1)\\]',
    #     '## Step 8: Simplify the expression\nWe simplify the expression to get the final result:\n\\[p - 1 - (q - 1) = p - q\\]',
    #     'Therefore, the final answer is: $\\boxed{p - q}$.'])

    # messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": res}]
    # scores = rm.score([messages] * 20)
    # print(len(scores))
    # print(scores)











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
    num_sample, N = 150, 32
    answer_parser = AnswerParsing("math500")
    correct_counter = 0


    with open("reward_model_config/qwen7b_prm800k.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    rm = QwenPRM(**cfg)



    outputs = defaultdict(dict)


    for i in range(num_sample):
        print('*'*10, f"SAMPLE:{i}")
        max_idx, max_score = 0, 0
        for j in range(N):
            with open(f"OutputInfo/en/CoE/llama3.1-8b-Instruct/math500/math500_{i}_{j}.pkl", "rb") as f:
                coe_score = pickle.load(f)
                coe_features[f'{i}_{j}'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]
            
            with open(f"OutputInfo/en/Output/llama3.1-8b-Instruct/math500/math500_{i}_{j}.pkl", "rb") as f:

                
                coe_res = pickle.load(f)
                orig_data = read_jsonl_line("Data/math500.jsonl", i)
                answer = orig_data['answer']

                extracted, binary = answer_parser.dataset_parse(coe_res['output_seq'], answer, "")

                if binary: correct_counter += 1
            
            chat = [{"role": "user", "content": orig_data['en']}, {"role": "assistant", "content": coe_res['output_seq']}]
            # chat = [{"role": "user", "content": coe_res['input_seq']}, {"role": "assistant", "content": coe_res['output_seq']}]

            score = rm.score([chat])[0]
            print('score/binary', score, binary)

            outputs[f'{i}_{j}']['reward'] = score
            outputs[f'{i}_{j}']['binary'] = binary
            outputs[f'{i}_{j}']['coe_features'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]



    with open("math500_outputs_qwen.pkl", "wb") as f:
        pickle.dump(outputs, f)
