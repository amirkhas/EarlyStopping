import logging
import os
import time
from typing import Dict, List, Literal, Optional, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from adsk_raylab.tools.aws import aws_s3_sync

from reward_model import RewardModel

logger = logging.getLogger("reward_model")
class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096, **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def batch_call(self, messages: List[List[Dict[str, str]]]) -> List[float]:

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            scores = output.score.cpu().float().tolist()
        
        return scores

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}
    
class ArmoRMRewardModel(RewardModel):

    def __init__(self, 
                 local_model_weights_dir,
                 s3_model_weights_dir,
                 batch_size=16,
                 device_map="auto", 
                 torch_dtype_name: Literal["torch.bfloat16"] = "torch.bfloat16", 
                 truncation=True, 
                 trust_remote_code=True, 
                 max_length=4096,
                 **kwargs):
        super().__init__()
        if torch_dtype_name == "torch.bfloat16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"torch_dtype_name {torch_dtype_name} not supported")
        

        model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        # Sync model-weights
        local_dir = os.path.join(local_model_weights_dir, model_id)
        remote_dir = os.path.join(s3_model_weights_dir, model_id)

        logger.info(f"Syncing model weights from {remote_dir} to {local_dir}")
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        start_time = time.time()
        aws_s3_sync(remote_dir, local_dir, wait=True, print_info=False)
        logger.info(f"Model weights synced in {time.time() - start_time:.2f} seconds")
        
        

        self.batch_size = batch_size
        self.rm = ArmoRMPipeline(local_dir, 
                device_map=device_map, 
                torch_dtype=torch_dtype, 
                truncation=truncation, 
                trust_remote_code=trust_remote_code, 
                max_length=max_length)


    def score(self, prompts: List[Dict[str, str]], doc_ids: Optional[List[Union[int, str]]] = None) -> List[float]:
        scores = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            scores.extend(self.rm.batch_call(batch_prompts))
        return scores
        
    
    
if __name__ == "__main__":

    # Create Reward Model Pipeline 
    prompt = 'What are some synonyms for the word "beautiful"?'
    response1 = 'Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant'
    response2 = '''Certainly! Here are some synonyms for the word "beautiful":
    1. Gorgeous
    2. Lovely
    3. Stunning
    4. Attractive
    5. Pretty
    6. Elegant
    7. Exquisite
    8. Handsome
    9. Charming
    10. Alluring
    11. Radiant
    12. Magnificent
    13. Graceful
    14. Enchanting
    15. Dazzling
    These synonyms can be used in various contexts to convey the idea of beauty.'''
    response3 = 'Sorry i cannot answer this.'

    rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    # score the messages
    chat1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
    chat2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]
    chat3 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response3}]
    score1 = rm(chat1)
    print(score1)

    score2 = rm(chat2)
    print(score2)

    score3 = rm(chat3)
    print(score3)

    scores = rm.batch_call([chat1, chat2, chat3])
    print(scores)

    rewward_model = ArmoRMRewardModel()
    print(rewward_model.score([chat1, chat2, chat3] * 16))