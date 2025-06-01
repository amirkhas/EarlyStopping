"""
Run Qwen-2.5-3B-Instruct on the MATH-500 benchmark with vLLM.

Requirements
------------
pip install "vllm[torch]" datasets tqdm
# 1× 16 GB GPU is enough for FP16; add --upgrade if you have an older vLLM.
"""

from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re, json, pathlib
from grading.grader import grade_answer
import os
from collections import defaultdict
import pickle

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")  # 500 rows  :contentReference[oaicite:0]{index=0}

# ---------------------------------------------------------------------
# 2. Build prompts
# ---------------------------------------------------------------------
# SYSTEM = (
#     "You are a helpful assistant who solves olympiad-style math questions "
#     "step-by-step. After the reasoning, print **only** the final answer on its "
#     "own line, preceded by 'Answer:'."
# )

SYSTEM = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {{ problem }}"


def make_prompt(problem: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

prompts = [make_prompt(ex["problem"]) for ex in ds]

# ---------------------------------------------------------------------
# 3. Init vLLM & generation parameters
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 8                         # adjust to fit GPU memory
N = 32 ### N for BoN
sampler = SamplingParams(
    temperature=1.0,                     # deterministic → better eval
    max_tokens=2048,
    n=N,
)

print(f"DEBUG: VLLM_DISABLE_TORCH_COMPILE = {os.environ.get('VLLM_DISABLE_TORCH_COMPILE')}")
llm = LLM(model=MODEL_NAME, dtype="float16", max_num_seqs=BATCH_SIZE)

# ---------------------------------------------------------------------
# 4. Batched inference
# ---------------------------------------------------------------------
preds = []
# for i in tqdm(range(0, len(prompts), BATCH_SIZE), ncols=80):
ALL_GEN = defaultdict(dict)

for i in tqdm(range(0, len(prompts), BATCH_SIZE), ncols=80): # iterate over data samples

    batch_prompts = prompts[i : i + BATCH_SIZE]
    outputs = llm.generate(batch_prompts, sampler, use_tqdm=False)
    
    for j, out in enumerate(outputs): # iterate over batches

        for k in range(N): # iterate over BoN
            txt = out.outputs[k].text.strip()
            if k == 0:
                ALL_GEN[ds[i]['unique_id']]['prompt_and_gen'] = [prompts[i] + txt]
            else:
                ALL_GEN[ds[i]['unique_id']]['prompt_and_gen'].append(prompts[i] + txt)
            ALL_GEN[ds[i]['unique_id']]['answer'] = ds[i]["answer"].strip()

        # Extract the last line after 'Answer:' (very simple parser)
        preds.append(txt)


with open(f'par-seq/results/{MODEL_NAME.split('/')[-1]}_GENs.pkl', 'wb') as f:
    pickle.dump(ALL_GEN, f)

# ---------------------------------------------------------------------
# 5. Simple exact-match accuracy
# ---------------------------------------------------------------------
gold = [str(a).strip() for a in ds["answer"]]
correct = sum(grade_answer(p, g) for p, g in zip(preds, gold))
print(f"Exact-match accuracy: {correct}/{len(ds)} = {correct/len(ds):.1%}")

# ---------------------------------------------------------------------
# 6. Save raw generations (optional)
# ---------------------------------------------------------------------
# out_file = pathlib.Path("qwen_math500_predictions.jsonl")
# with out_file.open("w", encoding="utf-8") as f:
#     for ex, pred in zip(ds, preds):
#         f.write(json.dumps({"id": ex["unique_id"], "prediction": pred,
#                             "gold": ex["answer"]}, ensure_ascii=False) + "\n")
# print(f"Predictions written to {out_file.resolve()}")
