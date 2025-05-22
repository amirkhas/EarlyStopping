import pickle
import numpy as np

R, C = [], []
for i in range(90):
    
    with open(f"OutputInfo/en/CoE/llama3.2-1b-instruct/test/test_{i}.pkl", "rb") as f:
        coe_score = pickle.load(f)
        R.append(coe_score['R'])
        C.append(coe_score['C'])



    with open(f"OutputInfo/en/Output/llama3.2-1b-instruct/test/test_{i}.pkl", "rb") as f:
        coe_res = pickle.load(f)
        print("\n", f"***********STRING OUTPUT******** {i}:", coe_res['output_seq'][-100:])
        print("\n", f"Sample {i}:", coe_score)







for i in range(0, len(R), 10):
    chunk_R = R[i:i+10]
    chunk_C = R[i:i+10]
    print("\n", np.mean(chunk_R), np.mean(chunk_C), np.std(chunk_R), np.std(chunk_C))