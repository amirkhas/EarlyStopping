import math # For distance, though abs() is fine for 1D
import numpy as np # Useful for centroid calculation if needed, though manual is easy for 1D
import heapq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go




def smallest_feat_change(feat1, feat2, feat3, labels, model_name, idx, reward_list):
    
    out_dir = f"Figure/en/{model_name}/2inference/"
    # Convert labels to colors
    colors = ['blue' if label else 'red' for label in labels]

    # Create scatter plot
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')

    plt.figure(figsize=(6, 6))
    plt.scatter(feat1, feat2, c=colors)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Feature Scatter Plot (Blue=True, Red=False)")
    plt.grid(True)

    # Draw lines between points with the same feat3
    n = len(feat1)
    dist = []
    for i in range(0, n, 2):
        for j in range(i+1, n):
            if feat3[i] == feat3[j]:
                dist.append(abs(feat1[i] - feat1[j]))
                if j != i+1 or labels[i] != labels[j]:
                    raise ValueError(f"feat3 not equal for {i} and {j}, feat3[i]: {feat3[i]}, feat3[j]: {feat3[j]}")

    # Add reward text annotations
    top_indices = np.argsort(dist)[-5:]
    # top_indices = np.concatenate([np.argsort(dist)[-5:], np.argsort(dist)[:5]])


    # top_10 = np.argsort(dist)[:10]

    # for i, ind in enumerate(2 * top_indices):
    #     color = 'red' if reward_list[ind] < 0 else 'blue' ## to visually see the negative numbers
    #     plt.text(feat1[ind], feat2[ind], s=f"{reward_list[ind]:.2f}", fontsize=8, color=color)
    #     j = ind + 1 ## because we have pairs
    #     plt.plot([feat1[ind], feat1[j]], [feat2[ind], feat2[j]], 'k--', linewidth=1, alpha=0.6)
    # # plt.text(feat1[0] + 0.01, feat2[0] + 0.01, f"{reward:.2f}", fontsize=8, ha='left', va='bottom')


    # # plt.tight_layout()
    # # plt.show()


    # plt.savefig(out_dir + f"/{idx}.png", dpi=300, bbox_inches='tight')
    plt.close('all')

    return top_indices, dist







def plot_labels(feat1, feat2, feat3, labels, model_name, idx, reward_list):
    
    out_dir = f"Figure/en/{model_name}/2inference/"
    # Convert labels to colors
    colors = ['blue' if label else 'red' for label in labels]

    # Create scatter plot
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')

    plt.figure(figsize=(6, 6))
    plt.scatter(feat1, feat2, c=colors)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Feature Scatter Plot (Blue=True, Red=False)")
    plt.grid(True)

    # Draw lines between points with the same feat3
    n = len(feat1)
    for i in range(n):
        for j in range(i+1, n):
            if feat3[i] == feat3[j]:
                plt.plot([feat1[i], feat1[j]], [feat2[i], feat2[j]], 'k--', linewidth=1, alpha=0.6)  

    # Add reward text annotations
    top_indices = np.argsort(reward_list)[-3:]

    for ind in top_indices:
        plt.text(feat1[ind], feat2[ind], s=f"{reward_list[ind]:.2f}", fontsize=8)
    # plt.text(feat1[0] + 0.01, feat2[0] + 0.01, f"{reward:.2f}", fontsize=8, ha='left', va='bottom')


    # plt.tight_layout()
    # plt.show()

    plt.savefig(out_dir + f"/{idx}.png", dpi=300, bbox_inches='tight')
    plt.close('all')



def plot_labels_3d_interactive(feat1, feat2, feat3, labels, model_name, idx, reward_list):
    
    out_dir = f"Figure/en/{model_name}/feat_vs_label/"
    # Convert labels to colors
    colors = ['blue' if label else 'red' for label in labels]

    feat1 = np.array(feat1)
    feat2 = np.array(feat2)
    feat3 = np.array(feat3)
    labels = np.array(labels)

    colors = ['blue' if label else 'red' for label in labels]

    fig = go.Figure(data=[go.Scatter3d(
        x=feat1,
        y=feat2,
        z=feat3,
        mode='markers',
        marker=dict(size=4, color=colors),
        text=[f"Label: {label}" for label in labels],
        hoverinfo='text'
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3',
        ),
        title='Interactive 3D Scatter (Blue=True, Red=False)',
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()




############### REAL USE OF REWARDS

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
num_sample, N = 240, 16
answer_parser = AnswerParsing("math500")
correct_counter = 0
outputs = defaultdict(dict)
G_T = defaultdict(dict) ### good-turing
BON = defaultdict(int)
import random

# Hyperparameters
# Use a slightly larger distance threshold for this method, as centroids move
distance_threshold = .001
stopping_alpha = 0.1   # Stop when P(new cluster) < 0.25
cropped_N = []
# model_name = "llama3.1-8b-Instruct"
model_name = "qwen2.5-3b-Instruct"

sweep_bon = []
with open(f"math500_outputs_{model_name}_skyworkRM.pkl", "rb") as f:
# with open(f"math500_outputs_{model_name}.pkl", "rb") as f:
    outputs = pickle.load(f)

for i in range(num_sample):

    correct_answer_distance, wrong_answer_distance = [], []


    if i == 108:
        print('stop')
    max_idx, max_reward = 0, -np.inf
    feat_list_1, feat_list_2, feat_list_3, feat_list_4, binary_list, feat_list_5, reward_list = [], [], [], [], [], [], []

    ANG, MAG = [], []
    for j in range(N):
        for k in range(2):
            with open(f"OutputInfo/en/CoE/{model_name}/math500/math500_{i}_{j}_{k}.pkl", "rb") as f:

                coe_score = pickle.load(f)
                coe_features[f'{i}_{j}'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]
                ANG.append(coe_score['Ang'])
                MAG.append(coe_score['Mag'])

                feat_list_1.append(coe_score['Mag'])
                feat_list_2.append(coe_score['Ang'])
                feat_list_3.append(coe_score['R'])
                feat_list_4.append(coe_score['C'])
                feat_list_5.append(str(i) + '_' + str(j)) ### 2 runs for the inference

                # feat_list.append(random.uniform(0.1, 0.3))
            
            with open(f"OutputInfo/en/Output/{model_name}/math500/math500_{i}_{j}.pkl", "rb") as f:
                coe_res = pickle.load(f)

            binary_list.append(outputs[f'{i}_{j}']['binary'])
            reward_list.append(outputs[f'{i}_{j}']['reward'])
        
        orig_data = read_jsonl_line("Data/math500.jsonl", i)
        answer = orig_data['answer']

        extracted, binary = answer_parser.dataset_parse(coe_res['output_seq'], answer, "")

        if binary and j == 0: correct_counter += 1 #best-of-one counter
        
        reward = outputs[f'{i}_{j}']['reward'] 
        if reward > max_reward:
            max_idx = j
            max_reward = reward
        
        if outputs[f'{i}_{max_idx}']['binary']:
            BON[j] += 1

        # plot_labels(feat_list, feat_list_2, feat_list_3, binary_list, model_name, i, reward_list)
    # plot_labels(feat_list_1, feat_list_2, feat_list_5, binary_list, model_name, i, reward_list)
    top_indices, dist = smallest_feat_change(feat_list_3, feat_list_2, feat_list_5, binary_list, model_name, i, reward_list)
    

    # processed = filter_points(ang=ANG, mag=MAG, percent_to_keep=.5)
    # processed = [cc for cc in range(0, N) if cc % 2 == 0]
    # top_indices = [cc for cc in range(32)]



    cropped_N.append(len(top_indices))
    print(f"Processed {len(top_indices)} scores out of {N}")


    max_idx, max_reward = 0, -np.inf
    # for jj in range(num_processed):
    
    print('top indicces', top_indices)
    for idxx, jj in enumerate(top_indices):
        reward = outputs[f'{i}_{jj}']['reward']
        binary = outputs[f'{i}_{jj}']['binary']

        if reward >= max_reward:
            max_idx = jj
            max_reward = reward
            G_T[i]['Mag'], G_T[i]['Ang'], G_T[i]['R'], G_T[i]['C'] = coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']
            G_T[i]['reward'], G_T[i]['binary'] = reward, binary



    ### creating distributions of correct and incorrect answers feat list
    for jj in range(N):
        if outputs[f'{i}_{jj}']['binary']:
            correct_answer_distance.append(dist[jj])
        else:
            wrong_answer_distance.append(dist[jj])
    
    plt.figure(figsize=(6, 6))
    plt.hist(correct_answer_distance, bins=10, alpha=0.7, color='blue', label='List 1')
    plt.hist(wrong_answer_distance, bins=10, alpha=0.7, color='red', label='List 2')
    plt.savefig(f"Figure/en/{model_name}/correct_wrong_dist/{i}.png", dpi=300, bbox_inches='tight')
    plt.close('all')


bon_counter = 0
for i in G_T.keys():
    if G_T[i]['binary']: bon_counter += 1

print("************* RESULTS ****************" )
print(correct_counter, bon_counter, np.mean(cropped_N))
m = int( np.mean(cropped_N))
print(BON[m], "*************")

for i in range(N): print(BON[i])