import math # For distance, though abs() is fine for 1D
import numpy as np # Useful for centroid calculation if needed, though manual is easy for 1D

def stop_on_gt_prob_streaming_cluster(scores, cluster_dist, alpha):
    """
    Processes scores sequentially using a streaming clustering method and stops
    when the simple Good-Turing probability of seeing a new cluster (N1/N)
    falls below alpha.

    Clusters are updated dynamically:
    - If a score is close enough (< cluster_dist) to the nearest existing centroid,
      it's added to that cluster, and the centroid is updated.
    - Otherwise, a new cluster is formed with the score as its centroid.

    Args:
        scores (list[float]): The list of reward scores to process.
        cluster_dist (float): Maximum absolute difference to the *closest* centroid
                              for a score to be added to an existing cluster. Must be >= 0.
        alpha (float): The probability threshold for stopping (e.g., 0.05).
                       Processing stops if P(new cluster) < alpha.

    Returns:
        tuple: A tuple containing:
            - list[float]: The scores processed up to the stopping point (or all scores).
            - float: The final calculated probability P(new cluster) = N1 / N.
            - list[tuple[float, int]]: The final clusters, represented as a list of
                                       (centroid, count) tuples.
            - int: The number of scores processed (N).
    """
    if not scores:
        return [], 1.0, [], 0 # Handle empty input

    if cluster_dist < 0:
        raise ValueError("cluster_dist cannot be negative")
    if not (0 <= alpha <= 1):
         print(f"Warning: alpha={alpha} is outside the typical [0, 1] range.")

    # List to store cluster info: [(centroid1, count1), (centroid2, count2), ...]
    clusters = []
    processed_scores = []
    N = 0  # Total number of items processed
    prob_new = 1.0 # Initial probability estimate

    print(f"--- Starting Streaming Clustering Process ---")
    print(f"Target alpha: {alpha}, Cluster distance threshold: {cluster_dist}\n")

    for i, score in enumerate(scores):
        N += 1
        processed_scores.append(score)

        min_dist = float('inf')
        closest_cluster_idx = -1

        # Find the closest existing cluster centroid
        if clusters: # Only search if clusters exist
            for idx, (centroid, count, _) in enumerate(clusters):
                # Use absolute difference for 1D scores
                dist = abs(score - centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster_idx = idx
        # else: first point will always create a new cluster

        assigned_to_existing = False
        # Decide whether to merge or create new
        # Merge if a closest cluster was found AND the distance is strictly less than the threshold
        if closest_cluster_idx != -1 and min_dist < cluster_dist:
            # --- Add to existing cluster and update centroid ---
            old_centroid, old_count, cluster_indices = clusters[closest_cluster_idx]
            new_count = old_count + 1
            # Update centroid: (old_sum + new_value) / new_count
            # old_sum = old_centroid * old_count
            new_centroid = (old_centroid * old_count + score) / new_count
            cluster_indices.append(i) # keeping a list of indices inside clusters
            clusters[closest_cluster_idx] = (new_centroid, new_count, cluster_indices)
            assigned_to_existing = True
            # print(f"Item {N}: Score {score:.4f} added to Cluster {closest_cluster_idx} (Dist: {min_dist:.4f}). New Centroid: {new_centroid:.4f}, New Count: {new_count}")

        else:
            # --- Create a new cluster ---
            # Happens if no clusters exist, or if min_dist >= cluster_dist
            new_centroid = score
            new_count = 1
            clusters.append((new_centroid, new_count, [i]))
            cluster_id = len(clusters) - 1
            # if closest_cluster_idx != -1:
            #      print(f"Item {N}: Score {score:.4f} (Dist to closest C{closest_cluster_idx}: {min_dist:.4f}) started NEW Cluster {cluster_id} (Centroid: {new_centroid:.4f})")
            # else:
            #      print(f"Item {N}: Score {score:.4f} started FIRST Cluster {cluster_id} (Centroid: {new_centroid:.4f})")


        # --- Good-Turing Calculation ---
        # Recalculate N1 (number of clusters with count == 1) based on *current* cluster counts
        N1 = 0
        for centroid, count, _ in clusters:
            if count == 1:
                N1 += 1

        # Calculate P(new) = N1 / N
        if N > 0:
            prob_new = N1 / N
        else:
            prob_new = 1.0 # Should not happen if scores is not empty

        # print(f"  Stats after item {N}: N={N}, N1={N1}, P(new) = {N1}/{N} = {prob_new:.4f}")
        # Optional: print cluster state for debugging
        # print(f"  Current Clusters: {[(f'{c:.2f}', n) for c,n in clusters]}")
        # print("-" * 20)

        # --- Check Stopping Condition ---
        # We check *after* processing the Nth item
        if N >=5 and prob_new < alpha: # we let the first 5 items to go w/o stopping criteria
        # if N == 32:
            print(f"\n--- Stopping Condition Met ---")
            print(f"P(new) = {prob_new:.4f} < alpha = {alpha} after processing {N} items.")
            return processed_scores, prob_new, clusters, N

    # If the loop finishes without stopping
    print(f"\n--- Processed All Scores ---")
    print(f"Finished processing all {N} items.")
    print(f"Final P(new) = {prob_new:.4f}")
    return processed_scores, prob_new, clusters, N

# # --- Example Usage ---
# # Sample reward scores (can be any sequence of numbers)
# reward_scores = [
#     10.1, 10.5, 15.5, 9.8, 20.1, 16.0, 15.6, 20.5, 20.3, 10.3, 25.0, 15.7, 20.2, 9.9, 24.8
# ]

# # Hyperparameters
# # Use a slightly larger distance threshold for this method, as centroids move
# distance_threshold = 1.0
# stopping_alpha = 0.25    # Stop when P(new cluster) < 0.25

# processed, final_prob, final_clusters, num_processed = stop_on_gt_prob_streaming_cluster(
#     reward_scores, distance_threshold, stopping_alpha
# )

# print("\n--- Final Results ---")
# print(f"Processed {num_processed} scores out of {len(reward_scores)}.")
# print(f"Final list of processed scores: {processed}")
# print(f"Final P(new cluster): {final_prob:.4f}")
# print(f"Final Clusters (Centroid, Count):")
# for i, (cent, count) in enumerate(final_clusters):
#     print(f"  Cluster {i}: Centroid={cent:.4f}, Count={count}")


import matplotlib.pyplot as plt

def plot_labels(feat1, feat2, labels, model_name, idx):
    
    out_dir = f"Figure/en/{model_name}/feat_vs_label/"
    # Convert labels to colors
    colors = ['blue' if label else 'red' for label in labels]

    # Create scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(feat1, feat2, c=colors)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Feature Scatter Plot (Blue=True, Red=False)")
    plt.grid(True)
    plt.savefig(out_dir + f"/{i}.png", dpi=300, bbox_inches='tight')
    plt.close('all')





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
num_sample, N = 200, 32
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
model_name =   "llama3.1-8b-Instruct" #"qwen2.5-3b-Instruct" 

sweep_bon = []
with open(f"math500_outputs_{model_name}_skyworkRM.pkl", "rb") as f:
    outputs = pickle.load(f)

for i in range(num_sample):

    max_idx, max_reward = 0, 0
    feat_list, feat_list_2, binary_list = [], [], []
    for j in range(N):
        with open(f"OutputInfo/en/CoE/{model_name}/math500/math500_{i}_{j}.pkl", "rb") as f:
            coe_score = pickle.load(f)
            coe_features[f'{i}_{j}'] = [coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']]
            feat_list.append(coe_score['Mag'])
            feat_list_2.append(coe_score['Ang'])
            # feat_list.append(random.uniform(0.1, 0.3))
        
        with open(f"OutputInfo/en/Output/{model_name}/math500/math500_{i}_{j}.pkl", "rb") as f:
            
            coe_res = pickle.load(f)
            orig_data = read_jsonl_line("Data/math500.jsonl", i)
            answer = orig_data['answer']

            extracted, binary = answer_parser.dataset_parse(coe_res['output_seq'], answer, "")

            if binary and j == 0: correct_counter += 1 #best-of-one counter
        
        reward = outputs[f'{i}_{j}']['reward'] 
        if reward >= max_reward:
            max_idx = j
            max_reward = reward
        
        if outputs[f'{i}_{max_idx}']['binary']:
            BON[j] += 1
        
        binary_list.append(outputs[f'{i}_{j}']['binary'])

        # plot_labels(feat_list, feat_list_2, binary_list, model_name, i)
    


    
    # print(feat_list)
    # min_val = min(feat_list)
    # max_val = max(feat_list)
    # feat_list = [(x - min_val) / (max_val - min_val) for x in feat_list]

    
    processed, final_prob, final_clusters, num_processed = stop_on_gt_prob_streaming_cluster(feat_list, distance_threshold, stopping_alpha)

    ########### taking the biggest cluster and do rm shit
    processed = []
    biggest_cluster = 0
    cluster_size = []
    for centroid, cluster_count, cluster_indices in final_clusters:
        min_dist = float('inf')
        for ci in cluster_indices:
            dist_to_centroid = abs(centroid - feat_list[ci])
            if dist_to_centroid < min_dist:
                min_idx = ci
                min_dist = dist_to_centroid
        
        cluster_size.append(cluster_count)
        processed.append(min_idx)

        # if cluster_count > biggest_cluster:
        #     biggest_cluster = cluster_count
        #     biggest_cluster_indices = cluster_indices

    # processed = biggest_cluster_indices
    num_processed = len(processed)
    ##########################

    cropped_N.append(len(processed))
    print(f"Processed {num_processed} scores out of {len(processed)} Final P(new cluster): {final_prob:.4f}")


    max_idx, max_reward = 0, 0
    # for jj in range(num_processed):
    
    print('processed', processed)
    for idxx, jj in enumerate(processed):
        reward = outputs[f'{i}_{jj}']['reward'] * cluster_size[idxx]
        binary = outputs[f'{i}_{jj}']['binary']

        if reward >= max_reward:
            max_idx = jj
            max_reward = reward
            G_T[i]['Mag'], G_T[i]['Ang'], G_T[i]['R'], G_T[i]['C'] = coe_score['Mag'], coe_score['Ang'], coe_score['R'], coe_score['C']
            G_T[i]['reward'], G_T[i]['binary'] = reward, binary

    
bon_counter = 0
for i in G_T.keys():
    if G_T[i]['binary']: bon_counter += 1

print(correct_counter, bon_counter, np.mean(cropped_N))
m = int( np.mean(cropped_N))
print(BON[m])

for i in range(m): print(BON[i])