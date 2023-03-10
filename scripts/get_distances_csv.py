import os
import csv
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

RACES = ['African', 'Asian', 'Caucasian', 'Indian']

rfw_embeddings_dir = '~/RFW_feature_embeddings/pair_race_simplex/vggface2/'
rfw_pair_dir = '~/RFW_dataset/txts/' # + race + '/' + race + '_pairs.txt'
vmer_embeddings_dir = '~/VMER_feature_embedddings/pair_race_simplex/vggface2/'
vmer_pair_dir = '~/VGG-Face2/data/' # + race + '_pairs.csv'

rfw_points_file = open('configs/points_config_pair_500_to_10000.json')
rfw_points = json.load(rfw_points_file)
rfw_points_file.close()

vmer_points_file = open('configs/points_config_pair_50_to_800.json')
vmer_points = json.load(vmer_points_file)
vmer_points_file.close()

def cosine_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def get_pairwise_sims(features_dict, pair_list):
    sims = []
    labels = []
    for pair in pair_list:
        features1 = features_dict[pair[0]]
        features2 = features_dict[pair[1]]
        label = pair[2]
        sim = cosine_metric(features1, features2)

        sims.append(sim)
        labels.append(label)
    return sims, labels

def get_rfw_list(pair_list_file):
    with open(pair_list_file, 'r') as fd:
        pair_lines = fd.readlines()
    pair_list = []
    for pair_line in pair_lines:
        pair_items = pair_line.split('\t')
        if len(pair_items) == 3:
            identity = pair_items[0].strip()
            first_image = identity + '_' + pair_items[1].strip().zfill(4)
            second_image = identity + '_' + pair_items[2].strip().zfill(4)
            pair_list.append((first_image, second_image, 1))
        elif len(pair_items) == 4:
            first_identity = pair_items[0].strip()
            first_image = first_identity + '_' + pair_items[1].strip().zfill(4)
            second_identity = pair_items[2].strip()
            second_image = second_identity + '_' + pair_items[3].strip().zfill(4)
            pair_list.append((first_image, second_image, 0))
        else:
            raise Exception('pair file not following expected format')

    return pair_list

def get_vmer_list(pair_list_file):
    test_pairs = pd.read_csv(pair_list_file)

    pair_list = []
    pair_lines = test_pairs[['p1', 'p2', 'label']].values.tolist()
    for pair_line in pair_lines:
        p1 = pair_line[0]
        p2 = pair_line[1]
        label = int(pair_line[2])

        first_image = '/'.join(p1.split('/')[-2:])[:-4]
        second_image = '/'.join(p2.split('/')[-2:])[:-4]
        pair_list.append((first_image, second_image, label))

    return pair_list

distances_dict = defaultdict(dict)

for race in RACES:
    parent_dir = vmer_embeddings_dir + race
    
    pair_list_path = vmer_pair_dir + race + '_pairs.csv'
    pair_list = get_vmer_list(pair_list_path)
    
    for file in os.listdir(parent_dir):
        features_path = os.path.join(parent_dir, file)
        name = file.split('.')[0]
        name_arr = name.split('_')
        total_num = name_arr[0]
        idx = int(name_arr[1])
        trial_num = name_arr[-1]
        
        distr_arr = vmer_points['distributions'][total_num][idx]
        distr_name = '-'.join([str(x) for x in distr_arr]) + '_' + trial_num
        print(distr_name, race)
        
        features_dict = torch.load(features_path)
        pairwise_sims, pair_labels = get_pairwise_sims(features_dict, pair_list)
        pos_pairwise_sims = [pairwise_sims[i] for i in range(len(pair_labels)) if pair_labels[i] == 1]
        neg_pairwise_sims = [pairwise_sims[i] for i in range(len(pair_labels)) if pair_labels[i] == 0]
        
        distances_dict[distr_name][race] = (
            str(np.mean(pos_pairwise_sims)),
            str(np.std(pos_pairwise_sims)),
            str(np.mean(neg_pairwise_sims)),
            str(np.std(neg_pairwise_sims))
        )


print('DONE generating dict: ', len(distances_dict.keys()))

out_json='~/vmer_vggface2_feature_distances.json'
with open(out_json, 'w') as fp:
    json.dump(distances_dict, fp)

out_file='~/vmer_vggface2_feature_distances.csv'
lines = []
with open(out_file, 'w', newline='') as file:
    writer = csv.writer(file)
    HEADER = [
        'Pivot group',
        'Pivot number',
        'Other group',
        'Other number',
        'Distribution',
        'Test group',
        'Pos distance mean',
        'Pos distance std',
        'Neg distance mean',
        'Neg distance std',
        'Trial',
    ]
    writer.writerow(HEADER)

    for key in distances_dict:
        value = distances_dict[key]
        
        trial_num = key.split('_')[1]
        distr_name = key.split('_')[0]
        distr_arr = distr_name.split('-')

        i = None
        j_single = None
        for idx in range(len(distr_arr)):
            if distr_arr[idx] != '0':
                if i is None:
                    i = idx
                else:
                    j_single = idx
                    break

        js = [idx for idx in range(len(RACES)) if idx != i]
        if j_single is not None:
            js = [j_single]
        
        for j in js:
            for test_group in RACES:
                if test_group not in value or value[test_group][0] == 'nan' or value[test_group][1] == 'nan':
                    continue
                pos_pair_mean, pos_pair_std, neg_pair_mean, neg_pair_std = value[test_group]
                
                output = [
                    RACES[i],
                    distr_arr[i],
                    RACES[j],
                    distr_arr[j],
                    distr_name,
                    test_group,
                    pos_pair_mean,
                    pos_pair_std,
                    neg_pair_mean,
                    neg_pair_std,
                    trial_num
                ]
                writer.writerow(output)
                lines.append(output)
                
                output = [
                    RACES[j],
                    distr_arr[j],
                    RACES[i],
                    distr_arr[i],
                    distr_name,
                    test_group,
                    pos_pair_mean,
                    pos_pair_std,
                    neg_pair_mean,
                    neg_pair_std,
                    trial_num
                ]
                writer.writerow(output)
print('DONE writing')
