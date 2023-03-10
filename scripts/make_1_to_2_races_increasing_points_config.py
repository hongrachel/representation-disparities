import json
import numpy as np

def list_of_points(total_num, base_per_group=2000, base_group_i=None):

    if base_group_i:
        points = []
        # keep i at total_num, others at 0
        p = [0] * 4
        p[base_group_i] = total_num
        points.append(p)

        # keep i at base_per_group, increase j, others at 0
        num_j = total_num - base_per_group
        if num_j > 0:
            for j in range(4):
                if j != base_group_i:
                    p = [0] * 4
                    p[base_group_i] = base_per_group
                    p[j] = num_j
                    points.append(p)

        for p in points:
            print(p)
        return points

    else:
        points = []

        # keep i at total_num, others at 0
        for i in range(4):
            p = [0] * 4
            p[i] = total_num
            points.append(p)

        # keep i at base_per_group, increase j, others at 0
        num_j = total_num - base_per_group
        if num_j > 0:
            for i in range(4):
                for j in range(4):
                    if i != j:
                        p = [0] * 4
                        p[i] = base_per_group
                        p[j] = num_j
                        points.append(p)

        for p in points:
            print(p)

        return points

total_nums = range(2000, 2441, 40)

ORDER = ['African', 'Asian', 'Caucasian', 'Indian']
config = {'order': ORDER,
          'distributions': {},
          }

for total_num in total_nums:
    config['distributions'][total_num] = list_of_points(total_num, base_per_group=total_nums[0])

with open('configs/points_config_increase_1_to_2_races_2000_to_2440.json', 'w') as f:
    json.dump(config, f, indent=2, sort_keys=True)
 
