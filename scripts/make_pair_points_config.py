import json
import numpy as np

MAX_NUM = 400

def list_of_points(total_num):
    points = []

    for i in range(0, MAX_NUM + 1, 50):
        j = total_num - i

        if j > 0 and j <= MAX_NUM:
            if i == 0:
                for g1 in range(4):
                    p = [0] * 4
                    p[g1] = j
                    points.append(p)
            elif j == 0:
                for g1 in range(4):
                    p = [0] * 4
                    p[g1] = i
                    points.append(p)
            else:
                for g1 in range(4):
                    for g2 in range(g1+1, 4):
                        p = [0] * 4
                        p[g1] = i
                        p[g2] = j
                        points.append(p)

    for p in points:
        print(p)

    return points

total_nums = range(50, 801, 50)

config = {'order': ['African', 'Asian', 'Caucasian', 'Indian'],
          'distributions': {},
          }

for total_num in total_nums:
    config['distributions'][total_num] = list_of_points(total_num)

num_points = 0
for t in config['distributions']:
    l = len(config['distributions'][t])
    print(t, l)
    num_points += l
print('Number of distributions', num_points)

with open('configs/points_config_pair_50_to_800.json', 'w') as f:
    json.dump(config, f, indent=2, sort_keys=True)
 
