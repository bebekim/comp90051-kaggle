import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
import matplotlib.pyplot as plt
import random

input_path = Path("./input/train.txt")
output_path = Path("./input/train.csv")

input_list = []
input_dict = dict()
# read txt file as a list of lines
with open(input_path) as f:
    for line in f:
        linestr_list = line.rstrip('\n').split("\t")
        linenum_list = list(map(int, linestr_list))
        input_list.append(linenum_list)
        k = linenum_list[0]
        input_dict[k] = linenum_list[1:]

# for l in input_list:
#     k = l[0]
#     input_dict[k] = l[1:]

edge_list = []
# convert list of lines to a list of tuples
for l in input_list:
    x = repeat(l[0], len(l)-1)
    y = l[1:]
    z = list(zip(x, y))
    for item in z:
        edge_list.append(item)

def get_value(input_diedct, u):
    if input_dict.get(u) is None:
        return [None]
    else:
        return input_dict.get(u)

edge_table = []
for u, v in edge_list:
    set_u = set(get_value(input_dict, u))
    set_v = set(get_value(input_dict, u))
    common_set = set_u and set_v
    union_set = set_u or set_v
    common_cnt = len(common_set)
    union_cnt = len(union_set)
    jacard_similarity = common_cnt/union_cnt
    row = (u, v, common_cnt, union_cnt, jacard_similarity)
    edge_table.append(row)


df = pd.DataFrame(edge_table, columns=['node_1', 'node_2', 'common_node_count', 'union_node_count', 'jacard_similarity'])
df.to_csv('node_similarity.csv', index=False, sep=',')

