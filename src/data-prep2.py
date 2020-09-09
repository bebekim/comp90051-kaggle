from pathlib import Path
from itertools import repeat
import csv

input_path = Path("./input/train.txt")
output_path = Path("./input/train.csv")


def read_file(input_path):
    input_dict = dict()
    # read txt file as a list of lines
    with open(input_path) as f:
        for line in f:
            linestr_list = line.rstrip('\n').split("\t")
            linenum_list = list(map(int, linestr_list))
            key = linenum_list[0]
            input_dict[key] = linenum_list[1:]
        
    return input_dict


def make_edge(input_dict):
    edge_list = []
    # convert list of lines to a list of tuples
    for key in input_dict:
        x = repeat(key, len(input_dict[key])-1)
        y = input_dict[key]
        z = list(zip(x, y))
        for item in z:
            edge_list.append(item)
            
    return edge_list


def get_value(input_dict, u):
    if input_dict.get(u) is None:
        return [None]
    else:
        return input_dict.get(u)


def write_csv(output_path, edge_list, input_dict, begin, end,):
    with open(output_path, 'a') as out:
        csv_out = csv.writer(out)
        # do something to avoid duplicate lookup on the dictionary
        for u, v in edge_list[begin:end]:
            set_u = set(get_value(input_dict, u))
            set_v = set(get_value(input_dict, v))
            row = (u, v, len(set_u), len(set_v), len(set_u & set_v), len(set_u | set_v), len(set_u & set_v)/len(set_u | set_v))
            csv_out.writerow(row)

if __name__ == "__main__":
    dicts = read_file(input_path)
    edges = make_edge(dicts)
    write_csv(output_path='./input/similarity.csv', edge_list=edges, input_dict=dicts, begin=0, end=100)

