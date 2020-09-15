import pandas as pd
from pathlib import Path
from itertools import repeat
import csv
import numpy as np
import random
from math import floor, ceil, log10
from copy import copy


def read_file(input: str):
    input_dict = dict()
    input_list = list()
    # read txt file as a list of lines
    with open(input) as f:
        for line in f:
            linestr_list = line.rstrip('\n').split("\t")
            linenum_list = list(map(int, linestr_list))
            input_list.append(linenum_list)
            key = linenum_list[0]
            input_dict[key] = linenum_list[1:]
        output = (input_dict, input_list)
    return output


def list_to_dict(input: list):
    output = dict()
    for line in input:
        key = line[0]
        output[key] = line[1:]
    return output


def sample_from_list(input: list, p: float):
    random.seed(42)
    k = floor(p * len(input))
    output = random.sample(input, k)
    return output


def get_dict_value(input: dict, u: int):
    if input.get(u) is None:
        return [None]
    else:
        return input.get(u)


def determine_node_class(edges: list):
    # u is not in training source list
    if edges == [None]:
        return -1
    # u is in training source but have no connecting edge
    elif edges == []:
        return 0
    # u is in training source and have 1 or more connecting edge
    else:
        log_value = log10(len(edges))
        return ceil(log_value)


def source_breakdown(input: dict):
    source_breakdown = list()
    for _, key in enumerate(input):
        edges = get_dict_value(input, key)
        edge_count = len(edges)
        node_class = determine_node_class(edges=edges)
        # deadend_edge = 0
        # unknown_edge = 0
        # connected_edge = 0

        if edge_count > 0:
            # for e in edges:
            #     if input.get(e) is None:
            #         unknown_edge = unknown_edge + 1
            #     elif len(input.get(e)) == 0:
            #          deadend_edge = deadend_edge + 1
            #     else:
            #         connected_edge = connected_edge + 1
            edge_present = 1
        else:
            edge_present = 0

        row = (key,
               node_class,
               # connected_edge,
               # deadend_edge,
               # unknown_edge,
               edge_present)
        source_breakdown.append(row)

    return source_breakdown


# def jacard_similarity(input: dict):
#     output_list = list()
#     for _, u in enumerate(input):
#         edges_u = get_dict_value(input, u)
#         set_u = set(edges_u)
#         for v in edges_u:
#             directed_edge = (u, v)
#             outcome = 1
#             set_v = set(get_dict_value(input, v))
#             common_set = set_u & set_v
#             union_set = set_u | set_v
#             jacard_similarity = len(common_set)/len(union_set)
#             row = (u, v, len(common_set), len(union_set), jacard_similarity, outcome)
#             output_list.append(row)

#     return output_list

def record_outcomes(input: dict):
    outcome = []
    sink_list = list(input.keys())
    for index, k in enumerate(input):
        u_edges = get_dict_value(input, k)
        u_class = determine_node_class(edges=u_edges)
        if u_class > 0:
            u_class_repeat = repeat(u_class, len(u_edges))
            u = repeat(k, len(u_edges))
            outcome = repeat(1, len(u_edges))
            z = list(zip(u, u_edges, u_class_repeat, outcome))
            for item in z:
                outcome.append(item)
        else:
            u_edges = copy(sink_list)
            u_edges.remove(k)

            u_class_repeat = repeat(0, len(u_edges))
            u = repeat(k, len(u_edges))
            outcome = repeat(0, len(u_edges))
            z = list(zip(u, u_edges, u_class_repeat, outcome))
            for item in z:
                outcome.append(item)

    return outcome


def counter(input: dict, n: int):
    count = 0
    for index, k in enumerate(input):
        if len(input[k]) == n:
            count = count + 1
    return count


def write_source_csv(output, data: list):
    with open(output, 'w') as out:
        csv_out = csv.writer(out)
        # do something to avoid duplicate lookup on the dictionary
        for row in data:
            csv_out.writerow(row)


def convert_df(data: list, cols):
    df = pd.DataFrame(data, columns=cols)
    return df


def make_edge(input: dict):
    edge_list = []
    # convert list of lines to a list of tuples
    for key in input:
        x = repeat(key, len(input[key])-1)
        y = input[key]
        z = list(zip(x, y))
        for item in z:
            edge_list.append(item)

    return edge_list


# dataset = df.values
# X = dataset[:, 1:4]
# Y = dataset[:, 4]


# baseline model
def create_baseline(feature_cnt):
    # create model
    model = Sequential()
    model.add(Dense(feature_cnt, input_dim=feature_cnt, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_frequency(input: dict):
    frequency = list()
    for _, k in enumerate(input):
        nodes_cnt = len(input[k])
        frequency.append(nodes_cnt)


if __name__ == "__main__":
    input_path = Path("./input/train.txt")
    output_path = Path("./input/train.csv")
    input_dict, _ = read_file(input=input_path)
    links = record_outcomes(input_dict)

    # test_path = Path("./input/test-public.txt")
    # test_dict, test_list = read_file(input=test_path)
    # test_distribution = []
    # for l in test_list:
    #     row = tuple(l)
    #     _, u, v = row
    #     edges_u = get_dict_value(input_dict, u)
    #     edges_v = get_dict_value(input_dict, v)
    #     u_class = determine_node_class(edges_u)
    #     v_class = determine_node_class(edges_v)
    #     writerow = (u, u_class, v, v_class)
    #     test_distribution.append(writerow)
        

    # df_test = pd.DataFrame(test_distribution)
    # df_test[0].value_counts()
    # df_test[1].value_counts()
    # df_test[(dt_test[0] == 0) & (df_test[1] == 3)]


    # training = []
    # for u, v in edge_list:
    #     edges_u = get_dict_value(input_dict, u)
    #     edges_v = get_dict_value(input_dict, v)
    #     u_class = determine_node_class(edges_u)
    #     v_class = determine_node_class(edges_v)
    #     row = (u, u_class, v, v_class)
    #     training.append(row)

    # df = pd.DataFrame(training)
    # df[0].value_counts()
    # df[1].value_counts()


    # output
    # source_features = source_breakdown(input_dict)
    # col_names = ['Source', 'SourceClass','Outcome']
    # df = convert_df(source_features, cols=col_names)
    # sample = sample_from_list(input=input_list, p=0.01)
    # sample_dict = list_to_dict(input=sample)
    # edge_similarity = jacard_similarity(sample_dict)

    # write_source_csv(output=output_path, data=source_features)
    # feature_dt = np.dtype('int, int, int, int, int, int')
    # source_arr = np.array(source_features, dtype=feature_dt)
    
    
