import pandas as pd
from pathlib import Path
from itertools import repeat
import csv
from math import ceil, log10
import swifter
from collections import defaultdict

def read_file(input: str):
    positive_dict = dict()
    negative_list = list()
    # read txt file as a list of lines
    with open(input) as f:
        for line in f:
            linestr_list = line.rstrip('\n').split("\t")
            linenum_list = list(map(int, linestr_list))
            # input_list.append(linenum_list)
            key = linenum_list[0]
            value = linenum_list[1:]
            # if node has no outgoing edge
            if not value:
                negative_list.append(key)
                pass
            # if node has outgoing edge    
            else:
                positive_dict[key] = value
        output = (positive_dict, negative_list)
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


def get_edge_count(input: dict, node_num: int):
    edge_cnt = get_dict_value(input, node_num)
    edge_class = determine_node_class(edges=edge_cnt)
    return (edge_cnt, edge_class)



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

def build_positive_edges(input: dict):
    positive_edges = []
    edge_profiles = defaultdict(set)
    for _, k in enumerate(input):
        u_edges, u_class = get_edge_count(input, k)
        u_class_repeat = repeat(u_class, len(u_edges))
        u = repeat(k, len(u_edges))
        edge_profiles[u_class] |= {k}
        outcome = repeat(1, len(u_edges))
        z = list(zip(u, u_edges, u_class_repeat, outcome))
        for item in z:
            positive_edges.append(item)

    output = (positive_edges, edge_profiles)
    return output


def build_negative_edges(negative_nodes: list, positive_nodes: dict):
    negative_edges = []
    u_nodes = negative_nodes
    v_nodes = list(positive_nodes.keys())

    u_class = 0
    outcome = 0
    for u in negative_nodes:
        for v in v_nodes:
            item = (u, v, u_class, outcome)
            negative_edges.append(item)

    return negative_edges


# traverse through source_class = 1



def build_edge_list(positive_link: dict, negative_node: list):
    positive_edge_list, edge_profiles = build_positive_edges(positive_link)
    negative_edge_list = build_negative_edges(negative_nodes=negative_node_list, positive_nodes=positive_link_dict)
    edge_list = positive_edge_list + negative_edge_list
    return edge_list
    

def construct_positive_edges(input: dict):
    positive_edges = []
    for _, source in enumerate(input):
        # u_edges is a list of edges
        source_edges, source_class = get_edge_count(input, source)
        outcome = 1

        for sink in source_edges:
            sink_edges, sink_class = get_edge_count(input, sink)
            item = (source , sink, source_class, sink_class, outcome)
            positive_edges.append(item)
    return positive_edges


def construct_negative_edges(negative_nodes: list, positive_nodes: dict):
    negative_edges = []
    for source in negative_nodes:
        source_class = 0
        outcome = 0     
        for _, sink in enumerate(positive_nodes):
            sink_edges = get_dict_value(positive_nodes, sink)
            sink_class = determine_node_class(sink_edges)    
            item = (source, sink, source_class, sink_class, outcome)
            negative_edges.append(item)

    return negative_edges


def write_source_csv(output, data: list):
    with open(output, 'w') as out:
        csv_out = csv.writer(out)
        # do something to avoid duplicate lookup on the dictionary
        for row in data:
            csv_out.writerow(row)


# this is an ill-defined function
# looks up positive_link_dict without explicitly saying so
def lookup_sink_class(node_num: int):
    # using positive_link_dict is troubling
    _, edge_class = get_edge_count(positive_link_dict, node_num)
    return edge_class



if __name__ == "__main__":
    input_path = Path("./input/train.txt")
    positive_link_dict, negative_node_list = read_file(input=input_path)
    table = build_edge_list(positive_link_dict, negative_node_list)

    cols = ['Source', 'Sink', 'SourceClass', 'Outcome']
    df = pd.DataFrame(table, columns=cols)

    output_path = Path("./input/table.csv")
    df.to_csv(output_path)

    SAMPLE_SIZE = 100000
    df_sample = df.sample(n=SAMPLE_SIZE)
    df_sample.to_csv('input/sample.csv')

    df_sample["SinkClass"] = df_sample.Sink.swifter.apply(lookup_sink_class)
    df_sample = df_sample[['Source', 'Sink', 'SourceClass', 'SinkClass', 'Outcome']]
    df_sample.to_csv('input/sample_fe.csv')
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
    
    
