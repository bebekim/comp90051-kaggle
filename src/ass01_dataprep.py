import pandas as pd
from pathlib import Path
from itertools import repeat
import csv
from math import ceil, log10
import swifter
from collections import defaultdict

def read_file(input: str):
    positive_dict = dict()
    # read txt file as a list of lines
    with open(input) as f:
        for line in f:
            linestr_list = line.rstrip('\n').split("\t")
            linenum_list = list(map(int, linestr_list))
            # input_list.append(linenum_list)
            key = linenum_list[0]
            value = linenum_list[1:]
            positive_dict[key] = value
            # if not value:
            #     # nodes are registered but have no child
            #     edge_profiles[0] |= {key}
            # # if node has outgoing edge    
            # else:
            #     positive_dict[key] = value
    return positive_dict


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
        # special treatment for edges with one connecting node so that they are not considered empty
        if log_value == 0:
            return 1
        else:
            return ceil(log_value)

# 2407470
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


# def build_positive_edges(input: dict, edges_categorized: dict):
#     positive_edges = []
#     for _, k in enumerate(input):
#         k_edges, k_class = get_edge_count(input, k)
#         k_class_repeat = repeat(k_class, len(k_edges))
#         u = repeat(k, len(k_edges))
#         outcome = repeat(1, len(k_edges))
#         z = list(zip(u, k_edges, k_class_repeat, outcome))
#         for item in z:
#             positive_edges.append(item)
#         edges_categorized[k_class] |= {k}

#     return positive_edges, edges_categorized

def update_edge_profile(input: dict):
    edge_profiles = defaultdict(set)
    for _, k in enumerate(input):
        _, k_class = get_edge_count(input, k)
        edge_profiles[k_class] |= {k}

    return edge_profiles

# build_negative_edges(positive_link, updated_edge_profiles)
def build_sample_edges(positive_nodes: dict, edges_categorized: dict, graph_size: int):
    edges_built = 0
    p_nodes = list(positive_nodes.keys())
    while edges_built < graph_size:
        # choose a source
            # randomly choose from p_nodes
        # determine source class
            # find source class
            # build positive links based on positive_nodes
        # reference check with edges_categorized:
            # randomly choose from a edges_categorized (k, v) 

        # register the pair
        # (u, v, u_class, outcome)
        edges_built += 1
    # u_nodes = negative_nodes
    u_class = 0
    outcome = 0
    for u in negative_nodes:
        for v in v_nodes:
            item = (u, v, u_class, outcome)
            negative_edges.append(item)

    return negative_edges


# def build_edge_list(positive_link: dict, negative_node: list):
#     positive_edge_list = build_positive_edges(positive_link)
#     negative_edge_list = build_negative_edges(negative_nodes=negative_node_list, positive_nodes=positive_link_dict)
#     edge_list = positive_edge_list + negative_edge_list
#     return edge_list
    

# build_positive_edges(positive_link, edges=edge_profiles)
def build_edge_list(positive_nodes: dict, edge_profiles: dict, graph_size: int):
    negative_edge_list = build_sample_edges(positive_nodes, edge_profiles, graph_size)
    edge_list += negative_edge_list
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
    # positive_link_dict, negative_node_list = read_file(input=input_path)
    # table = build_edge_list(positive_link_dict, negative_node_list)
    input_nodes = read_file(input=input_path)
    input_edge_profiles = update_edge_profile(input_nodes)
    GRAPH_SIZE = 100000
    graph = build_edge_list(input_nodes, input_edge_profiles, GRAPH_SIZE)

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
    
    
