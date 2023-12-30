
import json
import sys
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix
import pandas as pd

def load_blogcatalog(data_dir):
    with open(data_dir+"/nodes.csv", "r") as file:
        N_nodes = len(file.readlines())

    # this one is utilized for learning the embedding
    graph_dict = {"edges":{i+1:[] for i in range(N_nodes)}, "nodes":np.array([i+1 for i in range(N_nodes)]), 
                "groups":{i+1:[] for i in range(N_nodes)},  "N_nodes":N_nodes}

    N_classes = 0
    with open(data_dir+'/groups.csv', "r") as file:
        N_classes = len(file.readlines())
    
    graph_dict['N_classes'] = N_classes

    edges_list = []
    N_edges = 0
    #adj_matrix = np.zeros((N_nodes, N_nodes))
    with open(data_dir+"/edges.csv", "r") as file:
        for line in file.readlines():
            node1  = int(line.split(",")[0])
            node2 = int(line.split(",")[1])
            graph_dict['edges'][node1].append(node2)
            graph_dict['edges'][node2].append(node1)
            # adj_matrix[node1-1, node2-1] = 1
            #adj_matrix[node2-1, node1-1] = 1
            N_edges += 1
            # Each edge is added only one time since the edge representation (inner product of vertices) is symmetric.
            edges_list.append((node1, node2))

    #graph_dict['adj_matrix'] = adj_matrix
    graph_dict['edges_list'] = edges_list
    graph_dict['N_edges'] = N_edges

    for i in range(N_nodes):
        graph_dict['edges'][i+1] = np.array(graph_dict['edges'][i+1])
        graph_dict['groups'][i+1] = np.array(graph_dict['groups'][i+1])

    with open(data_dir+"/group-edges.csv", "r") as file:
        for line in file.readlines():
            node  = int(line.split(",")[0])
            group = int(line.split(",")[1])
            graph_dict['groups'][node].append(group)

    return graph_dict



def load_flickr(data_dir):
    """has the exact same file structure as blogcatalog"""
    return load_blogcatalog(data_dir)



def load_youtube(data_dir):
    mat_file = scipy.io.loadmat(data_dir+"/youtube.mat")
    groups = mat_file['group']
    adj_matrix = mat_file['network']
    N_groups = groups.shape[1]
    N_nodes = groups.shape[0]
    graph_dict = {"edges":{i+1:[] for i in range(N_nodes)}, "nodes":np.array([i+1 for i in range(N_nodes)]), 
                "groups":{i+1:[] for i in range(N_nodes)}, 'N_edges':0, 
                "N_nodes":N_nodes, "N_classes":N_groups}
    
    row_inds, col_inds = adj_matrix.nonzero()
    N_edges_double = row_inds.shape[0]

    for i in range(N_edges_double):
        row = row_inds[i]
        col = col_inds[i]
        graph_dict["edges"][row+1].append(col+1)

        if i%int(N_edges_double/10)==0:
            print(i/N_edges_double)

    graph_dict['N_edges'] = int(N_edges_double/2)
    for i in range(N_nodes):
        graph_dict['edges'][i+1] = np.array(graph_dict['edges'][i+1])
        graph_dict['groups'][i+1] = np.array(graph_dict['groups'][i+1])

    return graph_dict
    

def load_reddit(data_dir):
    with open(data_dir+"/reddit-id_map.json") as json_data:
        id_to_ind = json.load(json_data)
    nodes = [i+1 for i in id_to_ind.values()]
    ids = np.array(list(id_to_ind.keys()), dtype=str)
    N_nodes = len(nodes)
    graph_dict = {"edges":{i+1:[] for i in range(N_nodes)}, "nodes":nodes, 
                "groups":{i+1:[] for i in range(N_nodes)}, 'N_edges':0, 
                "N_nodes":N_nodes, "N_groups":0}

    with open(data_dir+"/reddit-class_map.json", "r") as json_file:
        id_to_group = json.load(json_file)
        for id in ids:
            group = id_to_group[id]
            node_indx = id_to_ind[id]+1
            graph_dict['groups'][node_indx].append(group)

    with open(data_dir+"/reddit-G.json", "r") as json_file:
        content = json.load(json_file)

    links = content['links']
    graph_dict['N_edges'] = len(links) 
    c = 0
    for edge in links:
        node1 = edge['source']+1
        node2 = edge['target']+1
        graph_dict['edges'][node1].append(node2)
        graph_dict['edges'][node2].append(node1)
        if c%int(graph_dict['N_edges']/10)==0:
            print(c/graph_dict['N_edges'])
        c += 1
    return graph_dict


def load_cora(data_dir):
    
    N_edges = 0
    edges = {}
    edge_list = pd.read_csv(data_dir+"/out.subelj_cora_cora", header=None, skiprows=2).to_numpy()
    unique_nodes = set()
    for edge in edge_list:
        edge = edge[0].strip().split()
        source = int(edge[0])
        target = int(edge[1])
        if not source in unique_nodes:
            unique_nodes.add(source)
        if not target in unique_nodes:
            unique_nodes.add(target)
        if not edges.get(source):
            edges[source] = []
        edges[source].append(target)
        N_edges += 1
    
    N_nodes = len(unique_nodes)
    for n in range(N_nodes):
        if not edges.get(n+1):
            edges[n+1] = []

    nodes = [i+1 for  i in range(N_nodes)]

    class_list = pd.read_csv(data_dir+"/ent.subelj_cora_cora.class.name", header=None).to_numpy()
    classname_to_ind = {}
    N_classes = 0
    classes_dict = {}
    for i,row in enumerate(class_list):
        c = row[0]
        if not classname_to_ind.get(c):
            N_classes += 1 
            classname_to_ind[c] = N_classes
        classes_dict[i+1] = [classname_to_ind[c]]    # indexed classes and and nodes
    
    print(classname_to_ind)
    graph_dict = {"N_nodes":len(nodes), "nodes":nodes, "edges":edges, "N_edges":N_edges,
                  "groups":classes_dict, "N_classes":N_classes}

    return graph_dict



if __name__=="__main__":
    #load_youtube("./Youtube")
    #load_reddit("./Reddit")
    load_cora("../Data/Cora")
