
import json
import sys
import numpy as np
import random
#import scipy.io
import pandas as pd
from torch_geometric.datasets import Coauthor, Actor, Reddit, Planetoid


def load_geometric_dataset(name):
    """"undirected"""
    if name=="Coauthor":
        data = Coauthor(root='/tmp/coauthor', name='CS')[0]
        N_classes = 15
    elif name=="Actor":
        data = Actor(root='/tmp/Actor')[0]
        N_classes = 5
    elif name=="Reddit":
        data = Reddit(root='/tmp/Reddit')[0]
        N_classes = 41
    elif name=="Citeseer":
        N_classes = 6
        data =  Planetoid(root='/tmp/citeseer', name='CiteSeer')

    node_features = data.x  # Node features
    node_labels = data.y  # Node labels (if available)
    N_nodes = node_features.shape[0]
    edges_dict = {i:[] for i in range(N_nodes)}
    adj_matrix = np.zeros((N_nodes, N_nodes))
    N_edges = 0
    for i in range(data.edge_index.shape[1]):
        edge = data.edge_index[:, i] 
        src = int(edge[0].item())
        target = int(edge[1].item())
        if src==target:
            continue
        edges_dict[src].append(target)
        edges_dict[target].append(src)
        adj_matrix[src, target] = 1
        adj_matrix[target, src] = 1
        N_edges += 1
    edges_dict = remove_doubles(edges_dict)
    labels_dict = {}
    for i in range(N_nodes):
        labels_dict[i] = [node_labels[i]]

    graph_object = {"edges":edges_dict, "nodes":[i for i in range(N_nodes)], "N_nodes":N_nodes, "N_classes":N_classes, "N_edges":N_edges,
                     "adj_matrix":adj_matrix, "groups":labels_dict, "Multioutput":False, "edges_list":data.edge_index}
    return graph_object


def remove_doubles(edge_dict):
    data = {}
    for node, neighbors in edge_dict.items():
        data[node] = list(np.unique(neighbors))
    return data

def determine_multioutput(graph_dict, indexing=0):
    graph_dict['Multioutput'] = False
    for i in range(graph_dict['N_nodes']):
        if indexing == 0:
            j = i
        else:
            j = i+1
        if len(graph_dict["groups"][j])>1:
            graph_dict['Multioutput'] = True
            break
    return graph_dict


def load_blogcatalog(data_dir):
    ## Edit: changed this one to zero indexing for convenience when developing graphsage
    with open(data_dir+"/nodes.csv", "r") as file:
        N_nodes = len(file.readlines())

    # this one is utilized for learning the embedding
    graph_dict = {"edges":{i:[] for i in range(N_nodes)}, "nodes":[i for i in range(N_nodes)], 
                "groups":{i:[] for i in range(N_nodes)},  "N_nodes":N_nodes}

    N_classes = 0
    with open(data_dir+'/groups.csv', "r") as file:
        N_classes = len(file.readlines())
    
    graph_dict['N_classes'] = N_classes
    graph_dict['adj_matrix'] = np.zeros((N_nodes, N_nodes))
    edges_list = []
    N_edges = 0
    #adj_matrix = np.zeros((N_nodes, N_nodes))
    with open(data_dir+"/edges.csv", "r") as file:
        for line in file.readlines():
            node1  = int(line.split(",")[0])-1
            node2 = int(line.split(",")[1])-1
            graph_dict['edges'][node1].append(node2)
            graph_dict['edges'][node2].append(node1)
            graph_dict['adj_matrix'][node1, node2] = 1
            graph_dict['adj_matrix'][node2, node1] = 1
            N_edges += 1
            # Each edge is added only one time since the edge representation (inner product of vertices) is symmetric.
            edges_list.append([node1, node2])
            edges_list.append([node2, node1])

    #graph_dict['adj_matrix'] = adj_matrix
    graph_dict['edges_list'] = edges_list
    graph_dict['N_edges'] = N_edges
    graph_dict['Multioutput'] = True
    graph_dict['edges'] = remove_doubles(graph_dict['edges'])

    with open(data_dir+"/group-edges.csv", "r") as file:
        for line in file.readlines():
            node  = int(line.split(",")[0])-1
            group = int(line.split(",")[1])-1
            graph_dict['groups'][node].append(group)

    return graph_dict



def load_flickr(data_dir):
    """has the exact same file structure as blogcatalog"""
    graph_dict = load_blogcatalog(data_dir)
    graph_dict = determine_multioutput(graph_dict)
    return graph_dict


def load_reddit(data_dir):
    with open(data_dir+"/reddit-id_map.json", "r") as json_data:
        id_to_ind = json.load(json_data)
    nodes = [i+1 for i in id_to_ind.values()]
    ids = np.array(list(id_to_ind.keys()), dtype=str)
    N_nodes = len(nodes)
    graph_dict = {"edges":{i+1:[] for i in range(N_nodes)}, "nodes":nodes, 
                "groups":{i+1:[] for i in range(N_nodes)}, 'N_edges':0, 
                "N_nodes":N_nodes}

    with open(data_dir+"/reddit-class_map.json", "r") as json_file:
        id_to_group = json.load(json_file)
        for id in ids:
            group = id_to_group[id]+1   # groups and nodes are 1 indexed
            node_indx = id_to_ind[id]+1
            graph_dict['groups'][node_indx].append(group)
    N_groups = len(np.unique(list(graph_dict['groups'].values())))
    graph_dict['N_classes'] = N_groups

    graph_dict = determine_multioutput(graph_dict)
    
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
    graph_dict['edges'] = remove_doubles(graph_dict['edges'])
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
    
    graph_dict = {"N_nodes":len(nodes), "nodes":nodes, "edges":edges, "N_edges":N_edges,
                  "groups":classes_dict, "N_classes":N_classes}
    
    graph_dict = determine_multioutput(graph_dict)
    graph_dict['edges'] = remove_doubles(graph_dict['edges'])
    return graph_dict



def load_dblp_ci(data_dir):
    """Has no node labels"""
    N_edges = 0
    edges = {}
    edge_list = pd.read_csv(data_dir+"/out.dblp-cite", header=None, skiprows=1).to_numpy()
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
    graph_dict = {"N_nodes":len(nodes), "nodes":nodes, "edges":edges, "N_edges":N_edges}
    graph_dict = determine_multioutput(graph_dict)
    graph_dict['edges'] = remove_doubles(graph_dict['edges'])
    return graph_dict



def load_pubmed(data_dir):
    ## edit: zero indexed 
    N_edges = 0
    edges_temp = {}
    edge_list = pd.read_csv(data_dir+"/PubMed.edges", header=None, skiprows=0).to_numpy()
    node_to_ind = {}
    N_nodes = 0
    for edge in edge_list:
        source = edge[0]
        target = edge[1]
        if not node_to_ind.get(source):
            node_to_ind[source] = N_nodes
            N_nodes += 1
        if not node_to_ind.get(target):
            node_to_ind[target] = N_nodes
            N_nodes += 1
        if not edges_temp.get(source):
            edges_temp[source] = []
        edges_temp[source].append(target)
        N_edges += 1
        
    # in case there are disconnected nodes, this is necessary so that other code doesn't break
    edges = {i:[] for i in range(N_nodes)}
    adj_matrix = np.zeros((N_nodes, N_nodes))
    edges_list = []
    for src, targets in edges_temp.items():
        true_neighbor_indices = []
        true_src_index = node_to_ind[src]
        for t in targets:
            true_target_index = node_to_ind[t]
            adj_matrix[true_src_index, true_target_index] = 1
            true_neighbor_indices.append(true_target_index)
            edges_list.append([true_src_index, true_target_index])
        edges[true_src_index] = true_neighbor_indices  
    
    nodes = [i for  i in range(N_nodes)]

    class_list = pd.read_csv(data_dir+"/PubMed.node_labels", header=None).to_numpy()
    classes_dict = {i:[] for i in range(N_nodes)}

    for i,row in enumerate(class_list):
        node_id, label = row
        node_index = node_to_ind[node_id]
        classes_dict[node_index].append(int(label))
        
    N_classes = 3
    graph_dict = {"N_nodes":len(nodes), "nodes":nodes, "edges":edges, "N_edges":N_edges,
                  "groups":classes_dict, "N_classes":N_classes, "adj_matrix":adj_matrix, "edges_list":edges_list}
    graph_dict = determine_multioutput(graph_dict)
    graph_dict['edges'] = remove_doubles(graph_dict['edges'])
    return graph_dict



def load_toy(data_dir):
    ### Obs this one has zero indexed nodes!
    with open(data_dir+"/toy-ppi-id_map.json", "r") as json_data:
        id_to_ind = json.load(json_data)
    nodes = [i for i in id_to_ind.values()]
    ids = np.array(list(id_to_ind.keys()), dtype=str)
    N_nodes = len(nodes)
    graph_dict = {"edges":{i:[] for i in range(N_nodes)}, "nodes":nodes, 
                "groups":{i:[] for i in range(N_nodes)}, 'N_edges':0, 
                "N_nodes":N_nodes}

    g = [i for i in range(121)]
    with open(data_dir+"/toy-ppi-class_map.json", "r") as json_file:
        id_to_group = json.load(json_file)
        for id in ids:
            groups = np.nonzero(id_to_group[id])[0]
            node_indx = id_to_ind[id]
            if len(groups):
                graph_dict['groups'][node_indx] = groups 
            else:
                random_group = [random.choice(g)]
                graph_dict['groups'][node_indx] = random_group 


    N_groups = len(np.unique(id_to_group.values()))
    graph_dict['N_classes'] = N_groups

    graph_dict = determine_multioutput(graph_dict)

    adj_matrix = np.zeros((N_nodes, N_nodes))

    with open(data_dir+"/toy-ppi-G.json", "r") as json_file:
        content = json.load(json_file)
    links = content['links']
    graph_dict['N_edges'] = len(links) 
    graph_dict['edges_list'] = []
    for edge in links:
        node1 = edge['source']
        node2 = edge['target']
        adj_matrix[node1, node2] = 1
        graph_dict['edges'][node1].append(node2)
        graph_dict['edges'][node2].append(node1)
        graph_dict['edges_list'].append(([node1, node2]))
        graph_dict['edges_list'].append(([node2, node1]))

    for i in range(N_nodes):
        graph_dict['edges'][i] = np.array(graph_dict['edges'][i], dtype=int)

    graph_dict['node_feats'] = np.load(data_dir+"/toy-ppi-feats.npy")
    graph_dict['adj_matrix'] = adj_matrix

    graph_dict['Multioutput'] = True
    graph_dict['edges'] = remove_doubles(graph_dict['edges'])
    return graph_dict

if __name__=="__main__":

    #load_youtube("../Data/Youtube")
    #load_toy("../Data/toy")
    #load_reddit("../Data/Reddit")
    #load_cora("../Data/Cora")
    load_pubmed("../Data/PubMed")
  
