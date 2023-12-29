
import json
import sys
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix


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
    nodes = np.array(list(id_to_ind.values()), dtype=int)+1
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
        node1 = edge['source']
        node2 = edge['target']
        graph_dict['edges'][node1].append(node2)
        graph_dict['edges'][node2].append(node1)
        if c%int(graph_dict['N_edges']/20)==0:
            print(c)
        c += 1
    return graph_dict
     

if __name__=="__main__":
    #load_youtube("./Youtube")
    load_reddit("./Reddit")

