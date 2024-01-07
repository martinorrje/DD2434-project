# Select 50% of the edges for training, leave remaining for testing.
# Want the remaining graph to still be connected, so we only remove edges if there are several neighbors
import random
from copy import deepcopy
import numpy as np
import torch



def ind_mapping(nodes):
    mapping = {}
    c = 0
    for node in nodes:
        mapping[node] = c
        c += 1
    return mapping

def compute_neighborhoods_subgraph(edge_dict, nodes, nb_size):
    unique_nodes = set(nodes)
    neighborhoods = {}  
    c = 0
    original_to_new = ind_mapping(nodes)
    for node in nodes:
        neighbors = edge_dict[node]
        # filter out neighbors that are not part of the subgraph   
        neighbors = [original_to_new[n] for n in neighbors if n in unique_nodes] 
        nb = len(neighbors)
        sample_size = min(nb_size, nb)
        if sample_size==0:
            neighborhoods[original_to_new[node]] = []
            continue
        if sample_size == 1:
            sample_neighborhood = [neighbors[0]]*nb_size
        else:
            neighborhood_ind = torch.randint(0, nb, (sample_size,))
            sample_neighborhood = [neighbors[i] for i in neighborhood_ind.tolist()]

        neighborhoods[original_to_new[node]] = sample_neighborhood
        c += 1
    return neighborhoods


def precision_and_recall(Y_true, Y_pred, nclasses):
    # count true positives and false positives and false negatives
    #nclasses = len(Y_true[0])
    TP_list = [0]*nclasses
    FP_list = [0]*nclasses
    FN_list = [0]*nclasses
    for j in range(nclasses):
       for i, pred in enumerate(Y_pred):
            if pred[j]==1 and Y_true[i][j]==1:
                TP_list[j] += 1
            elif pred[j]==1 and  Y_true[i][j]==0:
                FP_list[j] += 1
            elif pred[j]==0 and Y_true[i][j]==1:
                FN_list[j] += 1 

    return TP_list, FP_list, FN_list

def compute_f1_macro(Y_true, Y_pred, nclasses):
    #nclasses = len(Y_true[0])
    TP_list, FP_list, FN_list = precision_and_recall(Y_true, Y_pred, nclasses)
    f1_scores = [0]*nclasses
    for k in range(nclasses):
        if TP_list[k]==0:
            continue
        f1_scores[k] = TP_list[k]/(TP_list[k]+0.5*(FP_list[k]+FN_list[k])) 
    return np.sum(f1_scores)/nclasses


def compute_f1_micro(Y_true, Y_pred, nclasses):
    TP_list, FP_list, FN_list = precision_and_recall(Y_true, Y_pred, nclasses)
    TP = np.sum(TP_list)
    FP = np.sum(FP_list)
    FN = np.sum(FN_list)
    return TP/(TP + 0.5*(FN+FP))

def onehot(y, nclasses):
    Y = np.zeros((y.shape[0], nclasses), dtype=int)
    for i in range(y.shape[0]):
        c = y[i]
        Y[i,c-1] =  1
    return Y


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_edge_representation(fu,fv):
    return sigmoid(np.dot(fu,fv))



def split_graphs(total_graph, directed=False):
    print("splitting graphs")
    n_test_samples = int(total_graph['N_edges']*0.5)
    training_graph_unbalanced = deepcopy(total_graph["edges"])
    test_graph_unbalanced = {i:[] for i in range(total_graph['N_nodes'])}
    LP_test_X = [(1,1)]*n_test_samples*2
    LP_test_Y = [0]*n_test_samples*2
    counter = 0

    high_degree_nodes = []
    n_neighbors = {i:0 for i in range(total_graph['N_nodes'])}
    for i in range(total_graph['N_nodes']):
        nb_count = len(total_graph['edges'][i]) 
        n_neighbors[i] = nb_count


    while counter<n_test_samples:
        node1 = random.choice(total_graph['nodes'])
        if n_neighbors[node1]>1:
            node2 = random.choice(training_graph_unbalanced[node1])
            if n_neighbors[node2]>1:
                # Add to test data
                LP_test_X[counter] = (node1, node2)
                LP_test_Y[counter] = 1
                test_graph_unbalanced[node1].append(node2)

                # remove edge from training graph
                training_graph_unbalanced[node1].remove(node2)

                # add/remove reverse edge in case of undirected graphs                 
                if not directed:
                    test_graph_unbalanced[node2].append(node1)
                    training_graph_unbalanced[node2].remove(node1)
                    n_neighbors[node2] -= 1
    
                # decrease neighbor count
                n_neighbors[node1] -= 1
                counter += 1          
                if counter%int(n_test_samples/10)==0:
                    print(counter/n_test_samples)

    return LP_test_X, LP_test_Y, training_graph_unbalanced, test_graph_unbalanced


def balance_test_graph(total_graph, LP_test_X, LP_test_Y, test_graph_unbalanced, directed=False, reverse_fraction=0.5):
    print("balancing test graph")
    counter = 0
    n_test_samples = int(total_graph['N_edges']*0.5)
    # in case of directed graphs, a fraction of the negative edges are added by reversing true edges
    if directed:
        true_edges = LP_test_X[0:n_test_samples]
        while counter<int(n_test_samples*reverse_fraction):
            true_edge = random.choice(true_edges)
            src = true_edge[0]
            target = true_edge[1]
            if not src in test_graph_unbalanced.get(target):
                LP_test_X[n_test_samples+counter] = (target, src)
                counter += 1
            
            if counter%int(n_test_samples/10)==0:
                print(counter/n_test_samples)

    while counter<n_test_samples:
        node1, node2 = random.sample(total_graph['nodes'], 2)
        if not node1 in test_graph_unbalanced[node2]:
            try:
                LP_test_X[n_test_samples+counter] = (node1, node2)
                LP_test_Y[n_test_samples+counter] = 0
            except:
                LP_test_X.append((node1, node2))
                LP_test_Y.append(0)
                print("appended edge")
            counter += 1
    
        if counter%int(n_test_samples/5)==0:
            print(counter/n_test_samples)
    return LP_test_X, LP_test_Y

# When created the test set, we add remaining edges to the training set
# and add negative edges to balance the training data
def balance_training_graph(training_graph_unbalanced, total_graph, directed=False):
    print("balancing training graph")
    n_test_samples = int(total_graph['N_edges']*0.5)
    LP_train_X = []
    LP_train_Y = []
    added_edges = {i:{} for i in range(total_graph['N_nodes'])}
    for node, neighbors in training_graph_unbalanced.items():
        for nb in neighbors:
            if not added_edges[node].get(nb, False):
                added_edges[node][nb] = True
                if not directed:
                    added_edges[nb][node] = True
                LP_train_X.append((node, nb))
                LP_train_Y.append(1)

    n_negative_edges = 0
    while n_negative_edges < n_test_samples:
        node1, node2 = random.sample(total_graph['nodes'], 2)
        if not node1 in training_graph_unbalanced[node2]:
            LP_train_X.append((node1, node2))
            LP_train_Y.append(0)
            n_negative_edges += 1

        if n_negative_edges%int(n_test_samples/10)==0:
            print(n_negative_edges/n_test_samples)
        
    return LP_train_X, LP_train_Y