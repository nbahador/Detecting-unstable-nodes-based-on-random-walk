# ------------------------------------------ #
# ------------------------------------------ #
# Detecting important nodes based on random walk

# Code writer: Nooshin Bahador
# ------------------------------------------ #
# ------------------------------------------ #

import numpy as np
from numpy.linalg import inv

## ------------------------------------------------------------------------------ ##
## Defining the network, Node features, Labels for nodes' cluster
## ------------------------------------------------------------------------------ ##

delta = 1

## ------------------------------------------------------------------------------ ##
# numerical example
# https://hal.archives-ouvertes.fr/hal-02867840/document
## ------------------------------------------------------------------------------ ##

b10 = 0.1295
a11 = 517.0544
b11 = 115.5967
a12 = 4.2614
b12 = 4.6361
a13 = 1.3083
b13 = 1.4428
a14 = 2.7480
b14 = 3.1052
a21 = 0.9492
b20 = -0.5262
a22 = 2.6331
b21 = 3.7399
a41 = 239.0092
b22 = 9.6729
a42 = 1.7819
b40 = 0.1595
a43 = 3.6549
b41 = 26.8926
a44 = 5.2346
b42 = 1.9330
a31 = 191.8224
b43 = 4.0660
a32 = 49.4899
b44 = 5.9926
b31 = 0.9688
b30 = 3.3
b32 = 0.2043


adj_matrix_0 = np.array([[0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [-a14,a13,-a12,-a11,0,0,-a22,-a21],
                       [0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [0,0,-a32,-a31,-a44,-a43,-a42,-a41]])

adj_matrix = np.array([[0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [-a14,a13,-a12,-a11,0,0,-a22,-a21],
                       [0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [0,0,-a32,-a31,-a44,-a43,-a42,-a41]])




beta = 1   # perturbation factor

fea_matrix = np.array(([0.5*beta,-0.1*beta,0.3*beta],
                       [0.2,0.1,0.7],
                       [-0.5,0.7,-0.1],
                       [-0.1,-0.6,0.4],
                       [0.3,-0.5,-0.2],
                       [0.1,-0.1,-0.4],
                       [0.3,0.8,-0.1],
                       [0.1,-0.2,0.2]), dtype=float)




Target_output = np.array(([0.01],[0.2],[0.2],[0.01],[0.01],[0.2],[0.2],[0.01]), dtype=float)

## ------------------------------------------------------------------------------ ##

row_sums = adj_matrix.sum(axis=1)
adj_matrix = adj_matrix / row_sums[:, np.newaxis]

## ------------------------------------------------------------------------------ ##

## ------------------------------------------------------------------------------ ##
## Plotting the topology of the network
## ------------------------------------------------------------------------------ ##
import matplotlib.pyplot as plt
import networkx as nx
G = nx.DiGraph()

for i in range(len(adj_matrix_0[1,:])):
 for j in range(len(adj_matrix_0[:,1])):
   if adj_matrix_0[i,j] != 0:
       G.add_edge(i, j, weight=adj_matrix_0[i,j])
   if adj_matrix_0[j,i] != 0:
       G.add_edge(j,i,weight=adj_matrix_0[j,i])

pos = nx.spiral_layout(G) #nx.circular_layout(G)
nx.draw(G,pos=pos, with_labels = True )
edgeLabels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = edgeLabels)
plt.show()

## ------------------------------------------------------------------------------ ##
bet_centrality = nx.betweenness_centrality(G, normalized=True,
                                           endpoints=False)
close_centrality = nx.closeness_centrality(G)
deg_centrality = nx.degree_centrality(G)

result = deg_centrality.values()
# Convert object to a list
data = list(result)
# Convert list to an array
numpyArray = np.array(data)
## ------------------------------------------------------------------------------ ##


A = adj_matrix_0
num_iter = 8
s111 = (num_iter, 1)
sum_weights_total_2 = np.zeros(s111)

for initial_node in range(len(adj_matrix_0[1,:])): # Starting node
    ax = plt.gca()
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos, nodelist=[initial_node], node_color='r')
    ax.set_title('Start walking from this node')
    plt.show()
    p = 0.8  # the return parameter: controlling the likelihood of immediately returning to a node visited in a walk.
    q = 0.1  # the in-out parameter: controlling the likelihood of visiting nodes further away from a node visited in a walk.

    Aprim = nx.to_numpy_array(G)
    num_iter = 8
    s111 = (num_iter, 1)
    sum_weights_total = 0

    num_iter = 1
    s111 = (num_iter, 3)
    path_total = np.zeros(s111)

    for n in range(len(adj_matrix_0[1,:])):
        previous_node = initial_node
        current_node = n
        for m in range(len(adj_matrix_0[1,:])):
            next_node = m
            bias = 1  # bias in random walks

            if (previous_node != current_node) and (current_node != next_node) and (previous_node != next_node) and \
                    (A[previous_node, current_node] != 0) and \
                    (A[current_node, next_node] != 0):
                # Possible scenarios of graph walking:
                if next_node == previous_node:  # If coming back to starting node
                    bias = 1 / p  # A large p reduce possibility to return to our starting node
                elif A[
                    next_node, previous_node] == 0:  # If there is no edge between next node previous node, then the bias be 1/q
                    bias = 1 / q  # A small q increase possibility to move away from starting node
                elif A[
                    next_node, previous_node] == 1:  # If there is an edge between next node previous node, then the bias be 1
                    bias = 1
                Aprim[current_node, next_node] = bias * A[current_node, next_node]
                sum_weights = (Aprim[previous_node, current_node] + Aprim[current_node, next_node]) / A.sum() ** 2  # Normalized edge transition probability calculation
                ax = plt.gca()
                ax.set_title('Random walk path = ' + str([previous_node, current_node,next_node]) + ' \n Sum of normalized transition probability of path = ' + str("%3f" % sum_weights))
                nx.draw(G, pos, with_labels=True, ax=ax)
                path = [initial_node, current_node, next_node]
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r')
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=10)
                #plt.show()
                sum_weights_total += sum_weights

                xx = path_total
                yy = np.array(path).reshape([-1, 1]).T
                path_total = np.append(xx, yy, axis=0)

    if not sum_weights_total:
        sum_weights_total = 0


    plt.show()

    for i in range(len(path_total[:, 1])):
        plt.plot([1, 2, 3], path_total[i, :])
        print(path_total[i, :])
    my_xticks = ['Initial Location', 'First Step', 'Second Step']
    plt.xticks([1, 2, 3], my_xticks)
    plt.ylabel('Node on Graph')
    plt.show()

    sum_weights_total_2[initial_node] = np.sum(sum_weights_total)/(G.degree[initial_node]**2)

plt.close('all')
objects = ('Node 0', 'Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5', 'Node 6', 'Node 7')
y_pos = np.arange(len(objects))
performance = sum_weights_total_2.squeeze()
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Sum of normalized transition probability of paths')
plt.xlabel('Starting Node')
plt.show()

