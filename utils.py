import numpy as np
from io import BytesIO
from subprocess import Popen, PIPE
import sys

#@profile


def to_xnet(g, file_name, names=True):

    """
    Adapted from Filipi's code (https://github.com/filipinascimento)

    Convert igraph object to a .xnet format string. This string
    can then be written to a file or used in a pipe. The purpose
    of this function is to have a fast means to convert a graph
    without any attributes to the .xnet format.

    Parameters
    ----------
    g : igraph.Graph
        Input graph.

    Returns
    -------
    None
    """
    with open(file_name, 'w') as xnet_file:
        N = g.vcount()
        xnet_file.write('#vertices '+str(N)+' nonweighted\n')
        if names:
            for v in g.vs['name']:
                xnet_file.write('"' + v + '"' + '\n')
        xnet_file.write('#edges weighted undirected\n')

        for e, w in zip(g.es, g.es['weight']):
            xnet_file.write('%d %d %f\n' % (e.source, e.target, w))

    return None


def extract_motif(g):

    """
    Authored by: Vanessa Queiroz Marinho (https://github.com/vanessamarinho)
    """

    def get_incoming_nodes(matrix, size, node):
        neighbours = []
        for i in range(0, size):
            if matrix[i][node] == 1:
                neighbours.append(i)
        return neighbours

    def get_node_neighbours(matrix, size, node):
        neighbours = []
        for i in range(0, size):
            if matrix[node][i] == 1:
                neighbours.append(i)
        return neighbours

    def is_connected(matrix, source, target):
        if matrix[source][target] == 1:
            return True
        else:
            return False

    def calculate_motifs(matrix, size):
        motif = [0] * 14

        for i in range(0, size):
            A = i
            firstDegreeNeighbours = get_node_neighbours(matrix, size, A)
            while len(firstDegreeNeighbours) != 0:
                B = firstDegreeNeighbours[-1]
                if is_connected(matrix, B, A) == False:
                    secondDegreeNeighbours = get_node_neighbours(matrix, size, B)
                    while len(secondDegreeNeighbours) != 0:
                        C = secondDegreeNeighbours[-1]
                        if is_connected(matrix, C, B) == False:
                            if (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                motif[2] = motif[2] + 1
                            elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == True):
                                motif[5] = motif[5] + 1
                            elif (is_connected(matrix, C, A) == True and is_connected(matrix, A, C) == False):
                                motif[9] = motif[9] + 1

                        if is_connected(matrix, C, B):
                            if (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                motif[3] = motif[3] + 1
                            elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == True):
                                motif[6] = motif[6] + 1
                            elif (is_connected(matrix, C, A) == True and is_connected(matrix, A, C) == False):
                                motif[10] = motif[10] + 1
                        secondDegreeNeighbours.pop()
                    incomingNodes = get_incoming_nodes(matrix, size, B)
                    while len(incomingNodes) != 0:
                        C = incomingNodes[-1]
                        if (A != C and is_connected(matrix, B, C) == False and is_connected(matrix, A, C) == False
                                and is_connected(matrix, C, A) == False): motif[1] = motif[1] + 1
                        incomingNodes.pop()
                    secondDegreeNeighbours = get_node_neighbours(matrix, size, A)
                    while len(secondDegreeNeighbours) != 0:
                        C = secondDegreeNeighbours[-1]
                        if (C != B and is_connected(matrix, C, A) == False and is_connected(matrix, B, C) == False
                                and is_connected(matrix, C, B) == False):  motif[4] = motif[4] + 1
                        secondDegreeNeighbours.pop()

                if is_connected(matrix, B, A):
                    secondDegreeNeighbours = get_node_neighbours(matrix, size, B)
                    while len(secondDegreeNeighbours) != 0:
                        C = secondDegreeNeighbours[-1]
                        if C != A:
                            if is_connected(matrix, C, B) == False:
                                if (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                    motif[7] = motif[7] + 1
                                elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C)):
                                    motif[11] = motif[11] + 1
                            elif is_connected(matrix, C, B):
                                if (is_connected(matrix, C, A) == True and is_connected(matrix, A, C) == True):
                                    motif[13] = motif[13] + 1
                                elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == True):
                                    motif[12] = motif[12] + 1
                                elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                    motif[8] = motif[8] + 1
                        secondDegreeNeighbours.pop()
                firstDegreeNeighbours.pop()

        motif[1] = motif[1] / 2
        motif[4] = motif[4] / 2
        motif[6] = motif[6] / 2
        motif[8] = motif[8] / 2
        motif[11] = motif[11] / 2
        motif[9] = motif[9] / 3
        motif[13] = motif[13] / 6
        return motif

    g = g.simplify()
    matrix = g.get_adjacency()
    size = matrix.shape[0]
    motif_freq = calculate_motifs(matrix, size)

    total_motifs = sum(motif_freq)
    if total_motifs != 0:
        motif_perce = [x / float(total_motifs) for x in motif_freq]

    return motif_freq[1:], motif_perce[1:]


def extract_weighted_motif(g):

    """
        Authored by: Vanessa Queiroz Marinho (https://github.com/vanessamarinho)
    """

    def get_incoming_nodes(matrix, size, node):
        neighbours = []
        for i in range(0, size):
            if matrix[i][node] == 1:
                neighbours.append(i)
        return neighbours

    def get_node_neighbours(matrix, size, node):
        neighbours = []
        for i in range(0, size):
            if matrix[node][i] == 1:
                neighbours.append(i)
        return neighbours

    def is_connected(matrix, source, target):
        if matrix[source][target] == 1:
            return True
        else:
            return False

    def sum_weights(A, B, C, weight_matrix):
        return weight_matrix[A][B] + weight_matrix[A][C] + weight_matrix[B][C] + weight_matrix[B][A] + weight_matrix[C][
            A] + weight_matrix[C][B]

    def calculate_motifs(matrix, weight_matrix, size):
        motif = [0] * 14
        weighted_motif = [0] * 14

        for i in range(0, size):
            A = i
            firstDegreeNeighbours = get_node_neighbours(matrix, size, A)
            while len(firstDegreeNeighbours) != 0:
                B = firstDegreeNeighbours[-1]
                if is_connected(matrix, B, A) == False:
                    secondDegreeNeighbours = get_node_neighbours(matrix, size, B)
                    while len(secondDegreeNeighbours) != 0:
                        C = secondDegreeNeighbours[-1]
                        if is_connected(matrix, C, B) == False:
                            if (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                motif[2] = motif[2] + 1
                                weighted_motif[2] += sum_weights(A, B, C, weight_matrix)
                            elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == True):
                                motif[5] = motif[5] + 1
                                weighted_motif[5] += sum_weights(A, B, C, weight_matrix)
                            elif (is_connected(matrix, C, A) == True and is_connected(matrix, A, C) == False):
                                motif[9] = motif[9] + 1
                                weighted_motif[9] += sum_weights(A, B, C, weight_matrix)

                        if is_connected(matrix, C, B):
                            if (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                motif[3] = motif[3] + 1
                                weighted_motif[3] += sum_weights(A, B, C, weight_matrix)
                            elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == True):
                                motif[6] = motif[6] + 1
                                weighted_motif[6] += sum_weights(A, B, C, weight_matrix)
                            elif (is_connected(matrix, C, A) == True and is_connected(matrix, A, C) == False):
                                motif[10] = motif[10] + 1
                                weighted_motif[10] += sum_weights(A, B, C, weight_matrix)
                        secondDegreeNeighbours.pop()
                    incomingNodes = get_incoming_nodes(matrix, size, B)
                    while len(incomingNodes) != 0:
                        C = incomingNodes[-1]
                        if (A != C and is_connected(matrix, B, C) == False and is_connected(matrix, A, C) == False
                                and is_connected(matrix, C, A) == False):
                            motif[1] = motif[1] + 1
                            weighted_motif[1] += sum_weights(A, B, C, weight_matrix)
                        incomingNodes.pop()
                    secondDegreeNeighbours = get_node_neighbours(matrix, size, A)
                    while len(secondDegreeNeighbours) != 0:
                        C = secondDegreeNeighbours[-1]
                        if (C != B and is_connected(matrix, C, A) == False and is_connected(matrix, B, C) == False
                                and is_connected(matrix, C, B) == False):
                            motif[4] = motif[4] + 1
                            weighted_motif[4] += sum_weights(A, B, C, weight_matrix)
                        secondDegreeNeighbours.pop()

                if is_connected(matrix, B, A):
                    secondDegreeNeighbours = get_node_neighbours(matrix, size, B)
                    while len(secondDegreeNeighbours) != 0:
                        C = secondDegreeNeighbours[-1]
                        if C != A:
                            if is_connected(matrix, C, B) == False:
                                if (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                    motif[7] = motif[7] + 1
                                    weighted_motif[7] += sum_weights(A, B, C, weight_matrix)
                                elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C)):
                                    motif[11] = motif[11] + 1
                                    weighted_motif[11] += sum_weights(A, B, C, weight_matrix)
                            elif is_connected(matrix, C, B):
                                if (is_connected(matrix, C, A) == True and is_connected(matrix, A, C) == True):
                                    motif[13] = motif[13] + 1
                                    weighted_motif[13] += sum_weights(A, B, C, weight_matrix)
                                elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == True):
                                    motif[12] = motif[12] + 1
                                    weighted_motif[12] += sum_weights(A, B, C, weight_matrix)
                                elif (is_connected(matrix, C, A) == False and is_connected(matrix, A, C) == False):
                                    motif[8] = motif[8] + 1
                                    weighted_motif[8] += sum_weights(A, B, C, weight_matrix)
                        secondDegreeNeighbours.pop()
                firstDegreeNeighbours.pop()

        motif[1] = motif[1] / 2
        motif[4] = motif[4] / 2
        motif[6] = motif[6] / 2
        motif[8] = motif[8] / 2
        motif[11] = motif[11] / 2
        motif[9] = motif[9] / 3
        motif[13] = motif[13] / 6

        weighted_motif[1] = weighted_motif[1] / 2
        weighted_motif[4] = weighted_motif[4] / 2
        weighted_motif[6] = weighted_motif[6] / 2
        weighted_motif[8] = weighted_motif[8] / 2
        weighted_motif[11] = weighted_motif[11] / 2
        weighted_motif[9] = weighted_motif[9] / 3
        weighted_motif[13] = weighted_motif[13] / 6
        return motif, weighted_motif

    # simplify removes the weights if combine_edges = None
    g = g.simplify(combine_edges="sum")
    adjacency_matrix = g.get_adjacency()
    size = adjacency_matrix.shape[0]
    weight_matrix = [[0.0 for x in range(size)] for y in range(size)]
    for edge in g.es:
        weight_matrix[edge.source][edge.target] = edge["weight"]
    motif_freq, weighted_motifs = calculate_motifs(adjacency_matrix, weight_matrix, size)

    total_motifs = sum(motif_freq)
    motif_perce = []
    if total_motifs != 0:
        motif_perce = [x / float(total_motifs) for x in motif_freq]

    return motif_freq[1:], motif_perce[1:], weighted_motifs[1:]
