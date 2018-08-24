from pyro import sample
import torch
from numpy import zeros, array, fliplr, sum
from functools import reduce
from itertools import product
from pyro.distributions import Bernoulli, Uniform
import pprint
import sys

def foulis_randall_product():
    fr_edges = []
    H = [ [[[0,0],[1,0]],[[0,1],[1,1]]], [[[0,0],[1,0]],[[0,1],[1,1]]] ]
    for edge_a in H[0]:
        for edge_b in H[1]:
            fr_edge = []
            for vertex_a in edge_a:
                for vertex_b in edge_b:
                    fr_edge.append([ vertex_a[0], vertex_b[0], vertex_a[1], vertex_b[1]])
            fr_edges.append(fr_edge)
    for mc in range(0,2):
        mc_i = abs(1-mc)
        for edge in H[mc]:
            for j in range(0,2):
                fr_edge = []
                for i in range(0, len(edge)):
                    edge_b = H[mc_i][i]
                    vertex_a = edge[abs(i-j)]
                    vertex_b = edge_b[0]
                    vertex_c = edge_b[1]
                    vertices_a = [vertex_a[0], vertex_b[0], vertex_a[1], vertex_b[1]]
                    vertices_b = [vertex_a[0], vertex_c[0], vertex_a[1], vertex_c[1]]
                    fr_edge.append([ vertices_a[mc], vertices_a[mc_i], vertices_a[mc+2], vertices_a[mc_i+2]])
                    fr_edge.append([ vertices_b[mc], vertices_b[mc_i], vertices_b[mc+2], vertices_b[mc_i+2]])
                fr_edges.append(fr_edge)
    return fr_edges

def variable(v):
    return torch.autograd.Variable(torch.Tensor([v]))

def get_vertex(a, b, x, y):
    return (2*x+a, 2*y+b)

def get_hyperedges(H, n):
    l = []
    for idx, e in enumerate(H):
        if n in e:
            l.append(idx)
    return l

def organise(M):
    return M.reshape(4).tolist()

def generate_global_distribution(constraints,N):
    hyperedges = foulis_randall_product()
    hyperedges_tallies = zeros(12)
    global_distribution = zeros((4, 4))
    while sum(global_distribution) < N:
        a = int(sample('A', Bernoulli(variable(0.5))))
        b = int(sample('B', Bernoulli(variable(0.5))))
        x = int(sample('X', Bernoulli(variable(0.5))))
        y = int(sample('Y', Bernoulli(variable(0.5))))
        value = float(sample('C', Uniform(variable(0.0), variable(1.0))))
        print(value)
        sys.exit()
        if (value < constraints[x][y][a,b]):
            for edge in get_hyperedges(hyperedges, [a, b, x, y]):
                hyperedges_tallies[edge] += 1
            global_distribution[get_vertex(a, b, x, y)] += 1
    for a, b, x, y in product(range(2), range(2), range(2), range(2)):
        summed_tally = (sum(hyperedges_tallies[e] for e in get_hyperedges(hyperedges, [a, b, x, y])))
        global_distribution[get_vertex(a, b, x, y)] /= summed_tally
    global_distribution *= 3
    global_distribution = [0] + reduce(lambda x, y: x + y, [
        organise(global_distribution[:2, :2]), organise(global_distribution[:2, 2:]), 
        organise(global_distribution[2:, :2]), organise(global_distribution[2:, 2:])])
    return global_distribution










# execution

constraints = [[ array([[0.5, 0], [0., 0.5]]), array([[0.5, 0], [0., 0.5]]) ],
        [ array([[0.5, 0], [0., 0.5]]), array([[0, 0.5], [0.5, 0.]]) ]]

p = generate_global_distribution(constraints,5000)










# testing


print(p)

A11 = (2 * (p[1] + p[2])) - 1
A12 = (2 * (p[5] + p[6])) - 1
A21 = (2 * (p[9] + p[10])) - 1
A22 = (2 * (p[13] + p[14])) - 1
B11 = (2 * (p[1] + p[3])) - 1
B12 = (2 * (p[9] + p[11])) - 1
B21 = (2 * (p[5] + p[7])) - 1
B22 = (2 * (p[13] + p[15])) - 1
delta = (abs(A11 - A12) + abs(A21 - A22) + abs(B11 - B21) + abs(B12 - B22))/2
A11B11 = (p[1] + p[4]) - (p[2] + p[3])
A12B12 = (p[5] + p[8]) - (p[6] + p[7])
A21B21 = (p[9] + p[12]) - (p[10] + p[11])
A22B22 = (p[13] + p[16]) - (p[14] + p[15])

print("Normalization in contexts", [p[1]+p[2]+p[7]+p[8]])
print("Normalization in contexts", [p[3]+p[4]+p[5]+p[6]])
print("Normalization in contexts", [p[9]+p[10]+p[15]+p[16]])
print("Normalization in contexts", [p[11]+p[12]+p[13]+p[14]])

print("delta", delta)
print("Potential violations")
print(abs(A11B11 + A12B12 + A21B21 - A22B22), 2 * (1 + delta))
print(abs(A11B11 + A12B12 - A21B21 + A22B22), 2 * (1 + delta))
print(abs(A11B11 - A12B12 + A21B21 + A22B22), 2 * (1 + delta))
print(abs(-A11B11 + A12B12 + A21B21 + A22B22), 2 * (1 + delta))
