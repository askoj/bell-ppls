from pyro import sample
import torch
from numpy import zeros, array, fliplr, sum
from functools import reduce
from itertools import product
from pyro.distributions import Bernoulli, Uniform
import pprint
import sys
import time

def foulis_randall_product():
    fr_edges = []
    H = [ [[[0,0],[1,0]],[[0,1],[1,1]]], [[[0,0],[1,0]],[[0,1],
                                                         [1,1]]] ]
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
                    vertices_a = [
                        vertex_a[0], vertex_b[0], vertex_a[1], vertex_b[1]
                    ]
                    vertices_b = [
                        vertex_a[0], vertex_c[0], vertex_a[1], vertex_c[1]
                    ]
                    fr_edge.append([ 
                        vertices_a[mc], vertices_a[mc_i], vertices_a[mc+2], vertices_a[mc_i+2]
                    ])
                    fr_edge.append([ 
                        vertices_b[mc], vertices_b[mc_i], vertices_b[mc+2], vertices_b[mc_i+2]
                    ])
                fr_edges.append(fr_edge)
    return fr_edges

def variable(v):
    return torch.autograd.Variable(torch.Tensor([v]))

def get_vertex(a, b, x, y):
   return ((x*8)+(y*4))+(b+(a*2))

def get_hyperedges(H, n):
   l = []
   for idx, e in enumerate(H):
       if n in e:
           l.append(idx)
   return l


def generate_global_distribution(constraints,N):
    hyperedges = foulis_randall_product()
    hyperedges_tallies = zeros(12)
    global_distribution = zeros(16)
    while sum(global_distribution) < N:
        a = int(sample('A', Bernoulli(variable(0.5))))
        b = int(sample('B', Bernoulli(variable(0.5))))
        x = int(sample('X', Bernoulli(variable(0.5))))
        y = int(sample('Y', Bernoulli(variable(0.5))))
        value = float(sample('C', Uniform(variable(0.0), variable(1.0))))
        if (value < constraints[x][y][a,b]):
            for edge in get_hyperedges(hyperedges, [a, b, x, y]):
                hyperedges_tallies[edge] += 1
            global_distribution[get_vertex(a, b, x, y)] += 1
    for a, b, x, y in product(range(2), range(2), range(2), range(2)):
        summed_tally = (sum(hyperedges_tallies[e] for e in get_hyperedges(hyperedges, [a, b, x, y])))
        global_distribution[get_vertex(a, b, x, y)] /= summed_tally
    global_distribution *= 3
    return global_distribution








def accuracy_time(N):
    print("Iterations %s" % (N))
    constraints = [[ array([[0.5, 0], [0., 0.5]]), array([[0.5, 0], [0., 0.5]]) ],[ array([[0.5, 0], [0., 0.5]]), array([[0, 0.5], [0.5, 0.]]) ]]
    start = time.time()
    Q = generate_global_distribution(constraints,N)
    end = time.time()
    p = Q
    A11 = (2 * (p[0] + p[1])) - 1
    A12 = (2 * (p[4] + p[5])) - 1
    A21 = (2 * (p[8] + p[9])) - 1
    A22 = (2 * (p[12] + p[13])) - 1
    B11 = (2 * (p[0] + p[2])) - 1
    B12 = (2 * (p[8] + p[10])) - 1
    B21 = (2 * (p[4] + p[6])) - 1
    B22 = (2 * (p[12] + p[14])) - 1
    delta = (abs(A11 - A12) + abs(A21 - A22) + abs(B11 - B21) + abs(B12 - B22))/2
    A11B11 = (p[0] + p[3]) - (p[1] + p[2])
    A12B12 = (p[4] + p[7]) - (p[5] + p[6])
    A21B21 = (p[8] + p[11]) - (p[9] + p[10])
    A22B22 = (p[12] + p[15]) - (p[13] + p[14])
    print("Time:")
    print(end - start)

    print("Normalization in contexts: ", [p[0]+p[1]+p[6]+p[7]])
    print("Normalization in contexts: ", [p[2]+p[3]+p[4]+p[5]])
    print("Normalization in contexts: ", [p[8]+p[9]+p[14]+p[15]])
    print("Normalization in contexts: ", [p[10]+p[11]+p[12]+p[13]])

    print("delta: ", delta)
    print("Potential violations: ")
    print(abs(A11B11 + A12B12 + A21B21 - A22B22), 2 * (1 + delta))
    print(abs(A11B11 + A12B12 - A21B21 + A22B22), 2 * (1 + delta))
    print(abs(A11B11 - A12B12 + A21B21 + A22B22), 2 * (1 + delta))
    print(abs(-A11B11 + A12B12 + A21B21 + A22B22), 2 * (1 + delta))


accuracy_time(1000)
accuracy_time(2000)
accuracy_time(3000)
accuracy_time(4000)
accuracy_time(5000)
accuracy_time(6000)
accuracy_time(7000)
accuracy_time(8000)
accuracy_time(9000)
accuracy_time(10000)
accuracy_time(15000)
accuracy_time(20000)
accuracy_time(25000)
accuracy_time(30000)
accuracy_time(35000)
accuracy_time(40000)
accuracy_time(45000)
accuracy_time(50000)
accuracy_time(55000)
accuracy_time(60000)
accuracy_time(65000)
accuracy_time(70000)
accuracy_time(75000)
accuracy_time(80000)
accuracy_time(85000)
accuracy_time(90000)
accuracy_time(95000)
accuracy_time(100000)
