from pyro import sample
import torch
from numpy import zeros, array, fliplr, sum
from functools import reduce
from itertools import product
from pyro.distributions import Bernoulli, Uniform
import pprint
import sys

# get the foulis randall product as an array of values
def foulis_randall_product():
    fr_edges = []
    # hypergraphs are declared
    H = [ [[[0,0],[1,0]],[[0,1],[1,1]]], [[[0,0],[1,0]],[[0,1],[1,1]]] ]
    # append the hyperedges for the four explicit contexts
    for edge_a in H[0]:
        for edge_b in H[1]:
            fr_edge = []
            for vertex_a in edge_a:
                for vertex_b in edge_b:
                    fr_edge.append([ vertex_a[0], vertex_b[0], vertex_a[1], vertex_b[1]])
            fr_edges.append(fr_edge)
    # then append the hyperedges for both measurement choices (mc) e.g. alice -> bob OR bob -> alice
    # E_A->B \ E_A x E_B
    # E_B->A \ E_A x E_B
    for mc in range(0,2):
        mc_i = abs(1-mc)
        # for some edge in the first hypergraph
        for edge in H[mc]:
            for j in range(0,2):
                fr_edge = []
                # for length of said edge (where both vertices will imply two edges to form hyperedge)
                for i in range(0, len(edge)):
                    # run calculations to map vertex of said edge to vertices of other edge
                    edge_b = H[mc_i][i]
                    vertex_a = edge[abs(i-j)]
                    vertex_b = edge_b[0]
                    vertex_c = edge_b[1]
                    vertices_a = [vertex_a[0], vertex_b[0], vertex_a[1], vertex_b[1]]
                    vertices_b = [vertex_a[0], vertex_c[0], vertex_a[1], vertex_c[1]]
                    # fix two vertices to this edge (recall there are two edge implied by the for loop, constructing a hyperedge)
                    fr_edge.append([ vertices_a[mc], vertices_a[mc_i], vertices_a[mc+2], vertices_a[mc_i+2]])
                    fr_edge.append([ vertices_b[mc], vertices_b[mc_i], vertices_b[mc+2], vertices_b[mc_i+2]])
                # append constructed hyperedge to hypergraph
                fr_edges.append(fr_edge)
    return fr_edges

# returns a torch tensor of value v
def variable(v):
    return torch.autograd.Variable(torch.Tensor([v]))

# gets a vertex in global distribution by mapping values a b x y
def get_vertex(a, b, x, y):
    return (2*x+a, 2*y+b)

# gets hyperedges associated with array n
def get_hyperedges(H, n):
    l = []
    for idx, e in enumerate(H):
        if n in e:
            l.append(idx)
    return l

# reshapes an array
def organise(M):
    return M.reshape(4).tolist()

# generates the global distribution when given a constraint distribution, and a number of iterations to run
def generate_global_distribution(constraints,N):
    # get the hyperedges for this scenario
    hyperedges = foulis_randall_product()
    hyperedges_tallies = zeros(12)
    global_distribution = zeros((4, 4))
    # while the global distribution's sum of tallies is smaller than the number of iterations
    while sum(global_distribution) < N:
        # generate the samples that form the tallies for each vertex (or node) of the global distribution
        a = int(sample('A', Bernoulli(variable(0.5))))
        b = int(sample('B', Bernoulli(variable(0.5))))
        x = int(sample('X', Bernoulli(variable(0.5))))
        y = int(sample('Y', Bernoulli(variable(0.5))))
        # before adding any value to the global distribution, ensure that it won't disrupt the constraints
        value = float(sample('C', Uniform(variable(0.0), variable(1.0))))
        if (value < constraints[x][y][a,b]):
            #when adding a single tally to the global distribution, not only is the tally added to the distribution itself,
            #but also to indexes representing the hyperedges associated with the vertex being incremented.
            for edge in get_hyperedges(hyperedges, [a, b, x, y]):
                hyperedges_tallies[edge] += 1
            global_distribution[get_vertex(a, b, x, y)] += 1
    # upon completion of tallying, all values in the global distribution are normalised by the sum of the hyperedges
    # associated with them.
    for a, b, x, y in product(range(2), range(2), range(2), range(2)):
        summed_tally = (sum(hyperedges_tallies[e] for e in get_hyperedges(hyperedges, [a, b, x, y])))
        global_distribution[get_vertex(a, b, x, y)] /= summed_tally
    # then the global distribution is normalised once again (multiplied by 3, as each vertex has three hyperedges)
    # and the distribution is reorganised
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
