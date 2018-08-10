import pyro
import torch
import numpy as np
from functools import reduce
from itertools import product
import pyro.distributions as dist

def V(v):
    return torch.autograd.Variable(torch.Tensor([v]))

def s_to_n(a, b, x, y):
    return (2*x+a, 2*y+b)

def edges(H, n):
    l = []
    for idx, e in enumerate(H):
        if n in e:
            l.append(idx)
    return l

def mutate(M):
    return M.reshape(4).tolist()

def fr_product():
    a = 2
    j = []
    k = (a*a)-1
    s = [[x, y] for y in range(a) for x in range(a)]
    for g in range(0, a*a):
        for h in range(0, a*a):
            l = []
            if h != k:
                y = s[g]
                z = s[h]
                for i in range(0, len(s)):
                    x = s[i]
                    l.append(x+y) if i < a else l.append(x+z)
                j.append(l)
        k -= 1
    return j

def gl_dist(N):
    H = fr_product()
    N_e = np.zeros(12)
    D = np.zeros((4, 4))
    i_a = [0.5, 0.0]
    M_a = np.array([i_a, i_a[::-1]])
    c = [[M_a, M_a],[M_a, np.fliplr(M_a)]]
    while np.sum(D) < N:
        S = []
        for i in range(0, 4):
            S.append(int(pyro.sample(str(i), dist.Bernoulli(V(0.5)))))
        if float(pyro.sample('C', dist.Uniform(V(0.0), V(1.0)))) < c[S[2]][S[3]][S[0], S[1]]:
            for e in edges(H, [S[0], S[1], S[2], S[3]]):
                N_e[e] += 1
            D[s_to_n(S[0], S[1], S[2], S[3])] += 1
    c_r = range(2)
    for S[0], S[1], S[2], S[3] in product(c_r, c_r, c_r, c_r):
        D[s_to_n(S[0], S[1], S[2], S[3])] /= (sum(N_e[e] for e in edges(H, [S[0], S[1], S[2], S[3]])))
    D *= 3
    D = [0] + reduce(lambda x, y: x + y, 
        [mutate(D[:2, :2]), mutate(D[:2, 2:]), mutate(D[2:, :2]), mutate(D[2:, 2:])])
    return D


p = gl_dist(5000)

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
