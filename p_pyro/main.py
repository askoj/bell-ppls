import pyro
import torch
import pyro.distributions as dist
import sys

import time

start_time = time.time()


def context(R,N):
	A = []
	for r in R:
		p = torch.autograd.Variable(torch.Tensor([r[1]]))
		A.append([int(pyro.sample(r[0],dist.Bernoulli(p))) for j in range(N)])
	P = [0.0] * 4
	for c in range(0, N):
		a = A[0][c]
		b = A[1][c]
		P[(a==b and 2 or 0)-(b==0 and 2 or 3)%4] += 1
	for p in range(0, len(P)):
		P[p] /= N
	return P


def record():

	global_distribution = []
	global_distribution.append(context([['A1',0.6],['B1',0.5]],50000))
	global_distribution.append(context([['A1',0.6],['B2',0.3]],50000))
	global_distribution.append(context([['A2',0.2],['B1',0.5]],50000))
	global_distribution.append(context([['A2',0.2],['B2',0.3]],50000))
	global_distribution = reduce(lambda x,y :x+y ,global_distribution)
	p = global_distribution

	print(p)

	def signalling(a,b,c,d):
		print(str((p[a]+p[b])) + " == " + str((p[c]+p[d])))
		print(abs((p[a]+p[b])-(p[c]+p[d])))
		return abs((p[a]+p[b])-(p[c]+p[d])) < 0.01

	def equality(v1,v2,v3,v4):
		def f1(v1,v2):
			return abs((2 * (p[v1] + p[v2])) - 1)
		def f2(v1,v2,v3,v4):
			return (p[v1] + p[v2]) - (p[v3] + p[v4])
		delta = 0.5 * ( 
			(f1(0,1) - f1(4,5)) + (f1(8,9) - f1(12,13)) + 
			(f1(0,2) - f1(4,6)) + (f1(8,10) - f1(12,14)))
		return 2 * (1 + delta) >= abs(
			(v1*f2(0,3,1,2)) + (v2*f2(4,7,5,6)) +
			(v3*f2(8,11,9,10)) + (v4*f2(12,15,13,14)))

	tests = [
		signalling(0,1,4,5),
		signalling(8,9,12,13),
		signalling(0,2,8,10),
		signalling(4,6,12,14),
		equality(1,1,1,-1),
		equality(1,1,-1,1),
		equality(1,-1,1,1),
		equality(-1,1,1,1)
		]

	print(tests)

record()

print("--- %s seconds ---" % (time.time() - start_time))