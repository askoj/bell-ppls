import pyro
import ipdb
import torch
from torch.autograd import Variable
from pyro.distributions import Bernoulli
import collections

################################################################################################
################################################################################################
################################################################################################
################################################################################################

def main(rvs,N):
	d = []
	for i in range(4):
		d.append([int(pyro.sample(str(i), Bernoulli(Variable(torch.Tensor([rvs[i]]))))) for j in range(N)])
	r = []
	for a in range(2):
		for b in range(2):
			p = [0.0,0.0,0.0,0.0]
			for i in range(0, N):
				p[(d[a][i]==d[b+2][i] and -2 or 0)+(d[a][i] == 0 and 2 or 3)] += 1
			for f in range(0, len(p)):
				p[f] /= N
			r.append(p)
	return r

p_tables = main([0.6,0.5,0.2,0.3],100000)

################################################################################################
################################################################################################
################################################################################################
################################################################################################


p = [0.0]
for i in range(0,4):
	p.append(p_tables[i][0])
	p.append(p_tables[i][2])
	p.append(p_tables[i][3])
	p.append(p_tables[i][1])

# No Signalling

def no_signalling_test(val1,val2,val3,val4):
	if ((val1 + val2) == (val3 + val4)):
		print('No signalling condition passed')
	else:
		print('No signalling condition failed')

no_signalling_test(p[1],p[2],p[5],p[6])
no_signalling_test(p[9],p[10],p[13],p[14])
no_signalling_test(p[1],p[3],p[9],p[11])
no_signalling_test(p[5],p[7],p[13],p[15])

# Bell Scenario

A11 = (2 * (p[1] + p[2])) - 1
A12 = (2 * (p[5] + p[6])) - 1
A21 = (2 * (p[9] + p[10])) - 1
A22 = (2 * (p[13] + p[14])) - 1
B11 = (2 * (p[1] + p[3])) - 1
B12 = (2 * (p[9] + p[11])) - 1
B21 = (2 * (p[5] + p[7])) - 1
B22 = (2 * (p[13] + p[15])) - 1
delta = ((abs(A11) - abs(A12)) + (abs(A21) - abs(A22)) + (abs(B11) - abs(B12)) + (abs(B21) - abs(A22))) / 2

if delta >= 1:
	print("Delta is greater than or equal to 1")
else:
	print("Delta is smaller than 1 so contextuality may probably occur")

A11B11 = (p[1] + p[4]) - (p[2] + p[3])
A12B12 = (p[5] + p[8]) - (p[6] + p[7])
A21B21 = (p[9] + p[12]) - (p[10] + p[11])
A22B22 = (p[13] + p[16]) - (p[14] + p[15])

if (A11B11 + A12B12 + A21B21 - A22B22) <= 2*(1+delta):
	print("Bell scenario test 1 passed")
else:
	print("Bell scenario test 1 failed")

if (A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+delta):
	print("Bell scenario test 2 passed")
else:
	print("Bell scenario test 2 failed")

if (A11B11 - A12B12 + A21B21 + A22B22) <= 2*(1+delta):
	print("Bell scenario test 3 passed")
else:
	print("Bell scenario test 3 failed")

if (0 - A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+delta):
	print("Bell scenario test 4 passed")
else:
	print("Bell scenario test 4 failed")



