import pyro
import ipdb
import torch
from torch.autograd import Variable
from pyro.distributions import Bernoulli
import collections

#http://pyro.ai/examples/svi_part_i.html
#https://gist.github.com/rosinality/708209a317efe05e5131cdddf29f2ac5

N = 10000

# context
def ct(t1, t2):
	t3 = []
	for i in range(0, len(t1)):
		t3.append(str(t1[i])+str(t2[i]))
	return collections.Counter(t3)

def flip(sample_name, num_samples, bias):
	return [int(pyro.sample(sample_name, Bernoulli(Variable(torch.Tensor([bias]))))) for i in range(num_samples)]

# isolated probability
def iP(name):
	return (float(collections.Counter(name)[1])/N)

# logical and
def la(context):
	return (float(context['11'])/N)

def exp_1():

	# rvs
	A1 = 0.6
	B1 = 0.5
	A2 = 0.2
	B2 = 0.3
	A3 = 0.5

	# context 1
	c1_A1_dist = flip('c1_A1', N, A1)
	c1_B1_dist = flip('c1_B1', N, B1)

	# context 2
	c2_A2_dist = flip('c2_A2', N, A2)
	c2_B1_dist = c1_B1_dist

	# context 3
	c3_A1_dist = c1_A1_dist
	c3_B2_dist = flip('c3_B2', N, B2)

	# context 4
	c4_A3_dist = flip('c4_A3', N, A3)
	c4_B2_dist = c3_B2_dist

	# return contexts
	c1 = ct(c1_A1_dist,c1_B1_dist)
	c2 = ct(c2_A2_dist,c2_B1_dist) 
	c3 = ct(c3_A1_dist,c3_B2_dist) 
	c4 = ct(c4_A3_dist,c4_B2_dist)

	# get all isolated probabilities
	pA1 = iP(c1_A1_dist)
	pB1 = iP(c2_B1_dist)
	pA2 = iP(c2_A2_dist)
	pB2 = iP(c3_B2_dist)
	pA3 = iP(c4_A3_dist) 

	# get 'logical and' of contexts
	pA1_a_B1 = la(c1)
	pA2_a_B1 = la(c2)
	pA1_a_B2 = la(c3)
	pA3_a_B2 = la(c4)

	# bayes theorem LHS
	bt_l = (pA1*pB1*pA2*pB2*pA3)

	# bayes theorem RHS
	bt_r = (pA1_a_B1*pA2_a_B1*pA1_a_B2*pA3_a_B2)/(pB1*pA1*pB2)

	return { 'bt_l' : bt_l, 'bt_r' : bt_r }

print(exp_1())

ipdb.set_trace()