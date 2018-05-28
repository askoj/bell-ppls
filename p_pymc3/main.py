import ipdb
import pymc3 as pm
import collections

# sample size
N = 1000

# average
def avg(l):
	return float(sum(l))/len(l)

# context
def ct(t1, t2):
	t3 = []
	for i in range(0, len(t1)):
		t3.append(str(t1[i])+str(t2[i]))
	return collections.Counter(t3)

# isolated probability
def iP(name):
	return (float(collections.Counter(name)[1])/N)

# average probability
def aP(name_a, name_b):
	return avg([iP(name_a),iP(name_a)])

# logical and
def la(context):
	return (float(context['11'])/N)

def exp_1(contextual):
	'''
	
	Use Bayes Theorem on an acyclic join tree to determine contextuality
	arg: contextual (if True, replaces values for separate contexts)

	'''

	# rvs
	A1 = 0.6
	B1 = 0.5
	A2 = 0.2
	B2 = 0.3
	A3 = 0.5

	with pm.Model() as model:

		# context 1
		pm.Bernoulli('c1_A1', A1)
		pm.Bernoulli('c1_B1', B1)

		# context 2
		pm.Bernoulli('c2_A2', A2)
		pm.Bernoulli('c2_B1', (0.3 if contextual else B1))

		# context 3
		pm.Bernoulli('c3_A1', (0.3 if contextual else A1))
		pm.Bernoulli('c3_B2', B2)

		# context 4
		pm.Bernoulli('c4_A3', A3)
		pm.Bernoulli('c4_B2', (0.5 if contextual else B2))

		trace = pm.sample((N/2), step=pm.Metropolis()) 

	# return contexts
	c1 = ct(trace['c1_A1'],trace['c1_B1'])
	c2 = ct(trace['c2_A2'],trace['c2_B1']) 
	c3 = ct(trace['c3_A1'],trace['c3_B2']) 
	c4 = ct(trace['c4_A3'],trace['c4_B2']) 

	# get all isolated probabilities (and average probabilities for common rvs)
	pA1 = aP(trace['c1_A1'],trace['c3_A1'])
	pB1 = aP(trace['c1_B1'],trace['c2_B1'])
	pA2 = iP(trace['c2_A2'])
	pB2 = aP(trace['c3_B2'],trace['c4_B2'])
	pA3 = iP(trace['c4_A3'])

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

# executes 'non-contextual' empirical model
em_1 = exp_1(False)

def exp_2():
	'''
	
	Something to do with order effects

	'''

	# rvs
	A1 = 0.7
	B1 = 0.4

	with pm.Model() as model:

		# context 1
		c1_A1 = pm.Bernoulli('c1_A1', A1)
		pm.Bernoulli('c1_B1', pm.math.switch(c1_A1, 0.8, 0.1))

		# context 2
		c1_B1 = pm.Bernoulli('c2_B1', B1)
		pm.Bernoulli('c2_A1', pm.math.switch(c1_B1, 0.4, 0.6))

		trace = pm.sample((N/2), step=pm.Metropolis())

	# return contexts
	c1 = ct(trace['c1_A1'],trace['c1_B1'])
	c2 = ct(trace['c2_B1'],trace['c2_A1']) 

	return { 'c1' : c1, 'c2' : c2 }

# executes empirical model with order effects
em_2 = exp_2()


ipdb.set_trace()



