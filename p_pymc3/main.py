import ipdb
import pymc3 as pm
import collections


# sample size
N = 1000

# context
def c(x, y):
	pt = [0.0,0.0,0.0,0.0]
	for i in range(0, len(x)):
		a = x[i] == y[i] and -2 or 0
		b = x[i] == 0 and 2 or 3
		pt[a+b] += 1

	for i in range(0, len(pt)):
		pt[i] /= N

	return pt

def main(rvs):

	contexts = []

	with pm.Model() as model:
		for k in rvs:
			pm.Bernoulli(k,rvs[k])
		t = pm.sample((N/2), step=pm.Metropolis()) 

	Dl = list(rvs)
	for a in range(2):
		for b in range(2):
			p = [0.0,0.0,0.0,0.0]
			contexts.append(c(t[Dl[a]],t[Dl[b+2]]))

	print(contexts)



# executes 'non-contextual' empirical model
main(collections.OrderedDict(( ('A1',0.6), ('B1',0.5), ('A2',0.2), ('B2',0.3) )))
