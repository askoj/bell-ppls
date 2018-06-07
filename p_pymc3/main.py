import pymc3 as pm
import collections as cl
import json

################################################################################################
################################################################################################
################################################################################################
################################################################################################

def main(rvs,N):
	r = []
	with pm.Model() as model: # run the model
		for k in rvs:
			pm.Bernoulli(k,rvs[k])
		t = pm.sample((N/2), step=pm.Metropolis()) 
	q = list(rvs)
	for a in range(2):
		for b in range(2): # create the context combinations
			p = [0.0,0.0,0.0,0.0]
			for c in range(0, len(t[q[a]])):
				p[(t[q[a]][c]==t[q[b+2]][c] and -2 or 0)+(t[q[a]][c]==0 and 2 or 3)] += 1
			for f in range(0, len(p)): # convert tallies to probabilities
				p[f] /= N
			r.append(p)
	return r # return all results

p_tables = main(cl.OrderedDict(( ('A1',0.6), ('B1',0.5), ('A2',0.2), ('B2',0.3) )),1000)

################################################################################################
################################################################################################
################################################################################################
################################################################################################

	# NEW SCOPE FUNCTION


T = {}
def context(R,N):
	for r in R:
		if r[0] not in T:
			with pm.Model():
				pm.Bernoulli(r[0],r[1])
				T.update({r[0] : pm.sample(N,tune=0,
					step=pm.Metropolis()).get_values(r[0])})
	P = [0.0] * 4
	for c in range(0, N):
		a = T[R[0][0]][c]
		b = T[R[1][0]][c]
		P[(a==b and 2 or 0)-(b==0 and 2 or 3)%4] += 1
	for p in range(0, len(P)):
		P[p] /= N
	return P

contexts = []
contexts.append( context([['A1',0.6],['B1',0.5]],10000) )
contexts.append( context([['A1',0.6],['B2',0.3]],10000) )
contexts.append( context([['A2',0.2],['B1',0.5]],10000) )
contexts.append( context([['A2',0.2],['B2',0.3]],10000) )

################################################################################################
################################################################################################
################################################################################################
################################################################################################


def no_signalling_test(val1,val2,val3,val4):
	if ((val1 + val2) == (val3 + val4)):
		return True
	else:
		return False


'''
x - 0 y - 0 = p[0]
x - 1 y - 1 = p[1]
x - 0 y - 1 = p[2]
x - 1 y - 0 = p[3]
'''


wrString = ''

for j in range(1):
	a = 1
	'''
	p_tables = main(cl.OrderedDict(( ('A1',0.6), ('B1',0.5), ('A2',0.2), ('B2',0.3) )),1000)

	print(p_tables)

	p = [0.0]
	for i in range(0,4):
		p.append(p_tables[i][0])
		p.append(p_tables[i][2])
		p.append(p_tables[i][3])
		p.append(p_tables[i][1])

	print(no_signalling_test(p[1],p[2],p[5],p[6]))
	print(no_signalling_test(p[9],p[10],p[13],p[14])) 
	print(no_signalling_test(p[1],p[3],p[9],p[11])) 
	print(no_signalling_test(p[5],p[7],p[13],p[15]))
	'''
	'''

	p = [0.0]
	for i in range(0,4):
		p.append(p_tables[i][0])
		p.append(p_tables[i][2])
		p.append(p_tables[i][3])
		p.append(p_tables[i][1])

	A11 = (2 * (p[1] + p[2])) - 1
	A12 = (2 * (p[5] + p[6])) - 1
	A21 = (2 * (p[9] + p[10])) - 1
	A22 = (2 * (p[13] + p[14])) - 1
	B11 = (2 * (p[1] + p[3])) - 1
	B12 = (2 * (p[9] + p[11])) - 1
	B21 = (2 * (p[5] + p[7])) - 1
	B22 = (2 * (p[13] + p[15])) - 1
	delta = ((abs(A11) - abs(A12)) + (abs(A21) - abs(A22)) + (abs(B11) - abs(B12)) + (abs(B21) - abs(A22))) / 2
	A11B11 = (p[1] + p[4]) - (p[2] + p[3])
	A12B12 = (p[5] + p[8]) - (p[6] + p[7])
	A21B21 = (p[9] + p[12]) - (p[10] + p[11])
	A22B22 = (p[13] + p[16]) - (p[14] + p[15])

	for i in range(1, 16):
		wrString += str(p[i]) + ","
	wrString += str(no_signalling_test(p[1],p[2],p[5],p[6])) + ","
	wrString += str(no_signalling_test(p[9],p[10],p[13],p[14])) + ","
	wrString += str(no_signalling_test(p[1],p[3],p[9],p[11])) + ","
	wrString += str(no_signalling_test(p[5],p[7],p[13],p[15])) + ","
	wrString += str(delta) + ","
	wrString += str((delta >= 1)) + ","
	wrString += str(A11B11) + "," + str(A12B12) + "," + str(A21B21) + "," + str(A22B22) + ","
	wrString += str(((A11B11 + A12B12 + A21B21 - A22B22) <= 2*(1+delta))) + ","
	wrString += str(((A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+delta))) + ","
	wrString += str(((A11B11 - A12B12 + A21B21 + A22B22) <= 2*(1+delta))) + ","
	wrString += str(((0 - A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+delta)))
	#wrString += "\n"
	'''

#text_file = open("new.csv", "w")
#text_file.write(wrString)
#text_file.close()