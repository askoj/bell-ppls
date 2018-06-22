import pymc3 as pm
import collections as cl
import json
import sys
import time

start_time = time.time()






	# NEW SCOPE FUNCTION

'''
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
'''
'''
contexts = []
contexts.append( context([['A1',0.6],['B1',0.5]],10000) )
contexts.append( context([['A1',0.6],['B2',0.3]],10000) )
contexts.append( context([['A2',0.2],['B1',0.5]],10000) )
contexts.append( context([['A2',0.2],['B2',0.3]],10000) )

'''


################################################################################################
################################################################################################
################################################################################################
################################################################################################


# Scope with non-persistent samples

def context(R,N):
	A = []
	with pm.Model():
		for r in R:
			pm.Bernoulli(r[0],r[1])
			A.append(pm.sample(N,tune=0, step=pm.Metropolis()).get_values(r[0]))
	P = [0.0] * 4
	for c in range(0, N):
		a = A[0][c]
		b = A[1][c]
		P[(a==b and 2 or 0)-(b==0 and 2 or 3)%4] += 1
	for p in range(0, len(P)):
		P[p] /= N
	return P



################################################################################################
################################################################################################
################################################################################################
################################################################################################


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
	

	'''

	for j in range(1):
		p = reduce(lambda x,y :x+y ,contexts())
		p.insert(0,0)
		n_s_t_1 = (no_signalling_test(p[1],p[2],p[5],p[6]))
		n_s_t_2 = (no_signalling_test(p[9],p[10],p[13],p[14])) 
		n_s_t_3 = (no_signalling_test(p[1],p[3],p[9],p[11])) 
		n_s_t_4 = (no_signalling_test(p[5],p[7],p[13],p[15]))

		

		A11 = (2 * (p[1] + p[2])) - 1
		A12 = (2 * (p[5] + p[6])) - 1
		A21 = (2 * (p[9] + p[10])) - 1
		A22 = (2 * (p[13] + p[14])) - 1
		B11 = (2 * (p[1] + p[3])) - 1
		B12 = (2 * (p[9] + p[11])) - 1
		B21 = (2 * (p[5] + p[7])) - 1
		B22 = (2 * (p[13] + p[15])) - 1
		delta = 0.5 * ( abs( A11 - A12 ) + abs( A21 - A22 ) + abs( B11 - B21 ) + abs( B12 - B22 ) )
		A11B11 = (p[1] + p[4]) - (p[2] + p[3])
		A12B12 = (p[5] + p[8]) - (p[6] + p[7])
		A21B21 = (p[9] + p[12]) - (p[10] + p[11])
		A22B22 = (p[13] + p[16]) - (p[14] + p[15])
		for i in range(0, 17):
			output += str(p[i]) + ","
		output += str(n_s_t_1) + ","
		output += str(n_s_t_2) + ","
		output += str(n_s_t_3) + ","
		output += str(n_s_t_4) + ","
		output += str(delta) + ","
		output += str((delta >= 1)) + ","
		output += str(A11B11) + "," + str(A12B12) + "," + str(A21B21) + "," + str(A22B22) + ","
		output += str((abs(A11B11 + A12B12 + A21B21 - A22B22) <= 2*(1+delta))) + ","
		output += str((abs(A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+delta))) + ","
		output += str((abs(A11B11 - A12B12 + A21B21 + A22B22) <= 2*(1+delta))) + ","
		output += str((abs(0 - A11B11 + A12B12 + A21B21 + A22B22) <= 2*(1+delta)))
		output += "\n"
		'''
	'''
	text_file = open("new.csv", "w")
	text_file.write(output)
	text_file.close()
	'''

record()

print("--- %s seconds ---" % (time.time() - start_time))