using Turing
using Distributions

tic()

function context(rvs, itr)
	@model mdl(p) = begin
		z ~ Beta(1,1)
		x ~ Bernoulli(p[1])
		y ~ Bernoulli(p[2])
	end
	p = [0.0,0.0,0.0,0.0]
	d = Array{Array}(2)
	r = sample(mdl(rvs), SMC(itr))
	d[1] = r[:x]
	d[2] = r[:y]
	for i = 1:itr
		p[mod(((d[1][i] == d[2][i] ? 2 : 0)-(d[2][i] == 0 ? 2 : 3)),4)+1] += 1
	end
	for i = 1:4
		p[i] /= itr
	end
	p
end

A1 = 0.6
A2 = 0.2
B1 = 0.5
B2 = 0.3

global_distribution = zeros(0)
append!( global_distribution , context([ A1 , B1 ], 50000) )
append!( global_distribution , context([ A1 , B2 ], 50000) )
append!( global_distribution , context([ A2 , B1 ], 50000) )
append!( global_distribution , context([ A2 , B2 ], 50000) )
p = global_distribution

println(p)


function signalling(a,b,c,d)
	println(string("	(",string(p[a] + p[b]),") = (",string(p[c] + p[d]),")"))
	println(abs((p[a]+p[b])-(p[c]+p[d])))
	abs((p[a]+p[b])-(p[c]+p[d])) < 0.01
end

function equality(v1,v2,v3,v4)
	function f1(v1,v2)
		abs((2 * (p[v1] + p[v2])) - 1)
	end
	function f2(v1,v2,v3,v4)
		(p[v1] + p[v2]) - (p[v3] + p[v4])
	end
	delta = 0.5 * ( 
		(f1(1,2) - f1(5,6)) + (f1(9,10) - f1(13,14)) + 
		(f1(1,3) - f1(5,7)) + (f1(9,11) - f1(13,15)))
	(2 * (1 + delta)) >= abs(
		(v1*f2(1,4,2,3)) + (v2*f2(5,8,6,7)) +
		(v3*f2(9,12,10,11)) + (v4*f2(13,16,14,15)))
end

tests = [
	signalling(1,2,5,6),
	signalling(9,10,13,14),
	signalling(1,3,9,11),
	signalling(5,7,13,15),
	equality(1,1,1,-1),
	equality(1,1,-1,1),
	equality(1,-1,1,1),
	equality(-1,1,1,1)
	]

println(tests)

toc()