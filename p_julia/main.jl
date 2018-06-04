using Turing
using Distributions

################################################################################################
################################################################################################
################################################################################################
################################################################################################

@model bmdl(p) = begin
  z ~ Beta(1,1)
  x ~ Bernoulli(p)
  x
end

function main(rvs, itr)
	d = Array{Array}(length(rvs))
	for i = 1:length(rvs)
		r = sample(bmdl(rvs[i]), SMC(itr))
		d[i] = r[:x]
	end
	apt = Array{Array}(length(rvs))
	j = 0
	for a = 1:2
		for b = 3:4
			j += 1
			pt = [0.0,0.0,0.0,0.0]
			for i = 1:itr
				pt[(d[a][i] == d[b][i] ? -2 : 0)+(d[a][i] == 0 ? 3 : 4)] += 1
			end
			for i = 1:length(pt)
				pt[i] /= itr
			end
			apt[j] = pt
		end
	end
	apt
end

# A1, A2, B1, B2

data = main([ 0.6, 0.2, 0.5, 0.3 ], 100000)

################################################################################################
################################################################################################
################################################################################################
################################################################################################

p = zeros(0)
for i = 1:4
	append!( p , data[i][1] )
	append!( p , data[i][3] )
	append!( p , data[i][4] )
	append!( p , data[i][2] )
end

# Set Up

println("P-Array")
for i = 1:16
	println(string("\n	p",string(i)," = ",string(p[i])))
end
println("\n")

# No Signalling

function no_signalling_test(val1, val2, val3, val4)
	println("\n")
	if ((val1 + val2) == (val3 + val4))
		println(string("	(",string(val1)," + ",string(val2),") = (",string(val3)," + ",string(val4),")"))
		println(string("	(",string((val1+val2)),") = (",string((val3+val4)),")"))
		println("	Signalling Test Passed!")
	else
		println(string("	(",string(val1)," + ",string(val2),") <> (",string(val3)," + ",string(val4),")"))
		println(string("	(",string((val1+val2)),") <> (",string((val3+val4)),")"))
		println("	Signalling Test Failed!")
	end
end

println("Signalling Tests")
no_signalling_test(p[1],p[2],p[5],p[6])
no_signalling_test(p[9],p[10],p[13],p[14])
no_signalling_test(p[1],p[3],p[9],p[11])
no_signalling_test(p[5],p[7],p[13],p[15])
println("\n")

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

println("Bell Scenario Experiment")
println(string("\n	A11 = ",string(A11)))
println(string("\n	A12 = ",string(A12)))
println(string("\n	A21 = ",string(A21)))
println(string("\n	A22 = ",string(A22)))
println(string("\n	B11 = ",string(B11)))
println(string("\n	B12 = ",string(B12)))
println(string("\n	B21 = ",string(B21)))
println(string("\n	B22 = ",string(B22)))
println(string("\n	Delta = ",string(delta)))



if delta >= 1
	println("	Delta is greater than or equal to 1 so contextuality cannot occur\n")
else
	println("	Delta is smaller than 1 so contextuality may probably occur\n")
end

A11B11 = (p[1] + p[4]) - (p[2] + p[3])
A12B12 = (p[5] + p[8]) - (p[6] + p[7])
A21B21 = (p[9] + p[12]) - (p[10] + p[11])
A22B22 = (p[13] + p[16]) - (p[14] + p[15])

if (A11B11 + A12B12 + A21B21 - A22B22) <= 2*(1+delta)
	println(string("	(",A11B11," + ",A12B12," + ",A21B21," - ",A22B22,") <= 2*(1+",delta,")"))
	println(" 	Condition 13 Passed!\n")
else
	println(string("	(",A11B11," + ",A12B12," + ",A21B21," - ",A22B22,") is not <= 2*(1+",delta,")"))
	println(" 	Condition 13 Failed!\n")
end

if (A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+delta)
	println(string("	(",A11B11," + ",A12B12," - ",A21B21," + ",A22B22,") <= 2*(1+",delta,")"))
	println(" 	Condition 14 Passed!\n")
else
	println(string("	(",A11B11," + ",A12B12," - ",A21B21," + ",A22B22,") is not <= 2*(1+",delta,")"))
	println(" 	Condition 14 Failed!\n")
end

if (A11B11 - A12B12 + A21B21 + A22B22) <= 2*(1+delta)
	println(string("	(",A11B11," - ",A12B12," + ",A21B21," + ",A22B22,") <= 2*(1+",delta,")"))
	println(" 	Condition 15 Passed!\n")
else
	println(string("	(",A11B11," - ",A12B12," + ",A21B21," + ",A22B22,") is not <= 2*(1+",delta,")"))
	println(" 	Condition 15 Failed!\n")
end

if (0 - A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+delta)
	println(string("	( - ",A11B11," + ",A12B12," - ",A21B21," + ",A22B22,") <= 2*(1+",delta,")"))
	println(" 	Condition 16 Passed!\n")
else
	println(string("	( - ",A11B11," + ",A12B12," - ",A21B21," + ",A22B22,") is not <= 2*(1+",delta,")"))
	println(" 	Condition 16 Failed!\n")
end








