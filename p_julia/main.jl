using Turing
using Distributions

@model bmdl(p) = begin
  z ~ Beta(1,1)
  x ~ Bernoulli(p)
  x
end

function main(rvs)
	itr = 300
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
				x = d[a][i]
				y = d[b][i]
				if x == y && x == 0
					pt[1] += 1
				elseif x == y && x == 1
					pt[2] += 1
				elseif x != y && x == 0
					pt[3] += 1
				elseif x != y && x == 1
					pt[4] += 1
				end
			end
			for i = 1:length(pt)
				pt[i] /= itr
			end
			apt[j] = pt
		end
	end
	apt
end

vals = [ 0.6, 0.2, 0.5, 0.3 ] # A1, A2, B1, B2
println(main(vals))


#=

using Turing
using Distributions: Bernoulli
using Gadfly
using Mamba: describe, plot
using DualNumbers
using ForwardDiff

immutable Flat <: ContinuousUnivariateDistribution
end

Distributions.rand(d::Flat) = rand()
Distributions.logpdf{T<:Real}(d::Flat, x::T) = zero(x)

Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf




ta = TArray(Float64, 0)
#a = 0
mf(vi, sampler) = begin
	s = Turing.assume(sampler,
					Beta(1, 1),
					Turing.VarName(vi, [:c_s, :s], ""),
					vi)
	#push!(ta, eltype(Turing.observe(sampler,Bernoulli(0.5),0,vi))[1])

	a = Turing.observe(sampler,Bernoulli(0.5),0.5,vi)
	if typeof(a) != Float64
		#b = getindex(ForwardDiff.partials(a).values,1)
		b = a
		println(b)
	end
	#println(a)

	#push! ta
	vi
end

mf() = mf(Turing.VarInfo(), nothing, ta)

chain = sample(mf, HMC(1000, 0.1, 5))
println(ta)
#describe(chain)

#var_ = chain[:q]



function ds(d)
	a = length(find(d .== 1))/length(d)
	[a, 1-a]
end


Dual{Void}
(-Inf,
-0.8665862468851546,


0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)


p = (rand([0,100])/100)
	if p > 0.5
		y ~ Bernoulli(1)
	else
		y ~ Bernoulli(0)
	end


x = Array(Float64, 0)
@model simple_choice(x) = begin
	b ~ Beta(1, 1)
    push!(x, rand(Bernoulli(0.5),1))
end

chain = sample(simple_choice(x), HMC(300, 0.1, 5))

const berstandata = [
  Dict(
  "N" => 10,
  "y" => [0,1,0,0,0,0,0,0,0,1]
  )
]

@model bermodel(y) = begin
 theta ~ Beta(1,1)
 for n = 1:length(y)
   y[n] ~ Bernoulli(theta)
 end
 return theta
end

chain = sample(bermodel(berstandata[1]), HMC(300, 0.25, 5))

describe(chain)

println(berstandata)


y = []
@model simple_choice(y) = begin
	b ~ Beta(1, 1)
	y ~ logpdf(Bernoulli(0.5),0.5)
end

chain = sample(simple_choice(y), HMC(300, 0.1, 5))

var_1 = y#chain[:y]
#var_2 = chain[:z]

#println(mean(var_0))
println(var_1)
#println(var_2)

#k ~ Binomial(p, q)
#p ~ Beta(1, 1)
#q ~ Beta(1, 1)
#var_1 = chain[:k]
#println(mean(var_1))
#println(var_1)
#p2 = plot(chain);  draw(PNG(15cm, 10cm), gridstack([p2[1] p2[2]; p2[9] p2[10]]));
#@model model(N) = begin
#	α ~ Beta(1, 1)
#	θ = Vector{Vector{Real}}(N)
#	θ[m] ~ Bernoulli(α)
#end

#cflip = model(1)
#chain = sample(cflip, HMC(1000, 0.1, 5))
#describe(cflip)

#@model simple_choice() = begin
#	p ~ Beta(1, 1)
#	z ~ Bernoulli(p)
#	if z == 1
#		x ~ Normal(0, 1)
#	else
#		x ~ Normal(1, 1)
#	end
#end

#simple_choice_f = simple_choice

#chain = sample(simple_choice(), HMC(1000, 0.1, 5))
#println(chain)
#describe(chain)

#S = 1     # number of samplers
#N = 2000
#spls = [HMC(N,0.005,5)][1:S]

#c = sample(gdemo([1.5, 2]), spls)

#obs = [0, 1, 0, 1, 0, 0, 0, 0, 0, 1] # the observations

#@model betabinomial begin
#	@assume p ∼ Beta(1, 1) # define the prior
#	for i = 1:length(obs)
#		@observe obs[i] ∼ Bernoulli(p) # observe data points
#	end
#	@predict p # predict of p
#end

=#