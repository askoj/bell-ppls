using Turing
using Distributions
using Gadfly
using Mamba: describe, plot


@model simple_choice() = begin
	b ~ Beta(1, 1)
	
	p = (rand([0,100])/100)
	if p > 0.5
		y ~ Bernoulli(1)
	else
		y ~ Bernoulli(0)
	end
end

chain = sample(simple_choice(), HMC(300, 0.1, 5))
#describe(chain)

var_0 = chain[:y]
#var_1 = chain[:a]
#var_2 = chain[:z]
println(var_0)
#println(mean(var_0))
#println(var_1)
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