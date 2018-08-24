using Turing
using Distributions

function foulis_randall_product()
	fr_edges = Array{Array{Array{Float64}}}(0)
	H = [	[[[0.0,0.0],[1.0,0.0]],[[0.0,1.0],[1.0,1.0]]],
			[[[0.0,0.0],[1.0,0.0]],[[0.0,1.0],[1.0,1.0]]]	]
	for i = 1:size(H[1])[1]
		for j = 1:size(H[2])[1]
			fr_edge = Array{Array{Float64}}(0)
			for k = 1:size(H[1][i])[1]
				for l = 1:size(H[1][j])[1]
					append!( fr_edge, [[ H[1][i][k][1] , H[2][j][l][1] , H[1][i][k][2] , H[2][j][l][2] ]] )
				end
			end
			append!( fr_edges, [ fr_edge ] )
		end
	end
	for mc = 1:2
		mc_i = abs(3-mc)
		for k = 1:size(H[mc])[1]
			for j = 1:2
				fr_edge = Array{Array{Float64}}(0)
				for i = 1:size(H[mc][k])[1]
					edge_b = H[mc_i][i]
					vertex_a = H[mc][k][abs(i-j)+1]
					vertex_b = edge_b[1]
                    vertex_c = edge_b[2]
                    vertices_a = [vertex_a[1], vertex_b[1], vertex_a[2], vertex_b[2]]
                    vertices_b = [vertex_a[1], vertex_c[1], vertex_a[2], vertex_c[2]]
                    this_edge_b = Array{Float64}(0)
                    append!( fr_edge, [[ vertices_a[mc], vertices_a[mc_i], vertices_a[mc+2], vertices_a[mc_i+2] ]] )
                    append!( fr_edge, [[ vertices_b[mc], vertices_b[mc_i], vertices_b[mc+2], vertices_b[mc_i+2] ]] )
				end
				append!(fr_edges, [ fr_edge ])
			end
		end
	end
	fr_edges
end

function get_vertex(a,b,x,y)
	(x == 1 && y == 0 ? 4 : (x == 0 && y == 1 ? 8 : (x == 1 && y == 1 ? 12 : 0)))+
	(a == 0 && b == 0 ? 1 : (a == 1 && b == 0 ? 2 : (a == 0 && b == 1 ? 3 : 4)))
end

function float(n)
	convert(Float64,n)
end

function get_hyperedges(H, n)
	l = []
	for i = 1:size(H)[1]
	    if any(x->x==n, H[i])
	        append!(l,i)
	    end
	end
	l
end

@model mdl() = begin
	z ~ Beta(1,1)
	a ~ Bernoulli(0.5)
	b ~ Bernoulli(0.5)
	x ~ Bernoulli(0.5)
	y ~ Bernoulli(0.5)
	c ~ Uniform(0.0, 1.0)
end

function generate_global_distribution(constraints,N)
	hyperedges = foulis_randall_product()
	hyperedges_tallies = zeros(12)
	global_distribution = zeros(16)
	while sum(global_distribution) < N
		r = sample(mdl(), SMC(N))
		a = r[:a]
		b = r[:b]
		x = r[:x]
		y = r[:y]
		c = r[:c]
		for i = 1:N
			if (c[i] < constraints[x[i]+1][y[i]+1][a[i]+1][b[i]+1])
				associated_hyperedges = get_hyperedges(hyperedges, [float(a[i]), float(b[i]), float(x[i]), float(y[i])])
				for j = 1:size(associated_hyperedges)[1]
					hyperedges_tallies[associated_hyperedges[j]] += 1
				end
				global_distribution[get_vertex(a[i], b[i], x[i], y[i])] += 1
			end
		end
	end
	for a = 1:2
		for b = 1:2
			for x = 1:2
				for y = 1:2
					A = a-1
					B = b-1
					X = x-1
					Y = y-1
					summed_amount = 0
					associated_hyperedges = get_hyperedges(hyperedges, [ float(A), float(B), float(X), float(Y)])
					for edge_index = 1:size(associated_hyperedges)[1]
						summed_amount += hyperedges_tallies[edge_index]
					end
					global_distribution[get_vertex(A, B, X, Y)] /= summed_amount
				end
			end
		end
	end
	global_distribution = global_distribution .* 3
	prepend!(global_distribution, [0.0])
	global_distribution
end

constraints = [[ [[0.5, 0.0], [0.0, 0.5]], [[0.5, 0.0], [0.0, 0.5]] ], [ [[0.5, 0.0], [0.0, 0.5]], [[0.0, 0.5], [0.5, 0.0]] ]]

println(generate_global_distribution(constraints,5000))