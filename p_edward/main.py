from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


'''
import edward as ed
import tensorflow as tf
import ipdb
from edward.models import Bernoulli, Normal, Empirical, Beta
import numpy as np
FLAGS = tf.flags.FLAGS
'''
'''
N = 1500

bbbb = Bernoulli(probs=[0.9, 0.5], sample_shape=N)

print("Shape = Sample + Batch + Event: {}".format(bbbb.shape))
print("Sample shape: {}".format(bbbb.sample_shape))
print("Batch shape: {}".format(bbbb.batch_shape))
print("Event shape: {}".format(bbbb.event_shape))
ipdb.set_trace()
'''
'''
mu = Normal(loc=0.0, scale=1.0)
qmu = Empirical(tf.Variable(tf.zeros(1000)))

pi = Normal(loc=0.0, scale=1.0)
qpi = Empirical(tf.Variable(tf.zeros(1000)))

sigma = Normal(loc=0.0, scale=1.0)
qsigma = Empirical(tf.Variable(tf.zeros(1000)))

c = Normal(loc=0.0, scale=1.0)
qc = Empirical(tf.Variable(tf.zeros(1000)))

x = Normal(loc=mu, scale=0.1, sample_shape=1)
proposal_mu = Normal(loc=mu, scale=1.0)
inference = ed.MetropolisHastings(
	latent_vars={pi: qpi, mu: qmu, sigma: qsigma, c: qc},
	proposal_vars={pi: gpi, mu: gmu, sigma: gsigma, c: gc}, 
	data={ x: np.zeros(1, dtype=np.float32)})

inference.initialize()
sess = ed.get_session()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
	info_dict = inference.update()
	inference.print_progress(info_dict)

	t = info_dict['t']
	#if t == 1 or t % inference.n_print == 0:
		#qpi_mean, qmu_mean = sess.run([qpi.mean(), qmu.mean()])
		#print("")
		#print("Inferred membership probabilities:")
		#print(qpi_mean)
		#print("Inferred cluster means:")
		#print(qmu_mean)
print(info_dict)
print(inference)
#ipdb.set_trace()
'''

# MODEL
'''
p = Beta(1.0, 1.0)
x = Bernoulli(probs=p, sample_shape=10)

# INFERENCE
qp = Empirical(params=tf.get_variable(
  "qp/params", [1000], initializer=tf.constant_initializer(0.5)))

proposal_p = Beta(3.0, 9.0)

inference = ed.MetropolisHastings({p: qp}, {p: proposal_p})
inference.run()
sess = ed.get_session()
mean, stddev = sess.run([qp.mean(), qp.stddev()])
ipdb.set_trace()
'''
'''

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Bernoulli, Empirical, Normal

N = 581012  # number of data points
D = 54  # number of features
T = 100  # number of empirical samples

# DATA
x_data = np.zeros([N, D])
y_data = np.zeros([N])

# MODEL
x = tf.Variable(x_data, trainable=False)
beta = Normal(loc=tf.zeros(D), scale=tf.ones(D))
y = Bernoulli(logits=ed.dot(x, beta))

# INFERENCE
qbeta = Empirical(params=tf.Variable(tf.zeros([T, D])))
inference = ed.HMC({beta: qbeta}, data={y: y_data})
inference.run(step_size=0.5 / N, n_steps=10)
'''


"""A simple coin flipping example. Inspired by Stan's toy example.
"""


import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical


def main(_):
	ed.set_seed(42)

	# DATA
	x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

	# MODEL
	p = Beta(1.0, 1.0)
	x = Bernoulli(probs=p, sample_shape=10)

	# INFERENCE
	qp = Empirical(params=tf.get_variable(
	  "qp/params", [1000], initializer=tf.constant_initializer(0.5)))

	proposal_p = Beta(3.0, 9.0)

	inference = ed.MetropolisHastings({p: qp}, {p: proposal_p}, data={x: x_data})
	inference.run()

	# CRITICISM
	# exact posterior has mean 0.25 and std 0.12
	sess = ed.get_session()
	mean, stddev = sess.run([qp.mean(), qp.stddev()])
	print("Inferred posterior mean:")
	print(mean)
	print("Inferred posterior stddev:")
	print(stddev)

	x_post = ed.copy(x, {p: qp})
	tx_rep, tx = ed.ppc(
		lambda xs, zs: tf.reduce_mean(tf.cast(xs[x_post], tf.float32)),
		data={x_post: x_data})
	ed.ppc_stat_hist_plot(
	  tx[0], tx_rep, stat_name=r'$T \equiv$mean', bins=10)
	plt.show()

if __name__ == "__main__":
	tf.app.run()