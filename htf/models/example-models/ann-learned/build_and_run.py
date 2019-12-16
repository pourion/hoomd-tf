import hoomd
import tensorflow as tf
import numpy as np
import hoomd.htf
from math import sqrt
from sys import argv as argv

if(len(argv) != 4):
    print('Usage: build_and_run.py [N_PARTICLES] [N_TRAINING_STEPS] [model_dir]')
    exit(0)

N = int(argv[1])
N_STEPS = int(argv[2])
model_dir = argv[3]

# make a simple ANN
def build_neural_network(x, keep_prob,
                         N_nodes,
                         N_hidden_layers,
                         activation_func=lambda x: x):
    '''Builds the TensorFlow graph with chosen weights and biases of the
        chosen width (N_nodes) and depth (N_layers)'''
    input_layer = tf.layers.dense(x, N_nodes, activation=activation_func, name='input')
    tf.layers.dropout(input_layer, rate=1.0-keep_prob)
    hidden_layers = []
    for i in range(N_hidden_layers):
        hidden_layers.append(tf.layers.dense(
            (input_layer if i == 0 else hidden_layers[-1]),
            N_nodes,
            activation=activation_func))
        hidden_layers.append(tf.layers.dropout(
            (input_layer if i == 0 else hidden_layers[-1]),
            rate=1.0-keep_prob))
    output_layer = tf.layers.dense(
        (input_layer if N_hidden_layers == 0 else hidden_layers[-1]),
        1,
        name='output')
    return(output_layer)

def make_train_graph(NN, N_hidden_nodes, N_hidden_layers, r_cut, system):
    graph = hoomd.htf.graph_builder(NN, output_forces=False)
    # get sorted neighbor list
    nlist = graph.nlist[:, :, :3]  
    # get the interatomic radii
    cost_tracker = tf.Variable(0.0, name='cost')
    r = hoomd.htf.graph_builder.safe_norm(nlist, axis=2)
    r_inv = hoomd.htf.graph_builder.safe_div(1., r)
    print('r_inv shape: {}'.format(r_inv.shape))
    # make weights tensors, using our number of hidden nodes
    # NxNN out because we want pairwise forces
    r_inv = tf.reshape(r_inv, shape=[-1, 1], name='r_inv')

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob', shape=[])
    hoomd_energy = tf.placeholder(dtype=tf.float32, name='hoomd_energy', shape=[])
    # specify the network structure
    output_layer = build_neural_network(r_inv, keep_prob,
                                        N_hidden_nodes, N_hidden_layers,
                                        activation_func=tf.nn.tanh)
    nn_energies = tf.reshape(output_layer, shape=[-1, NN])  # recover structure
    print('nn_energies shape: {}'.format(nn_energies.shape))
    nn_tot_en = tf.reduce_sum(nn_energies)
    # calculate the forces
    calculated_energies = tf.reduce_sum(nn_energies, axis=1,
                                        name='calculated_energies')
    print('calculated_energies shape: {}'.format(calculated_energies.shape))
    print('graph forces shape: {}'.format(graph.forces.shape))
    calculated_forces = graph.compute_forces(nn_energies)
    print('calculated_forces shape is: {}'.format(calculated_forces.shape))
    # compare calculated forces to HOOMD's forces
    force_cost = tf.losses.mean_squared_error(graph.forces, calculated_forces)
    en_cost =  tf.losses.mean_squared_error(nn_tot_en, hoomd_energy)
    cost = force_cost#force_cost + en_cost
    # need to minimize the cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
    update_cost = tf.assign(cost_tracker, cost)
    # print summaries for tensorboard
    tf.summary.scalar('cost', cost)
    # print cost for a more granular plot
    # check = tf.add_check_numerics_ops()
    graph.save(model_directory=model_dir,
               out_nodes=[optimizer, update_cost],
               move_previous=False)

minval = -1.
maxval = 1.
NN = 128
N_hidden_nodes = 30
N_hidden_layers = 2

hoomd.context.initialize('--mode=cpu')#('--mode=gpu')
rcut = 3.0
sqrt_N = int(sqrt(N))
system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                   n=[sqrt_N, sqrt_N])
make_train_graph(NN, N_hidden_nodes, N_hidden_layers, rcut, system)
nlist = hoomd.md.nlist.cell()
# basic LJ forces from HOOMD
lj = hoomd.md.pair.lj(rcut, nlist)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
hoomd.md.integrate.mode_standard(dt=0.005)
logger = hoomd.analyze.log(filename=None, quantities=['potential_energy'], period=10)
hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.0, seed=42)
# hoomd.md.integrate.nve(group=hoomd.group.all()).
# randomize_velocities(kT=1.2, seed=42)
# equilibrate for 4k steps first
hoomd.run(4000)

with hoomd.htf.tfcompute(model_dir,
                         write_tensorboard=True) as tfcompute:
    # attach the trainable model
    tfcompute.attach(nlist,
                     r_cut=rcut,
                     save_period=100,
                     period=10,
                     feed_dict=dict({'keep_prob:0': 0.8,
                                     'hoomd_energy:0': logger.query('potential_energy')}))
    # train on specified number of timesteps
    hoomd.run(N_STEPS)
