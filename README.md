Tensorflow Plugin
==============

This plugin allows using tensorflow to compute forces in a simulation
or to compute other quantities, like collective variables to fit a
potential for coarse-graining. You must first construct your
tensorlfow graph using the `tensorflow_plugin.graph_builder` class and
then add the `tfcompute` compute to your hoomd simulation.

Building Graph
=====

To construct a graph, construct a graphbuilder:

```
from hoomd.tensorflow_plugin import graph_builder
graph = graph_builder(N, NN, output_forces)
```

where `N` is the number of particles in the simulation, `NN` is the maximum number of nearest neighbors to consider, and `output_forces` indicates if the graph will output forces to use in the simulation. After building the `graph`, it will have three tensors as attributes to use in constructing the tensorflow graph: `nlist`, `positions`, and `forces`. `nlist` is an `N` x `NN` x 4 tensor containing the nearest neighbors. An entry of all zeros indicates that less than `NN` nearest neighbors where present for a particular particle. The 4 right-most dimensions are `x,y,z` and `w`, which is the particle type. Note that the `x,y,z` values are a vector originating at the particle and ending at its neighbor. `positions` and `forces` are `N` x 4 tensors. `forces` *only* is available if the graph does not output forces via `output_forces=False`.

Computing Forces
-----
If you graph is outputting forces, you may either compute forces and pass them to `graph_builder.save(...)` or have them computed via automatic differentiation of a potential energy. Call `graph_builder.compute_forces(energy)` where `energy` is a scalar or tensor that depends on `nlist` and/or `positions`. A tensor of forces will be returned as sum(-dE / dn) - dE / dp where the sum is over the neighbor list. For example, to compute a `1 / r` potential:

```
graph = hoomd.tensorflow_plugin.graph_builder(N, N - 1)
#remove w since we don't care about types
nlist = graph.nlist[:, :, :3]
#get r
r = tf.norm(nlist, axis=2)
#compute 1. / r while safely treating r = 0.
# halve due to full nlist
rij_energy = 0.5 * graph.safe_div(1, r)
#sum over neighbors
energy = tf.reduce_sum(rij_energy, axis=1)
forces = graph.compute_forces(energy)
```

See in the above example that we have used the
`graph_builder.safe_div(numerator, denominator)` function which allows
us to safely treat a `1 / 0` due to using nearest neighbor distances,
which can arise because `nlist` contains 0s for when less than `NN`
nearest neighbors are found. Note that because `nlist` is a *full*
neighbor list, you should divide by 2 if your energy is a sum of
pairwise energies.

Virial
-----

The virial is computed and added to the graph if you use the
`compute_forces` function and your energy has a non-zero derivative
with respect to `nlist`. You may also explicitly pass the virial when
saving, or pass `None` to remove the automatically calculated virial.

Finalizing the Graph
----

To finalize and save your graph, you must call the `graph_builder.save(directory, force_tensor=forces, virial = None, out_node=None)` function. `force_tensor` should be your computed forces, either as computed by your graph or as the output from `compute_energy`. If your graph is not outputting forces, then you must provide a tensor which will be computed, `out_node`, at each timestep. Your forces should be an `N x 4` tensor with the 4th column indicating per-particle potential energy. The virial should be an `N` x 3 x 3 tensor.

Complete Examples
-----

See `tensorflow_plugin/models/test-models/build.py` for more.

### Lennard-Jones

```
graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
nlist = graph.nlist[:, :, :3]
#get r
r = tf.norm(nlist, axis=2)
#compute 1 / r while safely treating r = 0.
#pairwise energy. Double count -> divide by 2
p_energy = 4.0 / 2.0 * (graph.safe_div(1., r**12) - graph.safe_div(1., r**6))
#sum over pairwise energy
energy = tf.reduce_sum(p_energy, axis=1)
forces = graph.compute_forces(energy)
graph.save(force_tensor=forces, model_directory='/tmp/lj-model')
```



Using Graph in a Simulation
=====

You may use a saved tensorflow model via:

```
import hoomd
from hoomd.tensorflow_plugin import tfcompute

#must construct prior to initialization
tfcompute = tfcompute(model_loc)

...hoomd initialization code...

nlist = hoomd.md.nlist.cell()
tfcompute.attach(nlist, r_cut=r_cut, force_mode='output')

```

where `model_loc` is the directory where the tensorflow model was saved, `nlist` is a hoomd neighbor list object, `r_cut` is the maximum distance for to consider particles as being neighbors, and `force_mode` is a string that indicates how to treat forces. A value of `'output'` indicates forces will be output from hoomd and input into the tensorflow model. `'add'` means the forces output from the tensorflow model should be added with whatever forces are computed from hoomd, for example if biasing a simulation. `'ignore'` means the forces will not be modified and are not used the tensorflow model, for example if computing collective variables that do not depend on forces. `'overwrite'` means the forces from the tensorflow model will overwrite the forces from hoomd, for example if the tensorflow model is computing the forces instead.

Examples
-----
See `tensorflow_plugin/test-py/test_tensorflow.py`

Note on Building and Executing Tensorflow Models in Same Script
------

Due to the side-effects of importing tensorflow, you must build and save your graph in a separate python process first before running it hoomd.

Interprocess Communication
-----
*You must be on a system with at least two threads so that the tensorflow and hoomd process can run concurrently.*


Tensorboard
=====

You can visualize your models with tensorboard. First, add
`_write_tensorboard=True` the tensorflow plugin constructor. This will
add a new directory called `tensorboard` to your model directory.

After running, you can launch tensorboard like so:
```bash
tensorboard --logdir=/path/to/model/tensorboard
```

and then visit `http://localhost:6006` to view the graph. If you are
running on a server, before launching tensorboard use this ssh command to login:
```bash
ssh -L 6006:[remote ip or hostname]:6006 username@remote
```

and then you can view after launching on the server via your local web browser.

Interactive Mode
----

Experimental, but you can trace your graph in realtime in a simulation. Add both the `_write_tensorboard=True` to
the constructor and the `_debug_mode=True` flag to `attach` command.

Requirements
=====
* Latest tensorflow (cuda 9.0, cudnn7)
* numpy
* HOOMD


Docker Image for Development
====

To use the included docker image:

```
docker build -t hoomd-tf tensorflow_plugin
```

To run the container:

```
docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v tensorflow_plugin/:/srv/hoomd-blue/tensorflow_plugin hoomd-tf bash
```

The `cap--add` and `security-opt` flags are optional and allow `gdb` debugging.

Once in the container:

```
cd /srv/hoomd-blue && mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=OFF -DENABLE_MPI=OFF -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
make -j2
```

Tests
====
To run the unit tests, run
```
python tensorflow-plugin/test-py/test_tensorflow.py [test_class].[test_name]
```

Note that only one test can be run at a time due to the way gpu contexts/forks occur.

If you change C++/C code, remake. If you modify python code, copy the new version to the build directory.

Bluehive Install
====

Load the modules necessary:
```bash
module load anaconda cmake sqlite cuda cudnn git
```

Set-up virtual python environment *ONCE* to keep packages separate
```bash
conda create -n hoomd-tf python=3.6
```

Then whenever you login and have loaded modules:
```bash
source activate hoomd-tf
```

Now that we're Python, install some pre-requisites:

```bash
pip install tensorflow-gpu
```

Compiling
-----
```bash
git clone --recursive https://bitbucket.org/glotzer/hoomd-blue hoomd-blue
```

Put our plugin in the source directory. Make a softlink:
```
ln -s $HOME/hoomd-tf/tensorflow_plugin $HOME/hoomd-blue/hoomd
```

Now compile (from hoomd-blue directory). Modify options for speed if necessary.

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=OFF -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
```

Now compile with make
```bash
make
```
Put build directory on your python path:
```bash
export PYTHONPATH="$PYTHONPATH:`pwd`"
```

Note: if you modify C++ code, only run make (not cmake). If you modify python, just copy over py files.

Issues
====
* Use GPU event handles -> Depends on TF While
* Domain decomposition testing -> Low priority
* Refactor style/names, line endings -> Style
* Write better source doc -> Style
* Make ipc2tensor not stateful (use resource manager) -> Low priority
* Explore using ptrs instead of memory addresses, at least
  to get to python -> Style
* TF while -> Next optimization
* Feed dict  -> Feature required for learning
* Stride -> Feature required for learning
* Callbacks -> Feature required for learning
* context manager -> way to fix shutting down tfmanager bug

Style Issues
===

C++
---
balance between tf/myself/hoomd
C++ class -> Camel
C++ methods -> camel
C++ variables -> snake
C++ types -> camel _t
C++ class variables -> snake prefix
POD struct -> (c++ types) (since that is cuda style)
C++ functions -> snake (?) because they are only used in py or gpu kernels

Python
----

py class ->snake


Examples
====

Just made up, not sure if they work

Force-Matching
----
```python
import tensorflow as tf
import hoomd.tensorflow_plugin
graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
#we want to get mapped forces!
#map = tf.Variable(tf.ones([N, M]))
#zero_map_enforcer = ...
#restricted_map = zero_map_enforcer * map
# + add some normalization
map = tf.Placeholder((N, M), dtype=tf.float32)
#forces from HOOMD are fx,fy,fz,pe where pe is potential energy of particle
forces = graph.forces[:, :, :3]
mapped_forces = map * forces #think -> N x 3 * N x M
# sum_i m_ij * f_ik = cf_jk
mapped_forces = tf.einsum('ij,ik->jk', map, forces)
#get model forces
mapped_positions = tf.einsum('ij,ik->jk', map, graph.positions[:, :3])
#get mapped neighbor list
dist_r = tf.reduce_sum(mapped_positions * mapped_positions, axis=1)
# turn dist_r into column vector
dist_r = tf.reshape(dist_r, [-1, 1])
mapped_distances = dist_r - 2*tf.matmul(mapped_positions, tf.transpose(mapped_positions)) + tf.transpose(dist_r)
#compute our model forces on CG sites
#our model -> f(r) ->  r * w = f
#      0->0.5,0.5->1,1->1.5,1.5->infty
# r -> hr = [ 0,       0.1,    0.8,      0.1]
#f hr * w
# distance at each grid point from r
#send through RElu
grid = tf.range(0.5, 10, 0.1, dtype=tf.float32)
#want an N x N x G
grid_dist = grid - tf.tile(mapped_distances, grid.shape[0])
#one of the N x N grid distances -> [0 - r, 0.5 - r , 1.0 - r, 1.5 - r, 2.0 - r]
#want to do f(delta r) -> [0,1]
clip_high = tf.Variable(1, name='clip-high')
grid_clip = tf.clip_by_value(tf.abs(grid_dist), 0, clip_high)
#afterwards -> r = 1.4, [0, 0.9, 0.4, 0.1, 0.6,]
#r = 1.3, [0, 0.8, 0.3, 0.2, 0.7]
#TODO -> see if Prof White assumption is correct -> sum of grid_clip = 2 * clip-high
grid_normed = grid_clip / 2 / clip_high
force_weights = tf.Variable(tf.ones(grid.shape), name='force-weights')
#N x N x G * G x 1 = N x N
#TODO: we need actual rs
model_force_mag = tf.matmul(grid_normed, force_weights)
#once fixed....
model_forces = ....
error = tf.reduce_sum(tf.norm(mapped_forces - model_forces, axis=1), axis=0)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(error)

#need to tell tf to run optimizer
graph.save('/tmp/force_matching', out_nodes=[optimizer])
```

To run the model
```python
import hoomd, hoomd.md
from hoomd.tensorflow_plugin import tfcompute
tfcompute = tfcompute('/tmp/force_matching')

....setup simulation....
nlist = hoomd.md.nlist.cell()
tfcompute.attach(nlist, r_cut=r_cut, force_mode='output')
hoomd.run(1000)
```
