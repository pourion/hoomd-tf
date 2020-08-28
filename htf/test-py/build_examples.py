# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import os
import hoomd.htf as htf
import pickle


class SimplePotential(htf.SimModel):
    def compute(self, nlist, positions):
        nlist = nlist[:, :, :3]
        neighs_rs = tf.norm(tensor=nlist, axis=2, keepdims=True)
        # no need to use netwon's law because nlist should be double counted
        fr = tf.multiply(-1.0, tf.multiply(tf.math.reciprocal(neighs_rs), nlist),
                         name='nan-pairwise-forces')
        zeros = tf.zeros_like(nlist)
        real_fr = tf.where(tf.math.is_finite(fr), fr, zeros,
                           name='pairwise-forces')
        forces = tf.reduce_sum(input_tensor=real_fr, axis=1, name='forces')
        return forces


class BenchmarkPotential(htf.SimModel):
    def compute(self, nlist):
        rinv = htf.nlist_rinv(nlist)
        energy = rinv
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class NoForceModel(htf.SimModel):
    def compute(self, nlist, positions, box):
        neighs_rs = tf.norm(tensor=nlist[:, :, :3], axis=2)
        energy = tf.math.divide_no_nan(tf.ones_like(
            neighs_rs, dtype=neighs_rs.dtype),
            neighs_rs, name='energy')
        pos_norm = tf.norm(tensor=positions, axis=1)
        return energy, pos_norm


class TensorSaveModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        pos_norm = tf.norm(tensor=positions, axis=1)
        return pos_norm


class WrapModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        p1 = positions[0, :3]
        p2 = positions[-1, :3]
        r = p1 - p2
        rwrap = htf.wrap_vector(r, box)
        # TODO: Smoke test. Think of a better test.
        return rwrap


class BenchmarkNonlistModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        ps = tf.norm(tensor=positions, axis=1)
        energy = tf.math.divide_no_nan(1., ps)
        forces = htf.compute_positions_forces(positions, energy)
        return forces


class LJModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJVirialModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces_and_virial = htf.compute_nlist_forces(
            nlist, energy, virial=True)
        return forces_and_virial


class EDSModel(htf.SimModel):
    def setup(self, set_point):
        self.cv_avg = tf.keras.metrics.Mean()
        self.eds_bias = htf.EDSLayer(set_point, 5, 1/5)

    def compute(self, nlist, positions, box, sample_weight):
        # get distance from center
        rvec = htf.wrap_vector(positions[0, :3], box)
        cv = tf.norm(tensor=rvec)
        self.cv_avg.update_state(cv, sample_weight=sample_weight)
        alpha = self.eds_bias(cv)
        # eds + harmonic bond
        energy = (cv - 5) ** 2 + cv * alpha
        # energy  = cv^2 - 6cv + cv * alpha + C
        # energy = (cv - (3 + alpha / 2))^2 + C
        # alpha needs to be = 4
        forces = htf.compute_positions_forces(positions, energy)
        return forces, alpha


class MolFeatureModel(htf.MolSimModel):

    def mol_compute(self, nlist, positions, mol_nlist, mol_pos, box):
        r = htf.mol_bond_distance(mol_pos, 2, 1)
        a = htf.mol_angle(mol_pos, 1, 2, 3)
        d = htf.mol_dihedral(mol_pos, 1, 2, 3, 4)
        avg_r = tf.reduce_mean(input_tensor=r)
        avg_a = tf.reduce_mean(input_tensor=a)
        avg_d = tf.reduce_mean(input_tensor=d)
        return avg_r, avg_a, avg_d


class CustomNlist(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)

        # compute nlist
        cnlist = htf.compute_nlist(
            positions[:, :3], self.r_cut, self.nneighbor_cutoff, htf.box_size(box))
        cr = tf.norm(tensor=cnlist[:, :, :3], axis=2)
        return r, cr


class NlistNN(htf.SimModel):
    def setup(self, dim, top_neighs):
        self.dense1 = tf.keras.layers.Layer(dim)
        self.dense2 = tf.keras.layers.Layer(dim)
        self.last = tf.keras.layers.Layer(1)
        self.top_neighs = top_neighs

    def compute(self, nlist, positions, box, sample_weight):
        rinv = htf.nlist_rinv(nlist)
        # closest neighbors have largest value in 1/r, take top
        top_n = tf.sort(rinv, axis=1, direction='DESCENDING')[
            :, :self.top_neighs]
        # run through NN
        x = self.dense1(top_n)
        x = self.dense2(x)
        energy = self.last(x)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class TrainModel(htf.SimModel):
    def setup(self, dim, top_neighs):
        self.dense1 = tf.keras.layers.Layer(dim)
        self.dense2 = tf.keras.layers.Layer(dim)
        self.last = tf.keras.layers.Layer(1)
        self.top_neighs = top_neighs

    def compute(self, nlist, positions, training):
        rinv = htf.nlist_rinv(nlist)
        # closest neighbors have largest value in 1/r, take top
        top_n = tf.sort(rinv, axis=1, direction='DESCENDING')[
            :, :self.top_neighs]
        # run through NN
        x = self.dense1(top_n)
        x = self.dense2(x)
        energy = self.last(x)
        if training:
            energy *= 2
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJRunningMeanModel(htf.SimModel):
    def setup(self):
        self.avg_energy = tf.keras.metrics.Mean()

    def compute(self, nlist, positions, box, sample_weight):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        # compute running mean
        self.avg_energy.update_state(energy, sample_weight=sample_weight)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJRDF(htf.SimModel):
    def setup(self):
        self.avg_rdf = tf.keras.metrics.MeanTensor()

    def compute(self, nlist, positions, box, sample_weight):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # get rdf
        rdf, rs = htf.compute_rdf(nlist, [3, 5], positions[:, 3])
        # also compute without types
        _, _ = htf.compute_rdf(nlist, [3, 5])
        # compute running mean
        self.avg_rdf.update_state(rdf, sample_weight=sample_weight)
        forces = htf.compute_nlist_forces(nlist, p_energy)
        return forces


class LJMolModel(htf.MolSimModel):
    def mol_compute(self, nlist, positions, mol_nlist, mol_positions, box):
        # assume particle (w) is 0
        r = tf.norm(mol_nlist, axis=3)
        rinv = tf.math.divide_no_nan(1.0, r)
        mol_p_energy = 4.0 / 2.0 * (rinv**12 - rinv**6)
        total_e = tf.reduce_sum(input_tensor=mol_p_energy)
        forces = htf.compute_nlist_forces(nlist, total_e)
        return forces


class PrintModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        tf.print(energy)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJLayer(tf.keras.layers.Layer):
    def __init__(self, sig, eps):
        super().__init__(self, name='lj')
        self.start = [sig, eps]
        self.w = self.add_weight(
            shape=[2],
            initializer=tf.constant_initializer([sig, eps]),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True,
            name='lj-params'

        )

    def call(self, r):
        r6 = tf.math.divide_no_nan(self.w[1]**6, r**6)
        energy = self.w[0] * 4.0 * (r6**2 - r6)
        # divide by 2 to remove double count
        return energy / 2.

    def get_config(self):
        c = {}
        c['sig'] = self.start[0]
        c['eps'] = self.start[1]
        return c


class TrainableGraph(htf.SimModel):
    def __init__(self, NN, **kwargs):
        super().__init__(NN, **kwargs)
        self.lj = LJLayer(1.0, 1.0)

    def compute(self, nlist, positions, box, sample_weight):
        # get r
        r = htf.safe_norm(tensor=nlist[:, :, :3], axis=2)
        p_energy = self.lj(r)
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces
