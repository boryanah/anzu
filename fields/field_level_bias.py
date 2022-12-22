import sys
from fields.common_functions import get_snap_z
from fields.measure_basis import advect_fields
from fields.make_lagfields import make_lagfields
from mpi4py import MPI
from glob import glob
import numpy as np
import time, sys, gc, psutil, os, yaml
import pmesh, h5py
import yaml
from yaml import Loader
from pmesh.pm import RealField, ComplexField


def integrate_field_to_kmax(field, kmax, dk):

    kmax = np.atleast_1d(kmax)
    intg = np.zeros_like(kmax, dtype=field.dtype)
    
    for s in field.slabs:
        k = np.meshgrid(*s.x)
        knorm = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
        for i in range(kmax.shape[0]):
            intg[i] += np.sum(dk * s.T[np.newaxis, ...][knorm<kmax[i]])
            
    return intg


def measure_field_level_bias(comm, pm, tracerfield, field_dict, field_D, nmesh, kmax, Lbox, M=None):
    
    m = field_dict.pop('1m', None)
    
    if m is not None:
        field_D = field_D[1:]
        
    cb = field_dict.pop('1cb')
    field_D = field_D[1:]
    
    eps = tracerfield.copy()
    if type(cb) is RealField:
        cb = cb.r2c()
    
    dk = 1
    A = np.zeros((len(field_dict), len(kmax)), dtype=cb.dtype)    
    
    for (s0, s1, s2) in zip(eps.slabs, tracerfield.slabs, cb.slabs):
        s0[...] = s1 - s2
    
    for i, k1 in enumerate(field_dict):
        f1 = field_dict[k1]
        if type(f1) is RealField:
            f1 = f1.r2c()
        
        o = f1.copy()
        
        for (s0, s1, s2) in zip(o.slabs, f1.slabs, eps.slabs):
            s0[...] = field_D[i]*s1 * s2.conj()
            
        A[i,...] = integrate_field_to_kmax(o, kmax, dk)
    
    if M is None:
        M = np.zeros((len(field_dict), len(field_dict), len(kmax)), dtype=f1.dtype)

        for i, k1 in enumerate(field_dict):
            f1 = field_dict[k1]

            for j, k2 in enumerate(field_dict):
                f2 = field_dict[k2]
                if j>i: continue

                if type(f2) is RealField:
                    f2 = f2.r2c()

                o = f1.copy()

                for (s0, s1, s2) in zip(o.slabs, f1.slabs, f2.slabs):
                    s0[...] = field_D[i]*s1 * field_D[j]*s2.conj()

                M[i,j,...] = integrate_field_to_kmax(o, kmax, dk)
                if i!=j:
                    M[j,i,...] = M[i,j].conj()
        comm.Allreduce(MPI.IN_PLACE, M)
        
    comm.Allreduce(MPI.IN_PLACE, A)

    b_all = []
    for n in range(len(kmax)):

        b = np.dot(np.linalg.inv(M[...,n]), A[:,n]).real
        b_all.append(b)
        
    zafield = cb.copy()
    
    for i, k1 in enumerate(field_dict):
        zafield += b_all[0][i] * field_D[i] * field_dict[k1]
        
    field_dict['1cb'] = cb
    
    return M, A, b_all, zafield

def advect_and_measure_bias(config):
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size    

    scale_dependent_growth = bool(config.pop("scale_dependent_growth", False))
    config["scale_dependent_growth"] = scale_dependent_growth
    lattice_type = int(config.pop('lattice_type', 0))
    config['lattice_type'] = lattice_type
    if 'bias' not in config:
        bias_vec = None
    else:
        bias_vec = config['bias_vec']
    
    if scale_dependent_growth:
        z = get_snap_z(config["particledir"], config["sim_type"])
        lag_field_dict = make_lagfields(config, save_to_disk=False, z=z)
    else:
        lag_field_dict = None

    pm, field_dict, field_D, keynames, labelvec, zbox = advect_fields(config, lag_field_dict=lag_field_dict)

    tracer_file = config['tracer_file']
    kmax = np.atleast_1d(config['field_level_kmax'])
    nmesh = int(config['nmesh_out'])
    Lbox = float(config['lbox'])
    
    if config['rsd']:
        tracer_pos = h5py.File(tracer_file)['pos_zspace'][rank::size,:]
    else:
        tracer_pos = h5py.File(tracer_file)['pos_rspace'][rank::size,:]
        
    layout = pm.decompose(tracer_pos)
    p = layout.exchange(tracer_pos)
    tracerfield = pm.paint(p, mass=1, resampler="cic")
    tracerfield = tracerfield / tracerfield.cmean() - 1
    tracerfield = tracerfield.r2c()
    
    del tracer_pos, p        
        
    M, A, b_all, zafield = measure_field_level_bias(comm, pm, tracerfield, field_dict, field_D, nmesh, kmax, Lbox)
   
    np.save('{}/b_rsd={}_{}_a{:.4f}.npy'.format(config['outdir'], config['rsd'], tracer_file.split('/')[-1], 1 / (zbox + 1)), b_all)
    np.save('{}/M_rsd={}_{}_a{:.4f}.npy'.format(config['outdir'], config['rsd'], tracer_file.split('/')[-1], 1 / (zbox + 1)), M)
    np.save('{}/A_rsd={}_{}_a{:.4f}.npy'.format(config['outdir'], config['rsd'], tracer_file.split('/')[-1], 1 / (zbox + 1)), A)

    return zafield

if __name__ == "__main__":
    
    config = sys.argv[1]
    
    with open(config, "r") as fp:
        config = yaml.load(fp, Loader=Loader)
        
    advect_and_measure_bias(config)
