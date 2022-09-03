import numpy as np
import sys, os
from fields.make_lagfields import make_lagfields
from fields.measure_basis import advect_fields, measure_basis_spectra
from fields.common_functions import get_snap_z, measure_pk
from fields.field_level_bias import measure_field_level_bias
from mpi4py_fft import PFFT
from anzu.utils import combine_real_space_spectra, combine_measured_rsd_spectra
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from classy import Class
from mpi4py import MPI
import pmesh
from copy import copy
from glob import glob
from yaml import Loader
import sys, yaml
import h5py
import gc

def CompensateCICAliasing(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the CIC window function in configuration space,
            as well as the approximate aliasing correction
    From the nbodykit documentation.
    """
    for i in range(3):
        wi = w[i]
        v = v / (1 - 2.0 / 3 * np.sin(0.5 * wi) ** 2) ** 0.5
    return v


def pk_list_to_vec(pk_ij_list):

    nspec = len(pk_ij_list)
    keys = list(pk_ij_list[0].keys())
    k = pk_ij_list[0]['k']
    nk = k.shape[0]
    
    if 'mu' in keys:
        mu = pk_ij_list[0]['mu']  
        nmu = mu.shape[-1]
    else:
        nmu = 1
        mu = None
    
    if 'power_poles' in keys:
        npoles = pk_ij_list[0]['power_poles'].shape[0]
        has_poles = True
        pk_pole_array = np.zeros((nspec, npoles, nk))
        
    else:
        npoles = 1
        has_poles = False
        
        
    pk_wedge_array = np.zeros((nspec, nk, nmu))
        
    for i in range(nspec):
        #power_wedges is always defined, even if only using 1d pk (then wedge is [0,1])
        pk_wedges = pk_ij_list[i]['power_wedges']
        pk_wedge_array[i,...] = pk_wedges.reshape(nk,-1)
        
        if has_poles:
            pk_poles = pk_ij_list[i]['power_poles']
            pk_pole_array[i,...] = pk_poles
            
    if has_poles:
        return k, mu, pk_wedge_array, pk_pole_array
    else:
        return k, mu, pk_wedge_array, None

def measure_2pt_bias(k, pk_ij_heft, pk_tt, kmax, rsd=False):
    
    kidx = k.searchsorted(kmax)
    kcut = k[:kidx]
    pk_tt_kcut = pk_tt[:kidx]
    pk_ij_heft_kcut = pk_ij_heft[:,...,:kidx,np.newaxis]
    
    if not rsd:
        loss = lambda bvec : np.sum((pk_tt_kcut - combine_real_space_spectra(kcut, pk_ij_heft_kcut, bvec)[:,0])**2/(pk_tt_kcut**2))
        bvec0 = [1, 0, 0, 0, 0]
    else:
        loss = lambda bvec : np.sum((pk_tt_kcut - combine_measured_rsd_spectra(kcut, pk_ij_heft_kcut, None, bvec)[:,0])**2/(pk_tt_kcut**2))
        bvec0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    out = minimize(loss, bvec0)
    
    return out

def get_linear_field(config, lag_field_dict, rank, size, nmesh):
    
    boltz = Class()
    boltz.set(config["Cosmology"])
    boltz.compute()
    z_ic = config["z_ic"]        
    z_this = get_snap_z(config["particledir"], config["sim_type"])
    D = boltz.scale_independent_growth_factor(z_this)
    D = D / boltz.scale_independent_growth_factor(z_ic)        
    f = D / boltz.scale_independent_growth_factor_f(z_this)        

    if config['rsd']:
        delta = real_to_redshift_space(lag_field_dict['delta'], nmesh, Lbox, rank, size, f)
        
    grid = np.meshgrid(
        np.arange(rank, nmesh, size, dtype=np.float32),
        np.arange(nmesh, dtype=np.float32),
        np.arange(nmesh, dtype=np.float32),
        indexing="ij"
    )

    grid[0] *= Lbox / nmesh
    grid[1] *= Lbox / nmesh
    grid[2] *= Lbox / nmesh

    meshpos = np.vstack([grid[0].flatten(), grid[1].flatten(), grid[2].flatten()]).T
    #if rank==0:
    #    print('meshpos.shape: {}'.format(meshpos.shape))
    #    print('meshpos.shape: {}'.format(delta.flatten().shape))
        
    layout = pm.decompose(meshpos)
    p = layout.exchange(meshpos)
    d = layout.exchange(delta.flatten())

    mesh = pm.paint(p, mass=d)
    del p, d, meshpos
    
    mesh = mesh.r2c()
    mesh = mesh.apply(CompensateCICAliasing, kind='circular')    
    field_dict = {'delta':mesh}
    field_D = [D]
    
    return field_dict, field_D, z_this

def real_to_redshift_space(field, nmesh, lbox, rank, nranks, f, fft=None):
    
    if fft is None:
        N = np.array([nmesh, nmesh, nmesh], dtype=int)
        fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype="float32", grid=(-1,))

    field_k = fft.forward(field)

    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // nranks : (rank + 1) * nmesh // nranks]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    knorm = kx ** 2 + ky ** 2 + kz ** 2
    mu = kz / np.sqrt(knorm)
    
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1
        mu[0][0][0] = 0
    rsdfac = 1 + f * mu**2
    del kx, ky, kz, mu
        
    field_k_rsd = field_k * rsdfac
    field_rsd = fft.backward(field_k_rsd)

    return field_rsd

if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size    

    config = sys.argv[1]

    with open(config, "r") as fp:
        config = yaml.load(fp, Loader=Loader)
    
    config['compute_cv_surrogate'] = True
    config['scale_dependent_growth'] = False
    lattice_type = int(config.get('lattice_type', 0))
    config['lattice_type'] = lattice_type
    lindir = config["outdir"]
    tracer_file = config['tracer_file']
    kmax = np.atleast_1d(config['field_level_kmax'])
    nmesh = int(config['nmesh_out'])
    Lbox = float(config['lbox'])    
    linear_surrogate = config.get('linear_surrogate', False)
    basename = "mpi_icfields_nmesh_filt"

    if 'bias_vec' in config:
        bias_vec = config['bias_vec']
    else:
        bias_vec=None
        field_level_bias = config.get('field_level_bias', False)
        M_file = config.get('field_level_M', None)
        if M_file:
            M = np.load(M_file)
    
    #create/load surrogate linear fields
    linfields = glob(lindir + "{}_{}_*_np.npy".format(basename, nmesh))
    if len(linfields)==0:
        lag_field_dict = make_lagfields(config, save_to_disk=True)
    elif linear_surrogate:
        lag_field_dict = {}
        arr = np.load(
            lindir + "{}_{}_{}_np.npy".format(basename, nmesh, 'delta'),
            mmap_mode="r",
        )        
        lag_field_dict['delta'] = arr[np.arange(rank, nmesh, size), :, :]
        keynames = ['delta']
        labelvec = ['delta']        
        
    #advect ZA fields
    if not linear_surrogate:
        pm, field_dict, field_D, keynames, labelvec, zbox = advect_fields(config, lag_field_dict=lag_field_dict)
    else:
        pm = pmesh.pm.ParticleMesh(
            [nmesh, nmesh, nmesh], Lbox, dtype="float32", resampler="cic", comm=comm
        )
        field_dict, field_D, zbox = get_linear_field(config, lag_field_dict, rank, size, nmesh)
    # load tracers and deposit onto mesh.
    # TODO: generalize to accept different formats
    
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
        
    #measure tracer auto-power
    pk_tt_dict = measure_pk(tracerfield, tracerfield, Lbox, nmesh, config['rsd'], config['use_pypower'], 1, 1)
    
    field_dict2 = {'t':tracerfield}
    field_D2 = [1]
    
    pk_auto_vec, pk_cross_vec = measure_basis_spectra(
        config,
        field_dict,
        field_D,
        keynames,
        labelvec,
        zbox,
        field_dict2=field_dict2,
        field_D2=field_D2,
        save=False
    )
    
    if linear_surrogate:
        stype = 'l'
    else:
        stype = 'z'
        
    np.save(
        lindir
        + "{}cv_surrogate_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(stype,
            config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
        ),
        pk_auto_vec,
    )    
    
    np.save(
        lindir
        + "{}_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
            tracer_file.split('/')[-1], config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
        ),
        [pk_tt_dict],
    )
    
    np.save(
        lindir
        + "{}cv_cross_{}_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(stype,
            tracer_file.split('/')[-1], config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
        ),
        pk_cross_vec,
    )        

    if not linear_surrogate:    
        if bias_vec is None:
            if field_level_bias:
                bias_vec, M, A = measure_field_level_bias(comm, pm, tracerfield, field_dict, field_D, nmesh, kmax, Lbox, M=M)
            else:
                k, mu, pk_tt_wedge_array, pk_tt_pole_array = pk_list_to_vec([pk_tt_dict])
                k, mu, pk_ij_wedge_array, pk_ij_pole_array = pk_list_to_vec(pk_auto_vec)

                if config['rsd']:
                    bias_vec = measure_2pt_bias(k, pk_ij_pole_array, pk_tt_pole_array[0,...], kmax)
                else:
                    bias_vec = measure_2pt_bias(k, pk_ij_wedge_array, pk_tt_wedge_array[0,...], kmax)
                    
                


