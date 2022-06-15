import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
from classy import Class
from scipy.interpolate import interp1d
import time
import gc
import sys
import h5py
import yaml
import os
from .common_functions import get_memory, kroneckerdelta

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()


def MPI_mean(array, nmesh):
    """
    Computes the mean of an array that is slab-decomposed across multiple processes.
    """
    procsum = np.sum(array) * np.ones(1)
    recvbuf = None
    if rank == 0:
        recvbuf = np.zeros(shape=[nranks, 1])
    comm.Gather(procsum, recvbuf, root=0)
    if rank == 0:
        fieldmean = np.ones(1) * np.sum(recvbuf) / nmesh ** 3
    else:
        fieldmean = np.ones(1)
    comm.Bcast(fieldmean, root=0)
    return fieldmean[0]


def delta_to_tidesq(delta_k, nmesh, lbox, rank, nranks, fft):
    """
    Computes the square tidal field from the density FFT

    s^2 = s_ij s_ij

    where

    s_ij = (k_i k_j / k^2 - delta_ij / 3 ) * delta_k

    Inputs:
    delta_k: fft'd density, slab-decomposed.
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    nranks: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT.

    Outputs:
    tidesq: the s^2 field for the given slab.
    """

    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // nranks : (rank + 1) * nmesh // nranks]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    if rank == 0:
        print(kvals.shape, kvalsmpi.shape, kvalsr.shape, "shape of x, y, z")

    knorm = kx ** 2 + ky ** 2 + kz ** 2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    klist = [[kx, kx], [kx, ky], [kx, kz], [ky, ky], [ky, kz], [kz, kz]]

    del kx, ky, kz
    gc.collect()

    # Compute the symmetric tide at every Fourier mode which we'll reshape later
    # Order is xx, xy, xz, yy, yz, zz
    jvec = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]
    tidesq = np.zeros((nmesh // nranks, nmesh, nmesh), dtype="float32")

    if rank == 0:
        get_memory()
    for i in range(len(klist)):
        karray = (
            klist[i][0] * klist[i][1] / knorm
            - kroneckerdelta(jvec[i][0], jvec[i][1]) / 3.0
        )
        fft_tide = np.array(karray * (delta_k), dtype="complex64")

        # this is the local sij
        real_out = fft.backward(fft_tide)

        if rank == 0:
            get_memory()

        tidesq += 1.0 * real_out ** 2
        if jvec[i][0] != jvec[i][1]:
            tidesq += 1.0 * real_out ** 2

        del fft_tide, real_out
        gc.collect()

    return tidesq


def delta_to_gradsqdelta(delta_k, nmesh, lbox, rank, nranks, fft):
    """
    Computes the density curvature from the density FFT

    nabla^2 delta = IFFT(-k^2 delta_k)

    Inputs:
    delta_k: fft'd density, slab-decomposed.
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    nranks: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT.

    Outputs:
    real_gradsqdelta: the nabla^2delta field for the given slab.
    """

    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // nranks : (rank + 1) * nmesh // nranks]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    if rank == 0:
        print(kvals.shape, kvalsmpi.shape, kvalsr.shape, "shape of x, y, z")

    knorm = kx ** 2 + ky ** 2 + kz ** 2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    del kx, ky, kz
    gc.collect()

    # Compute -k^2 delta which is the gradient
    ksqdelta = -np.array(knorm * (delta_k), dtype="complex64")

    real_gradsqdelta = fft.backward(ksqdelta)

    return real_gradsqdelta


def gaussian_filter(field, nmesh, lbox, rank, nranks, fft, kcut):
    """
    Apply a fourier space gaussian filter to a field

    Inputs:
    field: the field to filter
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    nranks: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT
    kcut: The exponential cutoff to use in the gaussian filter

    Outputs:
    f_filt: Gaussian filtered version of field
    """

    fhat = fft.forward(field, normalize=True)
    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // nranks : (rank + 1) * nmesh // nranks]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    filter = np.exp(-(kx ** 2 + ky ** 2 + kz ** 2) / (2 * kcut ** 2))
    fhat = filter * fhat
    del filter, kx, ky, kz

    f_filt = fft.backward(fhat)

    return f_filt


def compute_transfer_function(configs, z, k_in, p_in):
    
    pkclass = Class()
    pkclass.set(configs["Cosmology"])
    pkclass.compute()

    h = configs["Cosmology"]["h"]

    p_cb_lin = np.array(
        [pkclass.pk_cb_lin(k, np.array([z])) * h ** 3 for k in k_in * h]
    )
    transfer = np.sqrt(p_cb_lin / p_in)

    return transfer, p_cb_lin


def apply_transfer_function(field, nmesh, lbox, rank, nranks, fft, k_t, transfer):

    transfer_interp = interp1d(k_t, transfer, kind="cubic", fill_value='extrapolate')

    fhat = fft.forward(field, normalize=True)
    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // nranks : (rank + 1) * nmesh // nranks]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    k_norm = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    transfer_k = transfer_interp(k_norm)
    transfer_k[0][0][0] = 1

    fhat = transfer_k * fhat
    f_filt = fft.backward(fhat)

    return f_filt


def apply_scale_dependent_growth(field, nmesh, lbox, rank, nranks, fft, configs, z):

    pk_in = np.genfromtxt(configs["p_lin_ic_file"])
    k_in = pk_in[:, 0]
    p_in = pk_in[:, 1] * (2 * np.pi)**3

    transfer, p_cb_lin = compute_transfer_function(configs, z, k_in, p_in)

    f_filt = apply_transfer_function(
        field, nmesh, lbox, rank, nranks, fft, k_in, transfer
    )

    return f_filt


def make_lagfields(configs, save_to_disk=False, z=None):

    if configs["ic_format"] == "monofonic":
        lindir = configs["icdir"]
    else:
        lindir = configs["outdir"]

    outdir = configs["outdir"]
    nmesh = configs["nmesh_in"]
    start_time = time.time()
    Lbox = configs["lbox"]
    compute_cv_surrogate = configs["compute_cv_surrogate"]
    scale_dependent_growth = configs["scale_dependent_growth"]
    if compute_cv_surrogate:
        basename = "mpi_icfields_nmesh_filt"
        if configs["surrogate_gaussian_cutoff"]:
            gaussian_kcut = np.pi * nmesh / Lbox
    else:
        basename = "mpi_icfields_nmesh"

    # load linear density field (and displacements for surrogates)
    try:
        if configs["ic_format"] == "monofonic":
            ics = h5py.File(lindir, "a", driver="mpio", comm=MPI.COMM_WORLD)
            bigmesh = ics["DM_delta"]

            if compute_cv_surrogate:
                psi_x = ics["DM_dx"]
                psi_y = ics["DM_dy"]
                psi_z = ics["DM_dz"]
        else:
            bigmesh = np.load(lindir + "linICfield.npy", mmap_mode="r")
    except Exception as e:
        if configs["ic_format"] == "monofonic":
            print(
                "Couldn't find {}. Make sure you've produced  \\\
                   with generic output format."
            )
        else:
            print(
                "Have you run ic_binary_to_field.py yet? Did not find the right file."
            )
        raise (e)

    N = np.array([nmesh, nmesh, nmesh], dtype=int)
    fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype="float32", grid=(-1,))
    u = newDistArray(fft, False)

    # Slab-decompose the noiseless ICs along the distributed array
    u[:] = -bigmesh[rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :].astype(
        u.dtype
    )

    if scale_dependent_growth:
        assert z is not None
        u = apply_scale_dependent_growth(u, nmesh, Lbox, rank, nranks, fft, configs, z)

    if compute_cv_surrogate:
        u_filt = gaussian_filter(u, nmesh, Lbox, rank, nranks, fft, gaussian_kcut)
        del u

        p_x = newDistArray(fft, False, val=1)
        p_y = newDistArray(fft, False, val=2)
        p_z = newDistArray(fft, False, val=3)

        p_x[:] = psi_x[
            rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
        ].astype(psi_x.dtype)
        p_y[:] = psi_y[
            rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
        ].astype(psi_y.dtype)
        p_z[:] = psi_z[
            rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
        ].astype(psi_z.dtype)
        ics.close()

        # have to write out after each filter step, since mpi4py-fft will
        # overwrite arrays otherwise

        with h5py.File(lindir, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
            dset_delta = ics.create_dataset(
                "DM_delta_filt", (nmesh, nmesh, nmesh), dtype=u_filt.dtype
            )
            dset_delta[
                rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
            ] = u_filt[:]
            del u_filt

        p_x_filt = gaussian_filter(p_x, nmesh, Lbox, rank, nranks, fft, gaussian_kcut)
        del p_x
        with h5py.File(lindir, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
            dset_dx = ics.create_dataset(
                "DM_dx_filt", (nmesh, nmesh, nmesh), dtype=psi_x.dtype
            )
            dset_dx[
                rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
            ] = p_x_filt[:]
            del p_x_filt

        p_y_filt = gaussian_filter(p_y, nmesh, Lbox, rank, nranks, fft, gaussian_kcut)
        del p_y
        with h5py.File(lindir, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
            dset_dy = ics.create_dataset(
                "DM_dy_filt", (nmesh, nmesh, nmesh), dtype=psi_y.dtype
            )
            dset_dy[
                rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
            ] = p_y_filt[:]
            del p_y_filt

        p_z_filt = gaussian_filter(p_z, nmesh, Lbox, rank, nranks, fft, gaussian_kcut)
        del p_z
        with h5py.File(lindir, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
            dset_dz = ics.create_dataset(
                "DM_dz_filt", (nmesh, nmesh, nmesh), dtype=psi_z.dtype
            )
            dset_dz[
                rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
            ] = p_z_filt[:]
            del p_z_filt

        if rank == 0:
            print("done filtering", flush=True)

        u = newDistArray(fft, False)
        with h5py.File(lindir, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
            u[:] = ics["DM_delta_filt"][
                rank * nmesh // nranks : (rank + 1) * nmesh // nranks, :, :
            ]

    # Compute the delta^2 field. This operation is local in real space.
    d2 = newDistArray(fft, False)
    d2[:] = u * u
    dmean = MPI_mean(d2, nmesh)

    # Mean-subtract delta^2
    d2 -= dmean
    if rank == 0:
        print(dmean, " mean deltasq")

    # Parallel-write delta^2 to hdf5 file
    if save_to_disk:
        d2.write(outdir + "{}_{}.h5".format(basename, nmesh), "deltasq", step=2)
        u.write(outdir + "{}_{}.h5".format(basename, nmesh), "delta", step=2)

    # Take a forward FFT of the linear density
    u_hat = fft.forward(u, normalize=True)
    if rank == 0:
        print("Did backwards FFT")

    # Make a copy of FFT'd linear density. Will be used to make s^2 field.
    deltak = u_hat.copy()
    if rank == 0:
        print("Did array copy")
    tinyfft = delta_to_tidesq(deltak, nmesh, Lbox, rank, nranks, fft)
    if rank == 0:
        print("Made the tidesq field")

    # Populate output with distarray
    v = newDistArray(fft, False)
    v[:] = tinyfft

    # Need to compute mean value of tidesq to subtract:
    vmean = MPI_mean(v, nmesh)
    if rank == 0:
        print(vmean, " mean tidesq")
    v -= vmean

    if save_to_disk:
        v.write(outdir + "{}_{}.h5".format(basename, nmesh), "tidesq", step=2)

    # clear up space yet again
#    del v, tinyfft, vmean
#    gc.collect()

    # Now make the nablasq field
    ns = newDistArray(fft, False)

    nablasq = delta_to_gradsqdelta(deltak, nmesh, Lbox, rank, nranks, fft)

    ns[:] = nablasq

    if save_to_disk:
        v.write(outdir + "{}_{}.h5".format(basename, nmesh), "nablasq", step=2)
    # Moar space
    #del u, bigmesh, deltak, u_hat, fft, v
    #gc.collect()

    if configs["np_weightfields"]:

        if rank == 0:

            print("Wrote successfully! Now must convert to .npy files")
            print(time.time() - start_time, " seconds!")
            get_memory()
            f = h5py.File(outdir + "{}_{}.h5".format(basename, nmesh), "r")
            fkeys = list(f.keys())
            for key in fkeys:
                arr = f[key]["3D"]["2"]
                print("converting " + key + " to numpy array")
                np.save(outdir + "{}_{}_{}_np".format(basename, nmesh, key), arr)
                print(time.time() - start_time, " seconds!")
                del arr
                gc.collect()
                get_memory()
            # Deletes the hdf5 file
            os.system("rm " + outdir + "{}_{}.h5".format(basename, nmesh))
    else:
        if rank == 0:
            print("Wrote successfully! Took %d seconds" % (time.time() - start_time))
            
            
    return u, d2, v, ns


if __name__ == "__main__":
    yamldir = sys.argv[1]
    configs = yaml.load(open(yamldir, "r"))

    make_lagfields(configs)
