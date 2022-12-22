import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
from classy import Class
from scipy.interpolate import interp1d
import time
import gc
import sys
import h5py
import asdf
import yaml
import os
from .common_functions import get_memory, kroneckerdelta

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_linear_field(ic_format, ic_path, nmesh, fft, cv=False):
    """Load a linear density field.

    Args:
        ic_format (string): The format of the IC file.
            Supported options are monofonic, ngenic, or abacus.
        ic_path (string): The file to read the linear density from.
    """

    delta_lin = newDistArray(fft, False)

    try:
        if ic_format == "monofonic":
            ics = h5py.File(ic_path, "a", driver="mpio", comm=MPI.COMM_WORLD)
            linfield = ics["DM_delta"]

            delta_lin[:] = -linfield[
                rank * nmesh // size : (rank + 1) * nmesh // size, :, :
            ].astype(linfield.dtype)

            if cv:
                psi_x = ics["DM_dx"]
                psi_y = ics["DM_dy"]
                psi_z = ics["DM_dz"]

                p_x = newDistArray(fft, False, val=1)
                p_y = newDistArray(fft, False, val=2)
                p_z = newDistArray(fft, False, val=3)

                p_x[:] = psi_x[
                    rank * nmesh // size : (rank + 1) * nmesh // size, :, :
                ].astype(psi_x.dtype)
                p_y[:] = psi_y[
                    rank * nmesh // size : (rank + 1) * nmesh // size, :, :
                ].astype(psi_y.dtype)
                p_z[:] = psi_z[
                    rank * nmesh // size : (rank + 1) * nmesh // size, :, :
                ].astype(psi_z.dtype)

            ics.close()

        elif ic_format == "ngenic":
            linfield = np.load(ic_path + "linICfield.npy", mmap_mode="r")
            delta_lin = -linfield[
                rank * nmesh // size : (rank + 1) * nmesh // size, :, :
            ].astype(linfield.dtype)

        elif ic_format == "abacus":
            print(ic_path)
            f = asdf.open(ic_path+f"ic_dens_N{nmesh:d}.asdf")
            delta_lin[:] = f['data']['density'][
                rank * nmesh // size : (rank + 1) * nmesh // size, :, :
            ].astype(f['data']['density'].dtype)
            f.close()

            if cv:
                f = asdf.open(ic_path+f"ic_disp_N{nmesh:d}.asdf")
                Lbox = f['header']['BoxSize']
                psi_x = f['data']['displacements'][:, :, :, 0]/Lbox
                psi_y = f['data']['displacements'][:, :, :, 1]/Lbox
                psi_z = f['data']['displacements'][:, :, :, 2]/Lbox
                f.close()
                
                p_x = newDistArray(fft, False, val=1)
                p_y = newDistArray(fft, False, val=2)
                p_z = newDistArray(fft, False, val=3)

                p_x[:] = psi_x[
                    rank * nmesh // size : (rank + 1) * nmesh // size, :, :
                ].astype(psi_x.dtype)
                p_y[:] = psi_y[
                    rank * nmesh // size : (rank + 1) * nmesh // size, :, :
                ].astype(psi_y.dtype)
                p_z[:] = psi_z[
                    rank * nmesh // size : (rank + 1) * nmesh // size, :, :
                ].astype(psi_z.dtype)

    except Exception as e:
        if ic_format == "monofonic":
            print(
                "Couldn't find {}. Make sure you've produced  \\\
                   with generic output format."
            )
        else:
            print(
                "Have you run ic_binary_to_field.py yet? Did not find the right file."
            )
        raise (e)

    # Slab-decompose the noiseless ICs along the distributed array

    if cv:
        return delta_lin, p_x, p_y, p_z
    else:
        return delta_lin


def filter_linear_fields(
    delta_lin, p_x, p_y, p_z, nmesh, Lbox, fft, gaussian_kcut, filt_ics_file
):
    d_filt = gaussian_filter(delta_lin, nmesh, Lbox, rank, size, fft, gaussian_kcut)
    del delta_lin

    # have to write out after each filter step, since mpi4py-fft will
    # overwrite arrays otherwise

    with h5py.File(filt_ics_file, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
        try:
            dset_delta = ics.create_dataset(
                "DM_delta_filt", (nmesh, nmesh, nmesh), dtype=d_filt.dtype
            )
        except Exception as e:
            print(e)
            dset_delta = ics["DM_delta_filt"]

        dset_delta[rank * nmesh // size : (rank + 1) * nmesh // size, :, :] = d_filt[:]
        del d_filt

    p_x_filt = gaussian_filter(p_x, nmesh, Lbox, rank, size, fft, gaussian_kcut)
    del p_x

    with h5py.File(filt_ics_file, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
        
        try:
            dset_dx = ics.create_dataset(
                "DM_dx_filt", (nmesh, nmesh, nmesh), dtype=p_x_filt.dtype
            )
        except Exception as e:
            print(e)
            dset_dx = ics["DM_dx_filt"]

        dset_dx[rank * nmesh // size : (rank + 1) * nmesh // size, :, :] = p_x_filt[:]
        del p_x_filt

    p_y_filt = gaussian_filter(p_y, nmesh, Lbox, rank, size, fft, gaussian_kcut)
    del p_y
    with h5py.File(filt_ics_file, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
        try:
            dset_dy = ics.create_dataset(
                "DM_dy_filt", (nmesh, nmesh, nmesh), dtype=p_y_filt.dtype
            )
        except Exception as e:
            print(e)
            dset_dy = ics["DM_dy_filt"]

        dset_dy[rank * nmesh // size : (rank + 1) * nmesh // size, :, :] = p_y_filt[:]
        del p_y_filt

    p_z_filt = gaussian_filter(p_z, nmesh, Lbox, rank, size, fft, gaussian_kcut)
    del p_z
    with h5py.File(filt_ics_file, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
        try:
            dset_dz = ics.create_dataset(
                "DM_dz_filt", (nmesh, nmesh, nmesh), dtype=p_z_filt.dtype
            )
        except Exception as e:
            print(e)
            dset_dz = ics["DM_dz_filt"]

        dset_dz[rank * nmesh // size : (rank + 1) * nmesh // size, :, :] = p_z_filt[:]
        del p_z_filt

    delta_lin = newDistArray(fft, False)
    with h5py.File(filt_ics_file, "a", driver="mpio", comm=MPI.COMM_WORLD) as ics:
        delta_lin[:] = ics["DM_delta_filt"][
            rank * nmesh // size : (rank + 1) * nmesh // size, :, :
        ]

    return delta_lin


def MPI_mean(array, nmesh):
    """
    Computes the mean of an array that is slab-decomposed across multiple processes.
    """
    procsum = np.sum(array) * np.ones(1)
    recvbuf = None
    if rank == 0:
        recvbuf = np.zeros(shape=[size, 1])
    comm.Gather(procsum, recvbuf, root=0)
    if rank == 0:
        fieldmean = np.ones(1) * np.sum(recvbuf) / nmesh**3
    else:
        fieldmean = np.ones(1)
    comm.Bcast(fieldmean, root=0)
    return fieldmean[0]


def delta_to_tidesq(delta_k, nmesh, lbox, rank, size, fft):
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
    size: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT.

    Outputs:
    tidesq: the s^2 field for the given slab.
    """

    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // size : (rank + 1) * nmesh // size]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    knorm = kx**2 + ky**2 + kz**2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    klist = [[kx, kx], [kx, ky], [kx, kz], [ky, ky], [ky, kz], [kz, kz]]

    del kx, ky, kz
    gc.collect()

    # Compute the symmetric tide at every Fourier mode which we'll reshape later
    # Order is xx, xy, xz, yy, yz, zz
    jvec = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]
    tidesq = np.zeros((nmesh // size, nmesh, nmesh), dtype="float32")

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

        tidesq += 1.0 * real_out**2
        if jvec[i][0] != jvec[i][1]:
            tidesq += 1.0 * real_out**2

        del fft_tide, real_out
        gc.collect()

    return tidesq


def delta_to_gradsqdelta(delta_k, nmesh, lbox, rank, size, fft):
    """
    Computes the density curvature from the density FFT

    nabla^2 delta = IFFT(-k^2 delta_k)

    Inputs:
    delta_k: fft'd density, slab-decomposed.
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    size: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT.

    Outputs:
    real_gradsqdelta: the nabla^2delta field for the given slab.
    """

    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // size : (rank + 1) * nmesh // size]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)

    knorm = kx**2 + ky**2 + kz**2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    del kx, ky, kz
    gc.collect()

    # Compute -k^2 delta which is the gradient
    ksqdelta = -np.array(knorm * (delta_k), dtype="complex64")

    real_gradsqdelta = fft.backward(ksqdelta)

    return real_gradsqdelta


def gaussian_filter(field, nmesh, lbox, rank, size, fft, kcut):
    """
    Apply a fourier space gaussian filter to a field

    Inputs:
    field: the field to filter
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    size: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT
    kcut: The exponential cutoff to use in the gaussian filter

    Outputs:
    f_filt: Gaussian filtered version of field
    """

    fhat = fft.forward(field, normalize=True)
    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // size : (rank + 1) * nmesh // size]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    filter = np.exp(-(kx**2 + ky**2 + kz**2) / (2 * kcut**2))
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
        [pkclass.pk_cb_lin(k, np.array([z])) * h**3 for k in k_in * h]
    )
    transfer = np.sqrt(p_cb_lin / p_in)

    return transfer, p_cb_lin


def apply_transfer_function(field, nmesh, lbox, rank, size, fft, k_t, transfer):

    transfer_interp = interp1d(k_t, transfer, kind="cubic", fill_value="extrapolate")

    fhat = fft.forward(field, normalize=True)
    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // size : (rank + 1) * nmesh // size]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    k_norm = np.sqrt(kx**2 + ky**2 + kz**2)
    transfer_k = transfer_interp(k_norm)
    transfer_k[0][0][0] = 1

    fhat = transfer_k * fhat
    f_filt = fft.backward(fhat)

    return f_filt


def apply_scale_dependent_growth(field, nmesh, lbox, rank, size, fft, configs, z):

    pk_in = np.genfromtxt(configs["p_lin_ic_file"])
    k_in = pk_in[:, 0]
    p_in = pk_in[:, 1] * (2 * np.pi) ** 3

    transfer, p_cb_lin = compute_transfer_function(configs, z, k_in, p_in)

    f_filt = apply_transfer_function(
        field, nmesh, lbox, rank, size, fft, k_in, transfer
    )

    return f_filt


def make_lagfields(configs, save_to_disk=False, z=None):

    if configs["ic_format"] == "monofonic":
        lindir = configs["icdir"]
    else:
        #lindir = configs["outdir"] 
        lindir = configs["icdir"]

    outdir = configs["outdir"]
    nmesh = configs["nmesh_in"]
    start_time = time.time()
    Lbox = configs["lbox"]
    compute_cv_surrogate = configs["compute_cv_surrogate"]
    scale_dependent_growth = configs["scale_dependent_growth"]
    if compute_cv_surrogate:
        basename = "mpi_icfields_nmesh_filt"
        if configs["surrogate_gaussian_cutoff"] is not False:
            if configs["surrogate_gaussian_cutoff"] == True:
                gaussian_kcut = np.pi * nmesh / Lbox
            else:
                gaussian_kcut = float(configs["surrogate_gaussian_cutoff"])
        else:
            gaussian_kcut = None
            if rank==0:
                print('gaussian_kcut:', gaussian_kcut, flush=True)
    else:
        basename = "mpi_icfields_nmesh"
        gaussian_kcut = None
    filt_ics_file = "/".join(outdir.split("/")[:-1]) + f"/filtered_ics_nmesh{nmesh:d}.h5"

    # set up fft objects
    N = np.array([nmesh, nmesh, nmesh], dtype=int)
    fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype="float32", grid=(-1,))

    # load linear density field (and displacements for surrogates)
    if not compute_cv_surrogate:
        delta_lin = get_linear_field(configs["ic_format"], lindir, nmesh, fft)
    else:
        delta_lin, p_x, p_y, p_z = get_linear_field(
            configs["ic_format"], lindir, nmesh, fft, cv=True
        )

    # We never apply scale dependent growth and compute cvs at the same time.
    if scale_dependent_growth:
        assert z is not None
        if rank == 0:
            print("Applying scale dependent growth transfer function", flush=True)
        delta_lin = apply_scale_dependent_growth(
            delta_lin, nmesh, Lbox, rank, size, fft, configs, z
        )

    elif compute_cv_surrogate:
        delta_lin = filter_linear_fields(
            delta_lin, p_x, p_y, p_z, nmesh, Lbox, fft, gaussian_kcut, filt_ics_file
        )

    d = newDistArray(fft, False)
    d[:] = delta_lin
    dmean = MPI_mean(d, nmesh)
    d -= dmean

    # Compute the delta^2 field. This operation is local in real space.
    d2 = newDistArray(fft, False)
    d2[:] = delta_lin * delta_lin
    dmean = MPI_mean(d2, nmesh)

    # Mean-subtract delta^2
    d2 -= dmean
    if rank == 0:
        print(dmean, " mean deltasq")

    # Parallel-write delta^2 to hdf5 file
    if save_to_disk:
        d2.write(outdir + "{}_{}.h5".format(basename, nmesh), "deltasq", step=2)
        d.write(outdir + "{}_{}.h5".format(basename, nmesh), "delta", step=2)

    u_hat = fft.forward(delta_lin, normalize=True)
    deltak = u_hat.copy()

    tidesq = delta_to_tidesq(deltak, nmesh, Lbox, rank, size, fft)

    # Populate output with distarray
    s2 = newDistArray(fft, False)
    s2[:] = tidesq

    # Need to compute mean value of tidesq to subtract:
    vmean = MPI_mean(s2, nmesh)
    if rank == 0:
        print(vmean, " mean tidesq")
    s2 -= vmean

    if save_to_disk:
        s2.write(outdir + "{}_{}.h5".format(basename, nmesh), "tidesq", step=2)

    # Now make the nablasq field
    ns = newDistArray(fft, False)
    nablasq = delta_to_gradsqdelta(deltak, nmesh, Lbox, rank, size, fft)
    ns[:] = nablasq

    if save_to_disk:
        ns.write(outdir + "{}_{}.h5".format(basename, nmesh), "nablasq", step=2)

    if save_to_disk & configs["np_weightfields"]:

        if rank == 0:

            print("Wrote successfully! Now must convert to .npy files")
            print(time.time() - start_time, " seconds!")
            get_memory()

            with h5py.File(outdir + "{}_{}.h5".format(basename, nmesh), "r") as f:
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

    fieldnames = ["delta", "deltasq", "tidesq", "nablasq"]
    lag_field_dict = dict(zip(fieldnames, [d, d2, s2, ns]))

    return lag_field_dict


if __name__ == "__main__":
    yamldir = sys.argv[1]
    configs = yaml.load(open(yamldir, "r"))

    make_lagfields(configs)
