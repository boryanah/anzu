import sys, os
from mpi4py_fft import PFFT

sys.path.insert(0, "/global/homes/b/boryanah/repos/velocileptors/")
sys.path.insert(0, "/global/homes/b/boryanah/repos/anzu/")
sys.path.insert(0, "/global/homes/b/boryanah/repos/anzu/boryanah/")

from anzu.utils import combine_real_space_spectra, combine_measured_rsd_spectra
from scipy.interpolate import interp1d
from classy import Class
from mpi4py import MPI
from glob import glob
from yaml import Loader

from fields.make_lagfields import make_lagfields
from fields.measure_basis import advect_fields, measure_basis_spectra
from fields.common_functions import (
    get_snap_z,
    measure_pk,
    _get_resampler,
    CompensateCICAliasing,
    CompensateInterlacedCICAliasing,
)
from fields.field_level_bias import measure_field_level_bias

import numpy as np
import yaml
import pmesh
import h5py
import gc


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def pk_list_to_vec(pk_ij_list):

    nspec = len(pk_ij_list)
    keys = list(pk_ij_list[0].keys())
    k = pk_ij_list[0]["k"]
    nk = k.shape[0]

    if "mu" in keys:
        mu = pk_ij_list[0]["mu"]
        nmu = mu.shape[-1]
    else:
        nmu = 1
        mu = None

    if "power_poles" in keys:
        npoles = pk_ij_list[0]["power_poles"].shape[0]
        has_poles = True
        pk_pole_array = np.zeros((nspec, npoles, nk))

    else:
        npoles = 1
        has_poles = False

    pk_wedge_array = np.zeros((nspec, nk, nmu))

    for i in range(nspec):
        # power_wedges is always defined, even if only using 1d pk (then wedge is [0,1])
        pk_wedges = pk_ij_list[i]["power_wedges"]
        pk_wedge_array[i, ...] = pk_wedges.reshape(nk, -1)

        if has_poles:
            pk_poles = pk_ij_list[i]["power_poles"]
            pk_pole_array[i, ...] = pk_poles

    if has_poles:
        return k, mu, pk_wedge_array, pk_pole_array
    else:
        return k, mu, pk_wedge_array, None


def get_linear_field(
    config, lag_field_dict, rank, size, nmesh, Lbox, pm, bias_vec=None
):

    boltz = Class()
    boltz.set(config["Cosmology"])
    boltz.compute()
    z_ic = config["z_ic"]
    z_this = get_snap_z(config["particledir"], config["sim_type"])
    D = boltz.scale_independent_growth_factor(z_this)
    D = D / boltz.scale_independent_growth_factor(z_ic)
    f = boltz.scale_independent_growth_factor_f(z_this)

    if bias_vec is not None:
        b = bias_vec[0]
    else:
        b = 1

    if config["rsd"]:
        delta, fft = real_to_redshift_space(
            lag_field_dict["delta"], nmesh, Lbox, rank, size, f, b=b
        )
    else:
        delta = lag_field_dict["delta"]

    grid = np.meshgrid(
        np.arange(nmesh)[rank * nmesh // size : (rank + 1) * nmesh // size],
        np.arange(nmesh),
        np.arange(nmesh),
        indexing="ij",
    )
    pos_x = ((grid[0] / nmesh) % 1) * Lbox
    pos_y = ((grid[1] / nmesh) % 1) * Lbox
    pos_z = ((grid[2] / nmesh) % 1) * Lbox
    pos = np.stack([pos_x, pos_y, pos_z])
    pos = pos.reshape(3, -1).T
    del pos_x, pos_y, pos_z

    layout = pm.decompose(pos)
    p = layout.exchange(pos)
    d = layout.exchange(delta.flatten())

    mesh = pm.paint(p, mass=d)
    del p, d, pos

    mesh = (
        mesh.r2c()
    )  # don't need to dealias, because 'particles' are all on grid points
    field_dict = {"delta": mesh}
    field_D = [D]

    return field_dict, field_D, z_this


def real_to_redshift_space(field, nmesh, lbox, rank, nranks, f, fft=None, b=1):

    if fft is None:
        N = np.array([nmesh, nmesh, nmesh], dtype=int)
        fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype="float32", grid=(-1,))

    field_k = fft.forward(field)

    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // nranks : (rank + 1) * nmesh // nranks]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    knorm = kx**2 + ky**2 + kz**2
    mu = kz / np.sqrt(knorm)

    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1
        mu[0][0][0] = 0

    rsdfac = b + f * mu**2
    del kx, ky, kz, mu

    field_k_rsd = field_k * rsdfac
    field_rsd = fft.backward(field_k_rsd)

    return field_rsd, fft


def get_cv_fields(
    config, lindir, basename, lbox, nmesh, linear_surrogate=False, bias_vec=[None]
):

    resampler_type = "cic"
    resampler = _get_resampler(resampler_type)

    # tuks
    linfields = glob(lindir + "{}_{}_*_np.npy".format(basename, nmesh))
    if len(linfields) == 0:
        make_lagfields(config, save_to_disk=True)
        lag_field_dict = None
    elif linear_surrogate:
        lag_field_dict = {}
        arr = np.load(
            lindir + "{}_{}_{}_np.npy".format(basename, nmesh, "delta"),
            mmap_mode="r",
        )
        lag_field_dict["delta"] = arr[
            rank * nmesh // size : (rank + 1) * nmesh // size, :, :
        ]
        keynames = ["delta"]
        labelvec = ["delta"]
    else:
        lag_field_dict = None

    # advect ZA fields
    if not linear_surrogate:
        pm, field_dict, field_D, keynames, labelvec, zbox = advect_fields(
            config, lag_field_dict=lag_field_dict
        )
    else:
        pm = pmesh.pm.ParticleMesh(
            [nmesh, nmesh, nmesh], lbox, dtype="float32", resampler=resampler, comm=comm
        )
        field_dict, field_D, zbox = get_linear_field(
            config, lag_field_dict, rank, size, nmesh, lbox, pm, bias_vec=bias_vec[0]
        )

    return pm, field_dict, field_D, keynames, labelvec, zbox


def tracer_power(
    tracer_pos, resampler, pm, Lbox, nmesh, rsd=False, use_pypower=True, interlaced=True
):
    layout = pm.decompose(tracer_pos)
    p = layout.exchange(tracer_pos)
    tracerfield = pm.paint(p, mass=1, resampler=resampler)

    if interlaced:
        H = Lbox / nmesh
        shifted = pm.affine.shift(0.5)
        field_interlaced = pm.create(type="real")
        field_interlaced[:] = 0
        pm.paint(
            p, mass=1, resampler=resampler, out=field_interlaced, transform=shifted
        )

        c1 = tracerfield.r2c()
        c2 = field_interlaced.r2c()

        for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
            kH = sum(k[i] * H for i in range(3))
            s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kH)

        c1.c2r(tracerfield)

    tracerfield = tracerfield / tracerfield.cmean() - 1
    tracerfield = tracerfield.r2c()

    if interlaced:
        tracerfield.apply(CompensateInterlacedCICAliasing, kind="circular")
    else:
        tracerfield.apply(CompensateCICAliasing, kind="circular")

    del tracer_pos, p

    # measure tracer auto-power
    pk_tt_dict = measure_pk(
        tracerfield, tracerfield, Lbox, nmesh, rsd, use_pypower, 1, 1
    )

    return tracerfield, pk_tt_dict


def field_level_bias(
    tracerfield,
    field_dict,
    field_D,
    nmesh,
    kmax,
    Lbox,
    pm,
    rsd,
    M=None,
    save=True,
    outdir=None,
    stype=None,
    tbase=None,
    zbox=None,
):

    if "1m" in field_dict:
        dm = field_dict.pop("1m")
        d = field_D[0]
        field_D = field_D[1:]
    else:
        dm = None
    M, A, bv, zafield = measure_field_level_bias(
        comm, pm, tracerfield, field_dict, field_D, nmesh, kmax, Lbox, M=M
    )

    if dm is not None:
        field_dict["1m"] = dm
        temp = np.copy(field_D)
        field_D = np.zeros(len(field_D) + 1)
        field_D[0] = d
        field_D[1:] = temp

    if save:
        np.save(
            "{}/b_{}cv_rsd={}_kmax{:.4f}_{}_a{:.4f}.npy".format(
                outdir, stype, rsd, kmax[0], tbase, 1 / (zbox + 1)
            ),
            bv,
        )
        np.save(
            "{}/M_{}cv_rsd={}_kmax{:.4f}_a{:.4f}.npy".format(
                outdir, stype, rsd, kmax[0], 1 / (zbox + 1)
            ),
            M,
        )
        np.save(
            "{}/A_{}cv_rsd={}_kmax{:.4f}_{}_a{:.4f}.npy".format(
                outdir, stype, rsd, kmax[0], tbase, 1 / (zbox + 1)
            ),
            A,
        )

    return bv, zafield


def error_power_spectrum(
    tracerfield,
    bias_vec,
    field_dict,
    field_D,
    nmesh,
    kmax,
    Lbox,
    rsd,
    save=False,
    lindir=None,
    use_pypower=True,
    stype=None,
    tbase=None,
    zbox=None,
):

    zafield = field_dict["1cb"].copy()
    counter = 0

    for j, k in enumerate(field_dict):
        if (k == "1m") | (k == "1cb"):
            continue

        try:
            zafield += bias_vec[counter] * field_D[counter] * field_dict[k]
            counter += 1
        except IndexError as e:
            continue

    eps = tracerfield - zafield
    pk_ee = measure_pk(eps, eps, Lbox, nmesh, rsd, use_pypower, 1, 1)
    pk_zz_fl = measure_pk(zafield, zafield, Lbox, nmesh, rsd, use_pypower, 1, 1)

    if save:
        np.save(
            lindir
            + "{}cv_surrogate_{}_resid_pk_rsd={}_kmax{:0.4f}_opmax{}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                stype, tbase, rsd, kmax[0], 4, use_pypower, 1 / (zbox + 1), nmesh
            ),
            [pk_ee],
        )

        np.save(
            lindir
            + "{}cv_surrogate_{}_fieldsum_pk_rsd={}_kmax{:.4f}_opmax{}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                stype, tbase, rsd, kmax[0], 4, use_pypower, 1 / (zbox + 1), nmesh
            ),
            [pk_zz_fl],
        )


def reduce_variance(
    config, tracer_files=None, tracer_pos_list=None, save=True, measure_perr=False
):

    # configuration
    if (tracer_files is None) & (tracer_pos_list is None):
        tracer_files = [config["tracer_file"]]
        read_tracers = True
        n_tracers = 1
    elif tracer_pos_list:
        read_tracers = False
        n_tracers = len(tracer_pos_list)
    else:
        read_tracers = True
        n_tracers = 1

    lattice_type = int(config.get("lattice_type", 0))
    config["lattice_type"] = lattice_type
    lindir = config["outdir"]

    rsd = config["rsd"]
    resampler_type = "cic"
    resampler = _get_resampler(resampler_type)
    interlaced = config.get("interlaced", False)
    kmax = np.atleast_1d(config['field_level_kmax'])
    nmesh = int(config['nmesh_in'])
    nmesh_out = int(config['nmesh_out'])
    Lbox = float(config['lbox'])    
    linear_surrogate = config.get('linear_surrogate', False)
    measure_cross_spectra = config.get('measure_cross_spectra', True)
    filt = config.get('surrogate_gaussian_cutoff', True)

    resampler_type = 'cic'
    resampler = _get_resampler(resampler_type)

    if config['compute_cv_surrogate']:
        if filt:
            basename = "mpi_icfields_nmesh_filt"
        else:
            basename = "mpi_icfields_nmesh"
    else:
        basename = "mpi_icfields_nmesh"

    # Optionally pass measured biases, or ask for them to
    # be fit for a the field level.
    if "bias_vec" in config:
        bias_vec = config["bias_vec"]
        field_level_bias = False
        M = None
    else:
        bias_vec = [None] * len(tracer_files)
        field_level_bias = config.get("field_level_bias", False)
        M_file = config.get("field_level_M", None)
        if M_file:
            M = np.load(M_file)
        else:
            M = None

    if linear_surrogate:
        stype = "l"
    elif config["compute_cv_surrogate"]:
        stype = "z"
    else:
        stype = "heft"

    # get the relevant cv fields
    pm, field_dict, field_D, keynames, labelvec, zbox = get_cv_fields(
        config,
        lindir,
        basename,
        Lbox,
        nmesh,
        linear_surrogate=linear_surrogate,
        bias_vec=bias_vec[0],
    )

    for ii in range(n_tracers):
        if read_tracers:
            tracer_file = tracer_files[ii]

            if rsd:
                tracer_pos = h5py.File(tracer_file)["pos_zspace"][rank::size, :]
            else:
                tracer_pos = h5py.File(tracer_file)["pos_rspace"][rank::size, :]
        else:
            tracer_pos = tracer_pos_list[ii]
            tracer_file = None

        tracerfield, pk_tt_dict = tracer_power(
            tracer_pos, resampler, pm, Lbox, nmesh_out, rsd=rsd, interlaced=interlaced
        )
        
        field_dict2 = {"t": tracerfield}
        field_D2 = [1]

        if tracer_file:
            tbase = tracer_file.split("/")[-1]
        else:
            tbase = "tt_{}".format(ii)

        np.save(
            lindir
            + "{}_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                tbase, config["rsd"], config["use_pypower"], 1 / (zbox + 1), nmesh_out
            ),
            [pk_tt_dict],
        )

        if measure_cross_spectra:

            pk_auto_vec, pk_cross_vec = measure_basis_spectra(
                config,
                field_dict,
                field_D,
                keynames,
                labelvec,
                zbox,
                field_dict2=field_dict2,
                field_D2=field_D2,
                save=False,
            )

            np.save(
                lindir
                + "{}cv_surrogate_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                    stype, config["rsd"], config["use_pypower"], 1 / (zbox + 1), nmesh_out
                ),
                pk_auto_vec,
            )

            np.save(
                lindir
                + "{}cv_cross_{}_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                    stype,
                    tbase,
                    config["rsd"],
                    config["use_pypower"],
                    1 / (zbox + 1),
                    nmesh_out,
                ),
                pk_cross_vec,
            )

        if (bias_vec[ii] is None) & (field_level_bias):
            bv, field_level_bias(
                tracerfield,
                field_dict,
                field_D,
                nmesh,
                kmax,
                Lbox,
                pm,
                rsd,
                M=M,
                save=save,
                outdir=lindir,
                stype=stype,
                tbase=tbase,
                zbox=zbox,
            )
        else:
            bv = bias_vec[ii]

        if measure_perr:
            error_power_spectrum(
                tracerfield,
                bv,
                field_dict,
                field_D,
                nmesh_out,
                kmax,
                Lbox,
                rsd,
                save=save,
                lindir=lindir,
                use_pypower=True,
                stype=stype,
                tbase=tbase,
                zbox=zbox,
            )

# CV estimator goes here
#        np.save(
#            lindir
#            + "{}_auto_zcved_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
#                tbase, config["rsd"], config["use_pypower"], 1 / (zbox + 1), nmesh
#            ),
#            [pk_tt_hat],
#        )


if __name__ == "__main__":

    config = sys.argv[1]
    if len(sys.argv) > 2:
        tracer_files = sys.argv[2:]
    else:
        tracer_files = None

    with open(config, "r") as fp:
        config = yaml.load(fp, Loader=Loader)

    if tracer_files is None:
        tracer_files = [config["tracer_file"]]

    reduce_variance(config, tracer_files=tracer_files)
