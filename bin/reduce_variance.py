from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from itertools import product
from classy import Class
from glob import glob
from mpi4py import MPI
import numpy as np
import h5py as h5
from yaml import Loader
import os, yaml


import sys

sys.path.append("/global/homes/j/jderose/project/ZeNBu/")
from zenbu import Zenbu


def _lpt_pk(
    k,
    p_lin,
    D=None,
    cleftobj=None,
    kecleft=False,
    zenbu=True,
    cutoff=np.pi * 700 / 525.0,
):
    """
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            If kecleft==True, then should be for z=0, and redshift evolution is
            handled by passing the appropriate linear growth factor to D.
        D: float
            Linear growth factor. Only required if kecleft==True.
        kecleft: bool
            Whether to use kecleft or not. Setting kecleft==True
            allows for faster computation of spectra keeping cosmology
            fixed and varying redshift if the cleftobj from the
            previous calculation at the same cosmology is provided to
            the cleftobj keyword.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    """

    if kecleft:
        if D is None:
            raise (ValueError("Must provide growth factor if using kecleft"))

        # If using kecleft, check if we're only varying the redshift

        if cleftobj is None:
            # Function to obtain the no-wiggle spectrum.
            # Not implemented yet, maybe Wallisch maybe B-Splines?
            # pnw = p_nwify(pk)
            # For now just use Stephen's standard savgol implementation.
            cleftobj = RKECLEFT(k, p_lin)

        # Adjust growth factors
        cleftobj.make_ptable(D=D, kmin=k[0], kmax=k[-1], nk=1000)
        cleftpk = cleftobj.pktable.T

    elif zenbu:
        zobj = Zenbu(k, p_lin, cutoff=cutoff, N=3000, jn=15)
        zobj.make_ptable()
        cleftpk = zobj.pktable.T

    else:
        # Using "full" CLEFT, have to always do calculation from scratch
        cleftobj = CLEFT(k, p_lin, N=2700, jn=10, cutoff=1)
        cleftobj.make_ptable()

        cleftpk = cleftobj.pktable.T

        # Different cutoff for other spectra, because otherwise different
        # large scale asymptote

        cleftobj = CLEFT(k, p_lin, N=2700, jn=5, cutoff=10)
        cleftobj.make_ptable()

    if not zenbu:
        cleftpk[2, :] /= 2
        cleftpk[6, :] /= 0.25
        cleftpk[7, :] /= 2
        cleftpk[8, :] /= 2

    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value="extrapolate")

    return cleftspline, cleftobj


def _cleft_pk(k, p_lin, D=None, cleftobj=None, kecleft=True):
    """
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            If kecleft==True, then should be for z=0, and redshift evolution is
            handled by passing the appropriate linear growth factor to D.
        D: float
            Linear growth factor. Only required if kecleft==True.
        kecleft: bool
            Whether to use kecleft or not. Setting kecleft==True
            allows for faster computation of spectra keeping cosmology
            fixed and varying redshift if the cleftobj from the
            previous calculation at the same cosmology is provided to
            the cleftobj keyword.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    """

    if kecleft:
        if D is None:
            raise (ValueError("Must provide growth factor if using kecleft"))

        # If using kecleft, check if we're only varying the redshift

        if cleftobj is None:
            # Function to obtain the no-wiggle spectrum.
            # Not implemented yet, maybe Wallisch maybe B-Splines?
            # pnw = p_nwify(pk)
            # For now just use Stephen's standard savgol implementation.
            cleftobj = RKECLEFT(k, p_lin)

        # Adjust growth factors
        cleftobj.make_ptable(D=D, kmin=k[0], kmax=k[-1], nk=1000)
        cleftpk = cleftobj.pktable.T

    else:
        # Using "full" CLEFT, have to always do calculation from scratch
        cleftobj = CLEFT(k, p_lin, N=2700, jn=10, cutoff=1)
        cleftobj.make_ptable()

        cleftpk = cleftobj.pktable.T

        # Different cutoff for other spectra, because otherwise different
        # large scale asymptote

        cleftobj = CLEFT(k, p_lin, N=2700, jn=5, cutoff=10)
        cleftobj.make_ptable()

    cleftpk[3:, :] = cleftobj.pktable.T[3:, :]
    cleftpk[2, :] /= 2
    cleftpk[6, :] /= 0.25
    cleftpk[7, :] /= 2
    cleftpk[8, :] /= 2

    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value="extrapolate")

    return cleftspline, cleftobj


def load_pk_from_dict(filename):

    pk_ij_list = np.load(filename, allow_pickle=True)
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
        return k, mu, pk_wedge_array


def get_spectra_from_fields(fields1, fields2, neutrinos=True):
    spectra = []
    for i, fi in enumerate(fields1):
        for j, fj in enumerate(fields2):
            if (i < j) | (neutrinos & (i == 1) & (j == 0)):
                continue
            spectra.append((fi, fj))

    return spectra


def compute_beta_and_reduce_variance(
    k,
    pk_ij_nn,
    pk_ij_zn,
    pk_ij_zz,
    pk_ij_zenbu,
    neutrinos=True,
    k0=0.618,
    dk=0.167,
    sg_window=21,
):

    if neutrinos:
        fields_n = ["1m", "1cb", "d", "d2", "s", "n2"]
        component_spectra_nn = get_spectra_from_fields(fields_n, fields_n)
        nspec_nn = len(component_spectra_nn)
        pk_ij_nn_dict = dict(zip(component_spectra_nn, pk_ij_nn))
    else:
        fields_n = ["1cb", "d", "d2", "s", "n2"]
        component_spectra_nn = get_spectra_from_fields(
            fields_n, fields_n, neutrinos=False
        )
        nspec_nn = len(component_spectra_nn)
        pk_ij_nn_dict = dict(zip(component_spectra_nn, pk_ij_nn))

    fields_z = ["1", "d", "d2", "s", "n2"]
    fields_zenbu = ["1", "d", "d2", "s"]

    component_spectra_zz = get_spectra_from_fields(fields_z, fields_z, neutrinos=False)
    component_spectra_zenbu = get_spectra_from_fields(
        fields_z, fields_zenbu, neutrinos=False
    )

    pk_ij_zz_dict = dict(zip(component_spectra_zz, pk_ij_zz))
    pk_ij_zenbu_dict = dict(zip(component_spectra_zenbu, pk_ij_zenbu))

    component_spectra_zn = list(product(fields_z, fields_n))
    pk_ij_zn_dict = dict(zip(component_spectra_zn, pk_ij_zn))

    beta_ij = []
    beta_ij_damp = []
    beta_ij_smooth = []

    pk_ij_nn_hat = []
    pk_ij_nn_smooth = []
    pk_ij_nn_beta1 = []

    pk_ij_zz_l = []
    pk_ij_zenbu_l = []
    pk_ij_zn_l = []
    pk_ij_nz_l = []
    for n in range(nspec_nn):
        if (component_spectra_nn[n][0] == "n2") | (component_spectra_nn[n][1] == "n2"):
            continue

        f_i_n = component_spectra_nn[n][0]
        f_i_z = component_spectra_nn[n][0]
        f_j_n = component_spectra_nn[n][1]
        f_j_z = component_spectra_nn[n][1]

        if f_i_n[0] == "1":
            f_i_z = "1"
        if f_j_n[0] == "1":
            f_j_z = "1"

        p_ii_zn = pk_ij_zn_dict[(f_i_z, f_i_n)]
        p_jj_zn = pk_ij_zn_dict[(f_j_z, f_j_n)]
        p_ij_zn = pk_ij_zn_dict[(f_i_z, f_j_n)]
        p_ij_nz = pk_ij_zn_dict[(f_j_z, f_i_n)]

        try:
            p_ii_zz = pk_ij_zz_dict[(f_i_z, f_i_z)]
            p_jj_zz = pk_ij_zz_dict[(f_j_z, f_j_z)]
            p_ij_zz = pk_ij_zz_dict[(f_j_z, f_i_z)]
            p_ij_zenbu = pk_ij_zenbu_dict[(f_j_z, f_i_z)]
        except:
            p_ii_zz = pk_ij_zz_dict[(f_i_z, f_i_z)]
            p_jj_zz = pk_ij_zz_dict[(f_j_z, f_j_z)]
            p_ij_zz = pk_ij_zz_dict[(f_i_z, f_j_z)]
            p_ij_zenbu = pk_ij_zenbu_dict[(f_i_z, f_j_z)]

        beta = (p_ii_zn * p_jj_zn + p_ij_zn * p_ij_nz) / (
            p_ij_zz**2 + p_ii_zz * p_jj_zz
        )
        beta_damp = 1 / 2 * (1 - np.tanh((k - k0) / dk)) * beta
        beta_smooth = savgol_filter(beta_damp, sg_window, 3)
        beta1 = 1 / 2 * (1 - np.tanh((k - k0) / dk)) * np.ones_like(beta)

        beta_ij.append(beta)
        beta_ij_damp.append(beta_damp)
        beta_ij_smooth.append(beta_damp)

        p_ij_nn_hat = pk_ij_nn[n] - beta_damp * (p_ij_zz - p_ij_zenbu)
        p_ij_nn_smooth = pk_ij_nn[n] - beta_smooth * (p_ij_zz - p_ij_zenbu)
        p_ij_nn_beta1 = pk_ij_nn[n] - beta1 * (p_ij_zz - p_ij_zenbu)

        pk_ij_nn_hat.append(p_ij_nn_hat)
        pk_ij_nn_smooth.append(p_ij_nn_smooth)
        pk_ij_nn_beta1.append(p_ij_nn_beta1)

        pk_ij_zz_l.append(p_ij_zz)
        pk_ij_zenbu_l.append(p_ij_zenbu)
        pk_ij_zn_l.append(p_ij_zn)
        pk_ij_nz_l.append(p_ij_nz)

    return (
        pk_ij_nn_hat,
        pk_ij_nn_smooth,
        pk_ij_nn_beta1,
        beta_ij,
        beta_ij_damp,
        beta_ij_smooth,
        pk_ij_zz_l,
        pk_ij_zenbu_l,
        pk_ij_zn_l,
        pk_ij_nz_l,
    )


def lpt_spectra(
    k,
    z,
    anzu_config,
    kin,
    p_lin_in,
    pkclass=None,
    kecleftobj_m=None,
    kecleftobj_cb=None,
):

    with open(anzu_config, "r") as fp:
        cfg = yaml.load(fp, Loader=Loader)

    if pkclass == None:
        pkclass = Class()
        pkclass.set(cfg["Cosmology"])
        pkclass.compute()

    if cfg["surrogate_gaussian_cutoff"] is not False:
        if cfg["surrogate_gaussian_cutoff"] == True:
            cutoff = np.pi * cfg["nmesh_in"] / cfg["lbox"]
        else:
            cutoff = float(cfg["surrogate_gaussian_cutoff"])
    else:
        cutoff = 10

    kt = np.logspace(-3, 1, 100)

    pk_cb_lin_z0 = np.array(
        [
            pkclass.pk_cb_lin(ki, np.array([0])) * cfg["Cosmology"]["h"] ** 3
            for ki in kt * cfg["Cosmology"]["h"]
        ]
    )
    pk_m_lin_z0 = np.array(
        [
            pkclass.pk_lin(ki, np.array([0])) * cfg["Cosmology"]["h"] ** 3
            for ki in kt * cfg["Cosmology"]["h"]
        ]
    )
    pk_m_lin = np.array(
        [
            pkclass.pk_lin(ki, np.array([z])) * cfg["Cosmology"]["h"] ** 3
            for ki in kt * cfg["Cosmology"]["h"]
        ]
    )
    pk_cb_lin_zb = np.array(
        [
            pkclass.pk_cb_lin(ki, np.array([z])) * cfg["Cosmology"]["h"] ** 3
            for ki in kt * cfg["Cosmology"]["h"]
        ]
    )

    Dthis = pkclass.scale_independent_growth_factor(z)
    Dic = pkclass.scale_independent_growth_factor(cfg["z_ic"])

    zbspline, cleftobj = _lpt_pk(kin, p_lin_in * (Dthis / Dic) ** 2, cutoff=cutoff)

    cleft_m_spline, cleftobj = _cleft_pk(kt, pk_m_lin, kecleft=False)
    cleft_cb_spline, cleftobj = _cleft_pk(kt, pk_cb_lin_zb, kecleft=False)
    kecleft_m_spline, kecleftobj_m = _cleft_pk(
        kt, pk_m_lin_z0, D=Dthis, kecleft=True, cleftobj=kecleftobj_m
    )
    kecleft_cb_spline, kecleftobj_cb = _cleft_pk(
        kt, pk_cb_lin_z0, D=Dthis, kecleft=True, cleftobj=kecleftobj_cb
    )

    pk_zenbu = zbspline(k)
    pk_cb_3lpt = cleft_cb_spline(k)
    pk_m_3lpt = cleft_m_spline(k)
    pk_cb_ke3lpt = kecleft_cb_spline(k)
    pk_m_ke3lpt = kecleft_m_spline(k)

    return (
        pk_zenbu[1:],
        pk_m_3lpt[1:12],
        pk_cb_3lpt[1:12],
        pk_m_ke3lpt[1:12],
        pk_cb_ke3lpt[1:12],
        pkclass,
        kecleftobj_m,
        kecleftobj_cb,
    )


def reduce_variance(
    k,
    pk_ij_nn,
    pk_ij_zn,
    pk_ij_zz,
    anzu_config,
    z,
    kt,
    p_lin_in,
    neutrinos=True,
    pkclass=None,
    kecleftobj_m=None,
    kecleftobj_cb=None,
):

    (
        pk_ij_zenbu,
        pk_m_3lpt,
        pk_cb_3lpt,
        pk_m_ke3lpt,
        pk_cb_ke3lpt,
        pkclass,
        kecleftobj_m,
        kecleftobj_cb,
    ) = lpt_spectra(
        k,
        z,
        anzu_config,
        kt,
        p_lin_in,
        pkclass=pkclass,
        kecleftobj_m=kecleftobj_m,
        kecleftobj_cb=kecleftobj_cb,
    )

    (
        pk_ij_nn_hat,
        pk_ij_nn_smooth,
        pk_ij_nn_beta1,
        beta_ij,
        beta_ij_damp,
        beta_ij_smooth,
        pk_ij_zz_l,
        pk_ij_zenbu_l,
        pk_ij_zn_l,
        pk_ij_nz_l,
    ) = compute_beta_and_reduce_variance(
        k,
        pk_ij_nn,
        pk_ij_zn,
        pk_ij_zz,
        pk_ij_zenbu,
        neutrinos=neutrinos,
        k0=0.618,
        dk=0.167,
    )  # k0=0.1, dk=0.167)

    return (
        pk_ij_nn_hat,
        pk_ij_nn_smooth,
        pk_ij_nn_beta1,
        beta_ij,
        beta_ij_damp,
        beta_ij_smooth,
        pk_ij_zz_l,
        pk_ij_zenbu_l,
        pk_ij_zn_l,
        pk_ij_nz_l,
        pk_m_3lpt,
        pk_cb_3lpt,
        pk_m_ke3lpt,
        pk_cb_ke3lpt,
        pkclass,
        kecleftobj_m,
        kecleftobj_cb,
    )


def reduce_variance_fullsim(configbase, rsd=False):

    with open("{}/anzu_fields.param".format(configbase), "r") as fp:
        cfg = yaml.load(fp, Loader=Loader)

    basename = cfg["outdir"]
    a_all = np.array(
        [
            float(f.split("_a")[-1].split("_")[0])
            for f in glob(
                "{}/basis_spectra_nbody_pk_rsd={}_pypower={}_a*_nmesh*.npy".format(
                    basename, rsd, cfg["use_pypower"]
                )
            )
            if len(f.split("_a")[-1].split("_")[0]) == 6
        ]
    )
    a_all.sort()
    print(len(a_all), flush=True)

    pk_ij_hat = np.zeros((len(a_all), 14, 699))
    pk_ij_smooth = np.zeros((len(a_all), 14, 699))
    pk_ij_beta1 = np.zeros((len(a_all), 14, 699))
    pk_ij_3lpt = np.zeros((len(a_all), 14, 699))
    pk_ij_ke3lpt = np.zeros((len(a_all), 14, 699))
    pk_ij_zenbu = np.zeros((len(a_all), 14, 699))
    pk_ij_zz_all = np.zeros((len(a_all), 14, 699))
    pk_ij_nn_all = np.zeros((len(a_all), 14, 699))
    pk_ij_zn_all = np.zeros((len(a_all), 14, 699))
    pk_ij_nz_all = np.zeros((len(a_all), 14, 699))
    beta_ij_all = np.zeros((len(a_all), 14, 699))
    beta_ij_smooth_all = np.zeros((len(a_all), 14, 699))
    anzu_config = configbase + "anzu_fields.param"

    s_m_map = {0: 0, 2: 1, 5: 3, 9: 6}
    s_cb_map = {1: 0, 3: 1, 4: 2, 6: 3, 7: 4, 8: 5, 10: 6, 11: 7, 12: 8, 13: 9}
    pkclass = None
    kecleftobj_m = None
    kecleftobj_cb = None

    for j, a_this in enumerate(np.array(a_all)):
        print("working on snapshot {}".format(j), flush=True)

        pk_zz_fname = (
            basename
            + "basis_spectra_za_surrogate_pk_rsd=False_pypower=True_a{:.4f}_nmesh1400.npy".format(
                a_this
            )
        )
        pk_nn_fname = (
            basename
            + "basis_spectra_nbody_pk_rsd=False_pypower=True_a{:.4f}_nmesh1400.npy".format(
                a_this
            )
        )
        pk_zn_fname = (
            basename
            + "basis_spectra_za_surrogate_crosscorr_pk_rsd=False_pypower=True_a{:.4f}_nmesh1400.npy".format(
                a_this
            )
        )
        z_this = 1 / a_this - 1
        k, mu, pk_ij_nn, pk_ij_nn_poles = load_pk_from_dict(pk_nn_fname)
        k, mu, pk_ij_zn, _ = load_pk_from_dict(pk_zn_fname)
        k, mu, pk_ij_zz, _ = load_pk_from_dict(pk_zz_fname)
        p_in = np.loadtxt("{}/input_powerspec.txt".format(configbase))
        (
            pk_ij_nn_hat,
            pk_ij_nn_smooth,
            pk_ij_nn_beta1,
            beta_ij,
            beta_ij_damp,
            beta_ij_smooth,
            pk_ij_zz_l,
            pk_ij_zenbu_l,
            pk_ij_zn_l,
            pk_ij_nz_l,
            pk_m_3lpt,
            pk_cb_3lpt,
            pk_m_ke3lpt,
            pk_cb_ke3lpt,
            pkclass,
            kecleftobj_m,
            kecleftobj_cb,
        ) = reduce_variance(
            k,
            pk_ij_nn[..., 0],
            pk_ij_zn[..., 0],
            pk_ij_zz[..., 0],
            anzu_config,
            z_this,
            p_in[:, 0],
            p_in[:, 1] * (2 * np.pi) ** 3,
            pkclass=pkclass,
            kecleftobj_m=kecleftobj_m,
            kecleftobj_cb=kecleftobj_cb,
        )

        for s in np.arange(len(pk_ij_nn_hat)):
            pk_ij_hat[j, s, :] = pk_ij_nn_hat[s]
            pk_ij_smooth[j, s, :] = pk_ij_nn_smooth[s]
            pk_ij_beta1[j, s, :] = pk_ij_nn_beta1[s]

            pk_ij_zenbu[j, s, :] = pk_ij_zenbu_l[s]
            pk_ij_zz_all[j, s, :] = pk_ij_zz_l[s]
            pk_ij_nn_all[j, s, :] = pk_ij_nn[s, ..., 0]
            pk_ij_zn_all[j, s, :] = pk_ij_zn_l[s]
            pk_ij_nz_all[j, s, :] = pk_ij_nz_l[s]
            beta_ij_all[j, s, :] = beta_ij[s]
            beta_ij_smooth_all[j, s, :] = beta_ij_smooth[s]

            if s in [0, 2, 5, 9]:
                pk_ij_3lpt[j, s, :] = pk_m_3lpt[s_m_map[s]]
            else:
                pk_ij_3lpt[j, s, :] = pk_cb_3lpt[s_cb_map[s]]

            if s in [0, 2, 5, 9]:
                pk_ij_ke3lpt[j, s, :] = pk_m_ke3lpt[s_m_map[s]]
            else:
                pk_ij_ke3lpt[j, s, :] = pk_cb_ke3lpt[s_cb_map[s]]

    np.save("{}/pk_ij_nn_hat_noexp_damp.npy".format(basename), pk_ij_hat)
    np.save("{}/pk_ij_nn_smooth_noexp_damp.npy".format(basename), pk_ij_smooth)
    np.save("{}/pk_ij_nn_beta1_noexp_damp.npy".format(basename), pk_ij_beta1)
    np.save("{}/pk_ij_zz_noexp_damp.npy".format(basename), pk_ij_zz_all)
    np.save("{}/pk_ij_nn_noexp_damp.npy".format(basename), pk_ij_nn_all)
    np.save("{}/pk_ij_zn_noexp_damp.npy".format(basename), pk_ij_zn_all)
    np.save("{}/pk_ij_nz_noexp_damp.npy".format(basename), pk_ij_nz_all)
    np.save("{}/beta_ij_noexp_damp.npy".format(basename), beta_ij_all)
    np.save("{}/beta_ij_smooth_noexp_damp.npy".format(basename), beta_ij_smooth_all)
    np.save("{}/pk_ij_zenbu_noexp_damp.npy".format(basename), pk_ij_zenbu)
    np.save("{}/pk_ij_3lpt_noexp_damp.npy".format(basename), pk_ij_3lpt)
    np.save("{}/pk_ij_ke3lpt_noexp_damp.npy".format(basename), pk_ij_ke3lpt)


if __name__ == "__main__":

    rsd = sys.argv[1]
    configbases = sys.argv[2:]

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

    configbases = configbases[rank::size]

    for configbase in configbases:
        print(configbase, flush=True)
        reduce_variance_fullsim(configbase, rsd=rsd)
