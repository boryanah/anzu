import os
import struct
from collections import namedtuple
from pmesh.window import FindResampler, ResampleWindow
from glob import glob
import psutil
import h5py
import asdf
import numpy as np
import gc

try:
    from nbodykit.algorithms.fftpower import FFTPower
except Exception as e:
    print(e)

from pypower import MeshFFTPower

__all__ = ["readGadgetSnapshot", "GadgetHeader"]

__GadgetHeader_fmt = "6I6dddii6Iiiddddii6Ii"

GadgetHeader = namedtuple(
    "GadgetHeader",
    "npart mass time redshift flag_sfr flag_feedback npartTotal flag_cooling num_files BoxSize Omega0 OmegaLambda HubbleParam flag_age flag_metals NallHW flag_entr_ics",
)


def get_memory():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1e9, "GB is current memory usage")  # in bytes


def _get_resampler(resampler):
    # Return :class:`ResampleWindow` from string or :class:`ResampleWindow` instance
    if isinstance(resampler, ResampleWindow):
        return resampler
    conversions = {"ngp": "nnb", "cic": "cic", "tsc": "tsc", "pcs": "pcs"}
    if resampler not in conversions:
        raise ValueError(
            "Unknown resampler {}, choices are {}".format(
                resampler, list(conversions.keys())
            )
        )
    resampler = conversions[resampler]
    return FindResampler(resampler)


def readGadgetSnapshot(
    filename,
    read_pos=False,
    read_vel=False,
    read_id=False,
    read_mass=False,
    print_header=False,
    single_type=-1,
    lgadget=False,
):
    """
    This function reads the Gadget-2 snapshot file.

    Parameters
    ----------
    filename : str
        path to the input file
    read_pos : bool, optional
        Whether to read the positions or not. Default is false.
    read_vel : bool, optional
        Whether to read the velocities or not. Default is false.
    read_id : bool, optional
        Whether to read the particle IDs or not. Default is false.
    read_mass : bool, optional
        Whether to read the masses or not. Default is false.
    print_header : bool, optional
        Whether to print out the header or not. Default is false.
    single_type : int, optional
        Set to -1 (default) to read in all particle types.
        Set to 0--5 to read in only the corresponding particle type.
    lgadget : bool, optional
        Set to True if the particle file comes from l-gadget.
        Default is false.

    Returns
    -------
    ret : tuple
        A tuple of the requested data.
        The first item in the returned tuple is always the header.
        The header is in the GadgetHeader namedtuple format.
    """
    blocks_to_read = (read_pos, read_vel, read_id, read_mass)
    ret = []
    with open(filename, "rb") as f:
        f.seek(4, 1)
        h = list(
            struct.unpack(
                __GadgetHeader_fmt, f.read(struct.calcsize(__GadgetHeader_fmt))
            )
        )
        if lgadget:
            h[30] = 0
            h[31] = h[18]
            h[18] = 0
            single_type = 1
        h = tuple(h)
        header = GadgetHeader._make(
            (h[0:6],)
            + (h[6:12],)
            + h[12:16]
            + (h[16:22],)
            + h[22:30]
            + (h[30:36],)
            + h[36:]
        )
        if print_header:
            print(header)
        if not any(blocks_to_read):
            return header
        ret.append(header)
        f.seek(256 - struct.calcsize(__GadgetHeader_fmt), 1)
        f.seek(4, 1)
        #
        mass_npart = [0 if m else n for m, n in zip(header.mass, header.npart)]
        if single_type not in set(range(6)):
            single_type = -1
        #
        for i, b in enumerate(blocks_to_read):
            fmt = np.dtype(np.float32)
            fmt_64 = np.dtype(np.float64)
            item_per_part = 1
            npart = header.npart
            #
            if i < 2:
                item_per_part = 3
            elif i == 2:
                fmt = np.dtype(np.uint32)
                fmt_64 = np.dtype(np.uint64)
            elif i == 3:
                if sum(mass_npart) == 0:
                    ret.append(np.array([], fmt))
                    break
                npart = mass_npart
            #
            size_check = struct.unpack("I", f.read(4))[0]
            #
            block_item_size = item_per_part * sum(npart)
            if size_check != block_item_size * fmt.itemsize:
                fmt = fmt_64
            if size_check != block_item_size * fmt.itemsize:
                raise ValueError("Invalid block size in file!")
            size_per_part = item_per_part * fmt.itemsize
            #
            if not b:
                f.seek(sum(npart) * size_per_part, 1)
            else:
                if single_type > -1:
                    f.seek(sum(npart[:single_type]) * size_per_part, 1)
                    npart_this = npart[single_type]
                else:
                    npart_this = sum(npart)
                data = np.fromstring(f.read(npart_this * size_per_part), fmt)
                if item_per_part > 1:
                    data.shape = (npart_this, item_per_part)
                ret.append(data)
                if not any(blocks_to_read[i + 1 :]):
                    break
                if single_type > -1:
                    f.seek(sum(npart[single_type + 1 :]) * size_per_part, 1)
            f.seek(4, 1)
    #
    return tuple(ret)


def get_snap_z(basedir, sim_type):
    """Count up the number of particles that will be read in by this rank.

    Args:
        snapfiles list: List of blocks assigned to this rank
        sim_type str: Type of simulation (format)

    Returns:
        npart int: Number of particles assigned to this rank.
    """

    if sim_type == "Gadget_hdf5":
        snapfiles = glob(basedir + "*hdf5")
        f = snapfiles[0]
        with h5py.File(f, "r") as block:
            z_this = block["Header"].attrs["Redshift"]

    elif sim_type == "Gadget":
        snapfiles = glob(basedir + "*")
        f = snapfiles[0]
        header = readGadgetSnapshot(f, read_id=False, read_pos=False)
        z_this = header["redshift"]

    elif sim_type == "AbacusSummit":
        snapfiles = glob(basedir + "*.asdf")
        f = snapfiles[0]
        header = asdf.open(f)["header"]
        z_this = header["Redshift"]

    return z_this


def get_Nparts(snapfiles, sim_type, parttype):
    """Count up the number of particles that will be read in by this rank.

    Args:
        snapfiles list: List of blocks assigned to this rank
        sim_type str: Type of simulation (format)

    Returns:
        npart int: Number of particles assigned to this rank.
    """
    npart = 0
    for f in snapfiles:
        if sim_type == "Gadget_hdf5":
            block = h5py.File(f, "r")
            npart += block["Header"].attrs["NumPart_ThisFile"][parttype]
            z_this = block["Header"].attrs["Redshift"]
            mass = block["Header"].attrs["MassTable"][parttype]
            block.close()
        elif sim_type == "Gadget":
            if parttype != 1:
                raise (
                    ValueError(
                        "Neutrino functionality not yet implemented for classic gadget outputs."
                    )
                )
            header = readGadgetSnapshot(f, read_id=False, read_pos=False)
            npart += header["npart"][1]
            z_this = header["redshift"]
            mass = 1

    if npart == 0:
        return npart, 0, 0
    else:
        return npart, z_this, mass


def load_particles(
    sim_type,
    rank,
    size,
    basedir=None,
    parttype=1,
    cv_surrogate=False,
    icfile=None,
    ic_format=None,
    D=None,
    nmesh=None,
    lbox=None,
    boltz=None,
    z_ic=None,
    rsd=False,
    z_this=None,
    Dic=None,
    gaussian_cutoff=True
):

    if not cv_surrogate: # B.H. ignoring for now for AbacusSummit
        if sim_type == "Gadget_hdf5":
            snapfiles = glob(basedir + "*hdf5")
        elif sim_type == "Gadget":
            snapfiles = glob(basedir + "*")

        snapfiles_this = snapfiles[rank::size]
        nfiles_this = len(snapfiles_this)
        npart_this, z_this, mass = get_Nparts(snapfiles_this, sim_type, parttype)
        pos = np.zeros((npart_this, 3))
        if parttype == 1:
            ids = np.zeros(npart_this, dtype=np.int)
        else:
            # don't need ids for neutrinos, since not weighting
            ids = None
    else:
        npart_this = None
        if z_this is None:
            assert basedir is not None
            z_this = get_snap_z(basedir, sim_type)

    if cv_surrogate:
        assert icfile is not None
        assert ic_format is not None
        assert nmesh is not None
        assert lbox is not None

    D = boltz.scale_independent_growth_factor(z_this)

    if Dic is None:
        D = D / boltz.scale_independent_growth_factor(z_ic)
    else:
        D = D / Dic
        
    Ha = boltz.Hubble(z_this) * 299792.458

    if rsd:
        f = boltz.scale_independent_growth_factor_f(z_this)
        v_fac = (1 + z_this) ** 0.5 / Ha * boltz.h()  # (v_p = v_gad * a^(1/2))
    else:
        f = 0
        v_fac = 0

    if not cv_surrogate:
        npart_counter = 0
        for i in range(nfiles_this):

            if sim_type == "Gadget_hdf5":
                block = h5py.File(snapfiles_this[i], "r")
                npart_block = block["Header"].attrs["NumPart_ThisFile"][parttype]
                pos[npart_counter : npart_counter + npart_block] = block[
                    "PartType{}/Coordinates".format(parttype)
                ]
                if rsd:
                    pos[npart_counter : npart_counter + npart_block, 2] += (
                        v_fac * block["PartType{}/Velocities".format(parttype)][:, 2]
                    )
                    pos[npart_counter : npart_counter + npart_block, 2] %= lbox

                if parttype == 1:
                    ids[npart_counter : npart_counter + npart_block] = block[
                        "PartType{}/ParticleIDs".format(parttype)
                    ]
                block.close()

            elif sim_type == "Gadget":
                if parttype != 1:
                    raise (
                        ValueError(
                            "Neutrino functionality not yet implemented for classic gadget outputs."
                        )
                    )
                if rsd:
                    hdr, pos_i, vel_i, ids_i = readGadgetSnapshot(
                        snapfiles_this[i], read_id=True, read_pos=True, read_vel=True
                    )
                    npart_block = hdr["npart"][1]
                    pos[npart_counter : npart_counter + npart_block] = pos_i
                    pos[npart_counter : npart_counter + npart_block, 2] += (
                        v_fac * vel_i[:, 2]
                    )
                    pos[npart_counter : npart_counter + npart_block, 2] %= lbox
                    ids[npart_counter : npart_counter + npart_block] = ids_i
                else:
                    hdr, pos_i, ids_i = readGadgetSnapshot(
                        snapfiles_this[i], read_id=True, read_pos=True
                    )
                    npart_block = hdr["npart"][1]
                    pos[npart_counter : npart_counter + npart_block] = pos_i
                    ids[npart_counter : npart_counter + npart_block] = ids_i

            else:
                raise (ValueError("Sim type must be either Gadget or GadgetHDF5"))

            npart_counter += npart_block
    else:
        n_ = [nmesh, nmesh, nmesh]
        get_cell_idx = lambda i, j, k: (i * n_[1] + j) * n_[2] + k
        filt_ics_file = "/".join(icfile.split("/")[:-1]) + f"/filtered_ics_nmesh{nmesh:d}.h5"

        with h5py.File(filt_ics_file, "r") as ics:
            if gaussian_cutoff:
                postfix = '_filt'
            else:
                postfix = ''
            
            # read in displacements, rescale by D=D(z_this)/D(z_ini)
            grid = np.meshgrid(
                np.arange(rank, nmesh, size),
                np.arange(nmesh),
                np.arange(nmesh),
                indexing="ij",
            )
            pos_x = (
                (grid[0] / nmesh + D * ics["DM_dx{}".format(postfix)][rank::size, ...]) % 1
            ) * lbox
            pos_y = (
                (grid[1] / nmesh + D * ics["DM_dy{}".format(postfix)][rank::size, ...]) % 1
            ) * lbox
            pos_z = (
                (grid[2] / nmesh + D * (1 + f) * ics["DM_dz{}".format(postfix)][rank::size, ...]) % 1
            ) * lbox

            pos = np.stack([pos_x, pos_y, pos_z])
            pos = pos.reshape(3, -1).T
            del pos_x, pos_y, pos_z
            gc.collect()

            ids = get_cell_idx(grid[0], grid[1], grid[2])
            del grid
            gc.collect()

            mass = 1

    return pos, ids, npart_this, z_this, mass, D


def position_to_index(pos, lbox, nmesh):

    deltax = lbox / nmesh

    idvec = np.floor((pos) / deltax)

    return (idvec % nmesh).astype("int16")


def kroneckerdelta(i, j):
    if i == j:
        return 1
    else:
        return 0


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


def CompensateInterlacedCICAliasing(w, v):

    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 2
        tmp[wi == 0.0] = 1.0
        v = v / tmp
    return v


def measure_pk(mesh1, mesh2, lbox, nmesh, rsd, use_pypower, D1, D2):

    k_edges = np.linspace(0, nmesh * np.pi / lbox, int(nmesh // 2))
    if not rsd:
        edges = k_edges
        poles = (0,)
        mode = "1d"
        Nmu = 1
    else:
        edges = k_edges
        poles = (0, 2, 4)
        mode = "2d"

    pkdict = {}

    if use_pypower:
        pk = MeshFFTPower(mesh1, mesh2=mesh2, edges=edges, los="z", ells=poles)

        pkdict["k"] = pk.poles.k
        pkdict["mu"] = pk.wedges.mu
        pkdict["nmodes"] = pk.poles.nmodes
        pkdict["nmodes_wedges"] = pk.wedges.nmodes
        pkdict["power_poles"] = pk.poles.power.real * D1 * D2
        pkdict["power_wedges"] = pk.wedges.power.real * D1 * D2
        pkdict["shotnoise"] = pk.poles.shotnoise_nonorm

    else:
        dk = 2 * np.pi / lbox
        pk = FFTPower(
            mesh1, mode, second=mesh2, BoxSize=lbox, Nmesh=nmesh, poles=poles, dk=dk, kmin=dk
        )

        pkdict["k"] = pk.power["k"].real
        pkdict["nmodes"] = pk.power["modes"].real
        pkdict["power_wedges"] = pk.power["power"].real * D1 * D2

        if rsd:
            pkdict["mu"] = pk.power["mu"].real
            pkdict["power_poles"] = np.stack(
                [pk.poles["power_{}".format(ell)].real * D1 * D2 for ell in poles]
            )

    return pkdict
