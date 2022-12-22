import numpy as np
from numba import jit

@jit(nopython=True)
def meshgrid(x, y, z):
    xx = np.empty(shape=(y.size, x.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(y.size, x.size, z.size), dtype=z.dtype)
    for i in range(y.size):
        for j in range(x.size):
            for k in range(z.size):
                xx[i,j,k] = x[i]  # change to x[k] if indexing xy
                yy[i,j,k] = y[j]  # change to y[j] if indexing xy
                zz[i,j,k] = z[k]  # change to z[i] if indexing xy
    return zz, yy, xx

@jit(nopython=True)
def meshgrid_2d(x, y, z):
    xx = np.empty(shape=(y.size, x.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size, z.size), dtype=y.dtype)
    for i in range(y.size):
        for j in range(x.size):
                xx[i,j] = x[i]  # change to x[k] if indexing xy
                yy[i,j] = y[j]  # change to y[j] if indexing xy
    return yy, xx

@jit(nopython=True)
def deconv_window_function(nmesh, lbox, kout):
    """Matrix to apply in order to deconvolve the window function from
    redshift-space power spectrum multipoles

    Args:
        nmesh (int): Size of the mesh used for power spectrum measurement
        lbox (float): Box length
        kout (np.array): k bins used for power spectrum measurement

    Returns:
        window (np.array): np.dot(np.linalg.inv(window), pell) gives unconvolved pell
        keff: Effective k value of each output k bin.
        
    """
    
    kvals = np.zeros(nmesh, dtype=np.float32)
    kvals[:nmesh//2] = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32)
    kvals[nmesh//2:] = np.arange(-2 * np.pi * nmesh / lbox / 2, 0, 2 * np.pi / lbox, dtype=np.float32)
    kvalsr = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32) 
    kz, ky, kx = meshgrid(kvals, kvals, kvalsr)    
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = kz / knorm
    mu[0,0,0] = 0
    
    ellmax = 3
    
    nkout = len(kout) - 1
    idx_o = np.digitize(knorm, kout) - 1
    nmodes_out = np.zeros(nkout * 3)
    keff = np.zeros(nkout, dtype=np.float32)
           
    L0 = np.ones_like(mu, dtype=np.float32)
    L2 = (3 * mu**2 - 1) / 2
    L4 = (35 * mu**4 - 30 * mu**2 + 3) / 8
    
    legs = [L0, L2, L4]
    pref = [(2 * (2 * i) + 1) for i in range(ellmax)]
    
    window = np.zeros((nkout * ellmax, nkout * ellmax), dtype=np.float32)
    
    for i in range(kx.shape[0]):
        for j in range(kx.shape[1]):
            for k in range(kx.shape[2]):
                if (idx_o[i,j,k]>=nkout): 
                    pass
                else:
                    if k==0:
                        nmodes_out[idx_o[i,j,k]::nkout] += 1
                        keff[idx_o[i,j,k]] += knorm[i,j,k]
                    else:
                        nmodes_out[idx_o[i,j,k]::nkout] += 2
                        keff[idx_o[i,j,k]] += 2 * knorm[i,j,k]      
                    for ell in range(ellmax):
                        for ellp in range(ellmax):
                            if k!=0:
                                window[int(ell * nkout) + int(idx_o[i,j,k]), int(ellp * nkout) + int(idx_o[i,j,k])] += 2 * pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k]
                            else:
                                window[int(ell * nkout) + int(idx_o[i,j,k]), int(ellp * nkout) + int(idx_o[i,j,k])] += pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k]

    norm_out = 1/nmodes_out
    norm_out[nmodes_out==0] = 0
    window = window * norm_out
    keff = keff * norm_out[:nkout]
    
    idx = np.where(keff>0)
    idx_grid_x, idx_grid_y = meshgrid_2d(idx,idx)
    window = window[idx_grid_x, idx_grid_y]
    keff = keff[idx]
    
    return window.T, keff

@jit(nopython=True)
def periodic_window_function(nmesh, lbox, kout, kin, k2weight=True):
    """Returns matrix appropriate for convolving a finely evaluated
    theory prediction with the 

    Args:
        nmesh (int): Size of the mesh used for power spectrum measurement
        lbox (float): Box length
        kout (np.array): k bins used for power spectrum measurement
        kin (np.array): . Defaults to None.
        k2weight (bool, optional): _description_. Defaults to True.

    Returns:
        window : np.dot(window, pell_th) gives convovled theory
        keff: Effective k value of each output k bin.
    """
    
    kvals = np.zeros(nmesh, dtype=np.float32)
    kvals[:nmesh//2] = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32)
    kvals[nmesh//2:] = np.arange(-2 * np.pi * nmesh / lbox / 2, 0, 2 * np.pi / lbox, dtype=np.float32)

    kvalsr = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32) 
    kx, ky, kz = meshgrid(kvals, kvals, kvalsr)    
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = kz / knorm
    mu[0,0,0] = 0
    
    ellmax = 3
    
    nkin = len(kin)
        
    if k2weight:
        dk = np.zeros_like(kin)
        dk[:-1] = kin[1:] - kin[:-1]
        dk[-1] = dk[-2]
    
    nkout = len(kout) - 1
    dkin = (kin[1:] - kin[:-1])[0]
    
    idx_o = np.digitize(knorm, kout) - 1
    nmodes_out = np.zeros(nkout * 3)

    idx_i = np.digitize(kin, kout) - 1
    nmodes_in = np.zeros(nkout, dtype=np.float32)

    for i in range(len(kout)):
        idx = i==idx_i
        if k2weight:
            nmodes_in[i] = np.sum(kin[idx]**2 * dk[idx])
        else:
            nmodes_in[i] = np.sum(idx)
            
    norm_in = 1/nmodes_in
    norm_in[nmodes_in==0] = 0
    norm_in_allell = np.zeros(3 * len(norm_in))
    norm_in_allell[:nkout] = norm_in
    norm_in_allell[nkout:2*nkout] = norm_in
    norm_in_allell[2*nkout:3*nkout] = norm_in
    
    window = np.zeros((nkout * 3, nkin * 3), dtype=np.float32)
    keff = np.zeros(nkout, dtype=np.float32)
    
    L0 = np.ones_like(mu, dtype=np.float32)
    L2 = (3 * mu**2 - 1) / 2
    L4 = (35 * mu**4 - 30 * mu**2 + 3) / 8
        
    legs = [L0, L2, L4]
    pref = [1, (2 * 2 + 1), (2 * 4 + 1)]
    
    for i in range(kx.shape[0]):
        for j in range(kx.shape[1]):
            for k in range(kx.shape[2]):
                if (idx_o[i,j,k]>=nkout): 
                    pass
                else:
                    if k==0:
                        nmodes_out[idx_o[i,j,k]::nkout] += 1
                        keff[idx_o[i,j,k]] += knorm[i,j,k]
                    else:
                        nmodes_out[idx_o[i,j,k]::nkout] += 2
                        keff[idx_o[i,j,k]] += 2 * knorm[i,j,k]

                    for beta in range(nkin):
                        if k2weight:
                            w = kin[beta]**2 * dk[beta]
                        else:
                            w = 1
                        if (idx_i[beta] == idx_o[i,j,k]):               
                            for ell in range(ellmax):
                                for ellp in range(ellmax):
                                    if k!=0:
                                        window[int(ell * nkout) + int(idx_o[i,j,k]), int(ellp * nkin) + int(beta)] += 2 * pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * w # * norm_in[idx_o[i,j,k]]
                                    else:
                                        window[int(ell * nkout) + int(idx_o[i,j,k]), int(ellp * nkin) + int(beta)] += pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * w # * norm_in[idx_o[i,j,k]]

    norm_out = 1/nmodes_out
    norm_out[nmodes_out==0] = 0
    window = window * norm_out.reshape(-1, 1) * norm_in_allell.reshape(-1, 1)
    keff = keff * norm_out[:nkout]
    
    return window, keff


@jit(nopython=True)
def conv_theory_window_function(nmesh, lbox, kout, plist, kth):
    """Exactly convolve the periodic box window function, without any 
        bin averaging uncertainty by evaluating a theory power spectrum
        at the k modes in the box.

    Args:
        nmesh (int): Size of the mesh used for power spectrum measurement
        lbox (float): Box length
        kout (np.array): k bins used for power spectrum measurement
        plist (list of np.arrays): List of theory multipoles evaluated at kth
        kth (np.array): k values that theory is evaluated at
        
    Returns:
        pell_conv : window convolved theory prediction
        keff: Effective k value of each output k bin. 
        
    """
    
    kvals = np.zeros(nmesh, dtype=np.float32)
    kvals[:nmesh//2] = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32)
    kvals[nmesh//2:] = np.arange(-2 * np.pi * nmesh / lbox / 2, 0, 2 * np.pi / lbox, dtype=np.float32)
    kvalsr = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32) 
    kx, ky, kz = meshgrid(kvals, kvals, kvalsr)    
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = kz / knorm
    mu[0,0,0] = 0
    
    ellmax = 3
    
    nkout = len(kout) - 1
    idx_o = np.digitize(knorm, kout) - 1
    nmodes_out = np.zeros(nkout * 3)
  
    pell_conv = np.zeros((nkout * 3), dtype=np.float32)
    keff = np.zeros(nkout, dtype=np.float32)
    
    ellmax_in = len(plist)

    pells = []
    for i in range(ellmax_in):
        pells.append(np.interp(knorm, kth, plist[i]))
        pells[i][0,0,0] = 0
        
    L0 = np.ones_like(mu, dtype=np.float32)
    L2 = (3 * mu**2 - 1) / 2
    L4 = (35 * mu**4 - 30 * mu**2 + 3) / 8
    L6 = (231 * mu**6 - 315 * mu**4 + 105 * mu**2 - 5) / 16
    
#    pk = (p0 * L0 + p2 * L2 + p4 * L4)

    legs = [L0, L2, L4, L6]
    pref = [(2 * (2 * i) + 1) for i in range(ellmax_in)]

    for i in range(kx.shape[0]):
        for j in range(kx.shape[1]):
            for k in range(kx.shape[2]):
                if (idx_o[i,j,k]>=nkout): 
                    pass
                else:
                    if k==0:
                        nmodes_out[idx_o[i,j,k]::nkout] += 1
                        keff[idx_o[i,j,k]] += knorm[i,j,k]
                    else:
                        nmodes_out[idx_o[i,j,k]::nkout] += 2
                        keff[idx_o[i,j,k]] += 2 * knorm[i,j,k]      
                    for ell in range(ellmax):
                        for ellp in range(ellmax_in):
                            if k!=0:
                                pell_conv[int(ell * nkout) + int(idx_o[i,j,k])] += 2 * pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * pells[ellp][i,j,k]
                            else:
                                pell_conv[int(ell * nkout) + int(idx_o[i,j,k])] += pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * pells[ellp][i,j,k]

    norm_out = 1/nmodes_out
    norm_out[nmodes_out==0] = 0
    pell_conv = pell_conv * norm_out
    keff = keff * norm_out[:nkout]
    
    return pell_conv, keff 