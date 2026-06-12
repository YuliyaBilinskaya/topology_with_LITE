import numpy as np
from collections import defaultdict
import itertools

def calc_opdm_operator(L,m,k):
    basic_operators = {
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
        "+": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        "-": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
        "1": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    }

    rho = {'eh': {}, 'hh': {}, 'he': {}, 'ee': {}}

    string_m = ['1'] * L
    string_k = ['1'] * L

    #    print('i', i)
    if m > 0:
        for s in range(m):
            string_m[s] = 'z'
    #    print('string_i', string_i)

    #    print('j', j)
    if k > 0:
        for s in range(k):
            string_k[s] = 'z'
    #    print('string_j', string_j)

    string_m[m] = '+'
    sigma_plus_string_m = string_m.copy()  # sigma_z kron sigma_z kron ... kron sigma_plus kron 1 ... 1
    sigma_plus_m = basic_operators[sigma_plus_string_m[0]]
    for op in sigma_plus_string_m[1:]:
        sigma_plus_m = np.kron(sigma_plus_m, basic_operators[op])

    string_m[m] = '-'
    sigma_minus_string_m = string_m.copy()  # sigma_z kron sigma_z kron ... kron sigma_minus kron 1 ... 1
    sigma_minus_m = basic_operators[sigma_minus_string_m[0]]
    for op in sigma_minus_string_m[1:]:
        sigma_minus_m = np.kron(sigma_minus_m, basic_operators[op])

    string_k[k] = '-'
    sigma_minus_string_k = string_k.copy()  # sigma_z kron sigma_z kron ... kron sigma_minus kron 1 ... 1
    sigma_minus_k = basic_operators[sigma_minus_string_k[0]]
    for op in sigma_minus_string_k[1:]:
        sigma_minus_k = np.kron(sigma_minus_k, basic_operators[op])

    string_k[k] = '+'
    sigma_plus_string_k = string_k.copy()  # sigma_z kron sigma_z kron ... kron sigma_minus kron 1 ... 1
    sigma_plus_k = basic_operators[sigma_plus_string_k[0]]
    for op in sigma_plus_string_k[1:]:
        sigma_plus_k = np.kron(sigma_plus_k, basic_operators[op])

    # print('i= ', i)
    # print('sigma_plus_string_i: ', sigma_plus_string_i)
    # print('sigma_minus_string_i: ', sigma_minus_string_i)
    # print('j= ', j)
    # print('sigma_minus_string_j: ', sigma_minus_string_j)

    rho['eh'][m, k] = sigma_plus_m @ sigma_minus_k
    rho['hh'][m, k] = sigma_minus_m @ sigma_minus_k
    rho['he'][m, k] = sigma_minus_m @ sigma_plus_k
    rho['ee'][m, k] = sigma_plus_m @ sigma_plus_k

    return rho

def calc_opdm_from_rho(many_body_rho):
    """
    Calculates the one-particle-density matrix from a local many-body rho.

    rho_opdm =  [[ Tr(rho_ij @ sigma^+_i @ sigma^-_j) , Tr(rho_ij @ sigma^-_i @ sigma^-_j)],
                 [ Tr(rho_ij @ sigma^+_i @ sigma^+_j),  Tr(rho_ij @ sigma^+_j @ sigma^-_i)]]

    Parameters
    ----------
    rho : numpy arrary
        The state for which we want the opdm.

    Returns
    -------
    rho_opdm : numpy array 2L x 2L
        the opdm.
    """
    L_loc = int(np.log2(len(many_body_rho)))

    rho_opdm = np.zeros((2 * L_loc, 2 * L_loc), dtype=np.complex128)

    #print('L_loc', L_loc)
    for i, j in itertools.product(range(L_loc), range(L_loc)):
        rho = calc_opdm_operator(L_loc, i, j)
        hopp_matrix = many_body_rho @ rho['eh'][i, j]
        rho_opdm[i, j] = np.trace(hopp_matrix)  # 1st quadrant: sigma_plus_i sigma_minus_j
        pair_matrix = many_body_rho @ rho['hh'][i, j]
        rho_opdm[i+ L_loc, j ] = np.trace(pair_matrix)  # 3rd quadrant: sigma_plus_i sigma_plus_j

    rho_opdm[: L_loc, L_loc : 2 * L_loc] = rho_opdm[L_loc : 2 * L_loc, : L_loc].T.conjugate()  # 2nd quadrant: sigma_plus_j sigma_plus_i
    rho_opdm[L_loc: 2 * L_loc, L_loc: 2 * L_loc] = np.eye(L_loc) - rho_opdm[:L_loc, :L_loc].conjugate() # 4th quadrant: sigma_minus_j sigma_plus_i

    return rho_opdm

def local_marker(L_x, x, P, S, site):
    """
    Calculates the local CS marker for the specified site on a 1d chain in the BdG basis

    ----------
    Parameters:
    L_x, L_y, L_z: Number of lattice sites in each direction
    n_orb : Number of orbitals
    n_sites: Number of lattice sites
    x, y, z: Vectors with the position of the lattice sites
    P : Valence band projector
    S: Chiral symmetry operator of the model
    site: Number of the site we calculate the marker on

    ----------
    Returns:
    Marker vector on each site
    """

    # Position operators  (take the particular site to be as far from the branch cut as possible)

    #This code rotatest the chain such that the site is in the middle of the chain. Effectively mapping the system onto the bulk only and omitting the edge states.
    #If one calculates this way than the average of the local markers gives the value of the chern number in the bulk.
#    half_Lx = np.floor(L_x / 2)
#    deltaLx = np.heaviside(x[site] - half_Lx, 0) * abs(x[site] - (half_Lx + L_x)) + np.heaviside(half_Lx - x[site], 0) * abs(half_Lx - x[site])
#    x = (x + deltaLx) % L_x                                      # Relabel of the operators

    #calculating the Chern marker with the below will include both the edges and the bulk.
    X = np.concatenate((x, x))                                   # X vector in BdG basis
    X = np.reshape(X, (len(X), 1))                               # Column vector x

    # Marker calculation
    M = P @ S @ (X * P)                                          # Marker operator (BdG x position space)
#    print('M[site, site]', M[site, site])
#    print('M[L_x + site, L_x + site]', M[L_x + site, L_x + site])
    if np.allclose(np.imag(M[site, site]), 0j, atol = 1e-10) and np.allclose(np.imag(M[L_x + site, L_x + site]), 0j, atol = 1e-10):
        marker = -2 * (np.real(M[site, site]) + np.real(M[L_x + site, L_x + site]))    # Trace over each site
    else:
        raise ValueError(f'Imaginary part of the Marker operator should be zero')

    return marker

def band_flattening(rho):
    values, vectors = np.linalg.eigh(rho)
    idx = np.min(np.where(values > 0.5)[0])
    U = np.zeros(rho.shape, dtype=np.complex128)
    U[:, : idx] = vectors[:, : idx]  # Filling the projector
    rho_flat = U @ np.conj(np.transpose(U))
    return rho_flat

def calc_marker_state(many_body_rho, L,  flatten = False):
    # Single-particle density matrix
    rho_state_opdm = calc_opdm_from_rho(many_body_rho)
    Id = np.eye(int(L), dtype=np.complex128)
    S = np.kron([[0.0, 1.0], [1.0, 0.0]], Id)  # Sigma_x: Chiral symmetry of the chain (BdG x position space)

    if flatten:
        rho_state_opdm = band_flattening(rho_state_opdm)

    # Local chiral marker
    marker = np.zeros((L,))

    for j in range(L):
        marker[j] = local_marker(L_x=L, x=np.arange(L), P=rho_state_opdm, S=S, site=j)

    return marker

def calc_ave_marker_per_t(loaded_dens_mat, L):
    all_markers = defaultdict(dict)
    ave_marker_per_t_subsyst = defaultdict(dict)
    ave_marker_per_t = defaultdict(dict)
    for idx in range(len(loaded_dens_mat)):  #for idx in range(0, int(len(loaded_dens_mat)), 10):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())
        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]
            for key in max_ell_keys:
                #print('key', key)
                many_body_rho = loaded_dens_mat[idx][key]
                L_loc = int(np.log2(len(many_body_rho)))
                _marker = calc_marker_state(many_body_rho, L_loc, flatten=False)
                #print('marker', _marker)
                all_markers[idx][key] = _marker  # all markers per time t and and subsystem (n, max_ell)
                #print('OPDM_eigvals', OPDM_eigvals[idx])
                ave_marker_per_t_subsyst[idx][key] = np.average(_marker[1:-1])  # average marker excluding mixed edge sites per t and subsystem
                #print('ave_marker_per_t_subsyst', ave_marker_per_t_subsyst[idx])


            vals_ave = list(ave_marker_per_t_subsyst[idx].values())
            ave_marker_per_t[idx] = np.sum(vals_ave) / len(vals_ave)  # average marker excluding mixed edge sites per t

    return ave_marker_per_t

def calc_marker_state_from_opdm(opdm, flatten=False):
    opdm = np.asarray(opdm, dtype=np.complex128)
    L_loc = opdm.shape[0] // 2
    Id = np.eye(int(L_loc), dtype=np.complex128)
    S = np.kron([[0.0, 1.0], [1.0, 0.0]], Id)

    if flatten:
        opdm = band_flattening(opdm)

    marker = np.zeros((L_loc,))
    for j in range(L_loc):
        marker[j] = local_marker(L_x=L_loc, x=np.arange(L_loc), P=opdm, S=S, site=j)

    return marker


def calc_ave_marker_per_t_from_opdm(opdm_t, L):
    ave_marker_per_t = {}
    var_marker_per_t = {}

    for idx, opdm in enumerate(opdm_t):
        marker = calc_marker_state_from_opdm(opdm, flatten=False)
        bulk_marker = marker[1:-1] if len(marker) > 2 else marker
        ave_marker_per_t[idx] = np.average(bulk_marker)

    return ave_marker_per_t

