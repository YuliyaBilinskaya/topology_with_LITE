import numpy as np
from collections import defaultdict
import itertools
from functools import lru_cache
import matplotlib.pyplot as plt

_BASIC_OPERATORS = {
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    "+": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
    "-": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
    "1": np.eye(2, dtype=np.complex128),
}


@lru_cache(maxsize=64)
def _cached_opdm_operators(L, m, k):
    """Build each pair of OPDM operators once per (L, m, k)."""

    def jordan_wigner_operator(site, ladder_operator):
        labels = ["z"] * site + [ladder_operator] + ["1"] * (L - site - 1)

        operator = _BASIC_OPERATORS[labels[0]].copy()
        for label in labels[1:]:
            operator = np.kron(operator, _BASIC_OPERATORS[label])

        operator.setflags(write=False)
        return operator

    sigma_plus_m = jordan_wigner_operator(m, "+")
    sigma_minus_m = jordan_wigner_operator(m, "-")
    sigma_minus_k = jordan_wigner_operator(k, "-")

    rho_eh = sigma_plus_m @ sigma_minus_k
    rho_hh = sigma_minus_m @ sigma_minus_k

    rho_eh.setflags(write=False)
    rho_hh.setflags(write=False)

    return rho_eh, rho_hh

def calc_opdm_operator_no_optimization(L,m,k):
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

def calc_opdm_from_rho_no_optimization(many_body_rho):
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
        rho = calc_opdm_operator_no_optimization(L_loc, i, j)
        hopp_matrix = many_body_rho @ rho['eh'][i, j]
        rho_opdm[i, j] = np.trace(hopp_matrix)  # 1st quadrant: sigma_plus_i sigma_minus_j
        pair_matrix = many_body_rho @ rho['hh'][i, j]
        rho_opdm[i+ L_loc, j ] = np.trace(pair_matrix)  # 3rd quadrant: sigma_plus_i sigma_plus_j

    rho_opdm[: L_loc, L_loc : 2 * L_loc] = rho_opdm[L_loc : 2 * L_loc, : L_loc].T.conjugate()  # 2nd quadrant: sigma_plus_j sigma_plus_i
    rho_opdm[L_loc: 2 * L_loc, L_loc: 2 * L_loc] = np.eye(L_loc) - rho_opdm[:L_loc, :L_loc].conjugate() # 4th quadrant: sigma_minus_j sigma_plus_i

    return rho_opdm

def calc_opdm_operator(L, m, k):
    rho_eh, rho_hh = _cached_opdm_operators(L, m, k)

    return {
        "eh": {(m, k): rho_eh},
        "hh": {(m, k): rho_hh},
    }

def calc_opdm_from_rho(many_body_rho):
    """
    Calculates the one-particle density matrix from a local many-body rho.
    """
    L_loc = int(np.log2(len(many_body_rho)))
    rho_opdm = np.zeros((2 * L_loc, 2 * L_loc), dtype=np.complex128)

    for i, j in itertools.product(range(L_loc), range(L_loc)):
        rho_eh, rho_hh = _cached_opdm_operators(L_loc, i, j)

        # Equivalent to trace(many_body_rho @ operator), without creating
        # the full temporary matrix from the matrix multiplication.
        rho_opdm[i, j] = np.einsum(
            "ij,ji->", many_body_rho, rho_eh, optimize=True
        )
        rho_opdm[i + L_loc, j] = np.einsum(
            "ij,ji->", many_body_rho, rho_hh, optimize=True
        )

    rho_opdm[:L_loc, L_loc : 2 * L_loc] = (
        rho_opdm[L_loc : 2 * L_loc, :L_loc].T.conjugate()
    )
    rho_opdm[L_loc : 2 * L_loc, L_loc : 2 * L_loc] = (
        np.eye(L_loc) - rho_opdm[:L_loc, :L_loc].conjugate()
    )

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
    #for idx in range(0, 150, 1):
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


def calc_ave_marker_onesite_per_t(loaded_dens_mat, L):
    ave_marker_per_t = {}

    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())
        ave_marker_per_t_subsyst = {}

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            for key in max_ell_keys:
                many_body_rho = loaded_dens_mat[idx][key]
                L_loc = int(np.log2(len(many_body_rho)))
                marker = calc_marker_state(many_body_rho, L_loc, flatten=False)

                # Choose the center site; for even L_loc this picks the smaller
                # of the two middle sites.
                center_site = (len(marker) - 1) // 2
                ave_marker_per_t_subsyst[key] = marker[center_site]

            vals_center = list(ave_marker_per_t_subsyst.values())
            ave_marker_per_t[idx] = np.sum(vals_center) / len(vals_center)

    return ave_marker_per_t

def calc_ave_marker_threesite_per_t(loaded_dens_mat, L):
    ave_marker_per_t = {}

    #for idx in range(0, 160, 1):
    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())
        ave_marker_per_t_subsyst = {}

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            for key in max_ell_keys:
                many_body_rho = loaded_dens_mat[idx][key]
                L_loc = int(np.log2(len(many_body_rho)))
                marker = calc_marker_state(many_body_rho, L_loc, flatten=False)

                # Take the three central sites, biased to the left when the
                # subsystem size is even:
                # 5 sites -> indices 1,2,3
                # 6 sites -> indices 1,2,3
                # 7 sites -> indices 2,3,4
                start = (len(marker) - 3) // 2
                center_marker = marker[start:start + 3]

                ave_marker_per_t_subsyst[key] = np.average(center_marker)

            vals_center = list(ave_marker_per_t_subsyst.values())
            ave_marker_per_t[idx] = np.sum(vals_center) / len(vals_center)

    return ave_marker_per_t

def calc_ave_marker_center_per_t(loaded_dens_mat, L):
    ave_marker_center = {}

    #for idx in range(0, 160, 1):
    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            # Choose the subsystem whose lattice coordinate is closest to the
            # center of the available max-level subsystems.
            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            center_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))

            many_body_rho = loaded_dens_mat[idx][center_key]
            L_loc = int(np.log2(len(many_body_rho)))
            marker = calc_marker_state(many_body_rho, L_loc, flatten=False)

            bulk_marker = marker[1:-1] if len(marker) > 2 else marker
            ave_marker_center[idx] = np.average(bulk_marker)

    return ave_marker_center

def calc_ave_marker_const_center_per_t(loaded_dens_mat, L):
    ave_marker_center = {}

    #for idx in range(0, 160, 1):
    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            # Choose the subsystem whose lattice coordinate is closest to the
            # center of the available max-level subsystems.
            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            center_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))

            many_body_rho = loaded_dens_mat[idx][center_key]
            L_loc = int(np.log2(len(many_body_rho)))
            marker = calc_marker_state(many_body_rho, L_loc, flatten=False)

            # Take the three central sites, biased to the left when the
            # subsystem size is even:
            # 5 sites  -> indices 1,2,3
            # 6 sites  -> indices 1,2,3
            # 7 sites  -> indices 2,3,4
            start = (len(marker) - 3) // 2
            center_marker = marker[start:start + 3]

            ave_marker_center[idx] = np.average(center_marker)

    return ave_marker_center

def calc_half_mode_weight_from_opdm(opdm, tol=1e-3, n_modes_fallback=2):
    opdm = np.asarray(opdm, dtype=np.complex128)
    evals, evecs = np.linalg.eigh(opdm)
    L_loc = opdm.shape[0] // 2

    mode_idxs = np.where(np.abs(evals - 0.5) < tol)[0]
    if len(mode_idxs) == 0:
        mode_idxs = np.argsort(np.abs(evals - 0.5))[:n_modes_fallback]

    weights = np.zeros(L_loc, dtype=float)
    for idx in mode_idxs:
        vec = evecs[:, idx]
        u = vec[:L_loc]
        v = vec[L_loc:]
        weights += np.abs(u) ** 2 + np.abs(v) ** 2

    total_weight = weights.sum()
    if total_weight > 0:
        weights /= total_weight

    return weights, np.asarray(evals[mode_idxs], dtype=float)

def plot_center_markers(loaded_dens_mat, loaded_times, L):
    times_plot = []
    markers_per_t = []
    marker_sums = []
    center_keys = []
    global_sites_per_t = []

    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            center_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))

            many_body_rho = loaded_dens_mat[idx][center_key]
            L_loc = int(np.log2(len(many_body_rho)))
            marker = np.real_if_close(calc_marker_state(many_body_rho, L_loc, flatten=False))

            if np.isnan(marker).any():
                raise ValueError(
                    f"NaN markers at time index {idx}, subsystem {(center_key.coord, center_key.level)}"
                )

            marker_array = np.asarray(marker, dtype=float)
            marker_sums.append(np.sum(marker_array))

            global_sites = np.arange(len(marker_array)) + int(center_key.coord - center_key.level / 2)
            global_sites_per_t.append(global_sites)

            times_plot.append(loaded_times[idx])
            markers_per_t.append(marker_array)
            center_keys.append((center_key.coord, center_key.level))

    if not markers_per_t:
        raise ValueError("No subsystem with max_ell >= 3 was found.")

    times_plot = np.asarray(times_plot, dtype=float)

    max_n_markers = max(len(marker) for marker in markers_per_t)
    markers_array = np.full((len(markers_per_t), max_n_markers), np.nan, dtype=float)

    for i, marker in enumerate(markers_per_t):
        markers_array[i, :len(marker)] = marker

    fig, ax = plt.subplots()

    for n in range(max_n_markers):
        valid = ~np.isnan(markers_array[:, n])
        ax.scatter(times_plot[valid], markers_array[valid, n], marker='.', s=1, color='blue', label='Marker per site' if n == 0 else None,)

    marker_sums = np.asarray(marker_sums, dtype=float)
    ax.scatter(times_plot, marker_sums, marker='.', s=1, color='orange', label='Sum of markers')

    final_markers = markers_array[-1]
    final_global_sites = global_sites_per_t[-1]
    valid_final = ~np.isnan(final_markers)

    site_value_pairs = [
        (site, value) for site, value in zip(final_global_sites, final_markers) if not np.isnan(value)
    ]
    site_value_pairs.sort(key=lambda x: x[1], reverse=True)

    textbox_lines = ["Final-time markers"]
    textbox_lines += [f"site {site}: {value:.3f}" for site, value in site_value_pairs]

    ax.text(
        1.02,
        0.5,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        fontsize=8,
        color='red',
        va='center',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
    )

    ax.set_xlabel('Time')
    coord, level = center_keys[-1]
    ax.set_title(
        f"Center subsystem markers for coordinates ({float(coord)},{int(level)})"
    )
    ax.legend()
    fig.tight_layout()

    plt.savefig('/Users/yuliyabilinskaya/Desktop/markers_of_center_subsyst.pdf', dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return times_plot, markers_array


def plot_center_half_mode_weights(loaded_dens_mat, loaded_times, L, tol=1e-3):
    "Plots the weight pers site of the eigenvectors corresponding to the 0.5 eigenvalues of the OPDM. The idea is to see whether the closed gap of the OPDM is only due to the edge modes, while all the open gap modes are in the bulk."
    times_plot = []
    weights_per_t = []
    center_keys = []
    selected_eigvals = []
    global_sites_per_t = []

    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            center_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))

            many_body_rho = loaded_dens_mat[idx][center_key]
            opdm = calc_opdm_from_rho(many_body_rho)
            weights, eigvals = calc_half_mode_weight_from_opdm(opdm, tol=tol)

            if np.isnan(weights).any():
                raise ValueError(
                    f"NaN half-mode weights at time index {idx}, subsystem {(center_key.coord, center_key.level)}"
                )

            weight_array = np.asarray(weights, dtype=float)
            global_sites = np.arange(len(weight_array)) + int(center_key.coord - center_key.level / 2)

            times_plot.append(loaded_times[idx])
            weights_per_t.append(weight_array)
            center_keys.append((center_key.coord, center_key.level))
            selected_eigvals.append(eigvals)
            global_sites_per_t.append(global_sites)

    if not weights_per_t:
        raise ValueError("No subsystem with max_ell >= 3 was found.")

    times_plot = np.asarray(times_plot, dtype=float)
    max_n_sites = max(len(weight) for weight in weights_per_t)
    weights_array = np.full((len(weights_per_t), max_n_sites), np.nan, dtype=float)

    for i, weight in enumerate(weights_per_t):
        weights_array[i, :len(weight)] = weight

    fig, ax = plt.subplots()

    for n in range(max_n_sites):
        valid = ~np.isnan(weights_array[:, n])
        ax.scatter(
            times_plot[valid],
            weights_array[valid, n],
            marker=".",
            s=1,
            color="purple",
            label="Half-mode weight per site" if n == 0 else None,
        )

    final_weights = weights_array[-1]
    final_global_sites = global_sites_per_t[-1]
    valid_final = ~np.isnan(final_weights)
    site_weight_pairs = [
        (site, value) for site, value in zip(final_global_sites, final_weights) if not np.isnan(value)
    ]
    site_weight_pairs.sort(key=lambda x: x[1], reverse=True)

    final_eigs = ", ".join(f"{value:.4f}" for value in selected_eigvals[-1])
    textbox_lines = [f"Final-time half-mode eigs:"]
    textbox_lines += [f"site {site}: {value:.3f}" for site, value in site_weight_pairs]

    ax.text(
        1.02,
        0.5,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        fontsize=8,
        color="red",
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.9),
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\sum_\alpha (|u^{(\alpha)}_j|^2 + |v^{(\alpha)}_j|^2)$")
    coord, level = center_keys[-1]
    ax.set_title(
        f"Center subsystem half-mode weights for coordinates ({float(coord)},{int(level)})"
    )
    #ax.legend()
    fig.tight_layout()

    plt.savefig("/Users/yuliyabilinskaya/Desktop/half_mode_weights_of_center_subsyst.pdf", dpi=200, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return times_plot, weights_array, selected_eigvals

def plot_center_markers_from_corr_mat_evo(opdm_t_corr, times_corr):
    "Plots the average of the Local Topological Markers in the central subsystem. The two edge sites are excluded"
    times_plot = []
    markers_per_t = []
    marker_sums = []

    for idx, opdm in enumerate(opdm_t_corr):
        marker = np.real_if_close(calc_marker_state_from_opdm(opdm, flatten=False))

        if np.isnan(marker).any():
            raise ValueError(f"NaN markers at time index {idx}")

        marker_array = np.asarray(marker, dtype=float)

        times_plot.append(times_corr[idx])
        markers_per_t.append(marker_array)
        marker_sums.append(np.sum(marker_array))

    if not markers_per_t:
        raise ValueError("No OPDM data was found.")

    times_plot = np.asarray(times_plot, dtype=float)

    max_n_markers = max(len(marker) for marker in markers_per_t)
    markers_array = np.full((len(markers_per_t), max_n_markers), np.nan, dtype=float)

    for i, marker in enumerate(markers_per_t):
        markers_array[i, :len(marker)] = marker

    fig, ax = plt.subplots()

    for n in range(max_n_markers):
        valid = ~np.isnan(markers_array[:, n])
        ax.scatter(
            times_plot[valid],
            markers_array[valid, n],
            marker='.',
            s=1,
            color='blue',
            label='Marker per site' if n == 0 else None,
        )

    marker_sums = np.asarray(marker_sums, dtype=float)
    ax.scatter(
        times_plot,
        marker_sums,
        marker='.',
        s=1,
        color='orange',
        label='Sum of markers',
    )

    final_markers = markers_array[-1]
    valid_final = ~np.isnan(final_markers)
    site_value_pairs = [(site, value) for site, value in enumerate(final_markers) if valid_final[site]]
    site_value_pairs.sort(key=lambda x: x[1], reverse=True)

    textbox_lines = ["Final-time markers"]
    textbox_lines += [f"site {site}: {value:.3f}" for site, value in site_value_pairs]

    fig.subplots_adjust(right=0.78)
    ax.text(
        1.02,
        0.5,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        fontsize=8,
        color='red',
        va='center',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
    )

    ax.set_xlabel('Time')
    ax.set_title('Markers from correlation matrix evolution')
    ax.legend()
    fig.tight_layout()

    plt.savefig('/Users/yuliyabilinskaya/Desktop/markers_of_corr_mat_evo.pdf', dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return times_plot, markers_array, marker_sums



def plot_opdm_eigvals(loaded_dens_mat, loaded_times, L):
    "Plots all OPDM eigenvalues at each time t."
    times_plot = []
    eigvals_per_t = []
    center_keys = []

    #for idx in range(0, 150, 1):
    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            center_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))

            many_body_rho = loaded_dens_mat[idx][center_key]
            opdm = calc_opdm_from_rho(many_body_rho)
            eigvals = np.sort(np.real_if_close(np.linalg.eigvalsh(opdm)))

            if np.isnan(eigvals).any():
                raise ValueError(
                    f"NaN eigenvalues at time index {idx}, subsystem {(center_key.coord, center_key.level)}"
                )

            times_plot.append(loaded_times[idx])
            eigvals_per_t.append(eigvals)
            center_keys.append((center_key.coord, center_key.level))

    if not eigvals_per_t:
        raise ValueError("No subsystem with max_ell >= 3 was found.")

    times_plot = np.asarray(times_plot, dtype=float)

    max_n_eigs = max(len(eigs) for eigs in eigvals_per_t)
    eigvals_array = np.full((len(eigvals_per_t), max_n_eigs), np.nan, dtype=float)

    for i, eigs in enumerate(eigvals_per_t):
        eigvals_array[i, :len(eigs)] = np.asarray(eigs, dtype=float)

    fig, ax = plt.subplots()

    for n in range(max_n_eigs):
        valid = ~np.isnan(eigvals_array[:, n])
        ax.scatter(times_plot[valid], eigvals_array[valid, n], marker='.', s = 1, color='green')

    ax.set_xlabel('Time')
    coord, level = center_keys[-1]
    ax.set_title(
        f'OPDM eigenvalues for subsystem with coordinates ({float(coord)},{int(level)})'
    )
    fig.tight_layout()

    plt.savefig('/Users/yuliyabilinskaya/Desktop/opdm_eigvals.pdf', dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return times_plot, eigvals_array



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
    "Calculates the local topological marker average from provided OPDM on all sites except the edges of the full system."
    ave_marker_per_t = {}
    var_marker_per_t = {}

    for idx, opdm in enumerate(opdm_t):
        marker = calc_marker_state_from_opdm(opdm, flatten=False)
        bulk_marker = marker[1:-1] if len(marker) > 2 else marker
        ave_marker_per_t[idx] = np.average(bulk_marker)

    return ave_marker_per_t


def calc_marker_onesite_per_t_from_opdm(opdm_t, L):
    "Calculates the local topological marker from provided OPDM only on the central site of the full system."

    marker_per_t = {}

    for idx, opdm in enumerate(opdm_t):
        marker = calc_marker_state_from_opdm(opdm, flatten=False)

        # Choose the center site; for even system size this picks the smaller
        # of the two middle sites.
        center_site = (len(marker) - 1) // 2
        marker_per_t[idx] = marker[center_site]

    return marker_per_t


def calc_marker_const_center_per_t_from_opdm(opdm_t, L):
    marker_per_t = {}

    for idx, opdm in enumerate(opdm_t):
        marker = calc_marker_state_from_opdm(opdm, flatten=False)

        # Take the three central sites, biased to the left when the
        # subsystem size is even:
        # 5 sites -> indices 1,2,3
        # 6 sites -> indices 1,2,3
        # 7 sites -> indices 2,3,4
        start = (len(marker) - 3) // 2
        center_marker = marker[start:start + 3]

        marker_per_t[idx] = np.average(center_marker)

    return marker_per_t


def plot_onsite_occupations_from_corr_mat_evo(opdm_t_corr, times_corr):
    times_plot = []
    occupations_per_t = []
    occupation_sums = []

    for idx, opdm in enumerate(opdm_t_corr):
        opdm = np.asarray(opdm, dtype=np.complex128)
        L_loc = opdm.shape[0] // 2

        # In the BdG OPDM used here, the upper-left block is <c_i^\dagger c_j>,
        # so the diagonal gives the on-site occupations <n_j>.
        occupations = np.real_if_close(np.diag(opdm[:L_loc, :L_loc]))

        if np.isnan(occupations).any():
            raise ValueError(f"NaN occupations at time index {idx}")

        occupations = np.asarray(occupations, dtype=float)

        times_plot.append(times_corr[idx])
        occupations_per_t.append(occupations)
        occupation_sums.append(np.sum(occupations))

    if not occupations_per_t:
        raise ValueError("No OPDM data was found.")

    times_plot = np.asarray(times_plot, dtype=float)

    max_n_sites = max(len(occ) for occ in occupations_per_t)
    occupations_array = np.full((len(occupations_per_t), max_n_sites), np.nan, dtype=float)

    for i, occ in enumerate(occupations_per_t):
        occupations_array[i, :len(occ)] = occ

    fig, ax = plt.subplots()

    for n in range(max_n_sites):
        valid = ~np.isnan(occupations_array[:, n])
        ax.scatter(
            times_plot[valid],
            occupations_array[valid, n],
            marker='.',
            s=1,
            color='blue',
            label='Occupation per site' if n == 0 else None,
        )

    occupation_sums = np.asarray(occupation_sums, dtype=float)
    ax.scatter(
        times_plot,
        occupation_sums,
        marker='.',
        s=1,
        color='orange',
        label='Sum of occupations',
    )

    final_occupations = occupations_array[-1]
    valid_final = ~np.isnan(final_occupations)
    site_value_pairs = [
        (site, value) for site, value in enumerate(final_occupations) if valid_final[site]
    ]
    site_value_pairs.sort(key=lambda x: x[1], reverse=True)

    textbox_lines = ["Final-time occupations"]
    textbox_lines += [f"site {site}: {value:.3f}" for site, value in site_value_pairs]

    fig.subplots_adjust(right=0.78)
    ax.text(
        1.02,
        0.5,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        fontsize=8,
        color='red',
        va='center',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
    )

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle n_j \rangle$')
    ax.set_title('On-site occupations from correlation matrix evolution')
    ax.legend()
    fig.tight_layout()

    plt.savefig('/Users/yuliyabilinskaya/Desktop/occupations_of_corr_mat_evo.pdf',
                dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return times_plot, occupations_array, occupation_sums

def plot_center_occupations(loaded_dens_mat, loaded_times, L):
    times_plot = []
    occupations_per_t = []
    occupation_sums = []
    center_keys = []

    for idx in range(len(loaded_dens_mat)):
        d = loaded_dens_mat[idx]
        max_ell = max(k.level for k in d.keys())

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            center_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))

            many_body_rho = loaded_dens_mat[idx][center_key]
            opdm = calc_opdm_from_rho(many_body_rho)
            L_loc = opdm.shape[0] // 2

            # Upper-left block is <c_i^\dagger c_j>, so its diagonal is <n_j>.
            occupations = np.real_if_close(np.diag(opdm[:L_loc, :L_loc]))

            if np.isnan(occupations).any():
                raise ValueError(
                    f"NaN occupations at time index {idx}, subsystem {(center_key.coord, center_key.level)}"
                )

            occupation_array = np.asarray(occupations, dtype=float)
            occupation_sums.append(np.sum(occupation_array))

            times_plot.append(loaded_times[idx])
            occupations_per_t.append(occupation_array)
            center_keys.append((center_key.coord, center_key.level))

    if not occupations_per_t:
        raise ValueError("No subsystem with max_ell >= 3 was found.")

    times_plot = np.asarray(times_plot, dtype=float)

    max_n_sites = max(len(occ) for occ in occupations_per_t)
    occupations_array = np.full((len(occupations_per_t), max_n_sites), np.nan, dtype=float)

    for i, occ in enumerate(occupations_per_t):
        occupations_array[i, :len(occ)] = occ

    fig, ax = plt.subplots()

    for n in range(max_n_sites):
        valid = ~np.isnan(occupations_array[:, n])
        ax.scatter(
            times_plot[valid],
            occupations_array[valid, n],
            marker='.',
            s=1,
            color='blue',
            label='Occupation per site' if n == 0 else None,
        )

    occupation_sums = np.asarray(occupation_sums, dtype=float)
    ax.scatter(
        times_plot,
        occupation_sums,
        marker='.',
        s=1,
        color='orange',
        label='Sum of occupations',
    )

    final_occupations = occupations_array[-1]
    valid_final = ~np.isnan(final_occupations)

    site_value_pairs = [
        (site, value) for site, value in enumerate(final_occupations) if valid_final[site]
    ]
    site_value_pairs.sort(key=lambda x: x[1], reverse=True)

    textbox_lines = ["Final-time occupations"]
    textbox_lines += [f"site {site}: {value:.3f}" for site, value in site_value_pairs]

    ax.text(
        1.02,
        0.5,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        fontsize=8,
        color='red',
        va='center',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
    )

    ax.set_xlabel('Time')
    coord, level = center_keys[-1]
    ax.set_title(
        f"Center subsystem occupations for coordinates ({float(coord)},{int(level)})"
    )
    ax.set_ylabel(r'$\langle n_j \rangle$')
    ax.legend()
    fig.tight_layout()

    plt.savefig('/Users/yuliyabilinskaya/Desktop/occupations_of_center_subsyst.pdf',
                dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return times_plot, occupations_array

def plot_center_occupations_from_file(on_site_density_data, loaded_times, save_path='/Users/yuliyabilinskaya/Desktop/occupations_of_center_subsyst.pdf'):
    times_plot = []
    occupations_per_t = []
    occupation_sums = []
    center_keys = []

    for idx in range(len(on_site_density_data)):
        d = on_site_density_data[idx]
        max_ell = max(k.level for k in d.keys())

        if max_ell >= 3:
            max_ell_keys = [k for k in d.keys() if k.level == max_ell]

            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            center_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))

            occupations = np.asarray(d[center_key], dtype=float)

            if np.isnan(occupations).any():
                raise ValueError(
                    f"NaN occupations at time index {idx}, subsystem {(center_key.coord, center_key.level)}"
                )

            occupation_sums.append(np.sum(occupations))
            times_plot.append(loaded_times[idx])
            occupations_per_t.append(occupations)
            center_keys.append((center_key.coord, center_key.level))

    if not occupations_per_t:
        raise ValueError("No subsystem with max_ell >= 3 was found.")

    times_plot = np.asarray(times_plot, dtype=float)

    max_n_sites = max(len(occ) for occ in occupations_per_t)
    occupations_array = np.full((len(occupations_per_t), max_n_sites), np.nan, dtype=float)

    for i, occ in enumerate(occupations_per_t):
        occupations_array[i, :len(occ)] = occ

    fig, ax = plt.subplots()

    for n in range(max_n_sites):
        valid = ~np.isnan(occupations_array[:, n])
        ax.scatter(
            times_plot[valid],
            occupations_array[valid, n],
            marker='.',
            s=1,
            color='blue',
            label='Occupation per site' if n == 0 else None,
        )

    occupation_sums = np.asarray(occupation_sums, dtype=float)
    ax.scatter(
        times_plot,
        occupation_sums,
        marker='.',
        s=1,
        color='orange',
        label='Sum of occupations',
    )

    final_occupations = occupations_array[-1]
    valid_final = ~np.isnan(final_occupations)

    site_value_pairs = [
        (site, value) for site, value in enumerate(final_occupations) if valid_final[site]
    ]
    site_value_pairs.sort(key=lambda x: x[1], reverse=True)

    textbox_lines = ["Final-time occupations"]
    textbox_lines += [f"site {site}: {value:.3f}" for site, value in site_value_pairs]

    ax.text(
        1.02,
        0.5,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        fontsize=8,
        color='red',
        va='center',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
    )

    ax.set_xlabel('Time')
    coord, level = center_keys[-1]
    ax.set_title(
        f"Center subsystem occupations for coordinates ({float(coord)},{int(level)})"
    )
    ax.set_ylabel(r'$\langle n_j \rangle$')
    ax.legend()
    fig.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return times_plot, occupations_array
