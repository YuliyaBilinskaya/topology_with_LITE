import numpy as np
from scipy.linalg import eigh
from infoLattice_functions import partial_trace_corr_mat


def _log_ratio_matrix(C_A):
    """Lambda_A = log2(C_A) - log2(I - C_A) via eigendecomposition."""
    eigvals, eigvecs = eigh(C_A)
    eigvals = np.clip(eigvals, 1e-12, 1 - 1e-12)
    lr = np.log2(eigvals) - np.log2(1 - eigvals)
    return (eigvecs * lr) @ eigvecs.conj().T


def _build_dCBdG(dC, dA_dag):
    """Assemble 2L x 2L BdG time-derivative from L x L blocks."""
    L = dC.shape[0]
    out = np.zeros((2*L, 2*L), dtype=complex)
    out[:L, :L] = -dC.T
    out[:L, L:] = dA_dag
    out[L:, :L] = -dA_dag.conj()
    out[L:, L:] = dC
    return out


def _bath_rhs(C, A_dag, gamma):
    """Bath-only contribution to (dC/dt, dA†/dt)."""
    L = C.shape[0]
    I = np.eye(L, dtype=complex)
    nu = np.full(L, 2.0, dtype=complex); nu[0] = nu[-1] = 1.0
    Nu = np.diag(nu)
    s = np.zeros(L, dtype=complex); s[0] = 1.0; s[-1] = -1.0
    S = np.diag(s)
    dmk_p1 = np.roll(I, -1, axis=1); dmk_p1[:, -1] = 0
    dm_p1k = np.roll(I, -1, axis=0); dm_p1k[-1, :] = 0
    dmk_m1 = np.roll(I, +1, axis=1); dmk_m1[:, 0] = 0
    dC = 0.25*gamma*(-(Nu@C + C@Nu) - S@A_dag + A_dag.conj()@S - dmk_p1 - dm_p1k + Nu)
    dA = 0.25*gamma*(-(Nu@A_dag + A_dag@Nu) - S@C + C.conj()@S + dmk_p1 - dmk_m1)
    return dC, dA


def _boundary_contributions(C, A_dag, n, ell, L, J):
    """
    Left (F_L) and right (F_R) Hamiltonian boundary contributions to
    dC_BdG restricted to A = {n,...,n+ell-1}. Returns (2ell x 2ell) matrices.

    Derived from the equations of motion in evolve_C_mk_A_mk_delta_t:
    boundary terms are those where the shifted index falls outside A.
    """
    dC_L = np.zeros((ell, ell), dtype=complex)
    dA_L = np.zeros((ell, ell), dtype=complex)
    dC_R = np.zeros((ell, ell), dtype=complex)
    dA_R = np.zeros((ell, ell), dtype=complex)

    if n > 0:
        cl = C[n-1, n:n+ell];   al = A_dag[n-1, n:n+ell]
        cc = C[n:n+ell, n-1];   ac = A_dag[n:n+ell, n-1]
        dC_L[0, :]  += 1j*J * (cl + al)
        dC_L[:, 0]  += -1j*J * cc + 1j*J * np.conj(ac)
        dA_L[0, :]  += -1j*J * (cl + al)
        dA_L[:, 0]  += 1j*J * np.conj(cc) - 1j*J * ac

    if n + ell < L:
        cr = C[n+ell, n:n+ell];  ar = A_dag[n+ell, n:n+ell]
        cd = C[n:n+ell, n+ell];  ad = A_dag[n:n+ell, n+ell]
        dC_R[-1, :] += 1j*J * (cr - ar)
        dC_R[:, -1] += -1j*J * cd - 1j*J * np.conj(ad)
        dA_R[-1, :] += 1j*J * (cr - ar)
        dA_R[:, -1] += -1j*J * np.conj(cd) - 1j*J * ad

    return _build_dCBdG(dC_L, dA_L), _build_dCBdG(dC_R, dA_R)


def calc_info_lattice(corr_mat):
    """
    Information lattice with 0-indexed scales:
      ell=0 → single-site subsystems, ell=1 → 2-site, ..., ell=L-1 → full system.
    Returns a dict {ell: array of length L-ell} (same structure as
    calc_info_from_corr_matr but shifted by -1).
    """
    from infoLattice_functions import calc_info_from_corr_matr
    raw = calc_info_from_corr_matr(corr_mat)   # keys 1..L
    return {ell - 1: raw[ell] for ell in raw}  # re-key to 0..L-1


def calc_all_info_currents(corr_mat, L, J, gamma, h):
    """
    Information currents at all nodes of the information lattice.

    Scales are 0-indexed: ell=0 → single-site, ell=1 → 2-site, ..., ell=L-1.

    Parameters
    ----------
    corr_mat : (2L, 2L) ndarray
        Full BdG correlation matrix (convention: [:L, L:] = A†, [L:, L:] = C).
    L : int; J, gamma, h : float

    Returns
    -------
    I_tilde_L, I_tilde_R, I_E : dicts keyed by (ell, n), ell in {0,...,L-1}
        Effective left/right boundary and bath currents (Nava et al. eq. 27).
    J_vert : (L,) ndarray
        Vertical information current per scale (eq. 30).
    """
    C = corr_mat[L:, L:]
    A_dag = corr_mat[:L, L:]

    dC_E, dA_E = _bath_rhs(C, A_dag, gamma)
    dCBdG_E = _build_dCBdG(dC_E, dA_E)

    I_L, I_R, I_E = {}, {}, {}
    for ell in range(0, L):          # 0-indexed scale
        size = ell + 1               # actual subsystem size
        for n in range(L - ell):     # L - ell valid positions
            C_A = partial_trace_corr_mat(corr_mat, n, size)
            Lam = _log_ratio_matrix(C_A)
            F_E = partial_trace_corr_mat(dCBdG_E, n, size)
            F_L, F_R = _boundary_contributions(C, A_dag, n, size, L, J)
            I_L[(ell, n)] = np.real(np.trace(F_L @ Lam))
            I_R[(ell, n)] = np.real(np.trace(F_R @ Lam))
            I_E[(ell, n)] = np.real(np.trace(F_E @ Lam))

    # Tilde currents
    I_tilde_L, I_tilde_R, I_tilde_E = {}, {}, {}
    for ell in range(0, L):
        for n in range(L - ell):
            I_tilde_L[(ell, n)] = I_L[(ell, n)] - I_L.get((ell-1, n), 0.0)
            I_tilde_R[(ell, n)] = I_R[(ell, n)] - I_R.get((ell-1, n+1), 0.0)
            I_tilde_E[(ell, n)] = -(I_E.get((ell-1, n), 0.0)
                                    + I_E.get((ell-1, n+1), 0.0)
                                    - I_E[(ell, n)]
                                    - I_E.get((ell-2, n+1), 0.0))

    J_vert = np.zeros(L)
    for ell in range(0, L):
        J_vert[ell] = sum(
            I_tilde_L[(ell, n)] + I_tilde_R[(ell, n)]
            for n in range(L - ell)
        )

    return I_tilde_L, I_tilde_R, I_tilde_E, J_vert


def check_continuity(corr_mat_t, corr_mat_t2, dt, L, J, gamma, h, max_ell=3):
    """
    Check node-level continuity (outgoing convention, derived from Eq. 26) for all (ell, n):
        dI(ell,n)/dt = Ĩ_L(ell,n) - Ĩ_L(ell-1,n+1)
                     + Ĩ_R(ell,n) - Ĩ_R(ell-1,n)
                     + Ĩ_E(ell,n)

    dI/dt uses _entropy (same normalisation as Lambda-based currents).

    corr_mat_t, corr_mat_t2 : (2L,2L) correlation matrices at times t and t+dt.
    max_ell : largest scale (0-indexed) to check (keep small for speed).

    Returns dict {(ell, n): residual} and prints a summary.
    """
    from scipy.linalg import eigh as _eigh

    def _entropy(C_A):
        ev = _eigh(C_A, eigvals_only=True)
        ev = np.clip(ev, 1e-12, 1 - 1e-12)
        return float(-np.sum(ev * np.log2(ev) + (1 - ev) * np.log2(1 - ev)))

    # Pre-compute entropies at both times for sizes 1 .. max_ell+1
    S_t, S_t2 = {}, {}
    for size in range(1, max_ell + 2):
        for n in range(L - size + 1):
            S_t [(size, n)] = _entropy(partial_trace_corr_mat(corr_mat_t,  n, size))
            S_t2[(size, n)] = _entropy(partial_trace_corr_mat(corr_mat_t2, n, size))

    # Raw currents at time t
    C = corr_mat_t[L:, L:]
    A_dag = corr_mat_t[:L, L:]
    dC_E, dA_E = _bath_rhs(C, A_dag, gamma)
    dCBdG_E = _build_dCBdG(dC_E, dA_E)

    I_L, I_R, I_E = {}, {}, {}
    for ell in range(max_ell + 1):
        size = ell + 1
        for n in range(L - ell):
            C_A = partial_trace_corr_mat(corr_mat_t, n, size)
            Lam = _log_ratio_matrix(C_A)
            F_E = partial_trace_corr_mat(dCBdG_E, n, size)
            F_L, F_R = _boundary_contributions(C, A_dag, n, size, L, J)
            I_L[(ell, n)] = np.real(np.trace(F_L @ Lam))
            I_R[(ell, n)] = np.real(np.trace(F_R @ Lam))
            I_E[(ell, n)] = np.real(np.trace(F_E @ Lam))

    # Tilde currents (Eq. 27, outgoing convention)
    I_tL, I_tR, I_tE = {}, {}, {}
    for ell in range(max_ell + 1):
        for n in range(L - ell):
            I_tL[(ell, n)] = I_L[(ell, n)] - I_L.get((ell-1, n),   0.0)
            I_tR[(ell, n)] = I_R[(ell, n)] - I_R.get((ell-1, n+1), 0.0)
            I_tE[(ell, n)] = -(I_E.get((ell-1, n),   0.0)
                               + I_E.get((ell-1, n+1), 0.0)
                               - I_E[(ell, n)]
                               - I_E.get((ell-2, n+1), 0.0))

    residuals = {}
    dI_list = []
    for ell in range(max_ell + 1):
        for n in range(L - ell):
            # dI/dt: finite difference of information lattice using _entropy
            if ell == 0:
                # I(0,n) = const - S(size=1, n)  =>  dI/dt = -dS/dt
                dI_dt = -(S_t2.get((1, n), 0.0) - S_t.get((1, n), 0.0)) / dt
            else:
                # I(ell,n) = S(ell,n) + S(ell,n+1) - S(ell+1,n) - S(ell-1,n+1)
                dI_dt = ((S_t2.get((ell,   n),   0.0) - S_t.get((ell,   n),   0.0))
                       + (S_t2.get((ell,   n+1), 0.0) - S_t.get((ell,   n+1), 0.0))
                       - (S_t2.get((ell+1, n),   0.0) - S_t.get((ell+1, n),   0.0))
                       - (S_t2.get((ell-1, n+1), 0.0) - S_t.get((ell-1, n+1), 0.0))) / dt

            rhs = (+ I_tL.get((ell,   n),   0.0)
                   - I_tL.get((ell-1, n+1), 0.0)
                   + I_tR.get((ell,   n),   0.0)
                   - I_tR.get((ell-1, n),   0.0)
                   + I_tE.get((ell,   n),   0.0))
            residuals[(ell, n)] = dI_dt - rhs
            dI_list.append(dI_dt)

    res = np.array(list(residuals.values()))
    dI_vals = np.array(dI_list)
    #print(f"Continuity check (max_ell={max_ell}, dt={dt:.2e}):")
    #print(f"  max |residual|        = {np.max(np.abs(res)):.3e}")
    #print(f"  mean|residual|        = {np.mean(np.abs(res)):.3e}")
    #print(f"  max |dI/dt|           = {np.max(np.abs(dI_vals)):.3e}  (scale of terms)")
    #print(f"  max |residual/dI_dt|  = {np.max(np.abs(res) / (np.abs(dI_vals) + 1e-30)):.3e}")
    return residuals


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


def plot_info_lattice_currents(corr_mat, L, J, gamma, h, t,
                               ell_range=None, n_range=None, ax=None,
                               arrow_scale=None, yscale=0.5, curr_clim=None):
    """
    Plot information lattice nodes (colored by I(ell,n)) with tilde current arrows.

    Arrow direction: toward larger ell (upward) if positive, downward if negative.
    Arrow color: RdBu_r (red=positive, blue=negative).
    Node color: viridis scaled to I(ell,n).

    Parameters
    ----------
    corr_mat : (2L,2L) ndarray at time t
    L, J, gamma, h : system parameters
    t : float — used for title only
    ell_range : (ell_min, ell_max) or None
    n_range   : (n_min, n_max) or None
    ax        : Axes or None (creates figure)
    arrow_scale : float or None (auto-normalised to max |current|)
    """
    I_tL, I_tR, I_E, _ = calc_all_info_currents(corr_mat, L, J, gamma, h)
    info = calc_info_lattice(corr_mat)   # 0-indexed: info[0] = single-site

    ell_lo, ell_hi = ell_range or (0, L - 1)
    ell_lo = max(ell_lo, 0)
    n_lo,   n_hi   = n_range   or (0, L - 1)

    # At scale ell (0-indexed), valid n: 0 <= n <= L - ell - 1
    nodes = [(l, n) for l in range(ell_lo, ell_hi + 1)
             for n in range(max(0, n_lo), min(n_hi, L - l - 1) + 1)]

    def pos(l, n):
        return n + l / 2.0, float(l) * yscale

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.5*(n_hi - n_lo + ell_hi)),
                                      max(3, 0.6*(ell_hi - ell_lo + 2))))

    # --- Lattice skeleton ---
    for l, n in nodes:
        if l > ell_lo:
            x, y = pos(l, n)
            for nb in [n, n + 1]:
                xc, yc = pos(l - 1, nb)
                ax.plot([x, xc], [y, yc], 'k-', lw=0.4, alpha=0.2, zorder=1)

    # --- Collect edges (one per left/right boundary) ---
    edges = []
    for l, n in nodes:
        if l <= ell_lo:
            continue
        x, y = pos(l, n)
        for val, nb in [(I_tL.get((l, n), 0.0), n), (I_tR.get((l, n), 0.0), n + 1)]:
            xc, yc = pos(l - 1, nb)
            edges.append((x, y, xc, yc, val))  # (parent, child, val)

    # --- Arrow scaling and colormap (blue=down, green=up) ---
    from matplotlib.colors import LinearSegmentedColormap
    # Nava convention: positive I_L = current leaving A (outgoing).
    # Upward flow → negative tilde current → green; downward → blue.
    arr_cmap = LinearSegmentedColormap.from_list('green_blue', ['#1a9850', 'white', '#2166ac'])
    if curr_clim is None:
        edge_vals = [abs(v) for *_, v in edges if abs(v) >= 1e-12]
        curr_clim = max(edge_vals) if edge_vals else 1.0
    arr_norm = TwoSlopeNorm(vmin=-curr_clim, vcenter=0.0, vmax=curr_clim)

    r_pts = np.sqrt(600 / np.pi)  # scatter marker radius in points

    for xp, yp, xc, yc, val in edges:
        if abs(val) < 1e-12:
            continue
        src, dst = ((xc, yc), (xp, yp)) if val < 0 else ((xp, yp), (xc, yc))
        ax.annotate('', xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle='->', lw=1.5, mutation_scale=12,
                                    color=arr_cmap(arr_norm(val)),
                                    shrinkA=r_pts, shrinkB=r_pts),
                    zorder=3)

    # --- Nodes (Oranges palette, white at zero) ---
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as mcm
    _base = plt.get_cmap("Oranges")(np.linspace(0, 1, 20))
    _base[0] = [1, 1, 1, 1]
    node_cmap = LinearSegmentedColormap.from_list("oranges_white", _base)
    xs = [pos(l, n)[0] for l, n in nodes]
    ys = [pos(l, n)[1] for l, n in nodes]
    ivals = np.array([info[l][n] for l, n in nodes])
    node_norm = Normalize(vmin=0, vmax=1.0)

    scat = ax.scatter(xs, ys, c=ivals, cmap=node_cmap, norm=node_norm,
                      s=300, zorder=5, edgecolors='k', lw=0.5)
    plt.colorbar(scat, ax=ax, label=r'$\mathcal{i}_{(\ell,n)}$', shrink=0.75, pad=0.02)

    ie_vals = [I_E.get((l, n), 0.0) for l, n in nodes]
    for (l, n), ie in zip(nodes, ie_vals):
        x, y = pos(l, n)
        ax.text(x, y, f'{ie:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+'), ha='center', va='center',
                fontsize=6, color='black', zorder=6)

    all_x = sorted({pos(l, n)[0] for l, n in nodes})
    ax.set_xticks(all_x)
    ax.set_xticklabels([f'{x:g}' for x in all_x], fontsize=5)
    ax.set_xlabel(r'$n$')

    sm = mcm.ScalarMappable(cmap=arr_cmap, norm=arr_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=r'$\tilde{\mathcal{I}}_{(n,\ell)}^{R/L}$', shrink=0.75, pad=0.12)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color='black', lw=1.5,
                               label=r'$\tilde{\mathcal{I}}_{(n,\ell)}^{E}$')],
              bbox_to_anchor=(1.02, 0.0), loc='lower left',
              bbox_transform=ax.transAxes, framealpha=0.7, borderaxespad=0)
    ax.set_title(rf'Information lattice local currents,  $t={t:.3g}$')
    ax.set_yticks([l * yscale for l in range(ell_lo, ell_hi + 1)])
    ax.set_yticklabels([str(l) for l in range(ell_lo, ell_hi + 1)])
    ax.set_ylabel(r'$\ell$')

    return ax


def plot_continuity_residuals(opdm_t_corr, times_corr, L, J, gamma, h,
                               max_ell=3, ax=None):
    """
    Compute and plot max/mean |residual| of the node-level continuity equation
    (Eq. 26) over time.

    Parameters
    ----------
    opdm_t_corr : (N, 2L, 2L) array of correlation matrices
    times_corr  : (N,) array of times
    L, J, gamma, h : system parameters
    max_ell : largest scale to include in the check
    ax : Axes or None (creates figure)

    Returns
    -------
    ax, t_vals, max_res, mean_res
    """
    n_steps = len(times_corr) - 1
    t_vals, max_res, mean_res = [], [], []

    for i in range(n_steps):
        dt = times_corr[i + 1] - times_corr[i]
        res = check_continuity(opdm_t_corr[i], opdm_t_corr[i + 1],
                               dt=dt, L=L, J=J, gamma=gamma, h=h,
                               max_ell=max_ell)
        vals = np.abs(list(res.values()))
        t_vals.append(times_corr[i])
        max_res.append(np.max(vals))
        mean_res.append(np.mean(vals))

    t_vals  = np.array(t_vals)
    max_res = np.array(max_res)
    mean_res = np.array(mean_res)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.semilogy(t_vals, max_res,  label=r'max $|\mathrm{residual}|$')
    ax.semilogy(t_vals, mean_res, label=r'mean $|\mathrm{residual}|$', linestyle='--')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$|\mathrm{residual}|$')
    ax.set_title(rf'Continuity check (max\_ell={max_ell}, dt={t_vals[1]-t_vals[0]:.2g})')
    ax.legend()

    return ax, t_vals, max_res, mean_res


def save_gif_info_lattice_currents(opdm_t_corr, times_corr, L, J, gamma, h,
                                   t_idx_start=0, t_idx_end=300,
                                   ell_range=(0, 7), n_range=(0, 24),
                                   fps=5, curr_clim=None,
                                   loop=0, save_path=None):
    """
    Save an animated GIF of the information lattice current plot over time.

    Parameters
    ----------
    opdm_t_corr : (N, 2L, 2L) array of correlation matrices
    times_corr  : (N,) array of times
    L, J, gamma, h : system parameters
    t_idx_start, t_idx_end : index range into opdm_t_corr/times_corr
    ell_range, n_range : passed to plot_info_lattice_currents
    fps : frames per second in the output GIF
    curr_clim : fixed current colour limit (None = per-frame auto)
    save_path : output path (default ~/Desktop/info_lattice_currents.gif)
    """
    import io, os
    from PIL import Image

    if save_path is None:
        save_path = os.path.expanduser('~/Desktop/info_lattice_currents.gif')

    indices = range(t_idx_start, min(t_idx_end + 1, len(times_corr)))
    frames = []

    for i in indices:
        fig, ax = plt.subplots(
            figsize=(max(6, 0.5 * (n_range[1] - n_range[0] + ell_range[1])),
                     max(3, 0.6 * (ell_range[1] - ell_range[0] + 2)))
        )
        plot_info_lattice_currents(
            opdm_t_corr[i], L=L, J=J, gamma=gamma, h=h, t=times_corr[i],
            ell_range=ell_range, n_range=n_range, ax=ax, curr_clim=curr_clim,
        )
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()

    frames[0].save(
        save_path, save_all=True, append_images=frames[1:],
        loop=loop, duration=int(1000 / fps),
    )
    print(f'Saved GIF ({len(frames)} frames) → {save_path}')