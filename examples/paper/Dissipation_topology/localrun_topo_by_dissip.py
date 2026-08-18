import os
import sys

os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
#np.__config__.show()
import pickle
import local_information as li
from marker_functions import calc_marker_state, calc_opdm_from_rho
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def plot_info_latt_from_state(initial_state, InfoLattice_Norm=2):
    """
    Plot the information lattice from a local_information State object.

    Parameters
    ----------
    initial_state : local_information.state.State
        For example the object returned by `li.State.build_finite(...)`.
    InfoLattice_Norm : float
        Upper bound of the color normalization.
    """

    density_matrix_all_levels = initial_state.get_all_levels()
    info_lattice = initial_state.get_information_lattice(density_matrix_all_levels)

    levels = sorted({key.level for key in info_lattice.keys()})
    info_latt_plot = {}

    for level in levels:
        items = sorted(
            (
                (key.coord, value)
                for key, value in info_lattice.items()
                if key.level == level
            ),
            key=lambda x: x[0],
        )
        info_latt_plot[level + 1] = np.asarray([val for _, val in items], dtype=float)

    L = len(info_latt_plot[1])

    colors = plt.get_cmap("Oranges")(np.linspace(0, 1, 20))
    colors[0] = [1, 1, 1, 1]
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    norm = Normalize(vmin=0, vmax=InfoLattice_Norm)

    r = 1 / (4 * L)
    fig, ax = plt.subplots(dpi=300)

    for l in sorted(info_latt_plot):
        for x, value in enumerate(info_latt_plot[l]):
            ax.add_artist(
                plt.Circle(
                    (x / L + l / (2 * L), (l - 0.5) / L),
                    r,
                    facecolor=custom_cmap(norm(value)),
                    edgecolor="black",
                    linewidth=0.2,
                )
            )

    x_ticks = [x / L + 1 / (2 * L) for x in range(len(info_latt_plot[1]))]
    y_ticks = [x / L + 1 / (2 * L) for x in range(len(info_latt_plot[1]))]

    plt.xlim([-2 * r, 1 + 2 * r])
    plt.ylim([-2 * r, 1 + 2 * r])

    plt.xticks(x_ticks, range(1, len(info_latt_plot[1]) + 1))
    plt.yticks(y_ticks, range(0, len(info_latt_plot[1])))
    plt.xlabel("Sites")
    plt.ylabel(r"Levels ($\ell$)")
    plt.title("Information lattice of initial state")
    ax.set_aspect("equal")
    plt.show()



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    min_l = 3
    max_l = 4
    L = 16
    dissipation_list = [0.0, 0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dissipation_strength = dissipation_list[2]

    J_list = [0.0, 0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -1.0]
    J = J_list[-1]

    J_list = [J for j in range(L)]
    hamiltonian_couplings = [['xx', J_list]]

    # lindblad part for the evolution
    list_dis = [dissipation_strength for j in range(L)]
    jump_couplings = [['tbd', list_dis]]

    # lindbaldians
    setup_lindbladian = li.operators.Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

    # initial state - product state of single site density matrices

    # homogeneous
    #prob_up = 1 - epsilon
    #site_1 = np.array([[prob_up, 0.0], [0.0, 1 - prob_up]])
    #site_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
    #bulk = [site_1 for _ in range(L)]

       # Mixed Bell pairs: sqrt(1/2)|00> + sqrt(1/2)|11>
    #bell = (1/2) * np.array([[1, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0],[1, 0, 0, 1]])
    #epsilon = 0.2  # small mixing
    #I4 = np.eye(4)
    #mixed_bell = (1 - epsilon) * bell + epsilon * I4 / 4  # a weighted sum of the Bell and the maximally mixed state (I4)/4
    #bulk = [mixed_bell for _ in range(int(L/2))]

       # Mixed imperfect Bell pairs: sqrt(1/5)|00> + sqrt(4/5)|11>
    #imperfect_bell = np.array([[1/5, 0, 0, np.sqrt(4)/5],[0, 0, 0, 0],[0, 0, 0, 0],[np.sqrt(4)/5, 0, 0, 4/5]])
    #epsilon = 0.2  # small mixing
    #I4 = np.eye(4)
    #mixed_bell = (1 - epsilon) * imperfect_bell + epsilon * I4 / 4  # a weighted sum of the Bell and the maximally mixed state (I4)/4
    #bulk = [mixed_bell for _ in range(int(L/2))]
#
#

        # Mixed single qubits
    #epsilon = 0.2
    #mixed_site = np.array([
    #    [1.0 - epsilon, 0.0],
    #    [0.0, epsilon],
    #])
    #bulk = [mixed_site for _ in range(L)]
##

        # mixed 3-qubit W blocks
    #w_vec = np.zeros(8, dtype=complex)
    #w_vec[1] = 1 / np.sqrt(3)  # |001>
    #w_vec[2] = 1 / np.sqrt(3)  # |010>
    #w_vec[4] = 1 / np.sqrt(3)  # |100>
##
    #w_state = np.outer(w_vec, np.conj(w_vec))
##
    #epsilon = 0.2
    #I8 = np.eye(8, dtype=complex)
    #mixed_w = (1.0 - epsilon) * w_state + epsilon * I8 / 8.0
##
    #bulk = [mixed_w for _ in range(L // 3)]


        # 4-qubit linear cluster state with local mixing:
        # start from |++++> and apply CZ on (0,1), (1,2), (2,3)
    psi = np.ones(16, dtype=np.complex128) / 4
    for x in range(16):
        b0 = (x >> 3) & 1
        b1 = (x >> 2) & 1
        b2 = (x >> 1) & 1
        b3 = x & 1
        psi[x] *= (-1) ** (b0 * b1 + b1 * b2 + b2 * b3)
#
    rho = np.outer(psi, np.conj(psi))

    epsilon = 0.2
#
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
#
    def embed_single(op, q, n=4):
        mats = [I2] * n
        mats[q] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out
#
    def local_dephase(rho, p, q):
        Zq = embed_single(Z, q)
        return (1 - p) * rho + p * (Zq @ rho @ Zq)
#
    for q in range(4):
        rho = local_dephase(rho, epsilon, q)
#
    mixed_block = rho

        # 5-qubit linear cluster state with global mixing depolarization:
        # start from |++++> and apply CZ on (0,1), (1,2), (2,3)
    #epsilon = 0.1
#
    #psi = np.ones(32, dtype=np.complex128) / np.sqrt(32)
#
    #for x in range(32):
    #    b0 = (x >> 4) & 1
    #    b1 = (x >> 3) & 1
    #    b2 = (x >> 2) & 1
    #    b3 = (x >> 1) & 1
    #    b4 = x & 1
    #    phase = (-1) ** (b0 * b1 + b1 * b2 + b2 * b3 + b3 * b4)
    #    psi[x] *= phase
#
    #cluster_state = np.outer(psi, np.conj(psi))
    #I32 = np.eye(32, dtype=np.complex128)
    #mixed_block = (1 - epsilon) * cluster_state + epsilon * I32 / 32

    bulk = [mixed_block for _ in range(L // 4)]



    initial_state = li.State.build_finite(bulk, 0)

    plot_info_latt_from_state(initial_state, InfoLattice_Norm=2)



    density_matrix_all_levels = initial_state.get_all_levels()
    info_lattice = initial_state.get_information_lattice(density_matrix_all_levels)
#
    for key, val in sorted(info_lattice.items(), key=lambda kv: (kv[0].level, kv[0].coord)):
        print(f"level={key.level + 1}, coord={key.coord}, value={val:.12f}")




    def save(data, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    ## time evolution loop
    checkpoint_folder = os.path.join(
        script_dir,
        "results",
        f'xx_diss={dissipation_strength}_J={J}_L={L}_lmin={min_l}_lmax={max_l}'
    )

    # Observables

    def local_marker(rho_dict):
        marker_dict = li.LatticeDict()
        for key, many_body_rho in rho_dict.items():
            L_loc = int(np.log2(len(many_body_rho)))
            marker = np.real_if_close(
                calc_marker_state(many_body_rho, L_loc, flatten=False)
            )
            marker_dict[key] = np.asarray(marker, dtype=float)
        return marker_dict

    def on_site_density(rho_dict):
        density_dict = li.LatticeDict()
        for key, many_body_rho in rho_dict.items():
            opdm = calc_opdm_from_rho(many_body_rho)
            L_loc = opdm.shape[0] // 2
            occupations = np.real_if_close(np.diag(opdm[:L_loc, :L_loc]))
            density_dict[key] = np.asarray(occupations, dtype=float)
        return density_dict

    data_config = li.DataConfig(
        info_lattice=True,
        density_matrix=True,
        observables=[local_marker, on_site_density],
    )

    data = li.DataContainer(config=data_config)

    config = li.config.TimeEvolutionConfig(
        save_checkpoint=True,
        checkpoint_folder=checkpoint_folder,
        min_l=min_l,
        max_l=max_l,
        shift=10,
        logging_config = li.config.LoggingConfig(logging_folder=checkpoint_folder),
    )

    system = li.OpenSystem(initial_state, setup_lindbladian, config=config,
                               data=data)
    steps = 1
    for i in range(steps):
        system.evolve(max_evolution_time=1.0, final_time=True)
        print(f'finished cycle {i} of {steps}')
        system.solver.step_size = 0.01



if __name__ == '__main__':
    main()
