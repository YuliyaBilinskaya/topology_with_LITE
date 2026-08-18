import os

from parser_topo_by_dissip import parse_args, build_run_configs
from marker_functions import calc_marker_state, calc_opdm_from_rho


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def configure_threads(num_threads: int):
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


args = parse_args()
configure_threads(args.threads)

import numpy as np
import local_information as li


def run_simulation(cfg, rank=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    J_values = [cfg.J for _ in range(cfg.L)]
    hamiltonian_couplings = [["xx", J_values]]

    list_dis = [cfg.dissipation for _ in range(cfg.L)]
    jump_couplings = [["tbd", list_dis]]

    setup_lindbladian = li.operators.Lindbladian(
        cfg.max_l, hamiltonian_couplings, jump_couplings
    )

    def build_initial_state(cfg):
        if cfg.initial_state == "Bell_mixed":
            bell = 0.5 * np.array([
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
            ])
            I4 = np.eye(4)
            mixed_block = (1 - cfg.epsilon) * bell + cfg.epsilon * I4 / 4

        elif cfg.initial_state == "imperfect_Bell_mixed": # imperfect_bell = sqrt(1/5)|00> + sqrt(4/5)|11>
            imperfect_bell = np.array([
                [1/5, 0, 0, np.sqrt(4)/5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [np.sqrt(4)/5, 0, 0, 4/5],
            ])
            I4 = np.eye(4)
            mixed_block = (1 - cfg.epsilon) * imperfect_bell + cfg.epsilon * I4 / 4


        elif cfg.initial_state == "triv_mixed":
            prob_up = 1.0 - cfg.epsilon
            rho_site = np.array([[prob_up, 0.0],[0.0, 1.0 - prob_up],], dtype=np.complex128)
            mixed_block = np.kron(rho_site, rho_site)

        elif cfg.initial_state == "W_mixed":
            if cfg.L % 3 != 0:
                raise ValueError("W_mixed requires L divisible by 3")

            w_vec = np.zeros(8, dtype=np.complex128)
            w_vec[1] = 1 / np.sqrt(3)  # |001>
            w_vec[2] = 1 / np.sqrt(3)  # |010>
            w_vec[4] = 1 / np.sqrt(3)  # |100>

            w_state = np.outer(w_vec, np.conj(w_vec))
            I8 = np.eye(8, dtype=np.complex128)
            mixed_block = (1 - cfg.epsilon) * w_state + cfg.epsilon * I8 / 8

        elif cfg.initial_state == "cluster_locally_mixed":
            if cfg.L % 4 != 0:
                raise ValueError("cluster_locally_mixed requires L divisible by 4")

            # 4-qubit linear cluster state mixed locally:
            # start from |++++> and apply CZ on (0,1), (1,2), (2,3)
            psi = np.ones(16, dtype=np.complex128) / 4
            for x in range(16):
                b0 = (x >> 3) & 1
                b1 = (x >> 2) & 1
                b2 = (x >> 1) & 1
                b3 = x & 1
                psi[x] *= (-1) ** (b0 * b1 + b1 * b2 + b2 * b3)

            rho = np.outer(psi, np.conj(psi))

            Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            I2 = np.eye(2, dtype=np.complex128)

            def embed_single(op, q, n=4):
                mats = [I2] * n
                mats[q] = op
                out = mats[0]
                for m in mats[1:]:
                    out = np.kron(out, m)
                return out

            def local_dephase(rho, p, q):
                Zq = embed_single(Z, q)
                return (1 - p) * rho + p * (Zq @ rho @ Zq)

            for q in range(4):
                rho = local_dephase(rho, cfg.epsilon, q)

            mixed_block = rho


        else:
            raise ValueError(f"Unknown initial_state: {cfg.initial_state}")

        if cfg.initial_state == "W_mixed":
            bulk = [mixed_block for _ in range(cfg.L // 3)]
        elif cfg.initial_state == "cluster_locally_mixed":
            bulk = [mixed_block for _ in range(cfg.L // 4)]
        else:
            bulk = [mixed_block for _ in range(cfg.L // 2)]

        return li.State.build_finite(bulk, 1)

    initial_state = build_initial_state(cfg)

    checkpoint_folder = os.path.join(
        script_dir,
        cfg.output_root,
        (
            f"xx_diss={cfg.dissipation}_J={cfg.J}_L={cfg.L}"
            f"_lmin={cfg.min_l}_lmax={cfg.max_l}_rank={rank}"
            f"_init={cfg.initial_state}"
        ),
    )

    # Observables:
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
        info_lattice=False,
        density_matrix=False,
        observables=[local_marker, on_site_density],
        #initial_state=cfg.initial_state,
    )

    data = li.DataContainer(config=data_config)

    config = li.config.TimeEvolutionConfig(
        save_checkpoint=True,
        checkpoint_folder=checkpoint_folder,
        min_l=cfg.min_l,
        max_l=cfg.max_l,
        shift=cfg.shift,
    )

    system = li.OpenSystem(
        initial_state,
        setup_lindbladian,
        config=config,
        data=data,
    )

    for i in range(cfg.steps):
        system.evolve(max_evolution_time=cfg.evolve_time, final_time=True)
        print(f"rank={rank} finished cycle {i + 1} of {cfg.steps}")
        system.solver.step_size = cfg.step_size


def main():
    configs = build_run_configs(args)

    if MPI is None:
        for cfg in configs:
            run_simulation(cfg, rank=0)
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for job_index, cfg in enumerate(configs):
        if job_index % size == rank:
            run_simulation(cfg, rank=rank)


if __name__ == "__main__":
    main()
