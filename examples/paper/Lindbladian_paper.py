import os
import sys

os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pickle
import local_information as li


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    min_l = 3
    max_l = 5
    L = 20
    dissipation_list = [0.0, 0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dissipation_strength = dissipation_list[0]

    J_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    P = 16
    J = J_list[-1]

    J_list = [J for j in range(L)]
    hamiltonian_couplings = [['xx', J_list], ['yy', J_list]]

    # lindblad part for the evolution
    z_list_dis = [dissipation_strength for j in range(L)]
    jump_couplings = [['z', z_list_dis]]

    # lindbaldians
    setup_lindbladian = li.operators.Lindbladian(max_l, hamiltonian_couplings, jump_couplings)

    # initial state - product state of single site density matrices

    # homogeneous
    prob_up = 0.9
    site_1 = np.array([[prob_up, 0.0], [0.0, 1 - prob_up]])
    site_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
    bulk = [site_1 for _ in range(P)]
    boundary = [site_0 for _ in range(2)]

   # build the initial state
    initial_state = li.State.build_asymptotic(bulk, boundary, 1)


    #thermal H
#    beta = 0.05
#    site_1 = (np.eye(2 ** P, dtype=np.complex128) - beta * setup_lindbladian.subsystem_hamiltonian[(L // 2, P-1)].toarray()) / 2**P
#    site_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
#    system = [[site_1], [site_0]]
#    initial_state = li.src.State.build(system, 1)

    #Additional observable:
    # x-magnetisation
    z_ = [1.0 for j in range(L)]
    z_list = [['z', z_]]
    z_mag = li.operators.Operator(z_list)

    def z_magnetization(rho):
        # return expect_value(rho, magnetization_dict, 0)
        return z_mag.expectation_value(rho)

    z_operator_current = setup_lindbladian.operator_current(z_mag)

    def z_mag_diff_cosnt(rho):
        return li.operators.observables.onsite_operator_diff_const(
            rho,
            z_mag.operator,
            z_operator_current,
            initial_state.anchor,
        )


    def save(data, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    ## time evolution loop
    checkpoint_folder = os.path.join(
        script_dir,
        "results",
        f'xx_yy_dissipation={dissipation_strength}_J={J}_L={L}'
    )

    data_config = li.DataConfig(observables = [z_magnetization, z_mag_diff_cosnt], diffusion_const = True, info_lattice = True)
    data = li.DataContainer(config=data_config)

    config = li.config.TimeEvolutionConfig(
        save_checkpoint=True,
        checkpoint_folder=checkpoint_folder,
        min_l=min_l,
        max_l=max_l,
        shift=10)

    system = li.OpenSystem(initial_state, setup_lindbladian, config=config,
                               data=data)
    steps = 1
    for i in range(steps):
        system.evolve(max_evolution_time=0.1, final_time=True)
        print(f'finished cycle {i} of {steps}')
        system.solver.step_size = 0.25



if __name__ == '__main__':
    main()
