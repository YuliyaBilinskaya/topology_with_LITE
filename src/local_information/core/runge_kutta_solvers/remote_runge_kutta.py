from __future__ import annotations

import logging
import sys
from copy import deepcopy
import numpy as np

from local_information.lattice.lattice_dict import LatticeKey
from local_information.mpi.mpi import *
from local_information.mpi.mpi_setup import COMM, RANK
from local_information.core.utils import commutator
from local_information.core.utils import get_higher_level
from local_information.core.petz_map import ptrace
from local_information.config import RungeKuttaConfig
from local_information.state.state import State
from local_information.core.runge_kutta_solvers.runge_kutta_solver import (
    RungeKuttaSolver,
)
from local_information.core.utils import anti_commutator
from local_information.typedefs import SystemOperator
from local_information.lattice.lattice_dict import LatticeKey, LatticeDict
from local_information.mpi.distribute import Distributor

logger = logging.getLogger()


class RemoteRungeKuttaSolver(RungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        hamiltonian: SystemOperator,
    ):
        super().__init__(
            runge_kutta_config=runge_kutta_config,
            range_=range_,
            system_operator=hamiltonian,
        )

    def dissipator(
            self,
            key: LatticeKey,
            density_matrix: np.ndarray,
            work_dict: LatticeDict | None = None,
            key_max_l_dim: int | None = None,
            m: int | None = None,
    ) -> np.ndarray | None:
        return None

    @property
    def hamiltonian(self):
        return self._system_operator

    def solve(
        self, state: State, final_time: float | None = None
    ) -> tuple[LatticeDict, float]:
        """! Compute one runge kutta step"""
        rho_dict, dyn_max_l, current_time = (
            state.density_matrix,
            state.dyn_max_l,
            state.current_time,
        )

        repeat = True
        while repeat:
            total_high = LatticeDict()
            total_low = LatticeDict()
            time = None
            k = None

            rhs = self.runge_kutta_func(rho_dict, dyn_max_l)

            if RANK != 0:
                if rhs is not None:
                    logger.debug(f"RANK {RANK} has {rhs}")
                    sys.stdout.flush()
                    raise ValueError

            if RANK == 0:
                k = [self.step_size * rhs]

            for j in range(1, len(self.b_higher)):
                addition = LatticeDict()
                for ell in range(j):
                    if RANK == 0:
                        addition = addition + self.a[j, ell] * k[ell]
                rhs = self.runge_kutta_func(rho_dict + addition, dyn_max_l)
                if RANK == 0:
                    k += [self.step_size * rhs]

            # sum up
            if RANK == 0:
                if self.order == 12:
                    for j, _ in enumerate(self.b_higher):
                        total_high = total_high + self.b_higher[j] * k[j]

                    # estimated error in the 10th order RK method according to:
                    # Neural, Parallel, and Scientific Computations 20 (2012) 437-458
                    error = 49 / 640 * (k[1] - k[23])

                elif self.order == 8:
                    for j, _ in enumerate(self.b_higher):
                        total_high = total_high + self.b_higher[j] * k[j]

                    # estimated error in the 10th order RK method according to:
                    # Neural, Parallel, and Scientific Computations 20 (2012) 437-458
                    error = 1 / 360 * (k[1] - k[15])

                else:
                    for j, _ in enumerate(self.b_higher):
                        total_high = total_high + self.b_higher[j] * k[j]
                        total_low = total_low + self.b_lower[j] * k[j]

                    error = total_high - total_low

                # compute error
                allowed_step_size = np.zeros(len(error))
                actual_error = np.linalg.norm(list(error.values()), np.inf, axis=(1, 2))

                for j, _ in enumerate(error):
                    if not actual_error[j] < 1e-16:
                        allowed_step_size[j] = self.step_size * (
                            self.max_error / actual_error[j]
                        ) ** (1 / self.order)
                    else:
                        allowed_step_size[j] = (self.step_size * 1.05) / 0.9

                # compare optimal step size to actual step size
                if np.any(allowed_step_size < self.step_size):
                    # redo calculation with smaller step size
                    self.step_size = np.min(allowed_step_size) * 0.9
                else:
                    # repeat to land exactly on the final time
                    if (
                        final_time is not None
                        and current_time + self.step_size > final_time
                    ):
                        self.step_size = final_time - current_time
                    else:
                        time = current_time + self.step_size
                        # update the step size
                        self.step_size = np.min(allowed_step_size) * 0.9
                        repeat = False

            repeat = COMM.bcast(repeat, root=0)

        time = COMM.bcast(time, root=0)
        self.step_size = COMM.bcast(self.step_size, root=0)

        if RANK == 0:
            tot = rho_dict + total_high
            tot_ct = (rho_dict + total_high).dagger()
            updated_rho_dict = 0.5 * (tot + tot_ct)
        else:
            updated_rho_dict = None

        updated_rho_dict = COMM.bcast(updated_rho_dict, root=0)

        logger.info(
            f"finished Runge-Kutta time step with adaptive stepsize {self.step_size}"
        )
        return updated_rho_dict, time

    def runge_kutta_func(self, rho_dict: LatticeDict, dyn_max_l: int):
        """Computes the right hand side of the von-Neumann equation"""

        sqrt_method = self.config.petz_map == "sqrt"
        work_dict = deepcopy(rho_dict)

        if len(work_dict) != 1:
            for r in range(self.range_):
                higher_level_dict = get_higher_level(
                    work_dict, dyn_max_l + r, sqrt_method=sqrt_method
                )
                if RANK == 0:
                    work_dict += higher_level_dict
                else:
                    work_dict = None
            if RANK != 0:
                work_dict = None

        root_work_dict = work_dict
        if RANK == 0:
            distributor = Distributor(root_work_dict, dyn_max_l)
        else:
            distributor = None

        if RANK == 0:
            block = distributor.scatter_with_halo(
                number_of_workers=COMM.Get_size(),
                left_halo=self.range_,
                right_halo=self.range_,
            )
        else:
            dummy = LatticeDict()
            if dyn_max_l not in []:
                pass
            distributor = Distributor(rho_dict, dyn_max_l)
            block = distributor.scatter_with_halo(
                number_of_workers=COMM.Get_size(),
                left_halo=self.range_,
                right_halo=self.range_,
            )

        local_rhs = self._runge_kutta_func_local_block(
            local_work_dict=block.local_lattice,
            owned_keys=block.owned_keys,
            dyn_max_l=dyn_max_l,
            global_level_dim=block.global_level_dim,
            global_index_by_key=block.global_index_by_key,
        )

        result = distributor.gather_owned(local_rhs)
        return result

    def _runge_kutta_func_local_block(
            self,
            local_work_dict: LatticeDict,
            owned_keys: list[LatticeKey],
            dyn_max_l: int,
            global_level_dim: int,
            global_index_by_key: dict[LatticeKey, int],
    ) -> LatticeDict:

        """
        Compute RHS on one MPI rank for its owned keys only.
        `local_work_dict` includes owned keys plus left/right halo keys.
        """
        rhs_dict = LatticeDict()

        if not owned_keys:
            return rhs_dict


        for key in owned_keys:
            m = global_index_by_key[key]

            if m >= self.range_:
                distance = self.range_
            else:
                distance = m

            key_l = key.left_up(distance)

            if key_l not in local_work_dict:
                raise KeyError(f"Missing left dependency key on rank {RANK}: {key_l}")

            DM_l = local_work_dict[key_l]
            H_max_l_range = self._system_operator.subsystem_hamiltonian[key_l]
            _com_l = ptrace(
                commutator(H_max_l_range.toarray(), DM_l), distance, end="left"
            )

            bar_m = (global_level_dim - 1) - m
            if bar_m >= self.range_:
                distance = self.range_
            else:
                distance = bar_m

            key_r = key.right_up(distance)

            if key_r not in local_work_dict:
                raise KeyError(f"Missing right dependency key on rank {RANK}: {key_r}")

            DM_r = local_work_dict[key_r]
            H_max_l_range = self._system_operator.subsystem_hamiltonian[key_r]
            _com_r = ptrace(
                commutator(H_max_l_range.toarray(), DM_r),
                distance,
                end="right",
            )

            DM_c = local_work_dict[key]
            H_max_l = self._system_operator.subsystem_hamiltonian[key]
            _com_c = commutator(H_max_l.toarray(), DM_c)
            rhs = _com_l + _com_r - _com_c

            D = self.dissipator(
                key,
                DM_c,
                work_dict=local_work_dict,
                key_max_l_dim=global_level_dim,
                m=m,
            )

            if D is not None:
                rhs += 1j * D

            rhs_dict[key] = -1j * rhs

        return rhs_dict


class RemoteLindbladRungeKuttaSolver(RemoteRungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        lindbladian: SystemOperator,
    ):
        super().__init__(
            runge_kutta_config=runge_kutta_config,
            range_=range_,
            hamiltonian=lindbladian,
        )

    def dissipator(
        self,
        key: LatticeKey,
        density_matrix: np.ndarray,
        work_dict: LatticeDict | None = None,
        key_max_l_dim: int | None = None,
        m: int | None = None,
    ) -> np.ndarray | None:
        has_tbd = any(term[0] == "tbd" for term in self._system_operator.jump_couplings)

        if has_tbd:
            if work_dict is None or key_max_l_dim is None or m is None:
                raise ValueError(
                    "tbd dissipator requires work_dict, key_max_l_dim, and m"
                )
            return self.dissipator_with_boundary_corrections(
                key,
                density_matrix,
                work_dict,
                key_max_l_dim,
                m,
            )

        return self._center_dissipator(key, density_matrix)

    def _center_dissipator(
        self,
        key: LatticeKey,
        density_matrix: np.ndarray,
    ) -> np.ndarray | None:
        lindbladian_dict_entry = self._system_operator.lindbladian_dict[key]
        D = np.zeros(
            (2 ** (key.level + 1), 2 ** (key.level + 1)),
            dtype=np.complex128,
        )

        count_non_zero_L = 0
        for e, dict_entry in enumerate(lindbladian_dict_entry):
            if dict_entry is None:
                continue

            count_non_zero_L += 1
            for entry in dict_entry:
                tpe = entry[0]
                coupling = entry[1]
                id_ = LatticeKey(level=key.level, coord=e, name=tpe)
                L = self._system_operator.L_operators[id_].toarray()
                L_dagger = np.conjugate(np.transpose(L))
                D += coupling * (
                    L @ density_matrix @ L_dagger
                    - 0.5 * anti_commutator(L_dagger @ L, density_matrix)
                )

        if count_non_zero_L == 0:
            return None
        return D

    def _single_jump_dissipator(
        self,
        key: LatticeKey,
        density_matrix: np.ndarray,
        local_index: int,
    ) -> np.ndarray:
        lindbladian_dict_entry = self._system_operator.lindbladian_dict[key]
        D = np.zeros_like(density_matrix, dtype=np.complex128)

        dict_entry = lindbladian_dict_entry[local_index]
        if dict_entry is None:
            return D

        for entry in dict_entry:
            tpe = entry[0]
            coupling = entry[1]
            id_ = LatticeKey(level=key.level, coord=local_index, name=tpe)
            L = self._system_operator.L_operators[id_].toarray()
            L_dagger = np.conjugate(np.transpose(L))
            D += coupling * (
                L @ density_matrix @ L_dagger
                - 0.5 * anti_commutator(L_dagger @ L, density_matrix)
            )

        return D

    def dissipator_with_boundary_corrections(
        self,
        key: LatticeKey,
        density_matrix: np.ndarray,
        work_dict: LatticeDict,
        key_max_l_dim: int,
        m: int,
    ) -> np.ndarray | None:
        D_c = self._center_dissipator(key, density_matrix)
        if D_c is None:
            D_c = np.zeros_like(density_matrix, dtype=np.complex128)

        D_l = np.zeros_like(density_matrix, dtype=np.complex128)
        D_r = np.zeros_like(density_matrix, dtype=np.complex128)

        if m >= self.range_:
            distance_l = self.range_
        else:
            distance_l = m

        if distance_l > 0:
            key_l = key.left_up(distance_l)
            DM_l = work_dict[key_l]
            D_l_full = self._single_jump_dissipator(key_l, DM_l, local_index=0)
            D_l = ptrace(D_l_full, distance_l, end="left")

        bar_m = (key_max_l_dim - 1) - m
        if bar_m >= self.range_:
            distance_r = self.range_
        else:
            distance_r = bar_m

        if distance_r > 0:
            key_r = key.right_up(distance_r)
            DM_r = work_dict[key_r]
            right_index = key_r.level - 1
            D_r_full = self._single_jump_dissipator(
                key_r, DM_r, local_index=right_index
            )
            D_r = ptrace(D_r_full, distance_r, end="right")

        return D_c + D_l + D_r

    @property
    def lindbladian(self):
        return self._system_operator