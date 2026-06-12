from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from local_information.config import RungeKuttaConfig
from local_information.core.runge_kutta_solvers.runge_kutta_solver import (
    RungeKuttaSolver,
)
from local_information.core.utils import anti_commutator
from local_information.lattice.lattice_dict import LatticeKey
from local_information.core.petz_map import ptrace



if TYPE_CHECKING:
    from local_information.operators.lindbladian import Lindbladian
    from local_information.operators.hamiltonian import Hamiltonian

logger = logging.getLogger()


class LocalRungeKuttaSolver(RungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        hamiltonian: Hamiltonian,
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
            work_dict=None,
            key_max_l_dim=None,
            m=None,
    ) -> np.ndarray | None:
        return None

    @property
    def hamiltonian(self):
        return self._system_operator


class LocalLindbladRungeKuttaSolver(RungeKuttaSolver):
    def __init__(
        self,
        runge_kutta_config: RungeKuttaConfig,
        range_: int,
        lindbladian: Lindbladian,
    ):
        super().__init__(
            runge_kutta_config=runge_kutta_config,
            range_=range_,
            system_operator=lindbladian,
        )

    def dissipator(
        self,
        key: LatticeKey,
        density_matrix: np.ndarray,
        work_dict=None,
        key_max_l_dim=None,
        m=None,
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
        """Center-block dissipator without boundary corrections."""
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
        """
        Dissipator contribution from exactly one local jump entry of lindbladian_dict[key].
        """
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
        work_dict,
        key_max_l_dim: int,
        m: int,
    ) -> np.ndarray | None:
        """
        D_eff = D_c + D_l + D_r for tbd jumps.
        """
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

