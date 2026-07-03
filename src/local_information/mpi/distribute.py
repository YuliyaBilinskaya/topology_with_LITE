from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import logging

import numpy as np

from local_information.lattice.lattice_dict import LatticeDict, LatticeKey
from local_information.mpi.mpi_setup import COMM, RANK, SIZE

logger = logging.getLogger()


@dataclass
class DistributedLatticeBlock:
    owned_keys: list[LatticeKey]
    local_lattice: LatticeDict
    global_level_dim: int
    global_index_by_key: dict[LatticeKey, int]



class Distributor:
    """
    Utilities to split a LatticeDict at one level across MPI ranks.

    Two modes exist:
    1. legacy one-sided overlap used by current @MultiProcessing helpers
    2. explicit halo blocks for distributed RK RHS evaluation
    """

    def __init__(self, density_matrices: LatticeDict, level: int):
        assert density_matrices, "Empty LatticeDict not allowed"
        if RANK == 0:
            assert list(density_matrices.keys_at_level(level)), (
                f"No density matrices at level {level}"
            )
        self._density_matrices = density_matrices
        self._level = level

    @property
    def ordered_keys(self) -> list[LatticeKey]:
        return [key for key in self._density_matrices.keys_at_level(self._level)]

    @cached_property
    def n_min(self) -> float:
        return self._density_matrices.smallest_at_level(self._level)

    @cached_property
    def n_max(self) -> float:
        return self._density_matrices.largest_at_level(self._level)

    def distribute(
        self, number_of_workers: int = SIZE, shift: int = 1
    ) -> list[LatticeDict] | None:
        self._check_shift(shift)

        if RANK == 0:
            split_up_lattice = self._split_lattice(
                number_of_splits=number_of_workers, shift=shift
            )
        else:
            split_up_lattice = None

        return split_up_lattice

    def _split_lattice(
        self, number_of_splits: int, shift: int = 1
    ) -> list[LatticeDict]:
        sub_lattices = []

        split_up_keys = self._split_keys(number_of_splits=number_of_splits, shift=shift)
        for key_block in split_up_keys:
            lattice_block = LatticeDict()
            for key in key_block:
                lattice_block[key] = self._density_matrices[key]

            sub_lattices.append(lattice_block)

        return sub_lattices

    def _split_keys(
        self, number_of_splits: int, shift: int = 1
    ) -> list[list[LatticeKey]]:
        split_up_keys = np.array_split(self.ordered_keys, number_of_splits)
        split_up_keys = list(map(lambda x: list(x), split_up_keys))
        self._add_keys_of_next_block(split_up_keys=split_up_keys, shift=shift)
        return split_up_keys

    @staticmethod
    def _add_keys_of_next_block(split_up_keys: list[list[LatticeKey]], shift: int = 1):
        for k, key_block in enumerate(split_up_keys[:-1]):
            next_key_block = split_up_keys[k + 1]
            for s in range(shift):
                if s < len(next_key_block):
                    key_block.append(next_key_block[s])

    def scatter(self, number_of_workers: int = SIZE, shift: int = 1) -> LatticeDict:
        split_up_lattice = self.distribute(
            number_of_workers=number_of_workers, shift=shift
        )
        return COMM.scatter(split_up_lattice, root=0)

    @staticmethod
    def gather(data: LatticeDict) -> LatticeDict | None:
        gathered_data = COMM.gather(data, root=0)

        if RANK == 0:
            joint_lattice = gathered_data[0]
            for lattice in gathered_data:
                joint_lattice.merge(lattice)
        else:
            joint_lattice = None

        return joint_lattice

    def scatter_with_halo(
        self,
        number_of_workers: int = SIZE,
        left_halo: int = 0,
        right_halo: int = 0,
    ) -> DistributedLatticeBlock:
        """
        Scatter owned keys together with left/right halo keys.
        Each rank computes outputs only for `owned_keys`.
        """
        if RANK == 0:
            blocks = self._build_halo_blocks(
                number_of_workers=number_of_workers,
                left_halo=left_halo,
                right_halo=right_halo,
            )
        else:
            blocks = None

        return COMM.scatter(blocks, root=0)

    def gather_owned(self, local_result: LatticeDict) -> LatticeDict | None:
        """
        Gather owned-only result lattices and merge at root.
        """
        gathered_data = COMM.gather(local_result, root=0)

        if RANK == 0:
            joint_lattice = LatticeDict()
            for lattice in gathered_data:
                joint_lattice.merge(lattice)
            return joint_lattice

        return None

    def _build_halo_blocks(
            self,
            number_of_workers: int,
            left_halo: int,
            right_halo: int,
    ) -> list[DistributedLatticeBlock]:
        ordered_keys = self.ordered_keys
        owned_splits = list(map(list, np.array_split(ordered_keys, number_of_workers)))
        global_index_by_key = {key: idx for idx, key in enumerate(ordered_keys)}
        global_level_dim = len(ordered_keys)

        blocks: list[DistributedLatticeBlock] = []
        for owned in owned_splits:
            if not owned:
                blocks.append(
                    DistributedLatticeBlock(
                        owned_keys=[],
                        local_lattice=LatticeDict(),
                        global_level_dim=global_level_dim,
                        global_index_by_key=global_index_by_key,
                    )
                )
                continue

            start = global_index_by_key[owned[0]]
            stop = global_index_by_key[owned[-1]]

            halo_start = max(0, start - left_halo)
            halo_stop = min(len(ordered_keys) - 1, stop + right_halo)

            base_keys = ordered_keys[halo_start: halo_stop + 1]

            needed_keys: set[LatticeKey] = set(base_keys)

            # Add all higher-level dependency keys needed by left_up/right_up lookups
            for key in base_keys:
                for d in range(1, max(left_halo, right_halo) + 1):
                    key_l = key.left_up(d)
                    key_r = key.right_up(d)

                    if key_l in self._density_matrices:
                        needed_keys.add(key_l)
                    if key_r in self._density_matrices:
                        needed_keys.add(key_r)

            lattice_block = LatticeDict()
            for key in self._density_matrices.keys():
                if key in needed_keys:
                    lattice_block[key] = self._density_matrices[key]

            blocks.append(
                DistributedLatticeBlock(
                    owned_keys=owned,
                    local_lattice=lattice_block,
                    global_level_dim=global_level_dim,
                    global_index_by_key=global_index_by_key,
                )
            )

        return blocks

    @staticmethod
    def _check_shift(shift: int):
        assert shift == 0 or shift == 1, (
            "shift values other than 0 or 1 are not supported."
        )
