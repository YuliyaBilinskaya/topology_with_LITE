import os
import sys
import pickle
import numpy as np
import h5py

from marker_functions import calc_marker_state


def load_from_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_markers_all_subsystems_per_timestep_hdf5(
    loaded_dens_mat,
    loaded_times,
    output_path,
    flatten=False,
):
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("times", data=np.asarray(loaded_times, dtype=float))

        for time_idx, subsystems_at_t in enumerate(loaded_dens_mat):
            time_group = h5f.create_group(f"timestep_{time_idx:06d}")
            time_group.create_dataset("time", data=float(loaded_times[time_idx]))

            sorted_items = sorted(
                subsystems_at_t.items(),
                key=lambda item: (item[0].level, item[0].coord),
            )

            for subsystem_idx, (key, many_body_rho) in enumerate(sorted_items):
                L_loc = int(np.log2(len(many_body_rho)))
                marker = np.real_if_close(
                    calc_marker_state(many_body_rho, L_loc, flatten=flatten)
                )
                marker = np.asarray(marker, dtype=float)

                subsystem_group = time_group.create_group(
                    f"subsystem_{subsystem_idx:03d}"
                )
                subsystem_group.create_dataset("coord", data=float(key.coord))
                subsystem_group.create_dataset("level", data=int(key.level))
                subsystem_group.create_dataset("markers", data=marker)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python save_markers_from_density.py /path/to/result_folder")

    result_folder = sys.argv[1]

    density_path = os.path.join(result_folder, "density_matrix.pkl")
    times_path = os.path.join(result_folder, "times.pkl")
    output_path = os.path.join(result_folder, "markers_all_subsystems.h5")

    loaded_dens_mat = load_from_file(density_path)
    loaded_times = load_from_file(times_path)

    save_markers_all_subsystems_per_timestep_hdf5(
        loaded_dens_mat=loaded_dens_mat,
        loaded_times=loaded_times,
        output_path=output_path,
        flatten=False,
    )

    print(f"Saved markers to {output_path}")


if __name__ == "__main__":
    main()
