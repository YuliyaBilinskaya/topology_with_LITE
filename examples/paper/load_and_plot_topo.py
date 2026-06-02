import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap, Normalize

L = 6
dissipation_strength = 0.5
J = 1.0
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_folder = os.path.join(
    script_dir,
    "results",
    f"xx_dissipation={dissipation_strength}_J={J}_L={L}"
)


def load_from_file(file_path: str):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def process_data(loaded_info_latt, time_indexes):
    max_ell = 0
    info_per_scale = {}

    for idx in time_indexes:
        info_per_scale[idx] = {}

    for idx, d in enumerate(loaded_info_latt):
        if idx in time_indexes:
            for key, value in d.items():
                ell = key.level
                if idx == 0 and ell > max_ell:
                    max_ell = ell
                info_per_scale[idx][ell] = (
                    info_per_scale[idx].get(ell, 0) + value
                )

    return info_per_scale


# Load the data
data_filepath = checkpoint_folder
times = load_from_file(os.path.join(data_filepath, "times.pkl"))
info_latt = load_from_file(os.path.join(data_filepath, "info_lattice.pkl"))

def plot_results(info_per_scale, loaded_info_latt, loaded_times, time_indexes, L, InfoLatt_norm):
    color_map = plt.get_cmap("Oranges")
    colors = color_map(np.linspace(0, 1, 20))
    colors[0] = [1, 1, 1, 1]  # Set the first color to white
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    norm = Normalize(vmin=0, vmax=InfoLatt_norm)
    r = 1 / (4 * L)

    for idx in time_indexes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=200)

        scales, info_per_scale_plot = zip(*sorted(info_per_scale[idx].items()))
        scales, info_per_scale_plot = np.array(scales), np.array(info_per_scale_plot)

        interp_func = PchipInterpolator(scales, info_per_scale_plot)
        interp_scales = np.arange(scales[0], scales[-1] + 0.1, 0.1)
        interp_average = interp_func(interp_scales)

        ax1.plot(scales, info_per_scale_plot, marker='x', linestyle='None', markersize=8, color='b')
        ax1.plot(interp_scales, interp_average, color='black')

        ax1.set_ylim([0, L])
        ax1.set_ylabel(r"$\mathscr{i}^\ell$", fontsize=14)
        ax1.set_xlabel(r"$\ell$", fontsize=12)
        ax1.set_title(f"Information per scale (Time {np.round(loaded_times[idx], 5)})", fontsize=12)

        # **Plot on ax2: Information Lattice**
        ax2.set_aspect('equal')

        for key, value in loaded_info_latt[idx].items():
            key_x = key.coord
            key_y = key.level
            ax2.add_artist(plt.Circle(
                (key_x / L, (key_y + 0.5) / L), r, facecolor=custom_cmap(norm(value)),
                edgecolor='black', linewidth=0.2))

        ax2.set_xlim([-2 * r, 1])
        ax2.set_ylim([-2 * r, 1 + 2 * r])

        ax2.set_xticks([x / L for x in range(L)], range(1, L + 1))
        ax2.set_yticks([y / L + 1 / (2 * L) for y in range(L)], range(0, L))

        ax2.set_xlabel('Site', fontsize=12)
        ax2.set_ylabel('$\ell$', fontsize=12)
        ax2.set_title(f"Information Lattice (Time {np.round(loaded_times[idx], 5)})", fontsize=12)

        fig.suptitle(f'Total info = {np.round(np.sum(info_per_scale_plot), 5)}')

        plt.show()
        plt.close(fig)  # Free memory after each plot


default_time_indexes = [len(times) - 1]
info_per_scale = process_data(info_latt, default_time_indexes)

plot_results(info_per_scale, info_latt, times, default_time_indexes, L,  InfoLatt_norm=1)
