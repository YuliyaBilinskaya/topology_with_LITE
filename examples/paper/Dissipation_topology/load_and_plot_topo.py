import os
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap, Normalize
from marker_functions import *


L = 10
dissipation_strength = 0.6
J = -1.0
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


# Load the data from LITE evolution
data_filepath = checkpoint_folder
times = load_from_file(os.path.join(data_filepath, "times.pkl"))
info_latt = load_from_file(os.path.join(data_filepath, "info_lattice.pkl"))
loaded_dens_mat = load_from_file(os.path.join(data_filepath, 'density_matrix.pkl'))
loaded_times = load_from_file(os.path.join(data_filepath, 'times.pkl'))
# Info Lattice and info per scale
default_time_indexes = [len(times) - 1]
info_per_scale = process_data(info_latt, default_time_indexes)

# Correlation matrix evolution
opdm_t_corr = np.load(os.path.join(data_filepath, "opdm_t_corr.npy"))
times_corr = np.load(os.path.join(data_filepath, "times_corr.npy"))


ave_marker_per_t_lite = calc_ave_marker_per_t(loaded_dens_mat, L)
ave_marker_per_t_corr =  calc_ave_marker_per_t_from_opdm(opdm_t_corr, times_corr)



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


def plot_local_marker(ave_marker_per_t, times):
    plot_times = {idx: np.round(times[idx], 3) for idx in ave_marker_per_t}

    fig, ax1 = plt.subplots()

    ax1.scatter(list(plot_times.values()), list(ave_marker_per_t.values()), marker='.', edgecolors='blue', facecolor='none',
                label='Average')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\sum_{i=2}^{L-1} m_i \,/\, (L-2)$', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    #ax1.set_ylim([0,1.1])

    # Title
    plt.title('Average Local Topological Marker over time')

    # Layout adjustment
    fig.tight_layout()

    plt.savefig("/Users/yuliyabilinskaya/Desktop/topo_marker.pdf", dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_local_marker_comparison(ave_marker_per_t_lite, loaded_times, ave_marker_per_t_corr, times_corr,
                                 lite_label='LITE evolution', corr_label='Correlation matrix evolution',
                                 save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
                                 use_lines=True, truncate_to_common_times=False):

    times_lite = np.array([loaded_times[idx] for idx in ave_marker_per_t_lite.keys()], dtype=float)
    values_lite = np.array(list(ave_marker_per_t_lite.values()), dtype=float)
    corr_values = np.array(list(ave_marker_per_t_corr.values()), dtype=float)

    if truncate_to_common_times and len(times_lite) > 0 and len(times_corr) > 0:
        t_max_common = min(times_lite[-1], times_corr[-1])
        lindblad_mask = times_lite <= t_max_common
        corr_mask = times_corr <= t_max_common
        times_lite = times_lite[lindblad_mask]
        values_lite = values_lite[lindblad_mask]
        corr_times = times_corr[corr_mask]
        corr_values = corr_values[corr_mask]

    fig, ax1 = plt.subplots()

    if use_lines:
        ax1.plot(times_lite, values_lite, color='blue', linewidth=1.0)
        ax1.plot(times_corr, corr_values, color='red', linewidth=1.0)

    ax1.scatter(times_lite, values_lite, marker='o', edgecolors='blue', facecolor='none',
                label=lite_label)
    ax1.scatter(times_corr, corr_values, marker='x', color='red',linewidths=0.5, s=10,
                label=corr_label)
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\sum_{i=2}^{L-1} m_i \,/\, (L-2)$', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend()
    plt.title('Average Local Topological Marker over time')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return times_lite, values_lite, times_corr, corr_values

################## PLOTTING ##############
#plot_results(info_per_scale, info_latt, times, default_time_indexes, L,  InfoLatt_norm=1)

#plot_local_marker(ave_marker_per_t_lite, loaded_times)

plot_local_marker_comparison(ave_marker_per_t_lite, loaded_times, ave_marker_per_t_corr, times_corr,
                                lite_label='LITE evolution', corr_label='Correlation matrix evolution',
                                save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
                                use_lines=True, truncate_to_common_times=False)