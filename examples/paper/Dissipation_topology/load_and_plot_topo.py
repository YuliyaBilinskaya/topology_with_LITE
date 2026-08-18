import os
import re
import pickle
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap, Normalize
from marker_functions import *





L = 9
dissipation_strength = 0.4
J = -1.0
min_l = 3
max_l= 4
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_folder = os.path.join(
    script_dir,
    "results",
    f"xx_diss={dissipation_strength}_J={J}_L={L}_lmin={min_l}_lmax={max_l}"
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
#loaded_dens_mat = load_from_file(os.path.join(data_filepath, 'density_matrix.pkl'))
#loaded_times = load_from_file(os.path.join(data_filepath, 'times.pkl'))
# Info Lattice and info per scale
default_time_indexes = [35]  #[len(times) - 1]
info_per_scale = process_data(info_latt, default_time_indexes)

# Correlation matrix evolution
#opdm_t_corr = np.load(os.path.join(data_filepath, "opdm_t_corr.npy"))
#times_corr = np.load(os.path.join(data_filepath, "times_corr.npy"))


#ave_marker_per_t_lite = calc_ave_marker_threesite_per_t(loaded_dens_mat, L)
#ave_marker_per_t_corr =  calc_marker_const_center_per_t_from_opdm(opdm_t_corr, times_corr)



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



def plot_local_marker_comparison_lmin_lmax(
    ave_marker_per_t_corr, times_corr,
    corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison_lmin_lmax.pdf",
    use_lines=True, truncate_to_common_times=False,
    results_dir=None, dissipation_strength=None, J=None, L=None
):
    corr_values = np.array(list(ave_marker_per_t_corr.values()), dtype=float)
    corr_times = np.array(times_corr, dtype=float)

    fig, ax1 = plt.subplots()

    if results_dir is not None:
        pattern = re.compile(
            rf"^xx_diss={dissipation_strength}_J={J}_L={L}_lmin=(\d+)_lmax=(\d+)$"
        )

        matching = []
        for folder in os.listdir(results_dir):
            full_path = os.path.join(results_dir, folder)
            if not os.path.isdir(full_path):
                continue

            match = pattern.match(folder)
            if match is None:
                continue

            lmin_val = int(match.group(1))
            lmax_val = int(match.group(2))

            if lmax_val - lmin_val == 1:
                matching.append((lmin_val, lmax_val, full_path))

        matching.sort(key=lambda x: (x[0], x[1]))

        colors = [
            "#1b5e20",
            "#1565c0",
            "#ef6c00",
            "#6a1b9a",
            "#c62828",
            "#00897b",
            "#8d6e63",
        ]

        for i, (lmin_val, lmax_val, folder_path) in enumerate(matching):
            color = colors[i % len(colors)]

            dens_mat = load_from_file(os.path.join(folder_path, "density_matrix.pkl"))
            times_other = load_from_file(os.path.join(folder_path, "times.pkl"))
            ave_marker_other = calc_ave_marker_threesite_per_t(dens_mat, L)

            times_other = np.array(
                [times_other[idx] for idx in ave_marker_other.keys()],
                dtype=float
            )
            values_other = np.array(list(ave_marker_other.values()), dtype=float)

            if truncate_to_common_times and len(corr_times) > 0:
                mask = times_other <= corr_times[-1]
                times_other = times_other[mask]
                values_other = values_other[mask]

            if use_lines:
                ax1.plot(times_other, values_other, linewidth=1.0, alpha=0.8, color=color)

            ax1.scatter(
                times_other,
                values_other,
                marker='o',
                edgecolors=color,
                facecolors='none',
                linewidths=1.0,
                s=25,
                zorder=3,
                label=rf'LITE evolution ($\ell_{{\min}}={lmin_val}, \ell_{{\max}}={lmax_val}$)'
            )

    if use_lines:
        ax1.plot(corr_times, corr_values, color='red', linewidth=1.0)

    ax1.scatter(
        corr_times,
        corr_values,
        marker='x',
        color='red',
        linewidths=0.5,
        s=10,
        label=corr_label
    )

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


################## PLOTTING ##############
plot_results(info_per_scale, info_latt, times, default_time_indexes, L,  InfoLatt_norm=1)

#plot_local_marker(ave_marker_per_t_lite, loaded_times)

#plot_local_marker_comparison(ave_marker_per_t_lite, loaded_times, ave_marker_per_t_corr, times_corr,
#                                lite_label='LITE evolution', corr_label='Correlation matrix evolution',
#                                save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
#                                use_lines=True, truncate_to_common_times=False)


#plot_local_marker_comparison_lmin_lmax(
#    ave_marker_per_t_corr, times_corr,
#    corr_label='Correlation matrix evolution',
#    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison_lmin_lmax.pdf",
#    use_lines=True,
#    truncate_to_common_times=False,
#    results_dir=os.path.join(script_dir, "results"),
#    dissipation_strength=dissipation_strength,
#    J=J,
#    L=L
#)


#plot_opdm_eigvals(loaded_dens_mat, loaded_times, L)

#plot_center_markers(loaded_dens_mat, loaded_times, L)

#plot_center_markers_from_corr_mat_evo(opdm_t_corr, times_corr)

#plot_center_half_mode_weights(loaded_dens_mat, loaded_times, L, tol=1e-3)

#plot_onsite_occupations_from_corr_mat_evo(opdm_t_corr, times_corr)

#plot_center_occupations(loaded_dens_mat, loaded_times, L)


with open(os.path.join(data_filepath, "on_site_density.pkl"), "rb") as f:
    on_site_density_data = pickle.load(f)

with open(os.path.join(data_filepath, "times.pkl"), "rb") as f:
    loaded_times = pickle.load(f)


#plot_center_occupations_from_file(on_site_density_data, loaded_times)


#def load_from_file(file_path):
#    with open(file_path, "rb") as file:
#        return pickle.load(file)


def plot_saved_site_observable(result_folder, observable_name, choose="center"):
    """
    Plot per-site values of a saved custom observable.

    Parameters
    ----------
    result_folder : str
        Folder containing times.pkl and <observable_name>.pkl
    observable_name : str
        "local_marker" or "on_site_density"
    choose : str
        Which subsystem to inspect at each timestep:
        - "center": subsystem closest to the center among the largest-level subsystems
        - "first": first subsystem after sorting by (level, coord)
    """
    times = load_from_file(os.path.join(result_folder, "times.pkl"))
    observable_per_t = load_from_file(
        os.path.join(result_folder, f"{observable_name}.pkl")
    )

    times_plot = []
    values_per_t = []
    chosen_keys = []

    for idx, obs_dict in enumerate(observable_per_t):
        if len(obs_dict) == 0:
            continue

        max_ell = max(key.level for key in obs_dict.keys())
        max_ell_keys = [key for key in obs_dict.keys() if key.level == max_ell]

        if choose == "center":
            coords = [key.coord for key in max_ell_keys]
            center_coord = 0.5 * (min(coords) + max(coords))
            chosen_key = min(max_ell_keys, key=lambda key: abs(key.coord - center_coord))
        elif choose == "first":
            chosen_key = sorted(max_ell_keys, key=lambda key: (key.level, key.coord))[0]
        else:
            raise ValueError("choose must be 'center' or 'first'")

        values = np.asarray(obs_dict[chosen_key], dtype=float)

        times_plot.append(times[idx])
        values_per_t.append(values)
        chosen_keys.append((chosen_key.coord, chosen_key.level))

    if not values_per_t:
        raise ValueError(f"No data found for observable '{observable_name}'")

    times_plot = np.asarray(times_plot, dtype=float)
    max_n_sites = max(len(v) for v in values_per_t)
    values_array = np.full((len(values_per_t), max_n_sites), np.nan, dtype=float)

    for i, values in enumerate(values_per_t):
        values_array[i, :len(values)] = values

    fig, ax = plt.subplots()

    for site in range(max_n_sites):
        valid = ~np.isnan(values_array[:, site])
        ax.scatter(
            times_plot[valid],
            values_array[valid, site],
            marker=".",
            s=8,
            label=f"site {site}" if site < 8 else None,
        )

    final_values = values_array[-1]
    valid_final = ~np.isnan(final_values)
    summary = ", ".join(
        f"{site}:{value:.3f}"
        for site, value in enumerate(final_values)
        if valid_final[site]
    )

    coord, level = chosen_keys[-1]
    ax.set_xlabel("Time")
    ax.set_ylabel(observable_name)
    ax.set_title(
        f"{observable_name} for subsystem (coord={float(coord)}, level={int(level)})"
    )

    if observable_name == "on_site_density":
        ax.set_ylabel(r"$\langle n_j \rangle$")
    elif observable_name == "local_marker":
        ax.set_ylabel("Local marker")

    ax.text(
        1.02,
        0.5,
        f"Final time values\n{summary}",
        transform=ax.transAxes,
        fontsize=8,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
    )

    ax.legend()
    fig.tight_layout()
    plt.show()

    return times_plot, values_array

#plot_saved_site_observable(checkpoint_folder, "local_marker")
#plot_saved_site_observable(checkpoint_folder, "on_site_density")

import pickle
import tables
import numpy as np

def plot_local_marker_comparison_h5(
    h5_path,
    local_marker_path,
    times_lite_path,
    lite_label='LITE evolution',
    corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
    use_lines=True,
    truncate_to_common_times=False,
):
    # Load correlation-matrix data from the .h5 file using PyTables
    with tables.open_file(h5_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())

    ave_marker_per_t_corr = calc_marker_const_center_per_t_from_opdm(opdm_t_corr, times_corr)

    # Load precomputed local markers and LITE times
    with open(local_marker_path, "rb") as f:
        local_marker_data = pickle.load(f)

    with open(times_lite_path, "rb") as f:
        loaded_times = pickle.load(f)

    # Average only the already-computed local markers
    ave_marker_per_t_lite = {}

    for idx, marker_t in enumerate(local_marker_data):
        if not isinstance(marker_t, dict):
            raise ValueError(
                f"Expected dict of subsystem markers at time index {idx}, got {type(marker_t)}"
            )

        max_level = max(key.level for key in marker_t.keys())
        max_level_keys = [key for key in marker_t.keys() if key.level == max_level]

        coords = [key.coord for key in max_level_keys]
        center_coord = 0.5 * (min(coords) + max(coords))
        center_key = min(max_level_keys, key=lambda key: abs(key.coord - center_coord))

        marker = np.asarray(marker_t[center_key], dtype=float)
        start = (len(marker) - 3) // 2
        ave_marker_per_t_lite[idx] = np.average(marker[start:start + 3])

    return plot_local_marker_comparison(
        ave_marker_per_t_lite,
        loaded_times,
        ave_marker_per_t_corr,
        times_corr,
        lite_label=lite_label,
        corr_label=corr_label,
        save_path=save_path,
        use_lines=use_lines,
        truncate_to_common_times=truncate_to_common_times,
    )



plot_local_marker_comparison_h5(
    h5_path=os.path.join(data_filepath, "opdm_t_corr_diss=0.2_J=-1_L=6_init=mixed_Bell.h5"),
    local_marker_path=os.path.join(data_filepath, "local_marker.pkl"),
    times_lite_path=os.path.join(data_filepath, "times.pkl"),
)



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

    L = max(info_latt_plot.keys())

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

    plt.xlim([-2 * r, 1])
    plt.ylim([-2 * r, 1 + 2 * r])
    plt.xticks(x_ticks, range(1, len(info_latt_plot[1]) + 1))
    plt.yticks(y_ticks, range(0, len(info_latt_plot[1])))
    plt.xlabel("Sites")
    plt.ylabel(r"Levels ($\ell$)")
    plt.title("Information lattice of initial state")
    ax.set_aspect("equal")
    plt.show()