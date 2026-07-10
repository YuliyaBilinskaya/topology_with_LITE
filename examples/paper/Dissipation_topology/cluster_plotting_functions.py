import pickle
import matplotlib.pyplot as plt
from marker_functions import *
import tables
import numpy as np



def load_all_data(h5_path, local_marker_path, times_lite_path):
    with tables.open_file(h5_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())

    with open(local_marker_path, "rb") as f:
        local_marker_data = pickle.load(f)

    with open(times_lite_path, "rb") as f:
        loaded_times = pickle.load(f)

    return opdm_t_corr, times_corr, local_marker_data, loaded_times


def calc_corr_evo_local_markers(opdm_t_corr, sites_corr):
    marker_per_t = {}

    start, stop = sites_corr
    for idx, opdm in enumerate(opdm_t_corr):
        marker = calc_marker_state_from_opdm(opdm, flatten=False)
        selected_marker = marker[start:stop+1]
        marker_per_t[idx] = np.average(selected_marker)

    return marker_per_t


def plot_local_marker_comparison(local_marker_data, loaded_times, ave_marker_per_t_corr, times_corr,
                                 subsys, sites_lite,
                                 lite_label='LITE evolution',
                                 corr_label='Correlation matrix evolution',
                                 save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison.pdf",
                                 use_lines=True,
                                 truncate_to_common_times=False):

    def in_range(value, selected_range):
        start, stop = selected_range
        return start <= value <= stop

    ave_marker_per_t_lite = {}
    for idx, marker_t in enumerate(local_marker_data):
        if not isinstance(marker_t, dict):
            raise ValueError(
                f"Expected dict of subsystem markers at time index {idx}, got {type(marker_t)}"
            )

        max_level = max(key.level for key in marker_t.keys())
        max_level_keys = [key for key in marker_t.keys() if key.level == max_level]
        selected_keys = [key for key in max_level_keys if in_range(key.coord, subsys)]

        if not selected_keys:
            raise ValueError(f"No subsystems found for subsys={subsys} at time index {idx}")

        start, stop = sites_lite
        marker_values = []
        for key in selected_keys:
            marker = np.asarray(marker_t[key], dtype=float)
            selected_marker = marker[start:stop+1]
            marker_values.append(np.average(selected_marker))

        ave_marker_per_t_lite[idx] = np.average(marker_values)

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
    else:
        corr_times = times_corr

    fig, ax1 = plt.subplots()

    if use_lines:
        ax1.plot(times_lite, values_lite, color='blue', linewidth=1.0)
        ax1.plot(corr_times, corr_values, color='red', linewidth=1.0)

    ax1.scatter(times_lite, values_lite, marker='o', edgecolors='blue', facecolor='none',
                label=lite_label)
    ax1.scatter(corr_times, corr_values, marker='x', color='red', linewidths=0.5, s=10,
                label=corr_label)
    ax1.set_xlabel('Time')
    #ax1.set_ylabel(r'$\sum_{i=2}^{L-1} m_i \,/\, (L-2)$', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend()
    plt.title('Average Local Topological Marker over time')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return times_lite, values_lite, corr_times, corr_values
