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
        subsys_eff = subsys
        if max_level % 2 == 1 and subsys[0] == subsys[1]:
            subsys_eff = (subsys[0] - 0.5, subsys[1] - 0.5)
        selected_keys = [key for key in max_level_keys if in_range(key.coord, subsys_eff)]

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

def plot_local_marker_comparison_lmin_lmax(
    local_marker_data, loaded_times, ave_marker_per_t_corr, times_corr, subsys, sites_lite,
    lite_labels=None, corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison_lmin_lmax.pdf",
    use_lines=True, truncate_to_common_times=False, t_min=None, t_max=None
):
    def in_range(value, selected_range):
        start, stop = selected_range
        return start <= value <= stop

    def get_item(container, key):
        if isinstance(container, dict):
            return container[key]
        if isinstance(container, (list, tuple)) and key is not None:
            return container[key]
        return container

    if isinstance(local_marker_data, dict):
        curve_keys = list(local_marker_data.keys())
    elif isinstance(local_marker_data, (list, tuple)):
        curve_keys = list(range(len(local_marker_data)))
    else:
        curve_keys = [None]

    corr_values = np.array(list(ave_marker_per_t_corr.values()), dtype=float)
    corr_times = np.array(times_corr, dtype=float)

    fig, ax1 = plt.subplots()
    colors = [
        "#1b5e20",
        "#1565c0",
        "#ef6c00",
        "#6a1b9a",
        "#c62828",
        "#00897b",
        "#8d6e63",
    ]

    for i, key in enumerate(curve_keys):
        marker_data_i = get_item(local_marker_data, key)
        loaded_times_i = get_item(loaded_times, key)
        subsys_i = get_item(subsys, key)
        sites_lite_i = get_item(sites_lite, key)

        ave_marker_per_t_lite = {}
        for idx, marker_t in enumerate(marker_data_i):
            if not isinstance(marker_t, dict):
                raise ValueError(
                    f"Expected dict of subsystem markers at time index {idx}, got {type(marker_t)}"
                )

            max_level = max(key_t.level for key_t in marker_t.keys())
            max_level_keys = [key_t for key_t in marker_t.keys() if key_t.level == max_level]
            subsys_eff = subsys_i
            if max_level % 2 == 1 and subsys_i[0] == subsys_i[1]:
                subsys_eff = (subsys_i[0] - 0.5, subsys_i[1] - 0.5)
            selected_keys = [key_t for key_t in max_level_keys if in_range(key_t.coord, subsys_eff)]

            if not selected_keys:
                raise ValueError(f"No subsystems found for subsys={subsys_i} at time index {idx}")

            start, stop = sites_lite_i
            marker_values = []
            for key_t in selected_keys:
                marker = np.asarray(marker_t[key_t], dtype=float)
                selected_marker = marker[start:stop+1]
                marker_values.append(np.average(selected_marker))

            ave_marker_per_t_lite[idx] = np.average(marker_values)

        times_lite = np.array([loaded_times_i[idx] for idx in ave_marker_per_t_lite.keys()], dtype=float)
        values_lite = np.array(list(ave_marker_per_t_lite.values()), dtype=float)

        if truncate_to_common_times and len(times_lite) > 0 and len(corr_times) > 0:
            t_max_common = min(times_lite[-1], corr_times[-1])
            lite_mask = times_lite <= t_max_common
            times_lite = times_lite[lite_mask]
            values_lite = values_lite[lite_mask]

        if t_min is not None:
            lite_mask = times_lite >= t_min
            times_lite = times_lite[lite_mask]
            values_lite = values_lite[lite_mask]

        if t_max is not None:
            lite_mask = times_lite <= t_max
            times_lite = times_lite[lite_mask]
            values_lite = values_lite[lite_mask]

        color = colors[i % len(colors)]

        if lite_labels is None:
            if isinstance(key, tuple) and len(key) == 2:
                lite_label = rf'LITE evolution ($\ell_{{\min}}={key[0]}, \ell_{{\max}}={key[1]}$)'
            elif key is None:
                lite_label = 'LITE evolution'
            else:
                lite_label = f'LITE evolution {key}'
        elif isinstance(lite_labels, dict):
            lite_label = lite_labels[key]
        else:
            lite_label = lite_labels[i]

        if use_lines:
            ax1.plot(times_lite, values_lite, color=color, linewidth=1.0)

        ax1.scatter(
            times_lite,
            values_lite,
            marker='o',
            edgecolors=color,
            facecolors='none',
            linewidths=1.0,
            s=25,
            label=lite_label,
        )

    if t_min is not None:
        corr_mask = corr_times >= t_min
        corr_times = corr_times[corr_mask]
        corr_values = corr_values[corr_mask]

    if t_max is not None:
        corr_mask = corr_times <= t_max
        corr_times = corr_times[corr_mask]
        corr_values = corr_values[corr_mask]

    if use_lines:
        ax1.plot(corr_times, corr_values, color='red', linewidth=1.0)

    ax1.scatter(
        corr_times,
        corr_values,
        marker='x',
        color='red',
        linewidths=0.5,
        s=10,
        label=corr_label,
    )

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


def plot_onsite_occupation_comparison_lmin_lmax(
    on_site_density_data, loaded_times, opdm_t_corr, times_corr, subsys, sites_lite,
    lite_labels=None, corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_onsite_occupation_comparison_lmin_lmax.pdf",
    use_lines=True, truncate_to_common_times=False, t_min=None, t_max=None
):
    def in_range(value, selected_range):
        start, stop = selected_range
        return start <= value <= stop

    def get_item(container, key):
        if isinstance(container, dict):
            return container[key]
        if isinstance(container, (list, tuple)) and key is not None:
            return container[key]
        return container

    if isinstance(on_site_density_data, dict):
        curve_keys = list(on_site_density_data.keys())
    elif isinstance(on_site_density_data, (list, tuple)):
        curve_keys = list(range(len(on_site_density_data)))
    else:
        curve_keys = [None]

    corr_times = np.array(times_corr, dtype=float)

    ref_key = next(iter(curve_keys))
    ref_density_data = get_item(on_site_density_data, ref_key)
    ref_subsys = get_item(subsys, ref_key)
    ref_sites_lite = get_item(sites_lite, ref_key)
    ref_density_t = ref_density_data[0]
    ref_max_level = max(key_t.level for key_t in ref_density_t.keys())
    ref_max_level_keys = [key_t for key_t in ref_density_t.keys() if key_t.level == ref_max_level]
    ref_subsys_eff = ref_subsys
    if ref_max_level % 2 == 1 and ref_subsys[0] == ref_subsys[1]:
        ref_subsys_eff = (ref_subsys[0] - 0.5, ref_subsys[1] - 0.5)
    ref_selected_keys = [key_t for key_t in ref_max_level_keys if in_range(key_t.coord, ref_subsys_eff)]

    corr_global_sites = []
    start_lite, stop_lite = ref_sites_lite
    for key_t in ref_selected_keys:
        start_site = int(round(key_t.coord - key_t.level / 2))
        corr_global_sites.extend(range(start_site + start_lite, start_site + stop_lite + 1))
    corr_global_sites = sorted(set(corr_global_sites))

    corr_density_per_t = {}
    for idx, opdm in enumerate(opdm_t_corr):
        density = np.real(np.diag(opdm[: opdm.shape[0] // 2, : opdm.shape[1] // 2]))
        corr_density_per_t[idx] = np.average(density[corr_global_sites])
    corr_values = np.array(list(corr_density_per_t.values()), dtype=float)

    fig, ax1 = plt.subplots()
    colors = [
        "#1b5e20",
        "#1565c0",
        "#ef6c00",
        "#6a1b9a",
        "#c62828",
        "#00897b",
        "#8d6e63",
    ]

    for i, key in enumerate(curve_keys):
        density_data_i = get_item(on_site_density_data, key)
        loaded_times_i = get_item(loaded_times, key)
        subsys_i = get_item(subsys, key)
        sites_lite_i = get_item(sites_lite, key)

        ave_density_per_t_lite = {}
        for idx, density_t in enumerate(density_data_i):
            if not isinstance(density_t, dict):
                raise ValueError(
                    f"Expected dict of subsystem densities at time index {idx}, got {type(density_t)}"
                )

            max_level = max(key_t.level for key_t in density_t.keys())
            max_level_keys = [key_t for key_t in density_t.keys() if key_t.level == max_level]
            subsys_eff = subsys_i
            if max_level % 2 == 1 and subsys_i[0] == subsys_i[1]:
                subsys_eff = (subsys_i[0] - 0.5, subsys_i[1] - 0.5)
            selected_keys = [key_t for key_t in max_level_keys if in_range(key_t.coord, subsys_eff)]

            if not selected_keys:
                raise ValueError(f"No subsystems found for subsys={subsys_i} at time index {idx}")

            start, stop = sites_lite_i
            density_values = []
            for key_t in selected_keys:
                density = np.asarray(density_t[key_t], dtype=float)
                selected_density = density[start:stop+1]
                density_values.append(np.average(selected_density))

            ave_density_per_t_lite[idx] = np.average(density_values)

        times_lite = np.array([loaded_times_i[idx] for idx in ave_density_per_t_lite.keys()], dtype=float)
        values_lite = np.array(list(ave_density_per_t_lite.values()), dtype=float)

        if truncate_to_common_times and len(times_lite) > 0 and len(corr_times) > 0:
            t_max_common = min(times_lite[-1], corr_times[-1])
            lite_mask = times_lite <= t_max_common
            times_lite = times_lite[lite_mask]
            values_lite = values_lite[lite_mask]

        if t_min is not None:
            lite_mask = times_lite >= t_min
            times_lite = times_lite[lite_mask]
            values_lite = values_lite[lite_mask]

        if t_max is not None:
            lite_mask = times_lite <= t_max
            times_lite = times_lite[lite_mask]
            values_lite = values_lite[lite_mask]

        color = colors[i % len(colors)]

        if lite_labels is None:
            if isinstance(key, tuple) and len(key) == 2:
                lite_label = rf'LITE evolution ($\ell_{{\min}}={key[0]}, \ell_{{\max}}={key[1]}$)'
            elif key is None:
                lite_label = 'LITE evolution'
            else:
                lite_label = f'LITE evolution {key}'
        elif isinstance(lite_labels, dict):
            lite_label = lite_labels[key]
        else:
            lite_label = lite_labels[i]

        if use_lines:
            ax1.plot(times_lite, values_lite, color=color, linewidth=1.0)

        ax1.scatter(
            times_lite,
            values_lite,
            marker='o',
            edgecolors=color,
            facecolors='none',
            linewidths=1.0,
            s=25,
            label=lite_label,
        )

    if t_min is not None:
        corr_mask = corr_times >= t_min
        corr_times = corr_times[corr_mask]
        corr_values = corr_values[corr_mask]

    if t_max is not None:
        corr_mask = corr_times <= t_max
        corr_times = corr_times[corr_mask]
        corr_values = corr_values[corr_mask]

    if use_lines:
        ax1.plot(corr_times, corr_values, color='red', linewidth=1.0)

    ax1.scatter(
        corr_times,
        corr_values,
        marker='x',
        color='red',
        linewidths=0.5,
        s=10,
        label=corr_label,
    )

    ax1.set_xlabel('Time')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend()
    plt.title('Average On-Site Density over time')
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig)



def plot_onsite_occupation_lmin_lmax_family(
    results_dir,
    diss_strength,
    J,
    L,
    init,
    subsys,
    sites_lite,
    sites_corr,
    corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_onsite_occupation_comparison_lmin_lmax.pdf",
    use_lines=True,
    truncate_to_common_times=False,
    t_min=None,
    t_max=None,
):
    import os
    import re

    corr_path = os.path.join(
        results_dir,
        f"opdm_t_corr_diss={diss_strength}_J={J}_L={L}_init=mixed_{init}.h5",
    )

    pattern = re.compile(
        rf"^xx_diss={diss_strength}_J={J}_L={L}_lmin=(\d+)_lmax=(\d+)_init={init}_mixed$"
    )

    on_site_density_data_all = {}
    loaded_times_all = {}
    subsys_all = {}
    sites_lite_all = {}

    for folder in sorted(os.listdir(results_dir)):
        match = pattern.match(folder)
        if match is None:
            continue

        lmin_val = int(match.group(1))
        lmax_val = int(match.group(2))
        folder_path = os.path.join(results_dir, folder)
        density_path = os.path.join(folder_path, "on_site_density.pkl")
        times_lite_path = os.path.join(folder_path, "times.pkl")

        if not os.path.isfile(density_path) or not os.path.isfile(times_lite_path):
            continue

        with open(density_path, "rb") as f:
            on_site_density_data = pickle.load(f)
        with open(times_lite_path, "rb") as f:
            loaded_times = pickle.load(f)

        key = (lmin_val, lmax_val)
        on_site_density_data_all[key] = on_site_density_data
        loaded_times_all[key] = loaded_times
        subsys_all[key] = subsys
        sites_lite_all[key] = sites_lite

    with tables.open_file(corr_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())

    return plot_onsite_occupation_comparison_lmin_lmax(
        on_site_density_data=on_site_density_data_all,
        loaded_times=loaded_times_all,
        opdm_t_corr=opdm_t_corr,
        times_corr=times_corr,
        subsys=subsys_all,
        sites_lite=sites_lite_all,
        corr_label=corr_label,
        save_path=save_path,
        use_lines=use_lines,
        truncate_to_common_times=truncate_to_common_times,
        t_min=t_min,
        t_max=t_max,
    )

def plot_local_marker_lmin_lmax_family(
    results_dir,
    diss_strength,
    J,
    L,
    init,
    subsys,
    sites_lite,
    sites_corr,
    corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison_lmin_lmax.pdf",
    use_lines=True,
    truncate_to_common_times=False,
    t_min=None,
    t_max=None,
):
    import os
    import re

    corr_path = os.path.join(
        results_dir,
        f"opdm_t_corr_diss={diss_strength}_J={J}_L={L}_init=mixed_{init}.h5",
    )

    pattern = re.compile(
        rf"^xx_diss={diss_strength}_J={J}_L={L}_lmin=(\d+)_lmax=(\d+)_init={init}_mixed$"
    )

    local_marker_data_all = {}
    loaded_times_all = {}
    subsys_all = {}
    sites_lite_all = {}

    for folder in sorted(os.listdir(results_dir)):
        match = pattern.match(folder)
        if match is None:
            continue

        lmin_val = int(match.group(1))
        lmax_val = int(match.group(2))
        folder_path = os.path.join(results_dir, folder)
        local_marker_path = os.path.join(folder_path, "local_marker.pkl")
        times_lite_path = os.path.join(folder_path, "times.pkl")

        if not os.path.isfile(local_marker_path) or not os.path.isfile(times_lite_path):
            continue

        _, _, local_marker_data, loaded_times = load_all_data(
            h5_path=corr_path,
            local_marker_path=local_marker_path,
            times_lite_path=times_lite_path,
        )

        key = (lmin_val, lmax_val)
        local_marker_data_all[key] = local_marker_data
        loaded_times_all[key] = loaded_times
        subsys_all[key] = subsys
        sites_lite_all[key] = sites_lite

    with tables.open_file(corr_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())

    ave_marker_per_t_corr = calc_corr_evo_local_markers(opdm_t_corr, sites_corr)

    return plot_local_marker_comparison_lmin_lmax(
        local_marker_data=local_marker_data_all,
        loaded_times=loaded_times_all,
        ave_marker_per_t_corr=ave_marker_per_t_corr,
        times_corr=times_corr,
        subsys=subsys_all,
        sites_lite=sites_lite_all,
        corr_label=corr_label,
        save_path=save_path,
        use_lines=use_lines,
        truncate_to_common_times=truncate_to_common_times,
        t_min=t_min,
        t_max=t_max,
    )
