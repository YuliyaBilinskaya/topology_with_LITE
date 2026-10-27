import pickle
import os
import matplotlib.pyplot as plt
from marker_functions import *
import tables
import numpy as np
import sys
import glob
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, Normalize



def load_all_data(h5_path, local_marker_path, times_lite_path):
    with tables.open_file(h5_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())

    with open(local_marker_path, "rb") as f:
        local_marker_data = pickle.load(f)

    with open(times_lite_path, "rb") as f:
        loaded_times = pickle.load(f)

    return opdm_t_corr, times_corr, local_marker_data, loaded_times

def load_info_data(info_latt_path, times_lite_path):

    with open(info_latt_path, "rb") as f:
        info_latt_data = pickle.load(f)

    with open(times_lite_path, "rb") as f:
        loaded_times = pickle.load(f)

    return info_latt_data, loaded_times


def calc_corr_evo_local_markers(
    opdm_t_corr,
    sites_corr,
    use_lite_subsystems=False,
    lite_level=None,
):
    """
    Calculate correlation-matrix local markers.

    Parameters
    ----------
    opdm_t_corr
        Sequence of full-system BdG correlation matrices with shape
        (2 * L, 2 * L).

    sites_corr
        Inclusive range of global sites, e.g. (22, 28).

    use_lite_subsystems
        False:
            Preserve the current behaviour. Calculate the marker from the
            full-system OPDM and average marker[sites_corr].

        True:
            For every requested global site, extract the centered subsystem
            used by LITE at `lite_level`, calculate the marker from that
            reduced OPDM, and select its central site.

    lite_level
        LITE subsystem level ell. The subsystem contains ell + 1 sites.
        Normally this should be `max_l`.
    """
    if use_lite_subsystems and lite_level is None:
        raise ValueError(
            "lite_level must be provided when use_lite_subsystems=True."
        )

    global_start, global_stop = sites_corr
    requested_sites = range(global_start, global_stop + 1)

    marker_per_t = {}

    for idx, full_opdm in enumerate(opdm_t_corr):
        full_opdm = np.asarray(full_opdm, dtype=np.complex128)

        if (
            full_opdm.ndim != 2
            or full_opdm.shape[0] != full_opdm.shape[1]
            or full_opdm.shape[0] % 2 != 0
        ):
            raise ValueError(
                f"OPDM at time index {idx} must be a square 2L x 2L "
                f"matrix; received shape {full_opdm.shape}."
            )

        system_size = full_opdm.shape[0] // 2

        if not use_lite_subsystems:
            # Current behaviour: calculate the marker in the full system.
            marker = calc_marker_state_from_opdm(
                full_opdm,
                flatten=False,
            )
            marker_per_t[idx] = float(
                np.mean(marker[global_start:global_stop + 1])
            )
            continue

        subsystem_size = lite_level + 1
        local_center = lite_level // 2
        selected_markers = []

        for global_site in requested_sites:
            # This reproduces the mapping used by
            # average_lite_marker_at_global_sites().
            #
            # Even level:
            #   coord = global_site
            #
            # Odd level:
            #   coord = global_site + 0.5
            #
            # In both cases this is the leftmost global subsystem site.
            left_site = global_site - local_center
            stop_site = left_site + subsystem_size

            if left_site < 0 or stop_site > system_size:
                raise ValueError(
                    f"Cannot center a level-{lite_level} subsystem on "
                    f"global site {global_site} in a system of size "
                    f"{system_size}. The requested subsystem would contain "
                    f"sites [{left_site}, {stop_site - 1}]."
                )

            physical_sites = np.arange(left_site, stop_site)

            # The BdG ordering used by calc_marker_state_from_opdm is
            # [particles for all sites, holes for all sites]. Therefore both
            # blocks must be selected; opdm[left_site:stop_site, ...] alone
            # would be incorrect.
            bdg_indices = np.concatenate(
                (
                    physical_sites,
                    system_size + physical_sites,
                )
            )

            subsystem_opdm = full_opdm[np.ix_(bdg_indices, bdg_indices)]

            subsystem_marker = calc_marker_state_from_opdm(
                subsystem_opdm,
                flatten=False,
            )

            selected_markers.append(
                subsystem_marker[local_center]
            )

        marker_per_t[idx] = float(np.mean(selected_markers))

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
    local_marker_data,
    loaded_times,
    ave_marker_per_t_corr,
    times_corr,
    global_sites,
    lite_labels=None,
    corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison_lmin_lmax.pdf",
    use_lines=True,
    truncate_to_common_times=False,
    t_min=None,
    t_max=None,
    t_line=None,
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
        #"#1b5e20", #green
        #"#1565c0", #blue
        #"#ef6c00", #orange
        "#6a1b9a", #purple
        #"#c62828",
        #"#00897b",
        #"#8d6e63",
    ]

    for i, key in enumerate(curve_keys):
        marker_data_i = get_item(local_marker_data, key)
        loaded_times_i = get_item(loaded_times, key)
        global_sites_i = get_item(global_sites, key)

        ave_marker_per_t_lite = {}

        for idx, marker_t in enumerate(marker_data_i):
            if not isinstance(marker_t, dict):
                raise ValueError(
                    f"Expected dict of subsystem markers at time index "
                    f"{idx}, got {type(marker_t)}"
                )

            value = average_lite_marker_at_global_sites(
                marker_t=marker_t,
                global_sites=global_sites_i,
                level=key[0] - 1,  # key=(lmin,lmax), so LTM are calculated from the center of subsystems at level l_min - 1
            )
            if value is None:
                continue
            ave_marker_per_t_lite[idx] = value

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
    if t_line is not None:
        ax1.axvline(t_line, color='black', linestyle='--', linewidth=1.0)
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


def average_lite_marker_at_global_sites(marker_t, global_sites, level):
    """
    Average the LITE marker over exactly the requested global sites.

    For a subsystem with coordinate c and level ell, local index j
    corresponds to global site:

        global_site = c - ell / 2 + j
    """
    global_start, global_stop = global_sites
    requested_sites = range(global_start, global_stop + 1)

    # Use the requested level if given; otherwise the highest stored level
    # (the level can change between timesteps).
    max_level = level if level is not None else max(key.level for key in marker_t)
    max_level_keys = [
        key for key in marker_t
        if key.level == max_level
    ]
    if not max_level_keys:
        return None  # requested level not stored at this timestep


    # Center index for this timestep's actual maximum level.
    local_center = max_level // 2

    # For odd levels, the subsystem coordinate is half-integer.
    coord_shift = 0.0 if max_level % 2 == 0 else 0.5

    marker_values = []

    for global_site in requested_sites:
        required_coord = global_site + coord_shift

        matching_keys = [
            key for key in max_level_keys
            if np.isclose(key.coord, required_coord)
        ]

        if len(matching_keys) != 1:
            raise ValueError(
                f"Cannot find a centered LITE subsystem for global site "
                f"{global_site} at level {max_level}. Expected subsystem "
                f"coordinate {required_coord}; found "
                f"{[key.coord for key in matching_keys]}."
            )

        key = matching_keys[0]
        marker = np.asarray(marker_t[key], dtype=float)

        if local_center >= len(marker):
            raise ValueError(
                f"Local center index {local_center} is outside a marker "
                f"array of length {len(marker)} for subsystem {key}."
            )

        # Verify that the chosen local site really is the requested global site.
        mapped_global_site = (
            key.coord - max_level / 2 + local_center
        )

        if not np.isclose(mapped_global_site, global_site):
            raise ValueError(
                f"Site-mapping error: requested global site {global_site}, "
                f"but subsystem {key} and local index {local_center} map "
                f"to global site {mapped_global_site}."
            )

        marker_values.append(marker[local_center])

    return float(np.mean(marker_values))


def plot_local_marker_lmin_lmax_family(
    results_dir,
    diss_strength,
    J,
    L,
    mu,
    init,
    global_sites,
    use_lite_subsystems_for_OPDM=False,
    use_lite_subsystems_at_level=None,
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_comparison_lmin_lmax.pdf",
    use_lines=True,
    truncate_to_common_times=False,
    t_min=None,
    t_max=None,
    t_line=None,
):
    import os
    import re

    corr_label = rf'Correlation matrix evolution ($\ell_{{\min}}-1={use_lite_subsystems_at_level}$)'

    corr_path = os.path.join(
        results_dir,
        f"opdm_t_corr_diss={diss_strength}_J={J:g}_mu={mu}_L={L}_init=mixed_{init}.h5",
    )

    pattern = re.compile(
        rf"^xx_diss={diss_strength}_J={J}_h={mu}_L={L}_lmin=(\d+)_lmax=(\d+)_init={init}_mixed$"
    )

    local_marker_data_all = {}
    loaded_times_all = {}
    global_sites_all = {}

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
        global_sites_all[key] = global_sites

    with tables.open_file(corr_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())

    ave_marker_per_t_corr = calc_corr_evo_local_markers(opdm_t_corr, global_sites, use_lite_subsystems=use_lite_subsystems_for_OPDM,
    lite_level=use_lite_subsystems_at_level)


    return plot_local_marker_comparison_lmin_lmax(
        local_marker_data=local_marker_data_all,
        loaded_times=loaded_times_all,
        ave_marker_per_t_corr=ave_marker_per_t_corr,
        times_corr=times_corr,
        global_sites=global_sites_all,
        corr_label=corr_label,
        save_path=save_path,
        use_lines=use_lines,
        truncate_to_common_times=truncate_to_common_times,
        t_min=t_min,
        t_max=t_max,
        t_line=t_line,
    )

def _match_common_times(times_a, times_b, atol=1e-6, rtol=1e-6):
    """
    Indices (idx_a, idx_b) of time points present in both arrays, matched with a
    tolerance |t_a - t_b| <= atol + rtol*|t_b| (correlation-matrix times are float32,
    LITE times float64). No interpolation: unmatched times are dropped.
    """
    times_a = np.asarray(times_a, dtype=float)
    times_b = np.asarray(times_b, dtype=float)
    if len(times_a) == 0 or len(times_b) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order_b = np.argsort(times_b)
    tb_sorted = times_b[order_b]
    n = len(tb_sorted)
    pos = np.searchsorted(tb_sorted, times_a)
    left = np.clip(pos - 1, 0, n - 1)
    right = np.clip(pos, 0, n - 1)
    nearest = np.where(
        np.abs(times_a - tb_sorted[left]) <= np.abs(times_a - tb_sorted[right]),
        left, right,
    )
    idx_b_all = order_b[nearest]
    matched = np.abs(times_a - times_b[idx_b_all]) <= atol + rtol * np.abs(times_b[idx_b_all])

    idx_a = np.where(matched)[0]
    return idx_a, idx_b_all[matched]


def plot_local_marker_diff_lmin_lmax(
    local_marker_data,
    loaded_times,
    ave_marker_per_t_corr,
    times_corr,
    global_sites,
    lite_labels=None,
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_difference_lmin_lmax.pdf",
    use_lines=True,
    t_min=None,
    t_max=None,
    t_line=None,
):
    """
    Plots  <marker>_corr(t) - <marker>_LITE(t)  for every (lmin, lmax) run, at the
    common time points only (no interpolation). The LITE marker is taken from
    the level lmin - 1 subsystems.
    local_marker_data, loaded_times, global_sites: dicts keyed by (lmin, lmax).
    """

    corr_times = np.array(times_corr, dtype=float)

    fig, ax1 = plt.subplots(figsize=(15, 5))
    colors = [
        "#1b5e20",  # green
        "#1565c0",  # blue
        "#ef6c00",  # orange
        "#6a1b9a",  # purple
        "#c62828",
        "#00897b",
        "#8d6e63",
    ]

    for i, key in enumerate(local_marker_data.keys()):
        corr_i = ave_marker_per_t_corr[key] if key in ave_marker_per_t_corr else ave_marker_per_t_corr  # ADD
        corr_values = np.array(list(corr_i.values()), dtype=float)
        marker_data_i = local_marker_data[key]
        loaded_times_i = loaded_times[key]
        global_sites_i = global_sites[key]

        ave_marker_per_t_lite = {}
        for idx, marker_t in enumerate(marker_data_i):
            if not isinstance(marker_t, dict):
                raise ValueError(
                    f"Expected dict of subsystem markers at time index "
                    f"{idx}, got {type(marker_t)}"
                )
            value = average_lite_marker_at_global_sites(
                marker_t=marker_t,
                global_sites=global_sites_i,
                level=key[0] - 1,          # l_min - 1; key = (lmin, lmax)
            )
            if value is None:
                continue                   # level lmin-1 not stored yet
            ave_marker_per_t_lite[idx] = value

        times_lite = np.array(
            [loaded_times_i[idx] for idx in ave_marker_per_t_lite.keys()], dtype=float
        )
        values_lite = np.array(list(ave_marker_per_t_lite.values()), dtype=float)

        # difference at the common time points only
        idx_corr, idx_lite = _match_common_times(corr_times, times_lite)
        t_common = corr_times[idx_corr]
        diff = corr_values[idx_corr] - values_lite[idx_lite]

        mask = np.ones(len(t_common), dtype=bool)
        if t_min is not None:
            mask &= t_common >= t_min
        if t_max is not None:
            mask &= t_common <= t_max
        t_common, diff = t_common[mask], diff[mask]

        color = colors[i % len(colors)]

        if lite_labels is None:
            label = rf'$\ell_{{\min}}={key[0]}, \ell_{{\max}}={key[1]}$'
        elif isinstance(lite_labels, dict):
            label = lite_labels[key]
        else:
            label = lite_labels[i]

        if use_lines:
            ax1.plot(t_common, diff, '-', color=color, linewidth=1.4, label=label)

        #ax1.scatter(
        #    t_common,
        #    diff,
        #    marker='o',
        #    edgecolors=color,
        #    facecolors='none',
        #    linewidths=1.0,
        #    s=25,
        #    label=label,
        #)

    ax1.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    if t_line is not None:
        ax1.axvline(t_line, color='black', linestyle='--', linewidth=1.0)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average local marker: correlation matrix − LITE')
    ax1.legend()
    plt.title('Difference in average local topological marker')
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig)


def plot_local_marker_diff_lmin_lmax_family(
    results_dir,
    diss_strength,
    J,
    L,
    mu,
    init,
    dt,
    global_sites,
    use_lite_subsystems_for_OPDM=False,
    use_lite_subsystems_at_level=None,
    save_path="/Users/yuliyabilinskaya/Desktop/topo_marker_difference_lmin_lmax.pdf",
    use_lines=True,
    t_min=None,
    t_max=None,
    t_line=None,
):
    import os
    import re

    corr_path = os.path.join(
        results_dir,
        f"opdm_t_corr_diss={diss_strength}_J={J:g}_mu={mu}_L={L}_init=mixed_{init}_dt={dt}.h5",
    )

    pattern = re.compile(
        rf"^xx_diss={diss_strength}_J={J}_h={mu}_L={L}_lmin=(\d+)_lmax=(\d+)_init={init}_mixed_dt={dt}$"
    )

    local_marker_data_all = {}
    loaded_times_all = {}
    global_sites_all = {}

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
        global_sites_all[key] = global_sites

    with tables.open_file(corr_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())

    # one correlation-matrix reference per run, at that run's own level lmin - 1
    if use_lite_subsystems_for_OPDM:
        levels = sorted({lmin - 1 for (lmin, lmax) in local_marker_data_all})
        corr_by_level = {
            lvl: calc_corr_evo_local_markers(
                opdm_t_corr, global_sites,
                use_lite_subsystems=True, lite_level=lvl,
            )
            for lvl in levels
        }
        ave_marker_per_t_corr = {key: corr_by_level[key[0] - 1] for key in local_marker_data_all}
    else:
        ref = calc_corr_evo_local_markers(opdm_t_corr, global_sites, use_lite_subsystems=False)
        ave_marker_per_t_corr = {key: ref for key in local_marker_data_all}

    return plot_local_marker_diff_lmin_lmax(
        local_marker_data=local_marker_data_all,
        loaded_times=loaded_times_all,
        ave_marker_per_t_corr=ave_marker_per_t_corr,
        times_corr=times_corr,
        global_sites=global_sites_all,
        save_path=save_path,
        use_lines=use_lines,
        t_min=t_min,
        t_max=t_max,
        t_line=t_line,
    )


def load_lite_onsite_density(folder, L):
    """
    On-site density of one LITE run as an array (n_times, L) (column = global site),
    together with the LITE times.

    1. If the observable `on_site_density.pkl` was saved, its level-0 entries are used.
    2. Otherwise it is computed from `density_matrix_fix_l.pkl`: the level-0 (single-site)
       density matrices, with the same definition as the `on_site_density` observable
       (diagonal of the particle block of the OPDM).
    """
    import os

    with open(os.path.join(folder, "times.pkl"), "rb") as f:
        times = np.asarray(pickle.load(f), dtype=float)

    density_path = os.path.join(folder, "on_site_density.pkl")
    dm_path = os.path.join(folder, "density_matrix_fix_l.pkl")

    density = np.full((len(times), L), np.nan)

    if os.path.isfile(density_path):
        with open(density_path, "rb") as f:
            data = pickle.load(f)
        for t_idx, snap in enumerate(data):
            for key, values in snap.items():
                if key.level == 0:
                    density[t_idx, int(round(key.coord))] = float(np.asarray(values)[0])
        source = "on_site_density.pkl"
    elif os.path.isfile(dm_path):
        with open(dm_path, "rb") as f:
            data = pickle.load(f)
        for t_idx, snap in enumerate(data):
            for key, rho in snap.items():
                if key.level == 0:                       # single-site reduced density matrix
                    opdm = calc_opdm_from_rho(rho)
                    density[t_idx, int(round(key.coord))] = float(np.real(opdm[0, 0]))
        source = "density_matrix_fix_l.pkl"
    else:
        raise FileNotFoundError(
            f"Neither on_site_density.pkl nor density_matrix_fix_l.pkl found in {folder}"
        )

    if len(data) != len(times):
        raise ValueError(f"{folder}: {len(data)} snapshots but {len(times)} times.")
    print(f"    on-site density from {source}")
    return times, density


def plot_onsite_density_diff_lmin_lmax(
    onsite_density_data,
    loaded_times,
    corr_density,
    times_corr,
    sites=None,
    average_sites=False,
    save_path="/Users/yuliyabilinskaya/Desktop/topo_onsite_density_difference_lmin_lmax.pdf",
    t_min=None,
    t_max=None,
    t_line=None,
    figsize=(8, 4),
    linewidth=0.8,
    ylim=None,
    quantity="diff",  # "diff" = n_corr - n_LITE, "lite" = n_LITE only
):
    """
    Plots  n_corr(t, x) - n_LITE(t, x)  for every (lmin, lmax) run, at the common time
    points only (no interpolation). Colour -> (lmin, lmax); one line per site.

    onsite_density_data : dict {(lmin, lmax): array (n_times_LITE, L)}
    loaded_times        : dict {(lmin, lmax): LITE times}
    corr_density        : array (n_times_corr, L), full-system on-site density
    sites               : None (all sites) | int (one site) | (first, last) inclusive range
    average_sites       : if True, average the difference over the selected sites (one line per run)
    """

    if quantity not in ("diff", "lite"):
        raise ValueError("quantity must be 'diff' or 'lite'")

    corr_density = np.asarray(corr_density, dtype=float)
    corr_times = np.array(times_corr, dtype=float)
    L = corr_density.shape[1]

    if sites is None:
        site_list = list(range(L))
    elif np.isscalar(sites):
        site_list = [int(sites)]
    else:
        site_list = list(range(int(sites[0]), int(sites[1]) + 1))
    if min(site_list) < 0 or max(site_list) >= L:
        raise ValueError(f"sites must lie in [0, {L - 1}], got {sites}.")

    fig, ax1 = plt.subplots(figsize=figsize)
    colors = [
        "#1b5e20",  # green
        "#1565c0",  # blue
        "#ef6c00",  # orange
        "#6a1b9a",  # purple
        "#c62828",
        "#00897b",
        "#8d6e63",
    ]

    for i, key in enumerate(onsite_density_data.keys()):
        density_lite = np.asarray(onsite_density_data[key], dtype=float)
        times_lite = np.asarray(loaded_times[key], dtype=float)

        idx_corr, idx_lite = _match_common_times(corr_times, times_lite)
        t_common = corr_times[idx_corr]
        lite_part = density_lite[np.ix_(idx_lite, site_list)]
        diff = (corr_density[np.ix_(idx_corr, site_list)] - lite_part
                if quantity == "diff" else lite_part)

        mask = np.ones(len(t_common), dtype=bool)
        if t_min is not None:
            mask &= t_common >= t_min
        if t_max is not None:
            mask &= t_common <= t_max
        t_common, diff = t_common[mask], diff[mask]

        if average_sites:
            diff = diff.mean(axis=1, keepdims=True)

        color = colors[i % len(colors)]
        label = rf'$\ell_{{\min}}={key[0]}, \ell_{{\max}}={key[1]}$'
        for s in range(diff.shape[1]):
            ax1.plot(t_common, diff[:, s], '-', color=color, linewidth=linewidth,
                     alpha=1.0 if diff.shape[1] == 1 else 0.6,
                     label=label if s == 0 else None)       # one legend entry per run

    if quantity == "diff":
        ax1.axhline(0, color="black", linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    if t_line is not None:
        ax1.axvline(t_line, color='black', linestyle='--', linewidth=1.0)
    if ylim is not None:
        ax1.set_ylim(ylim)

    if average_sites:
        site_txt = f"mean over sites {site_list[0]}-{site_list[-1]}"
    elif len(site_list) == 1:
        site_txt = f"site {site_list[0]}"
    else:
        site_txt = f"sites {site_list[0]}-{site_list[-1]}, one line per site"
    ax1.set_xlabel('Time')
    ax1.set_ylabel('On-site density: correlation matrix − LITE' if quantity == "diff"
                   else 'On-site density: LITE')
    ax1.legend()
    plt.title(f"{'Difference in on-site density' if quantity == 'diff' else 'On-site density (LITE)'} ({site_txt})")
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()
    plt.close(fig)


def _load_onsite_family(results_dir, diss_strength, J, L, mu, init, dt=None):
    """Load every (lmin, lmax) LITE run and the correlation-matrix on-site density."""
    import os
    import re

    dt_suffix = f"_dt={dt}" if dt is not None else ""
    corr_path = os.path.join(
        results_dir,
        f"opdm_t_corr_diss={diss_strength}_J={J:g}_mu={mu}_L={L}_init=mixed_{init}{dt_suffix}.h5",
    )
    pattern = re.compile(
        rf"^xx_diss={diss_strength}_J={J}_h={mu}_L={L}_lmin=(\d+)_lmax=(\d+)_init={init}_mixed{re.escape(dt_suffix)}$"
    )

    onsite_density_all, loaded_times_all = {}, {}
    for folder in sorted(os.listdir(results_dir)):
        match = pattern.match(folder)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        print(f"(lmin={key[0]}, lmax={key[1]}):")
        times_lite, density_lite = load_lite_onsite_density(os.path.join(results_dir, folder), L)
        loaded_times_all[key] = times_lite
        onsite_density_all[key] = density_lite

    if not onsite_density_all:
        raise FileNotFoundError(f"No LITE folders matching {pattern.pattern} in {results_dir}")

    with tables.open_file(corr_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())
    corr_density = np.real(np.einsum("tii->ti", opdm_t_corr[:, :L, :L]))
    del opdm_t_corr
    return onsite_density_all, loaded_times_all, corr_density, times_corr


def plot_onsite_density_3D_diff_lmin_lmax_family(
    results_dir,
    diss_strength,
    J,
    L,
    mu,
    init,
    dt=None,
    sites=None,
    t_min=None,
    t_max=None,
    elev=50,
    azim=-45,
    shared_zlim=True,
    quantity="diff",
    max_time_points=400,
    linewidth=0.8,
    alpha=0.5,
    save_path=None,

):
    """
    Plot Delta n(t, x) = n_corr(t, x) - n_LITE(t, x) as 3D lines, in the same style as
    plot_info_per_scale_diff_3d: one thin black line per time slice, running across the
    sites (x axis) at fixed time (y axis), with the difference on the z axis.
    One figure per (lmin, lmax) run.

    sites           : None (all sites) or (first, last) inclusive range
    shared_zlim     : same symmetric z-range in all figures, so the runs can be compared
    max_time_points : the time axis is thinned to about this many slices
    save_path       : if given, "<name>_lmin=<a>_lmax=<b><ext>" is written for every run
    """

    onsite_density_all, loaded_times_all, corr_density, times_corr = _load_onsite_family(
        results_dir, diss_strength, J, L, mu, init, dt
    )
    corr_times = np.asarray(times_corr, dtype=float)

    if sites is None:
        site_list = list(range(L))
    elif np.isscalar(sites):
        raise ValueError("sites must be None or a (first, last) range for the 3D plot.")
    else:
        site_list = list(range(int(sites[0]), int(sites[1]) + 1))
    if min(site_list) < 0 or max(site_list) >= L:
        raise ValueError(f"sites must lie in [0, {L - 1}], got {sites}.")
    sites_arr = np.array(site_list, dtype=float)

    # ---- difference at the common times, for every run ----
    panels = {}
    for key, density_lite in onsite_density_all.items():
        idx_corr, idx_lite = _match_common_times(corr_times, loaded_times_all[key])
        t_common = corr_times[idx_corr]
        corr_part = corr_density[np.ix_(idx_corr, site_list)]
        lite_part = np.asarray(density_lite)[np.ix_(idx_lite, site_list)]
        data = {"diff": corr_part - lite_part,
                "corr": corr_part,
                "lite": lite_part}[quantity]

        mask = np.ones(len(t_common), dtype=bool)
        if t_min is not None:
            mask &= t_common >= t_min
        if t_max is not None:
            mask &= t_common <= t_max
        t_common, data = t_common[mask], data[mask]

        stride = max(1, len(t_common) // max_time_points)
        panels[key] = (t_common[::stride], data[::stride])

    allv = np.concatenate([d[np.isfinite(d)].ravel() for _, d in panels.values()])
    if quantity == "diff":
        zlim = (-np.abs(allv).max(), np.abs(allv).max())
    else:
        zlim = (allv.min(), allv.max())

    label = {"diff": r"$n_{\rm corr}-n_{\rm LITE}$",
             "corr": r"$n_{\rm corr}$",
             "lite": r"$n_{\rm LITE}$"}[quantity]

    for key, (t_plot, diff) in panels.items():
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(111, projection="3d")

        # one line per site, running along time
        for j, site in enumerate(site_list):
            z = diff[:, j]
            finite = np.isfinite(z)
            if finite.sum() == 0:
                continue
            ax.plot(np.full(finite.sum(), float(site)), t_plot[finite], z[finite],
                    color="black", linewidth=linewidth, alpha=alpha)

        ax.set_xlabel("Site", fontsize=14)
        ax.set_ylabel("Time", fontsize=14)
        ax.set_zlabel(label, fontsize=14)
        ax.set_xlim(site_list[0], site_list[-1])  # NEW
        ax.set_ylim(t_plot.min(), t_plot.max())
        if shared_zlim:
            ax.set_zlim(*zlim)
        ax.set_title(rf"$\ell_{{\min}}={key[0]},\ \ell_{{\max}}={key[1]}$", fontsize=12)
        ax.view_init(elev=elev, azim=azim)

        # ---- GIF with one frame per (lmin, lmax) run ----
        import io
        from PIL import Image
        gif_path = "/Users/yuliyabilinskaya/Desktop/onsite_diff.gif"   # output file
        sec_per_frame = 1.0
        if key == list(panels.keys())[0]:
            frames = []
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        if key == list(panels.keys())[-1]:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=int(1000 * sec_per_frame), loop=0)
            print(f"Saved {gif_path}")

        if save_path is not None:
            import os
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_lmin={key[0]}_lmax={key[1]}{ext}", dpi=200, bbox_inches="tight")

        plt.show()
        plt.close(fig)

def plot_onsite_density_2D_diff_lmin_lmax_family(
    results_dir, diss_strength, J, L, mu, init, dt=None, sites=None,
    t_min=None, t_max=None, shared_clim=True, cmap="RdBu_r",
    max_time_points=400, figsize=(8, 6), save_path=None,
):
    """
    Heatmap (site on x, time on y) of n_corr - n_LITE for every (lmin, lmax) run.
    One figure per run; colour scale is symmetric around zero.
    """
    onsite_density_all, loaded_times_all, corr_density, times_corr = _load_onsite_family(
        results_dir, diss_strength, J, L, mu, init, dt)
    corr_times = np.asarray(times_corr, dtype=float)

    # --- site selection: None -> all, (a, b) -> inclusive range
    if sites is None:
        site_list = list(range(L))
    elif isinstance(sites, (tuple, list)) and len(sites) == 2:
        site_list = list(range(int(sites[0]), int(sites[1]) + 1))
    else:
        raise ValueError("sites must be None or a (first, last) range for a heatmap.")

    # --- common-time matching, identical to the 3D function
    panels = {}
    for key, density_lite in onsite_density_all.items():
        t_lite = np.asarray(loaded_times_all[key], dtype=float)
        idx_lite, idx_corr = _match_common_times(t_lite, corr_times)
        if len(idx_lite) == 0:
            continue
        t_common = t_lite[idx_lite]
        if t_min is not None or t_max is not None:
            mask = np.ones_like(t_common, dtype=bool)
            if t_min is not None: mask &= t_common >= t_min
            if t_max is not None: mask &= t_common <= t_max
            idx_lite, idx_corr, t_common = idx_lite[mask], idx_corr[mask], t_common[mask]
        if len(t_common) == 0:
            continue
        stride = max(1, int(np.ceil(len(t_common) / max_time_points)))
        diff = (corr_density[np.ix_(idx_corr, site_list)]
                - np.asarray(density_lite)[np.ix_(idx_lite, site_list)])
        panels[key] = (t_common[::stride], diff[::stride])

    if not panels:
        print("No common times found.")
        return

    global_max = max(np.nanmax(np.abs(d)) for _, d in panels.values())
    if global_max == 0:
        global_max = 1e-16

    for key, (t_plot, diff) in panels.items():
        vmax = global_max if shared_clim else np.nanmax(np.abs(diff))
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        # pcolormesh with cell edges so non-uniform time grids are drawn correctly
        x_edges = np.arange(site_list[0] - 0.5, site_list[-1] + 1.5, 1.0)
        if len(t_plot) > 1:
            mid = 0.5 * (t_plot[1:] + t_plot[:-1])
            y_edges = np.concatenate(([t_plot[0] - (mid[0] - t_plot[0])], mid,
                                      [t_plot[-1] + (t_plot[-1] - mid[-1])]))
        else:
            y_edges = np.array([t_plot[0] - 0.5, t_plot[0] + 0.5])
        mesh = ax.pcolormesh(x_edges, y_edges, diff, cmap=cmap,
                             vmin=-vmax, vmax=vmax, shading="flat", rasterized=True)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(r"$n_{\rm corr}-n_{\rm LITE}$", fontsize=14)
        ax.set_xlabel("Site", fontsize=14)
        ax.set_ylabel("Time", fontsize=14)
        ax.set_title(rf"$\ell_{{\min}}={key[0]},\ \ell_{{\max}}={key[1]}$", fontsize=12)
        fig.tight_layout()

        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_lmin={key[0]}_lmax={key[1]}{ext or '.png'}",
                        bbox_inches="tight")
        plt.show()
        plt.close(fig)

def plot_onsite_density_diff_lmin_lmax_family(
    results_dir,
    diss_strength,
    J,
    L,
    mu,
    init,
    dt=None,
    sites=None,
    average_sites=False,
    save_path="/Users/yuliyabilinskaya/Desktop/topo_onsite_density_difference_lmin_lmax.pdf",
    t_min=None,
    t_max=None,
    t_line=None,
    figsize=(8, 4),
    linewidth=0.8,
    ylim=None,
    quantity='diff',
):
    import os
    import re

    dt_suffix = f"_dt={dt}" if dt is not None else ""

    corr_path = os.path.join(
        results_dir,
        f"opdm_t_corr_diss={diss_strength}_J={J:g}_mu={mu}_L={L}_init=mixed_{init}{dt_suffix}.h5",
    )
    pattern = re.compile(
        rf"^xx_diss={diss_strength}_J={J}_h={mu}_L={L}_lmin=(\d+)_lmax=(\d+)_init={init}_mixed{re.escape(dt_suffix)}$"
    )

    onsite_density_all = {}
    loaded_times_all = {}

    for folder in sorted(os.listdir(results_dir)):
        match = pattern.match(folder)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        print(f"(lmin={key[0]}, lmax={key[1]}):")
        times_lite, density_lite = load_lite_onsite_density(os.path.join(results_dir, folder), L)
        loaded_times_all[key] = times_lite
        onsite_density_all[key] = density_lite

    if not onsite_density_all:
        raise FileNotFoundError(f"No LITE folders matching {pattern.pattern} in {results_dir}")

    with tables.open_file(corr_path, mode="r") as h5:
        opdm_t_corr = np.array(h5.root.opdm_t_corr.read())
        times_corr = np.array(h5.root.times_corr.read())
    # full-system on-site density: diagonal of the particle block of the 2L x 2L correlation matrix
    corr_density = np.real(np.einsum("tii->ti", opdm_t_corr[:, :L, :L]))
    del opdm_t_corr

    return plot_onsite_density_diff_lmin_lmax(
        onsite_density_data=onsite_density_all,
        loaded_times=loaded_times_all,
        corr_density=corr_density,
        times_corr=times_corr,
        sites=sites,
        average_sites=average_sites,
        save_path=save_path,
        t_min=t_min,
        t_max=t_max,
        t_line=t_line,
        figsize=figsize,
        linewidth=linewidth,
        ylim=ylim,
        quantity=quantity,
    )



def plot_info_latt_from_state(results_dir, diss_strength, time=None, InfoLattice_Norm=2):
    """
    Load a saved `state.pkl`, reconstruct the information lattice from the saved
    density-matrix checkpoint, and plot it.

    Notes
    -----
    - `state.pkl` stores one checkpointed state, not the whole time evolution.
    - Therefore this function can only plot the saved checkpoint time.
    - If `time` is given and does not match the checkpoint time, an error is raised.

    Example
    -------
    plot_info_latt_from_state(
        results_dir=data_filepath,
        diss_strength=diss_strength,
        time=0.5,
        InfoLattice_Norm=2,
    )
    """

    results_dir = Path(results_dir).expanduser().resolve()

    project_root = results_dir.parents[4]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import local_information as li

    run_dirs = sorted(glob.glob(str(results_dir / f"xx_diss={diss_strength}_*")))
    if not run_dirs:
        raise FileNotFoundError(
            f"No run folder found under {results_dir} for diss_strength={diss_strength}."
        )

    checkpoint_folder = Path(run_dirs[0])

    state_path = checkpoint_folder / "state.pkl"
    meta_path = checkpoint_folder / "state_meta_data.yaml"
    times_path = checkpoint_folder / "times.pkl"

    if not state_path.is_file():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing state metadata file: {meta_path}")

    state = li.State.from_checkpoint(str(checkpoint_folder))

    checkpoint_time = float(state.current_time)

    if times_path.is_file():
        with open(times_path, "rb") as f:
            saved_times = np.asarray(pickle.load(f), dtype=float)
        if saved_times.size > 0:
            checkpoint_time = float(saved_times[-1])

    if time is not None and not np.isclose(time, checkpoint_time, atol=1e-10):
        raise ValueError(
            f"`state.pkl` contains only one checkpointed state at t={checkpoint_time}, "
            f"but you requested time={time}."
        )

    density_matrix_all_levels = state.get_all_levels()
    info_lattice = state.get_information_lattice(density_matrix_all_levels)

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
    plt.title(f"Info lattice from state checkpoint at t={checkpoint_time:.6g}")
    ax.set_aspect("equal")
    plt.show()

    print(f"Loaded checkpoint: {checkpoint_folder}")
    print(f"Plotted checkpoint time: {checkpoint_time}")



def plot_onsite_occupation_lmin_lmax_family(
    results_dir,
    diss_strength,
    J,
    L,
    init,
    global_sites,
    corr_label='Correlation matrix evolution',
    save_path="/Users/yuliyabilinskaya/Desktop/topo_onsite_occupation_comparison_lmin_lmax.pdf",
    use_lines=True,
    truncate_to_common_times=False,
    t_min=None,
    t_max=None,
):
    import os
    import re

    def lite_selection_from_global_sites(global_sites, lmax_val):
        if lmax_val % 2 == 0:
            center_site = lmax_val // 2
            subsys = global_sites
            sites_lite = (center_site, center_site)
        else:
            center_site = lmax_val // 2
            subsys = (global_sites[0] + 0.5, global_sites[1] + 0.5)
            sites_lite = (center_site, center_site)

        return subsys, sites_lite

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

        subsys_i, sites_lite_i = lite_selection_from_global_sites(global_sites, lmax_val)

        key = (lmin_val, lmax_val)
        on_site_density_data_all[key] = on_site_density_data
        loaded_times_all[key] = loaded_times
        subsys_all[key] = subsys_i
        sites_lite_all[key] = sites_lite_i

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