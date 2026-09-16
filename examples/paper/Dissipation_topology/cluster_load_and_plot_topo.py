import os
import re
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap, Normalize
from marker_functions import *
import tables
import numpy as np
from cluster_plotting_functions import *

################### Parameters ##################
L = 50
diss_strength = 0.05
J = -1.0
mu = -0.3
min_l = 6
max_l= 7
init = 'triv'


##################################################
# average over a single center site from a range of subsystems in the middle. The correlation matrix averages over the same sites

# what subsystems in LITE to average the local marker over
center_subsys = L  // 2
#global_sites = (center_subsys - 15, center_subsys + 15)

global_sites = (20,30) # (2,2)

# selects the same sites in Corrlation Matrix evo to average the marker over as the selected sites in LITE
sites_corr = global_sites


################### Data paths ##################
script_dir = os.path.dirname(os.path.abspath(__file__))
data_filepath = os.path.join(script_dir, "results/cluster_results",)
checkpoint_folder = os.path.join(data_filepath, f"xx_diss={diss_strength}_J={J}_h={mu}_L={L}_lmin={min_l}_lmax={max_l}_init={init}_mixed")


################### Load data ##################

opdm_t_corr, times_corr, ave_marker_per_t_lite, loaded_times = load_all_data(
    h5_path=os.path.join(data_filepath, f"opdm_t_corr_diss={diss_strength}_J={J:g}_mu={mu}_L={L}_init=mixed_{init}.h5"),
    local_marker_path=os.path.join(checkpoint_folder, "local_marker.pkl"),
    times_lite_path=os.path.join(checkpoint_folder, "times.pkl"),
)

################### Calculate observables ##################

#ave_marker_per_t_corr = calc_corr_evo_local_markers(
#    opdm_t_corr,
#    sites_corr,
#    use_lite_subsystems=True,
#    lite_level=7,
#)


################### Plot ##################

# Plots local marker averages for the exact evolution VS a specific LITE evolution
#plot_local_marker_comparison(
#    ave_marker_per_t_lite,
#    loaded_times,
#    ave_marker_per_t_corr,
#    times_corr,
#    subsys=subsys,
#    sites_lite=sites_lite,
#)

# Plots local marker averages for the exact evolution VS all LITE evolutions with the same parameters except for \ell_min and \ell_max

#plot_local_marker_lmin_lmax_family(
#    results_dir=data_filepath,
#    diss_strength=diss_strength,
#    J=J,
#    L=L,
#    mu=mu,
#    init=init,
#    global_sites=global_sites,
#    use_lite_subsystems_for_OPDM=True,
#    use_lite_subsystems_at_level=7,
#    t_min=30,
#    t_max=50,
#)


#plot_onsite_occupation_lmin_lmax_family(
#    results_dir=data_filepath,
#    diss_strength=diss_strength,
#    J=J,
#    L=L,
#    init=init,
#    global_sites=global_sites,
#    t_min=0,
#    t_max=4,
#)



#################### To check exact results ##############
#subsys = (2, 3.5)
#sites_lite = (2, 3)
#sites_corr = (2, 3)
#ave_marker_per_t_corr = calc_corr_evo_local_markers(opdm_t_corr, sites_corr)

#plot_local_marker_comparison(
#    ave_marker_per_t_lite,
#    loaded_times,
#    ave_marker_per_t_corr,
#    times_corr,
#    subsys=subsys,
#    sites_lite=sites_lite,
#)


#################### To print out local marker values for LITE ##############
#def print_lite_markers_for_timestep(
#    local_marker_data,
#    loaded_times,
#    timestep_idx,
#    only_max_level=True,
#    decimals=8,
#):
#    marker_t = local_marker_data[timestep_idx]
#
#    if not isinstance(marker_t, dict):
#        raise TypeError(
#            f"Expected a dictionary at timestep {timestep_idx}, "
#            f"got {type(marker_t)}"
#        )
#
#    max_level = max(key.level for key in marker_t)
#
#    keys = sorted(
#        marker_t,
#        key=lambda key: (key.level, key.coord),
#    )
#
#    if only_max_level:
#        keys = [key for key in keys if key.level == max_level]
#
#    print()
#    print("=" * 100)
#    print(
#        f"LITE markers at timestep index {timestep_idx}, "
#        f"time = {loaded_times[timestep_idx]:.8f}"
#    )
#    print(f"Maximum level at this timestep: {max_level}")
#    print("=" * 100)
#
#    current_level = None
#
#    for key in keys:
#        marker = np.asarray(marker_t[key], dtype=float)
#
#        if key.level != current_level:
#            current_level = key.level
#
#            print()
#            print(
#                f"LEVEL {current_level} "
#                f"({current_level + 1} sites per subsystem)"
#            )
#            print("-" * 100)
#            print(f"n {'coord':>8}  all local markers")
#            print("-" * 100)
#
#        marker_text = "[" + ", ".join(
#            f"{value:+.{decimals}f}" for value in marker
#        ) + "]"
#
#        print(f"{key.coord:8.1f}  {marker_text}")
#
#    print("=" * 100)
#
#timestep_idx = 300
#
#print_lite_markers_for_timestep(
#    local_marker_data=ave_marker_per_t_lite,
#    loaded_times=loaded_times,
#    timestep_idx=timestep_idx,
#    only_max_level=True,
#    decimals=8,
#)