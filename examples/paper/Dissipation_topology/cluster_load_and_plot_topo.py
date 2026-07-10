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
diss_strength = 0.2
J = -1.0
min_l = 3
max_l= 4
init = 'triv'

##################################################
# CASE 1: fix center system, average over the same three sites as max_l increases

# what subsystems in LITE to average the local marker over
#center_subsys = L  // 2
#subsys = (center_subsys, center_subsys)
#
## what sites in each subsystem in LITE to average the local marker over
#center_site_lite = (max_l + 1) // 2
#sites_lite = (center_site_lite-1, center_site_lite+1)
#
## selects the same sites in Corrlation Matrix evo to average the marker over as the selected sites in LITE
#sites_corr = (subsys[0]-15, subsys[1]+15 ) # "this interval is lagom away from the physical edges of the system


##################################################
# CASE 2: average over a single center site from a range of subsystems in the middle. The correlation matrix averages over the same sites

# what subsystems in LITE to average the local marker over
center_subsys = L  // 2
subsys = (center_subsys - 15, center_subsys + 15)

# what sites in each subsystem in LITE to average the local marker over
center_site_lite = (max_l + 1) // 2
sites_lite = (center_site_lite-1, center_site_lite+1)

# selects the same sites in Corrlation Matrix evo to average the marker over as the selected sites in LITE
sites_corr = (subsys[0], subsys[1] )

print('subsys', subsys, 'sites_lite', sites_lite, 'sites_corr', sites_corr)

################### Data paths ##################
script_dir = os.path.dirname(os.path.abspath(__file__))
data_filepath = os.path.join(script_dir, "results/cluster_results",)
checkpoint_folder = os.path.join(data_filepath, f"xx_diss={diss_strength}_J={J}_L={L}_lmin={min_l}_lmax={max_l}_init={init}_mixed")


################### Load data ##################

opdm_t_corr, times_corr, ave_marker_per_t_lite, loaded_times = load_all_data(
    h5_path=os.path.join(data_filepath, f"opdm_t_corr_diss={diss_strength}_J={J}_L={L}_init=mixed_{init}.h5"),
    local_marker_path=os.path.join(checkpoint_folder, "local_marker.pkl"),
    times_lite_path=os.path.join(checkpoint_folder, "times.pkl"),
)

################### Calculate observables ##################

ave_marker_per_t_corr = calc_corr_evo_local_markers(opdm_t_corr, sites_corr)


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
#    init=init,
#    subsys=subsys,
#    sites_lite=sites_lite,
#    sites_corr=sites_corr,
#    t_min=None,
#    t_max=None,
#)

# Plots on site density averages for the exact evolution VS all LITE evolutions with the same parameters except for \ell_min and \ell_max
plot_onsite_occupation_lmin_lmax_family(
    results_dir=data_filepath,
    diss_strength=diss_strength,
    J=J,
    L=L,
    init=init,
    subsys=subsys,
    sites_lite=sites_lite,
    sites_corr=sites_corr,
    t_min=None,
    t_max=4,
)


