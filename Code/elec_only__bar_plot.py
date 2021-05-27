import numpy as np
import pypsa
from elec_only_functions import BAR, PCA
#import elec_only_functions as EOF

# Load data - CO2 constraint
# Folder name of data files
directory = "C:/Users/jense/.spyder-py3/data_files/elec_only/"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
            "postnetwork-elec_only_0.125_0.5.h5",
            "postnetwork-elec_only_0.125_0.4.h5",
            "postnetwork-elec_only_0.125_0.3.h5",
            "postnetwork-elec_only_0.125_0.2.h5",
            "postnetwork-elec_only_0.125_0.1.h5",
            "postnetwork-elec_only_0.125_0.05.h5"]

# Variable for principal components for plotting later
PCA_bar_CO2 = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename_CO2)):
    
    # Load network
    network = pypsa.Network(directory+filename_CO2[i])
    
    # Array of 30 points load
    load = network.loads_t.p_set
        
    # Array of 30 point summed generation
    generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
        
    # Array of mismatch for all 30 points
    mismatch = generation - load
    
    # PCA analysis of mismatch
    variance_explained = PCA(mismatch)[2]

    # Add variance_explained to PCA_bar_CO2
    PCA_bar_CO2.append(variance_explained)

#%% Bar plot - CO2 constraint
matrix_CO2 = PCA_bar_CO2
PC_max_CO2 = 10
constraints_CO2 = ['40%', '50%', '60%', '70%', '80%', '90%', '95%']
title_CO2 = 'Variance for each PC as a function of $CO_{2}$ constraint with constant transmission size (2x current)'
xlabel_CO2 = '$CO_{2}$ Constraint'
suptitle_CO2 = 'Mismatch'

BAR(matrix_CO2, PC_max_CO2, filename_CO2, constraints_CO2, title_CO2, xlabel_CO2, suptitle_CO2)


#%% Load data - Transmission constraint
# Folder name of data files
directory = "C:/Users/jense/.spyder-py3/data_files/elec_only/"

# Name of file (starts from zero to 6-times-curren-size)
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
            "postnetwork-elec_only_0.0625_0.05.h5",
            "postnetwork-elec_only_0.125_0.05.h5",
            "postnetwork-elec_only_0.25_0.05.h5",
            "postnetwork-elec_only_0.375_0.05.h5"]

# Variable for principal components for plotting later
PCA_bar_transmission = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename_trans)):
    network = pypsa.Network(directory+filename_trans[i])
    
    # Array of 30 points load
    load = network.loads_t.p_set
        
    # Array of 30 point summed generation
    generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
        
    # Array of mismatch for all 30 points
    mismatch = generation - load
    
    # PCA analysis of mismatch
    variance_explained = PCA(mismatch)[2]

    PCA_bar_transmission.append(variance_explained)

#%% Bar plot - Transmission constraint
matrix_trans = PCA_bar_transmission
PC_max_trans = 10
constraints_trans = ['Zero', 'Current', '2x Current', '4x Current', '6x Current']
title_trans = 'Variance for each PC as a function of transmission size with constant $CO_{2}$ constraint (95%))'
xlabel_trans = 'Transmission size'
suptitle_trans = 'Mismatch'

BAR(matrix_trans, PC_max_trans, filename_trans, constraints_trans, title_trans, xlabel_trans, suptitle_trans)
