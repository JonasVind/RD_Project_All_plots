import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
from elec_only_functions import BAR, PCA

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
    network = pypsa.Network(directory+filename_CO2[i])
    
    # Get the names of the data
    data_names = network.loads_t.p.columns
    
    # Get time stamps
    time_index = network.loads_t.p.index
    
    # List of prices
    prices = network.buses_t.marginal_price
    
    # List of nodal prices for each country
    country_price = prices[data_names] # [€/MWh]
    #country_price_gas = prices[(data_names + ' gas')] # [€/MWh]
    #country_price_H2 = prices[(data_names + ' H2')] # [€/MWh]
    #country_price_battery = prices[(data_names + ' battery')] # [€/MWh]
    
    # Sum up all the prices into one for every country
    #nodal_price = country_price.values + country_price_gas.values + country_price_H2.values + country_price_battery.values
    nodal_price = country_price.values
    nodal_price = pd.DataFrame(data=nodal_price, index=time_index, columns=data_names)
    
    # Restricts the values to 0 to 1000
    nodal_price = np.clip(nodal_price, 0, 1000)
    
    # PCA analysis of mismatch
    variance_explained = PCA(nodal_price)[2]

    PCA_bar_CO2.append(variance_explained)
       


#%% Bar plot - CO2 constraint
matrix_CO2 = PCA_bar_CO2
PC_max_CO2 = 12
constraints_CO2 = ['40%', '50%', '60%', '70%', '80%', '90%', '95%']
title_CO2 = 'Variance for each PC as a function of $CO_{2}$ constraint with constant transmission size (2x current)'
xlabel_CO2 = '$CO_{2}$ Constraint'
suptitle_CO2 = 'Electricity Nodal Prices'

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
    
    # Get the names of the data
    data_names = network.loads_t.p.columns
    
    # Get time stamps
    time_index = network.loads_t.p.index
    
    # List of prices
    prices = network.buses_t.marginal_price
    
    # List of nodal prices for each country
    country_price = prices[data_names] # [€/MWh]
    #country_price_gas = prices[(data_names + ' gas')] # [€/MWh]
    #country_price_H2 = prices[(data_names + ' H2')] # [€/MWh]
    #country_price_battery = prices[(data_names + ' battery')] # [€/MWh]
    
    # Sum up all the prices into one for every country
    #nodal_price = country_price.values + country_price_gas.values + country_price_H2.values + country_price_battery.values
    nodal_price = country_price.values
    nodal_price = pd.DataFrame(data=nodal_price, index=time_index, columns=data_names)
    
    # Restricts the values to 0 to 1000
    nodal_price = np.clip(nodal_price, 0, 1000)
    
    # PCA analysis of mismatch
    variance_explained = PCA(nodal_price)[2]

    PCA_bar_transmission.append(variance_explained)


#%% Bar plot - Transmission constraint
matrix_trans = PCA_bar_transmission
PC_max_trans = 12
constraints_trans = ['Zero', 'Current', '2x Current', '4x Current', '6x Current']
title_trans = 'Variance for each PC as a function of transmission size with constant $CO_{2}$ constraint (95%))'
xlabel_trans = 'Transmission size'
suptitle_trans = 'Electricity Nodal Prices'

BAR(matrix_trans, PC_max_trans, filename_trans, constraints_trans, title_trans, xlabel_trans, suptitle_trans)