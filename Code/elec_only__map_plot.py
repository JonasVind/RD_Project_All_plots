import numpy as np
import pandas as pd
import pypsa
from elec_only_functions import PCA, MAP

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


filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]

file = filename_CO2[-1]

# import Network
network = pypsa.Network(directory+file)

# Get the names of the data
data_names = network.loads_t.p.columns

# Get time stamps
time_index = network.loads_t.p.index

# # Array of 30 points load
# load = network.loads_t.p_set
    
# # Array of 30 point summed generation
# generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
    
# # Array of mismatch for all 30 points
# mismatch = generation - load

# # PCA analysis
# eigen_values, eigen_vectors = PCA(mismatch)[0:2]

# # Title for mismatch plot
# title_plot = 'Colormap of PC for mismatch'

# List of prices
prices = network.buses_t.marginal_price

# List of nodal prices for each country
country_price = prices[data_names] # [€/MWh]
###country_price_gas = prices[(data_names + ' gas')] # [€/MWh]
###country_price_H2 = prices[(data_names + ' H2')] # [€/MWh]
###country_price_battery = prices[(data_names + ' battery')] # [€/MWh]

# Sum up all the prices into one for every country
###nodal_price = country_price.values + country_price_gas.values + country_price_H2.values + country_price_battery.values
nodal_price = country_price.values
nodal_price = pd.DataFrame(data=nodal_price, index=time_index, columns=data_names)

# Restricts the values to 0 to 1000
nodal_price = np.clip(nodal_price, 0, 1000)

# PCA analysis
eigen_values, eigen_vectors = PCA(nodal_price)[0:2]

# Title for mismatch plot
title_plot = 'Colormap of PC for Electricity Nodal Prices'



# Define the eigen vectors in a new variable with names
VT = pd.DataFrame(data=eigen_vectors, index=data_names)

filename_plot = file

for i in np.arange(6):
    MAP(eigen_vectors, eigen_values, data_names, (i+1), title_plot, filename_plot)
    
