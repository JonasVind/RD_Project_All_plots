# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 08:48:54 2021

@author: jones
"""

#%% Libraries

import numpy as np 
#import h5py 
import pypsa
import pandas as pd
import tables
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from sklearn.preprocessing import Normalizer
from RD_func import BAR, PCA, season_plot, FFT_plot


#%% Data Import

# Filename:
#filename = "Data\postnetwork-elec_only_0.125_0.6.h5" # 40% CO2
filename = "Data\postnetwork-elec_only_0.125_0.05.h5" # 95% CO2 constrain
# Import network
network = pypsa.Network(filename)

print("\nFile loaded: ", filename)

# Get the names of the data
data_names = network.loads_t.p.columns

# Get time stamps
time_index = network.loads_t.p.index

#%% Calculating mismatch

# Defining dispatched electricity generation
generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()

# Defining load 
load = network.loads_t.p_set

# Calculate mismatch
mismatch = generation - load # Using available electricity generation
#mismatch = generation - load # Using dispatched electricity generation

#%% PCA mismatch

eigen_values_mis, eigen_vectors_mis, variance_explained_mis, norm_const_mis, T_mis = PCA(mismatch)

T_mis = pd.DataFrame(data=T_mis,index=time_index)

# Season Plot
season_plot(T_mis, time_index, filename)

#%% PCA Electricity Nodal Price

# List of prices
prices = network.buses_t.marginal_price

# List of nodal prices for each country
country_price = prices[data_names] # [€/MWh]
country_price_gas = prices[(data_names + ' gas')] # [€/MWh]
country_price_H2 = prices[(data_names + ' H2')] # [€/MWh]
country_price_battery = prices[(data_names + ' battery')] # [€/MWh]

# Sum up all the prices into one for every country
nodal_price = country_price.values #+ country_price_gas.values + country_price_H2.values + country_price_battery.values
nodal_price = pd.DataFrame(data=nodal_price, index=time_index, columns=data_names)


# Limit
nodal_price = np.clip(nodal_price, 0, 1000) 

#%% PCA Electricity nodal price

eigen_values_ENP, eigen_vectors_ENP, variance_explained_ENP, norm_const_ENP, T_ENP = PCA(nodal_price)

T_ENP = pd.DataFrame(data=T_ENP,index=time_index)

# Season Plot
season_plot(T_ENP, time_index, filename)


#%% Comparison

C = np.zeros((12,7))
C_without_scale = np.zeros((12,7))

C_test = np.zeros((12,7))
C_test2 = np.zeros((12,7))
C_test3 = np.zeros((12,7))

for n in range(C.shape[0]):
    for k in range(C.shape[1]):
        
        lambda1 = eigen_values_ENP[n]
        lambda2 = eigen_values_mis[k]
        p_k1 = eigen_vectors_ENP[:,n]
        p_k2 = eigen_vectors_mis[:,k]
        
        C[n,k] = np.sqrt(lambda1 * lambda2)*abs(np.dot(p_k1, p_k2))
        
        C_test[n,k] = abs(np.dot(p_k1, p_k2))
        C_test2[n,k] = lambda1 * lambda2
        C_test3[n,k] = np.sqrt(lambda1 * lambda2)
        
        C_without_scale[n,k] = abs(np.dot(p_k1, p_k2))
        
print("ENP has "+str(C.shape[0])+" PC")
print("Mismatch has "+str(C.shape[1])+" PC")

C = np.around(C, 3)
C_without_scale = np.around(C_without_scale, 3)


