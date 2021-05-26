# Imported libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

#----------------------------------------------------------------------------%
# CO2 CONSTRAINTS
# Load data - CO2 constraint
# Folder name of data files
directory = "Data\\"
# Name of file (must be in correct folder location)
filename = ["postnetwork-elec_only_0.125_0.6.h5",
            "postnetwork-elec_only_0.125_0.5.h5",
            "postnetwork-elec_only_0.125_0.4.h5",
            "postnetwork-elec_only_0.125_0.3.h5",
            "postnetwork-elec_only_0.125_0.2.h5",
            "postnetwork-elec_only_0.125_0.1.h5",
            "postnetwork-elec_only_0.125_0.05.h5"]

# Network
network = pypsa.Network(directory+filename[-1])

# Get the names of the data
data_names = network.loads_t.p.columns

#%% Define index and columns

# Index
time_index = network.loads_t.p_set.index 

# Columns
country_column = network.loads_t.p_set.columns

#%% Calculating mismatch

# Defining dispatched electricity generation
generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()

# Defining load 
load = network.loads_t.p_set

# Calculate mismatch
mismatch = generation - load # Using available electricity generation

#%% Storage

# Efficiency for different storages
# Can be seen at (network.links.efficiency)
eff_gas = 1 # 0.39
eff_H2_charge = 0.8
eff_H2_discharge = 1 # 0.58
eff_battery_charge = 0.9
eff_battery_discharge = 1 # 0.9

# Store of gas/H2/battery for each country at each time step
store_all = network.stores_t.p
store_gas = pd.DataFrame((np.array(network.stores_t.p)[:,0:30] * (1/eff_gas)), columns=(data_names))
store_H2 = pd.DataFrame((np.array(network.stores_t.p)[:,30:60]), columns=(data_names))
store_battery = pd.DataFrame((np.array(network.stores_t.p)[:,60:90]), columns=(data_names))

# Multiply charge/discharge values with correct efficiencies
for name in data_names:
    
    # H2 store
    for i in np.arange(0,8760):
        val = store_H2[name][i]
        if (val > 0):
            store_H2[name][i] *= 1/(eff_H2_discharge)
            
        elif (val < 0):
            store_H2[name][i] *= 1/(eff_H2_charge)
    
    # Battery store
    for j in np.arange(0,8760):
        val = store_battery[name][j]
        if (val > 0):
            store_battery[name][j] *= 1/(eff_battery_discharge)
            
        elif (val < 0):
            store_battery[name][j] *= 1/(eff_battery_charge)

#%% Import/export (links)

# Country links (p0/p1 = active power at bus0/bus1)
# Values of p0 links:   network.links_t.p0
# Values of p0 links:   network.links_t.p1
# Names of all link:    network.links_t.p0.columns
country_link_names = np.array(network.links_t.p0.columns)[150:]

# Link p0 is the same as negative (-) p1
# Turn into array
country_links_p0 = np.array(network.links_t.p0)[:,150:]

# Turn into dataframe
country_links_p0 = pd.DataFrame(country_links_p0, columns=(country_link_names))

# Turn into array
country_links_p1 = np.array(network.links_t.p1)[:,150:]

# Turn into dataframe
country_links_p1 = pd.DataFrame(country_links_p1, columns=(country_link_names))

# Sum all bus0 exports values
sum_bus0 = country_links_p0.groupby(network.links.bus0,axis=1).sum()

# Sum all bus1 imports values and minus
sum_bus1 = (country_links_p1.groupby(network.links.bus1,axis=1).sum())

# Creating empty matrix
country_links = np.zeros((8760,30))

# loop that adds bus0 and bus1 together (remember bus1 is minus value)
for i in range(len(data_names)):   
    
    if data_names[i] in sum_bus0.columns: # Checks if the country is in bus0
        country_links[:,i] += sum_bus0[data_names[i]] # Adds the value for the country to the collected link function
        
    if data_names[i] in sum_bus1.columns: # Checks if the country is in bus1
        country_links[:,i] += sum_bus1[data_names[i]] # Adds the value for the country to the collected link function

# Define the data as a pandas dataframe with country names and timestampes
country_links = pd.DataFrame(data=country_links, index=network.loads_t.p.index, columns=data_names)

#%% Principal Component Analysis

# Defining data
X = mismatch

# Mean of data
X_mean = np.mean(X,axis=0) # axis=0, mean at each colume 
X_mean = np.array(X_mean.values).reshape(30,1) # Define as an array

# Calculate centered data
X_cent = np.subtract(X,X_mean.T)

# Calculate normalization constant
c = 1/np.sqrt(np.sum(np.mean(((X_cent.values)**2),axis=0)))

# Normalize the centered data
B = c*(X_cent.values)

# Convariance of normalized and centered data
C_new = np.dot(B.T,B)*1/(8760-1)
C = np.cov(B.T,bias=True) 

# Calculate eigen values and eigen vectors 
assert np.size(C) <= 900, "C is too big" # Checks convariance size, if to large then python will be stuck on the eigen problem
eig_val, eig_vec = np.linalg.eig(C) # using numpy's function as it scales eigen values

# Calculate amplitude
T = np.dot(B,eig_vec)

#%% Centering responce data

mismatch_check = store_gas.values + store_battery.values + store_H2.values + country_links.values

# Mean values
gas_mean = np.mean(store_gas,axis=0)
battery_mean = np.mean(store_battery,axis=0)
H2_mean = np.mean(store_H2,axis=0)
inex_mean = np.mean(country_links,axis=0)

# Centering data
gas_cent = np.subtract(store_gas,gas_mean.T)
battery_cent = np.subtract(store_battery,battery_mean.T)
H2_cent = np.subtract(store_H2,H2_mean.T)
inex_cent = np.subtract(country_links,inex_mean.T)

check = gas_cent.values + battery_cent.values + H2_cent.values + inex_cent.values
check = pd.DataFrame(data=check, index=network.loads_t.p.index, columns=data_names)


#%% eigen values contribution

# Component contribution
# Backup generator gas
gas_con = np.dot(store_gas,eig_vec)
# Battery
battery_con = np.dot(store_battery,eig_vec)
# hydrogen
H2_con = np.dot(store_H2,eig_vec)
# inport/export
inex_con = np.dot(country_links,eig_vec)
# Sum (with -L) of this is equal to T

# Eigenvalues contribution

test = c*(gas_con+battery_con+H2_con+inex_con)




