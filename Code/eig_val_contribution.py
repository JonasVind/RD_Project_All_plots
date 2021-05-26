# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:51:58 2021

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


#%% Data Import

# Filename:
#filename = "Data\postnetwork-elec_only_0.125_0.6.h5" # 40% CO2
filename = "Data\postnetwork-elec_only_0.125_0.05.h5" # 95% CO2 constrain
# Import network
network = pypsa.Network(filename)

# User message
print("\nFile loaded: ", filename)

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

# Collecting mismatch terms
gen_grouped = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum().values
mismatch_terms = pd.DataFrame({'wind': gen_grouped[:,0]+gen_grouped[:,1],
                                'ror': gen_grouped[:,2],
                                'solar': gen_grouped[:,3],
                                'load': load.sum(axis=1).values},index=time_index)

#%% Collecting technologies per country

# Combine the load at every timestep for all countries
load_EU = np.sum(load, axis=1)

# Dataframe (array) for different generator technologies
generator_wind = pd.DataFrame(np.zeros([8760, 30]), columns=country_column)
generator_solar = pd.DataFrame(np.zeros([8760, 30]), columns=country_column)
generator_hydro = pd.DataFrame(np.zeros([8760, 30]), columns=country_column)

# Counter for positioning in generator data
counter = 0
for i in network.generators.index:
    
    # Current value to insert into correct array and position
    value = np.array(network.generators_t.p)[:,counter]
    
    # Check for wind, solar and hydro
    if (i[-4:] == "wind"):
        generator_wind[i[0:2]] = generator_wind[i[0:2]] + value
    elif (i[-5:] == "solar"):
        generator_solar[i[0:2]] = generator_solar[i[0:2]] + value
    elif (i[-3:] == "ror"):
        generator_hydro[i[0:2]] = generator_hydro[i[0:2]] + value
    
    # Increase value of counter by 1
    counter +=1


# Mean values
wind_mean = np.mean(generator_wind,axis=0)
solar_mean = np.mean(generator_solar,axis=0)
hydro_mean = np.mean(generator_hydro,axis=0)
load_mean = np.mean(load,axis=0)

# Centering data
wind_cent = np.subtract(generator_wind,wind_mean.T)
solar_cent = np.subtract(generator_solar,solar_mean.T)
hydro_cent = np.subtract(generator_hydro,hydro_mean.T)
load_cent = np.subtract(load,load_mean.T)

check = wind_cent+solar_cent+hydro_cent-load_cent.values

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

#%% eigen values contribution

# Component contribution
# Wind
wind_con = np.dot(wind_cent,eig_vec)
# Solar
solar_con = np.dot(solar_cent,eig_vec)
# Hydro
ror_con = np.dot(hydro_cent,eig_vec)
# Load
load_con = np.dot(load_cent,eig_vec)
# Sum (with -L) of this is equal to T

check = c*(wind_con+solar_con+ror_con-load_con)

# Eigenvalues contribution
# Wind
lambda_W = (c**2)*(np.mean((wind_con**2),axis=0))
# Solar
lambda_S = (c**2)*(np.mean((solar_con**2),axis=0))
# Hydro
lambda_H = (c**2)*(np.mean((ror_con**2),axis=0))
# Load
lambda_L = (c**2)*(np.mean((load_con**2),axis=0))
# Wind+Solar
lambda_WS = (c**2)*2*(np.mean((wind_con*solar_con),axis=0))
# Wind+Hydro
lambda_WH = (c**2)*2*(np.mean((wind_con*ror_con),axis=0))
# Hydro+Solar
lambda_HS = (c**2)*2*(np.mean((ror_con*solar_con),axis=0))
# Wind+Load
lambda_WL = (c**2)*2*(np.mean((wind_con*load_con),axis=0))
# Load+Solar
lambda_LS = (c**2)*2*(np.mean((load_con*solar_con),axis=0))
# Load+Hydro
lambda_LH = (c**2)*2*(np.mean((load_con*ror_con),axis=0))

# Collecting terms
lambda_collect_wmin = pd.DataFrame({'wind':             lambda_W,
                                   'solar':             lambda_S,
                                   'RoR':               lambda_H,
                                   'load':              lambda_L,
                                   'wind/\nsolar':      lambda_WS,
                                   'wind/\nRoR':        lambda_WH,
                                   'RoR/\nsolar':       lambda_HS,
                                   'wind/\nload':      -lambda_WL,
                                   'load/\nsolar':     -lambda_LS,
                                   'load/\nRoR':       -lambda_LH,
                                   })
lambda_collect_nmin = pd.DataFrame({'wind':             lambda_W,
                                   'solar':             lambda_S,
                                   'RoR':               lambda_H,
                                   'load':              lambda_L,
                                   'wind/\nsolar':      lambda_WS,
                                   'wind/\nRoR':        lambda_WH,
                                   'RoR/\nsolar':       lambda_HS,
                                   'wind/\nload':       lambda_WL,
                                   'load/\nsolar':      lambda_LS,
                                   'load/\nRoR':        lambda_LH,
                                   })



lambda_tot = sum([+lambda_W,
                 +lambda_S,
                 +lambda_H,
                 +lambda_L,
                 +lambda_WS,
                 +lambda_WH,
                 +lambda_HS,
                 -lambda_WL,
                 -lambda_LS,
                 -lambda_LH
                 ])

#%% Plotting of eigen values contribution

plt.figure(figsize=[14,16])
for n in range(6):
    lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # percentage
    #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
    plt.subplot(3,2,n+1)
    plt.bar(lambda_collect_wmin.columns,lambda_collect_procent.values[0])
    plt.title('PC'+str(n+1)+': '+str(round(lambda_tot[n],3)))
    plt.ylabel('Influance [%]')
    plt.ylim([-50,125])
    plt.grid(axis='y',alpha=0.5)
    for k in range(10):
        if lambda_collect_procent.values[:,k] < 0:
            v = lambda_collect_procent.values[:,k] - 6.5
        else:
            v = lambda_collect_procent.values[:,k] + 2.5
        plt.text(x=k,y=v,s=str(round(float(lambda_collect_procent.values[:,k]),2))+'%',ha='center',size='small')
plt.suptitle(filename,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07     

#%% Plotting of eigen values contribution

individual_plots = True

if individual_plots==True:
    for n in range(6):
        lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # percentage
        #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
        plt.figure()
        plt.bar(lambda_collect_wmin.columns,lambda_collect_procent.values[0])
        plt.title('$\lambda_{'+str(n+1)+'}$: '+str(round(lambda_tot[n]*100,1))+'%')
        plt.ylabel('Influance [%]')
        plt.ylim([-60,150])
        plt.grid(axis='y',alpha=0.5)
        for k in range(10):
            if lambda_collect_procent.values[:,k] < 0:
                v = lambda_collect_procent.values[:,k] - 6.5
            else:
                v = lambda_collect_procent.values[:,k] + 2.5
            plt.text(x=k,y=v,s=str(round(float(lambda_collect_procent.values[:,k]),2))+'%',ha='center',size='small')
        
        title = "PC"+str(n+1)
        plt.savefig("Figures/overleaf/mismatch/contribution/contribution_"+title+".png", bbox_inches='tight')
    #plt.suptitle(filename,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07  
