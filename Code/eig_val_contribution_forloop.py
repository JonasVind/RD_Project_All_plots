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
# Name of file (must be in same folder)
filename = ["postnetwork-elec_only_0.125_0.6.h5",
            "postnetwork-elec_only_0.125_0.5.h5",
            "postnetwork-elec_only_0.125_0.4.h5",
            "postnetwork-elec_only_0.125_0.3.h5",
            "postnetwork-elec_only_0.125_0.2.h5",
            "postnetwork-elec_only_0.125_0.1.h5",
            "postnetwork-elec_only_0.125_0.05.h5"]

dic = 'Data\\' # Location of files

print("Files loaded")

#%% Starting loop

# Variable for principal components for plotting later
PC1_con = np.zeros((7,11))
PC2_con = np.zeros((7,11))
PC3_con = np.zeros((7,11))
PC4_con = np.zeros((7,11))
PC5_con = np.zeros((7,11))
PC6_con = np.zeros((7,11))
PC_con = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename)):
    # User info:
    print("\nCalculating for: ",filename[i])
    # Create network from previous file
    network = pypsa.Network(dic+filename[i])

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
    for j in network.generators.index:
        
        # Current value to insert into correct array and position
        value = np.array(network.generators_t.p)[:,counter]
        
        # Check for wind, solar and hydro
        if (j[-4:] == "wind"):
            generator_wind[j[0:2]] = generator_wind[j[0:2]] + value
        elif (j[-5:] == "solar"):
            generator_solar[j[0:2]] = generator_solar[j[0:2]] + value
        elif (j[-3:] == "ror"):
            generator_hydro[j[0:2]] = generator_hydro[j[0:2]] + value
        
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
    
    lambda_collect_all = pd.DataFrame({'wind':              lambda_W,
                                       'solar':             lambda_S,
                                       'RoR':               lambda_H,
                                       'load':              lambda_L,
                                       'wind/\nsolar':      lambda_WS,
                                       'wind/\nRoR':        lambda_WH,
                                       'RoR/\nsolar':       lambda_HS,
                                       'wind/\nload':      -lambda_WL,
                                       'load/\nsolar':     -lambda_LS,
                                       'load/\nRoR':       -lambda_LH,
                                       'total':             lambda_tot
                                       })
    
    
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
    plt.suptitle(filename[i],fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07     
   
    plt.show(all)

    #%% Save data for PC1-PC6
    
    PC1_con[i] = lambda_collect_all[0:1].values
    PC2_con[i] = lambda_collect_all[1:2].values
    PC3_con[i] = lambda_collect_all[2:3].values
    PC4_con[i] = lambda_collect_all[3:4].values
    PC5_con[i] = lambda_collect_all[4:5].values
    PC6_con[i] = lambda_collect_all[5:6].values

#%% Data handling

PC_con.append(PC1_con)
PC_con.append(PC2_con)
PC_con.append(PC3_con)
PC_con.append(PC4_con)
PC_con.append(PC5_con)
PC_con.append(PC6_con)

#%% Plot

plt.figure(figsize=(14,16))#,dpi=500)

for i in range(6):
    # y functions comulated
    wind_con_data  = PC_con[i][:,:1].sum(axis=1)
    solar_con_data = PC_con[i][:,:2].sum(axis=1)
    hydro_con_data = PC_con[i][:,:3].sum(axis=1)
    load_con_data  = PC_con[i][:,:4].sum(axis=1)
    gen_cov_data   = PC_con[i][:,:7].sum(axis=1)
    load_cov_data  = PC_con[i][:,8:10].sum(axis=1)
    # plot function
    plt.subplot(3,2,i+1)
    # Plot lines
    plt.plot(wind_con_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(solar_con_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(hydro_con_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(load_con_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(gen_cov_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(load_cov_data,color='k',alpha=1,linewidth=0.5)
    # Plot fill inbetween lines
    plt.fill_between(range(7), np.zeros(7), wind_con_data,
                     label='Wind',
                     color='cornflowerblue') # Because it is a beutiful color
    plt.fill_between(range(7), wind_con_data, solar_con_data,
                     label='Solar',
                     color='yellow')
    plt.fill_between(range(7), solar_con_data, hydro_con_data,
                     label='RoR',
                     color='darkslateblue')
    plt.fill_between(range(7), hydro_con_data, load_con_data,
                     label='Load',
                     color='slategray')
    plt.fill_between(range(7), load_con_data, gen_cov_data,
                     label='Generator\ncovariance',
                     color='brown',
                     alpha=0.5)
    plt.fill_between(range(7), load_cov_data, np.zeros(7),
                     label='Load\ncovariance',
                     color='orange',
                     alpha=0.5)
    # y/x-axis and title
    #plt.legend(bbox_to_anchor = (1,1))
    plt.ylabel('$\lambda_k$')
    plt.xticks(np.arange(0,7),['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    plt.title('Principle component '+str(i+1))
    if i == 4: # Create legend of figure 4 (lower left)
        plt.legend(loc = 'center', # How the label should be places according to the placement
                   bbox_to_anchor = (1.1,-0.17), # placement relative to the graph
                   ncol = 6, # Amount of columns
                   fontsize = 'large', # Size of text
                   framealpha = 1, # Box edge alpha
                   columnspacing = 2.5 # Horizontal spacing between labels
                   )
        
plt.suptitle("Contribution as a Function of CO2 Constrain",fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 

plt.show(all)





