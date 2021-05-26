# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:07:07 2021

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

name = ["Response for Mismatch with 40% CO_2 Emission and 2x Current Transmission",
        "Response for Mismatch with 50% CO_2 Emission and 2x Current Transmission",
        "Response for Mismatch with 60% CO_2 Emission and 2x Current Transmission",
        "Response for Mismatch with 70% CO_2 Emission and 2x Current Transmission",
        "Response for Mismatch with 80% CO_2 Emission and 2x Current Transmission",
        "Response for Mismatch with 90% CO_2 Emission and 2x Current Transmission",
        "Response for Mismatch with 95% CO_2 Emission and 2x Current Transmission"]

place = ["40","50","60","70","80","90","95"]

dic = 'Data\\' # Location of files

print("Files loaded")

#%% Starting loop

# Variable for principal components for plotting later
PC1_con = np.zeros((7,7))
PC2_con = np.zeros((7,7))
PC3_con = np.zeros((7,7))
PC4_con = np.zeros((7,7))
PC5_con = np.zeros((7,7))
PC6_con = np.zeros((7,7))
PC_con = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename)):
    # User info:
    print("\nCalculating for: ",name[i])
    # Create network from previous file
    network = pypsa.Network(dic+filename[i])
    
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
    
    #%% Backup generator
    
    # Efficiency (can be seen at (network.links.efficiency))
    eff_gas = network.links.efficiency['DK OCGT'] #0.39
    
    # output gas backup generator
    store_gas_out = pd.DataFrame(np.array(network.links_t.p0)[:,0:30], columns=(data_names))
    
    # With efficiency
    store_gas_out_eff = store_gas_out * eff_gas
    
    #%% Storage
    
    # Efficiency for different storages (can be seen at (network.links.efficiency))
    eff_H2_charge = network.links.efficiency['DK H2 Electrolysis'] #0.8
    eff_H2_discharge = network.links.efficiency['DK H2 Fuel Cell'] #0.58
    eff_battery_charge = network.links.efficiency['DK battery charger'] #0.9
    eff_battery_discharge = network.links.efficiency['DK battery discharger'] #0.9
    eff_PHS = network.storage_units.efficiency_dispatch['AT PHS'] #0.866
    
    # H2 storage
    store_H2_in = pd.DataFrame(np.array(network.links_t.p0)[:,30:60], columns=(data_names))
    store_H2_out = pd.DataFrame(np.array(network.links_t.p0)[:,60:90], columns=(data_names))
    
    # Battery stoage
    store_batttery_in = pd.DataFrame(np.array(network.links_t.p0)[:,90:120], columns=(data_names))
    store_battery_out = pd.DataFrame(np.array(network.links_t.p0)[:,120:150], columns=(data_names))
    
    # PHS and hydro
    PHS_and_hydro = network.storage_units_t.p.groupby([network.storage_units.carrier, network.storage_units.bus],axis=1).sum()
    PHS_and_hydro_val = PHS_and_hydro.values
    PHS = PHS_and_hydro["PHS"]
    hydro = PHS_and_hydro["hydro"]
    store_PHS = np.zeros((8760,30)) # All countries
    store_hydro = np.zeros((8760,30)) # All countries
    for k in range(len(data_names)):
        if data_names[k] in PHS.columns: # Checks if the country is in PHS
            store_PHS[:,k] += PHS[data_names[k]] # Adds the value
    for k in range(len(data_names)):
        if data_names[k] in hydro.columns: # Checks if the country is in PHS
            store_hydro[:,k] += hydro[data_names[k]] # Adds the value
    store_PHS = pd.DataFrame(store_PHS, columns=data_names)
    store_hydro = pd.DataFrame(store_hydro, columns=data_names)
    
    # With efficiency
    store_H2_in_eff = store_H2_in #* 1/eff_H2_charge
    store_H2_out_eff = store_H2_out * eff_H2_discharge
    store_battery_in_eff = store_batttery_in #* 1/eff_battery_charge
    store_battery_out_eff = store_battery_out * eff_battery_discharge
    store_PHS_eff = store_PHS #* eff_PHS
    store_hydro_eff = store_hydro 
    
    # Sum of all storage
    store_sum = - store_H2_in + store_H2_out - store_batttery_in + store_battery_out + store_PHS + store_hydro
    
    # Sum of all storage including efficicency
    store_sum_eff = - store_H2_in_eff + store_H2_out_eff - store_battery_in_eff + store_battery_out_eff + store_PHS_eff + store_hydro_eff
    
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
    sum_bus0 = (country_links_p0.groupby(network.links.bus0,axis=1).sum())
    
    # Sum all bus1 imports values and minus
    sum_bus1 = (country_links_p1.groupby(network.links.bus1,axis=1).sum())
    
    # Creating empty matrix
    country_links = np.zeros((8760,30))
    
    # loop that adds bus0 and bus1 together (remember bus1 is minus value)
    for k in range(len(data_names)):   
        
        if data_names[k] in sum_bus0.columns: # Checks if the country is in bus0
            country_links[:,k] += sum_bus0[data_names[k]] # Adds the value for the country to the collected link function
            
        if data_names[k] in sum_bus1.columns: # Checks if the country is in bus1
            country_links[:,k] += sum_bus1[data_names[k]] # Adds the value for the country to the collected link function
    
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
    
    #%% CHECKER
    
    checker = country_links - store_sum_eff.values - store_gas_out_eff.values
    
    checker2 = checker - X.values
    
    #%% Centering responce data
    
    # Mean values
    backup_mean = np.mean(store_gas_out_eff,axis=0)
    inport_export_mean = np.mean(country_links,axis=0)
    storage_mean = np.mean(store_sum_eff,axis=0)
    
    # Centering data
    backup_cent = - np.subtract(store_gas_out_eff,backup_mean.T) # MINUS ADDED?
    inport_export_cent = np.subtract(country_links,inport_export_mean.T)
    storage_cent = - np.subtract(store_sum_eff,storage_mean.T) # MINUS ADDED?
    
    #check = backup_cent + inport_export_cent.values + storage_cent.values
    
    #%% eigen values contribution
    
    # Component contribution
    # Backup generator
    backup_con = np.dot(backup_cent,eig_vec)
    # inport/export
    inport_export_con = np.dot(inport_export_cent,eig_vec)
    # storage technologies
    storage_con = np.dot(storage_cent,eig_vec)
    # Sum (with -L) of this is equal to T
    
    #check = (backup_con + inport_export_con + storage_con)*c
    
    # Eigenvalues contribution
    # Backup
    lambda_B = (c**2)*(np.mean((backup_con**2),axis=0))
    # inport/export
    lambda_P = (c**2)*(np.mean((inport_export_con**2),axis=0))
    # storage technologies
    lambda_DeltaS = (c**2)*(np.mean((storage_con**2),axis=0))
    # Backup + inport/export
    lambda_BP = (c**2)*2*(np.mean((backup_con*inport_export_con),axis=0))
    # Backup + storage technologies
    lambda_BdeltaS = (c**2)*2*(np.mean((backup_con*storage_con),axis=0))
    # inport/export + storage technologies
    lambda_PdeltaS = (c**2)*2*(np.mean((inport_export_con*storage_con),axis=0))
    
    # Collecting terms
    lambda_collect = pd.DataFrame({'backup':                    lambda_B,
                                   'import &\nexport':          lambda_P,
                                   'storage':                   lambda_DeltaS,
                                   'backup/\ninport/export':    lambda_BP,
                                   'backup/\nstorage':          lambda_BdeltaS,
                                   'inport/export/\nstorage':   lambda_PdeltaS,
                                   })
    
    
    lambda_tot = sum([+lambda_B,
                      +lambda_P,
                      +lambda_DeltaS,
                      +lambda_BP,
                      +lambda_BdeltaS,
                      +lambda_PdeltaS
                     ])
    
    lambda_collect_all = pd.DataFrame({'backup':                    lambda_B,
                                       'import &\nexport':          lambda_P,
                                       'storage':                   lambda_DeltaS,
                                       'backup/\ninport/export':    lambda_BP,
                                       'backup/\nstorage':          lambda_BdeltaS,
                                       'inport/export/\nstorage':   lambda_PdeltaS,
                                       'total':                     lambda_tot
                                   })
    
    #%% Plotting of eigen values contribution
    
    plt.figure(figsize=[14,16])
    for n in range(6):
        lambda_collect_procent = lambda_collect[n:n+1]/lambda_tot[n]*100 # percentage
        #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
        plt.subplot(3,2,n+1)
        plt.bar(lambda_collect.columns,lambda_collect_procent.values[0])
        plt.title('PC'+str(n+1)+': '+str(round(lambda_tot[n],3)))
        plt.ylabel('Influance [%]')
        plt.ylim([-100,140])
        plt.grid(axis='y',alpha=0.5)
        for k in range(6):
            if lambda_collect_procent.values[:,k] < 0:
                v = lambda_collect_procent.values[:,k] - 6.5
            else:
                v = lambda_collect_procent.values[:,k] + 2.5
            plt.text(x=k,y=v,s=str(round(float(lambda_collect_procent.values[:,k]),2))+'%',ha='center',size='small')
    plt.suptitle(name[i],fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 
    
    
    title = name[i]
    
    placement = place[i]
    
    plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 

    plt.savefig("Figures/CO2_constraint/"+placement+"/Mismatch/"+title+".png", bbox_inches='tight')
    
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
    backup_con_data  = PC_con[i][:,:1].sum(axis=1)
    inport_export_con_con_data = PC_con[i][:,:2].sum(axis=1)
    storage_con_data = PC_con[i][:,:3].sum(axis=1)
    backup_inport_cov_data  = PC_con[i][:,:4].sum(axis=1)
    backup_store_cov_data   = PC_con[i][:,:5].sum(axis=1)
    inport_store_cov_data   = PC_con[i][:,:6].sum(axis=1)
    # plot function
    plt.subplot(3,2,i+1)
    # Plot lines
    plt.plot(backup_con_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(inport_export_con_con_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(storage_con_data,color='k',alpha=1,linewidth=0.5)
    #plt.plot(backup_inport_cov_data,color='k',alpha=1,linewidth=0.5)
    #plt.plot(backup_store_cov_data,color='k',alpha=1,linewidth=0.5)
    plt.plot(inport_store_cov_data,color='k',alpha=1,linewidth=0.5)
    # Plot fill inbetween lines
    plt.fill_between(range(7), np.zeros(7), backup_con_data,
                     label='backup',
                     color='cornflowerblue') # Because it is a beutiful color
    plt.fill_between(range(7), backup_con_data, inport_export_con_con_data,
                     label='import & export',
                     color='yellow')
    plt.fill_between(range(7), inport_export_con_con_data, storage_con_data,
                     label='storage',
                     color='darkslateblue')
    plt.fill_between(range(7), storage_con_data, inport_store_cov_data,
                     label='covariance',
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

title = "PCA of Mismatch - Responses as a Function of CO2 Constrain"

plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 

plt.savefig("Figures/CO2_constraint/"+title+".png", bbox_inches='tight')

plt.show(all)
    