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
filename = ["postnetwork-elec_only_0_0.05.h5",
            "postnetwork-elec_only_0.0625_0.05.h5",
            "postnetwork-elec_only_0.125_0.05.h5",
            "postnetwork-elec_only_0.25_0.05.h5",
            "postnetwork-elec_only_0.375_0.05.h5"]

dic = 'Data\\' # Location of files

print("Files loaded")

#%% Starting loop

# Variable for principal components for plotting later
PC1_con = np.zeros((5,7))
PC2_con = np.zeros((5,7))
PC3_con = np.zeros((5,7))
PC4_con = np.zeros((5,7))
PC5_con = np.zeros((5,7))
PC6_con = np.zeros((5,7))
PC_con = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename)):
    # User info:
    print("\nCalculating for: ",filename[i])
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
    
    #%% FIX FOR 0x CURRENT TRANSMISSION LINKS IS UNSORTED
    
    links_unsorted = network.links_t.p0
    
    if i == 0:
        sorted_list = ['AT OCGT', 'BA OCGT', 'BE OCGT', 'BG OCGT', 'CH OCGT', 'CZ OCGT',
               'DE OCGT', 'DK OCGT', 'EE OCGT', 'ES OCGT', 'FI OCGT', 'FR OCGT',
               'GB OCGT', 'GR OCGT', 'HR OCGT', 'HU OCGT', 'IE OCGT', 'IT OCGT',
               'LT OCGT', 'LU OCGT', 'LV OCGT', 'NL OCGT', 'NO OCGT', 'PL OCGT',
               'PT OCGT', 'RO OCGT', 'RS OCGT', 'SE OCGT', 'SI OCGT', 'SK OCGT',
               'AT H2 Electrolysis', 'BA H2 Electrolysis', 'BE H2 Electrolysis',
               'BG H2 Electrolysis', 'CH H2 Electrolysis', 'CZ H2 Electrolysis',
               'DE H2 Electrolysis', 'DK H2 Electrolysis', 'EE H2 Electrolysis',
               'ES H2 Electrolysis', 'FI H2 Electrolysis', 'FR H2 Electrolysis',
               'GB H2 Electrolysis', 'GR H2 Electrolysis', 'HR H2 Electrolysis',
               'HU H2 Electrolysis', 'IE H2 Electrolysis', 'IT H2 Electrolysis',
               'LT H2 Electrolysis', 'LU H2 Electrolysis', 'LV H2 Electrolysis',
               'NL H2 Electrolysis', 'NO H2 Electrolysis', 'PL H2 Electrolysis',
               'PT H2 Electrolysis', 'RO H2 Electrolysis', 'RS H2 Electrolysis',
               'SE H2 Electrolysis', 'SI H2 Electrolysis', 'SK H2 Electrolysis',
               'AT H2 Fuel Cell', 'BA H2 Fuel Cell', 'BE H2 Fuel Cell',
               'BG H2 Fuel Cell', 'CH H2 Fuel Cell', 'CZ H2 Fuel Cell',
               'DE H2 Fuel Cell', 'DK H2 Fuel Cell', 'EE H2 Fuel Cell',
               'ES H2 Fuel Cell', 'FI H2 Fuel Cell', 'FR H2 Fuel Cell',
               'GB H2 Fuel Cell', 'GR H2 Fuel Cell', 'HR H2 Fuel Cell',
               'HU H2 Fuel Cell', 'IE H2 Fuel Cell', 'IT H2 Fuel Cell',
               'LT H2 Fuel Cell', 'LU H2 Fuel Cell', 'LV H2 Fuel Cell',
               'NL H2 Fuel Cell', 'NO H2 Fuel Cell', 'PL H2 Fuel Cell',
               'PT H2 Fuel Cell', 'RO H2 Fuel Cell', 'RS H2 Fuel Cell',
               'SE H2 Fuel Cell', 'SI H2 Fuel Cell', 'SK H2 Fuel Cell',
               'AT battery charger', 'BA battery charger', 'BE battery charger',
               'BG battery charger', 'CH battery charger', 'CZ battery charger',
               'DE battery charger', 'DK battery charger', 'EE battery charger',
               'ES battery charger', 'FI battery charger', 'FR battery charger',
               'GB battery charger', 'GR battery charger', 'HR battery charger',
               'HU battery charger', 'IE battery charger', 'IT battery charger',
               'LT battery charger', 'LU battery charger', 'LV battery charger',
               'NL battery charger', 'NO battery charger', 'PL battery charger',
               'PT battery charger', 'RO battery charger', 'RS battery charger',
               'SE battery charger', 'SI battery charger', 'SK battery charger',
               'AT battery discharger', 'BA battery discharger',
               'BE battery discharger', 'BG battery discharger',
               'CH battery discharger', 'CZ battery discharger',
               'DE battery discharger', 'DK battery discharger',
               'EE battery discharger', 'ES battery discharger',
               'FI battery discharger', 'FR battery discharger',
               'GB battery discharger', 'GR battery discharger',
               'HR battery discharger', 'HU battery discharger',
               'IE battery discharger', 'IT battery discharger',
               'LT battery discharger', 'LU battery discharger',
               'LV battery discharger', 'NL battery discharger',
               'NO battery discharger', 'PL battery discharger',
               'PT battery discharger', 'RO battery discharger',
               'RS battery discharger', 'SE battery discharger',
               'SI battery discharger', 'SK battery discharger', 'AT-CH', 'AT-CZ',
               'AT-DE', 'AT-HU', 'AT-IT', 'AT-SI', 'BA-HR', 'BA-RS', 'BG-GR',
               'BG-RO', 'BG-RS', 'CH-DE', 'CH-IT', 'CZ-DE', 'CZ-SK', 'DE-DK',
               'DE-LU', 'DE-SE', 'EE-LV', 'FI-EE', 'FI-SE', 'FR-BE', 'FR-CH',
               'FR-DE', 'FR-ES', 'FR-GB', 'FR-IT', 'GB-IE', 'GR-IT', 'HR-HU',
               'HR-RS', 'HR-SI', 'HU-RS', 'HU-SK', 'IT-SI', 'LV-LT', 'NL-BE',
               'NL-DE', 'NL-GB', 'NL-NO', 'NO-DK', 'NO-SE', 'PL-CZ', 'PL-DE',
               'PL-LT', 'PL-SE', 'PL-SK', 'PT-ES', 'RO-HU', 'RO-RS', 'SE-DK',
               'SE-LT']
        
        links_sorted = pd.DataFrame(data=0, index=links_unsorted.index, columns=sorted_list)
        
        for k in np.arange(0,len(sorted_list)):
            link = links_sorted.columns[k]
            links_sorted[link] = links_unsorted[link]
        
        # Updated with sorted list
        network.links_t.p0 = links_sorted
    
    
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
    plt.suptitle(filename[i],fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 
    
    plt.show(all)    
    
    #%% Save data for PC1-PC6
    
    PC1_con[i] = lambda_collect_all[0:1].values
    PC2_con[i] = lambda_collect_all[1:2].values
    PC3_con[i] = lambda_collect_all[2:3].values
    PC4_con[i] = lambda_collect_all[3:4].values
    PC5_con[i] = lambda_collect_all[4:5].values
    PC6_con[i] = lambda_collect_all[5:6].values   
    
    #%%
    if i == 0:
        test0 = network.links_t.p0
    elif i == 1:
        test1 = network.links_t.p0
    elif i == 2:
        test2 = network.links_t.p0    
    elif i == 3:
        test3 = network.links_t.p0
    elif i == 4:
        test4 = network.links_t.p0
    elif i == 5:
        test5 = network.links_t.p0
assert False
    
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
    plt.fill_between(range(5), np.zeros(5), backup_con_data,
                     label='backup',
                     color='cornflowerblue') # Because it is a beutiful color
    plt.fill_between(range(5), backup_con_data, inport_export_con_con_data,
                     label='import & export',
                     color='yellow')
    plt.fill_between(range(5), inport_export_con_con_data, storage_con_data,
                     label='storage',
                     color='darkslateblue')
    plt.fill_between(range(5), storage_con_data, inport_store_cov_data,
                     label='covariance',
                     color='orange',
                     alpha=0.5)
    # y/x-axis and title
    #plt.legend(bbox_to_anchor = (1,1))
    plt.ylabel('$\lambda_k$')
    plt.xticks(np.arange(0,5),['Zero', 'Current', '2x Current', '4x Current', '6x Current'])
    plt.title('Principle component '+str(i+1))
    if i == 4: # Create legend of figure 4 (lower left)
        plt.legend(loc = 'center', # How the label should be places according to the placement
                   bbox_to_anchor = (1.1,-0.17), # placement relative to the graph
                   ncol = 6, # Amount of columns
                   fontsize = 'large', # Size of text
                   framealpha = 1, # Box edge alpha
                   columnspacing = 2.5 # Horizontal spacing between labels
                   )

plt.suptitle("Responses as a Function of Transmission Link Sizes",fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 

plt.show(all)
    