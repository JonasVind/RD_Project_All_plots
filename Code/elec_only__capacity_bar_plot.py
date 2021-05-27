import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import pypsa
from elec_only_functions import BAR


directory = "C:/Users/jense/.spyder-py3/data_files/elec_only/"

filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]

file = filename_CO2[-1]

generators = []
storages = []
backup = []

for file in filename_CO2:
    # Network
    network = pypsa.Network(directory+file)
    
    # Get the names of the data
    data_names = network.loads_t.p.columns
    
    generators.append(network.generators.groupby('carrier').sum().p_nom_opt)
    
    # Storage 1 (PHS + hydro)
    storage1 = network.storage_units_t.state_of_charge.max()
    
    PHS = 0
    hydro = 0
    
    for i in np.arange(0, len(storage1)):
        if (storage1.index[i][-3:] == "PHS"):
            PHS += storage1.values[i]
        
        elif (storage1.index[i][-5:] == "hydro"):
            hydro += storage1.values[i]
            #hydro += 0
         
        
    # Storage 2 (batter + H2)
    storage2 = network.stores.e_nom_opt[30:]
    
    battery = 0
    H2 = 0
    
    for i in np.arange(0, len(storage2)):
        if (storage2.index[i][3:5] == "H2"):
            H2 += storage2.values[i]
        
        elif (storage2.index[i][3:10] == "battery"):
            battery += storage2.values[i]
    
    storages.append(pd.Series(data=[hydro, PHS, H2, battery], index=["Hydro", "PHS", "H2", "Battery"]))
    
    
    #backup.append(network.stores.e_nom_opt[0:30].sum())
    backup.append(network.stores_t.p.values[:,0:30].max(axis=0).sum())

#%%

matrix = [x * (1e-3) for x in generators]
constraints = ['40%', '50%', '60%', '70%', '80%', '90%', '95%']
title = 'Installed renewable capacity for each technology \n as a function of $CO_{2}$ constraint'
xlabel = '$CO_{2}$ Constraint'
ylabel = 'Installed capacity [GW]'


fig1 = plt.figure(dpi=200)
# fig.add_axes([x1,y1,x2,y2])
ax = fig1.add_axes([0,0,1,1.5])    

# List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
#cmap = plt.get_cmap("tab10")
cmap = plt.get_cmap("Accent")

# Number of "color-bars" / PC to show (k1, k2, k3...)
j_max = len(generators[0])
colour_map = cmap(np.arange(j_max)) # Must be equal to the number of bars plotted

# Label variable
lns_fig1 = []

for i in np.arange(0,len(filename_CO2)):
    
    # Plotting 'j' components in total
    for j in np.arange(0,j_max):
        if j == 0:
            lns_plot = ax.bar(constraints[i], matrix[i][j], color=colour_map[j], edgecolor='black', linewidth=1.2, label=(generators[0].index[j]))
            
        elif (j > 0 and j < (j_max-1)):
            lns_plot = ax.bar(constraints[i], matrix[i][j], bottom = sum(matrix[i][0:j]), color=colour_map[j], edgecolor='black', label=(generators[0].index[j]))
        
        else:
            lns_plot = ax.bar(constraints[i], sum(matrix[i][j:29]), bottom = sum(matrix[i][0:j]), color=colour_map[j], edgecolor='black', label=(generators[0].index[j]))
            
            
        if (i==0):
            lns_fig1.append(lns_plot)


# Add labels
labs = [l.get_label() for l in lns_fig1]
ax.legend(lns_fig1, labs, bbox_to_anchor = (1,1))

#plt.yticks(range(0, 120, 25))
plt.title(title, fontsize=18, fontweight='bold')
plt.xlabel(xlabel, fontsize=16)
plt.ylabel(ylabel, fontsize=16)
plt.grid(axis='y')


#%% Storage energy capacity
matrix = [x * (1e-3) for x in storages]
constraints = ['40%', '50%', '60%', '70%', '80%', '90%', '95%']
title = 'Installed storage energy capacity for each technology \n as a function of $CO_{2}$ constraint'
xlabel = '$CO_{2}$ Constraint'
ylabel = 'Installed energy capacity [GWh]'


fig1 = plt.figure(dpi=200)
# fig.add_axes([x1,y1,x2,y2])
ax = fig1.add_axes([0,0,1,1.5])    

# List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
#cmap = plt.get_cmap("tab10")
cmap = plt.get_cmap("Dark2")

# Number of "color-bars" / PC to show (k1, k2, k3...)
j_max = len(storages[0])
colour_map = cmap(np.arange(j_max)) # Must be equal to the number of bars plotted

# Label variable
lns_fig1 = []

for i in np.arange(0,len(filename_CO2)):
    
    # Plotting 'j' components in total
    for j in np.arange(0,j_max):
        if j == 0:
            lns_plot = ax.bar(constraints[i], matrix[i][j], color=colour_map[j], edgecolor='black', linewidth=1.2, label=(storages[0].index[j]))
            
        elif (j > 0 and j < (j_max-1)):
            lns_plot = ax.bar(constraints[i], matrix[i][j], bottom = sum(matrix[i][0:j]), color=colour_map[j], edgecolor='black', label=(storages[0].index[j]))
        
        else:
            lns_plot = ax.bar(constraints[i], sum(matrix[i][j:29]), bottom = sum(matrix[i][0:j]), color=colour_map[j], edgecolor='black', label=(storages[0].index[j]))
            
            
        if (i==0):
            lns_fig1.append(lns_plot)


# Add labels
labs = [l.get_label() for l in lns_fig1]
ax.legend(lns_fig1, labs, bbox_to_anchor = (1,1))

#plt.yticks(range(0, 120, 25))
plt.title(title, fontsize=18, fontweight='bold')
plt.xlabel(xlabel, fontsize=16)
plt.ylabel(ylabel, fontsize=16)
plt.ylim(130000, 150000)
plt.grid(axis='y')

#%% Backup capacity
matrix = [x * (1e-3) for x in backup]
constraints = ['40%', '50%', '60%', '70%', '80%', '90%', '95%']
title = 'Installed backup capacity as a function of $CO_{2}$ constraint'
xlabel = '$CO_{2}$ Constraint'
ylabel = 'Installed backup capacity [GW]'

fig1 = plt.figure(dpi=200)
ax = fig1.add_axes([0,0,1,1.5]) 
plt.bar(constraints, matrix, edgecolor="black", color="indianred")
plt.title(title, fontsize=16, fontweight='bold')
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
plt.legend(["Gas"])
plt.ylim(0, 550)
plt.grid(axis='y')