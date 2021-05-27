import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import pypsa


directory = "C:/Users/jense/.spyder-py3/data_files/elec_only/"

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

#file = filename_CO2[-1]

for file in filename_CO2:
    file = filename_CO2[-1]
    
    # Load network
    network = pypsa.Network(directory+file)
    
    # Get the names of the data
    data_names = network.loads_t.p.columns
    
    # Remove AC
    network.buses_copy = network.buses.copy()
    network.buses.drop(network.buses.index[network.buses.carrier != "AC"], inplace=True)
    
    # Plot figure
    size = 15.0
    plt.figure(figsize=[size, size], dpi=200)
    data = network.generators.assign(g = network.generators.p_nom_opt).groupby(['bus', 'carrier']).g.sum()
    data = data * 3e-5
    network.plot(bus_sizes=data, bus_colors={'onwind' : 'dodgerblue',
                                             'offwind' : 'dodgerblue',
                                             'ror' : 'limegreen',
                                             'solar' : 'gold'}, projection=(ccrs.PlateCarree()))#, link_colors='black')
    
    plt.suptitle(file, fontsize=26, x=.55, y=0.84)
    plt.title("Installed generator capacity [MW]", fontsize=20)
    legend_elements = [Line2D([0], [0], marker='o', color='white', label='Wind', markerfacecolor='dodgerblue', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='Solar', markerfacecolor='gold', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='ROR', markerfacecolor='limegreen', markersize=12)]              
    
    plt.legend(handles=legend_elements, fontsize=15, bbox_to_anchor = (0.13,0.96), framealpha=1)
    
    
    # STORAGE ENERGY CAPACITY (MWh)
    # Take storage units from network.stores (minus gas)
    storage_energy_1 = network.stores.e_nom_opt[30:].rename("capacity")
    
    # Take maximum SOC from storage units
    storage_energy_2 = network.storage_units_t.state_of_charge.max().rename("capacity")
    
    # Combine all storage units
    storage_energy = pd.concat([storage_energy_1, storage_energy_2]).sort_index(axis=0)
    
    # Create list for names to group by 
    storage_energy_names = []
    storage_energy_tech = []
    
    # Add country code and tech-type to different lists
    for i in np.arange(0,len(storage_energy.index)):
        storage_energy_names.append(storage_energy.index[i][0:2])
        storage_energy_tech.append(storage_energy.index[i][3:])
    
    # Make two DataFrames to turn into the series later, with the correct index names
    part1_energy = pd.DataFrame(data=storage_energy.values, index=storage_energy_names)
    part1_energy.index.name = 'bus'
    
    part2_energy = pd.DataFrame(data=storage_energy.values, index=storage_energy_tech)
    part2_energy.index.name = 'carrier'
    
    # Make a series for correct input-type for plotting
    data_storage_energy = pd.Series(data=part1_energy.values[:,0], index=[part1_energy.index, part2_energy.index])
    
    # Down-scale values
    data_storage_energy = data_storage_energy * 1e-7
    
    size = 15.0
    plt.figure(figsize=[size, size], dpi=200)
    network.plot(bus_sizes=data_storage_energy, bus_colors={'H2 Store' : 'purple', 
                                                            'PHS' : 'aqua',
                                                            'battery' : 'springgreen', 
                                                            'gas Store' : 'red', 
                                                            'hydro' : 'lightcoral'}, projection=(ccrs.PlateCarree()))#, link_colors='black')
    
    plt.suptitle(file, fontsize=26, x=.55, y=0.84)
    plt.title("Installed storage energy capacity [MWh]", fontsize=20)
    legend_elements = [Line2D([0], [0], marker='o', color='white', label='H2 Storage', markerfacecolor='purple', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='Battery', markerfacecolor='springgreen', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='PHS', markerfacecolor='aqua', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='Hydro', markerfacecolor='lightcoral', markersize=12)]              
    
    plt.legend(handles=legend_elements, fontsize=15, bbox_to_anchor = (0.19,0.96), framealpha=1)

    
    # STORAGE POWER CAPACITY (MW)
    # Take storage units from network.stores (minus gas)
    #storage_power_1 = network.stores.e_nom_opt[30:].rename("capacity")
    storage_power_1 = network.stores_t.p.max()[30:].rename("capacity")
    
    # Take maximum SOC from storage units
    storage_power_2 = network.storage_units.p_nom.rename("capacity")
    
    # Combine all storage units
    storage_power = pd.concat([storage_power_1, storage_power_2]).sort_index(axis=0)
    
    # Create list for names to group by 
    storage_power_names = []
    storage_power_tech = []
    
    # Add country code and tech-type to different lists
    for i in np.arange(0,len(storage_power.index)):
        storage_power_names.append(storage_power.index[i][0:2])
        storage_power_tech.append(storage_power.index[i][3:])
    
    # Make two DataFrames to turn into the series later, with the correct index names
    part1_power = pd.DataFrame(data=storage_power.values, index=storage_power_names)
    part1_power.index.name = 'bus'
    
    part2_power = pd.DataFrame(data=storage_power.values, index=storage_power_tech)
    part2_power.index.name = 'carrier'
    
    # Make a series for correct input-type for plotting
    data_storage_power = pd.Series(data=part1_power.values[:,0], index=[part1_power.index, part2_power.index])
    
    # Down-scale values
    data_storage_power = data_storage_power * 6e-5
    
    size = 15.0
    plt.figure(figsize=[size, size], dpi=200)
    network.plot(bus_sizes=data_storage_power, bus_colors={'H2 Store' : 'purple', 
                                                           'PHS' : 'aqua',
                                                           'battery' : 'springgreen', 
                                                           'gas Store' : 'red', 
                                                           'hydro' : 'lightcoral'}, projection=(ccrs.PlateCarree()))#, link_colors='black')
    
    plt.suptitle(file, fontsize=26, x=.55, y=0.84)
    plt.title("Installed storage power capacity [MW]", fontsize=20)
    legend_elements = [Line2D([0], [0], marker='o', color='white', label='H2 Storage', markerfacecolor='purple', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='Battery', markerfacecolor='springgreen', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='PHS', markerfacecolor='aqua', markersize=12),
                        Line2D([0], [0], marker='o', color='white', label='Hydro', markerfacecolor='lightcoral', markersize=12)]              
    
    plt.legend(handles=legend_elements, fontsize=15, bbox_to_anchor = (0.19,0.96), framealpha=1)

    break

# To describe how much % each technology contributes
no_wind = 0
no_solar = 0
no_ror = 0
for i in np.arange(0,len(data)):
    x = data.index[i][1]
    
    if ( (x == 'onwind') or (x == 'offwind') ):
        no_wind += data[i]
    
    elif (x == 'solar'):
        no_solar += data[i]
    
    elif (x == 'ror'):
        no_ror += data[i]

print("Generator capacity ([MW]) distribution:")
print('Wind: ' + str(round(no_wind/data.sum() * 100,3)) + ' %.')
print('Solar: ' + str(round(no_solar/data.sum() * 100,3)) + ' %.')
print('ROR: ' + str(round(no_ror/data.sum() * 100,3)) + ' %.')

no_battery_energy = 0
no_H2_energy = 0
no_hydro_energy = 0

for i in np.arange(0,len(data_storage_energy)):
    x = data_storage_energy.index[i][1]
    
    if (x == 'battery'):
        no_battery_energy += data_storage_energy[i]
    
    elif (x == 'H2 Store'):
        no_H2_energy += data_storage_energy[i]
        
    elif ((x == 'PHS') or (x == 'hydro')):
        no_hydro_energy += data_storage_energy[i]

print('\n')
print("Storage energy capacity ([MWh]) distribution:")        
print('Battery energy: ' + str(round(no_battery_energy/data_storage_energy.sum() * 100,3)) + ' %.')
print('H2 energy: ' + str(round(no_H2_energy/data_storage_energy.sum() * 100,3)) + ' %.')
print('PHS + Hydro energy: ' + str(round(no_hydro_energy/data_storage_energy.sum() * 100,3)) + ' %.')

no_battery_power = 0
no_H2_power = 0
no_hydro_power = 0

for i in np.arange(0,len(data_storage_power)):
    x = data_storage_power.index[i][1]
    
    if (x == 'battery'):
        no_battery_power += data_storage_power[i]
    
    elif (x == 'H2 Store'):
        no_H2_power += data_storage_power[i]
        
    elif ((x == 'PHS') or (x == 'hydro')):
        no_hydro_power += data_storage_power[i]

print('\n')
print("Storage power capacity ([MW]) distribution:")        
print('Battery: ' + str(round(no_battery_power/data_storage_power.sum() * 100,3)) + ' %.')
print('H2: ' + str(round(no_H2_power/data_storage_power.sum() * 100,3)) + ' %.')
print('PHS + Hydro: ' + str(round(no_hydro_power/data_storage_power.sum() * 100,3)) + ' %.')
    
