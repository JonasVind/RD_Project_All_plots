# Imported libraries
import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import time
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

#----------------------------------------------------------------------------%
# CO2 CONSTRAINTS
# Load data - CO2 constraint
# Folder name of data files
directory = "C:/Users/jense/.spyder-py3/data_files/elec_only/"
# Name of file (must be in correct folder location)
filename = ["postnetwork-elec_only_0.125_0.6.h5",
            "postnetwork-elec_only_0.125_0.5.h5",
            "postnetwork-elec_only_0.125_0.4.h5",
            "postnetwork-elec_only_0.125_0.3.h5",
            "postnetwork-elec_only_0.125_0.2.h5",
            "postnetwork-elec_only_0.125_0.1.h5",
            "postnetwork-elec_only_0.125_0.05.h5"]

filename = filename[-1]

# Network
network = pypsa.Network(directory+filename)

# Get the names of the data
data_names = network.loads_t.p.columns

# Get time stamps
time_index = network.loads_t.p.index

# Array of 30 country load
load = network.loads_t.p_set

# Dataframe (array) for different generator technologies
wind = pd.DataFrame(np.zeros([8760, 30]), index=time_index, columns=(data_names))
solar = pd.DataFrame(np.zeros([8760, 30]), index=time_index, columns=(data_names))
hydro = pd.DataFrame(np.zeros([8760, 30]), index=time_index, columns=(data_names))

# Counter for positioning in generator data
counter = 0
for i in network.generators.index:
    
    # Current value to insert into correct array and position
    value = np.array(network.generators_t.p)[:,counter]
    
    # Check for wind, solar and hydro
    if (i[-4:] == "wind"):
        wind[i[0:2]] += value
    
    elif (i[-5:] == "solar"):
        solar[i[0:2]] += value
    
    elif (i[-3:] == "ror"):
        hydro[i[0:2]] += value
    
    # Increase value of counter by 1
    counter += 1

# List of prices
prices = network.buses_t.marginal_price

# List of nodal prices for each country
country_price = prices[data_names] # [€/MWh]
country_price_gas = prices[(data_names + ' gas')] # [€/MWh]
country_price_H2 = prices[(data_names + ' H2')] # [€/MWh]
country_price_battery = prices[(data_names + ' battery')] # [€/MWh]

# Sum up all the prices into one for every country
nodal_price = country_price.values + country_price_gas.values + country_price_H2.values + country_price_battery.values
nodal_price = pd.DataFrame(data=nodal_price, index=time_index, columns=data_names)

# Total price of the system (for comparison)
total_price = network.objective

# Mean of nodal price
nodal_price_avg = np.mean(nodal_price, axis=0)

# Subtract the average for a mean centered distribution 
B = nodal_price.values - nodal_price_avg.values

# Normalisation constant
c = (1 / (np.sqrt( np.sum( np.mean( ( (nodal_price - nodal_price_avg)**2 ), axis=0 ) ) ) ) )
    
# Covariance matrix "It is a measure of how much each of the dimensions varies from the mean with respect to each other."
C = np.cov(B.T)

# Stops if C is larger than [30 x 30]
assert np.size(C) <= 900, "C is too big"

# Eigen vector and values
eigen_values, eigen_vectors = eig(C)

# Creating array to describe variance explained by each of the eigen values
variance_explained = []

for i in eigen_values:
     variance_explained.append((i/sum(eigen_values))*100)

# Cumulative variance explained
variance_explained_cumulative = np.cumsum(variance_explained)

# Define the eigen vectors in a new variable with names
VT = pd.DataFrame(data=eigen_vectors, index=data_names)




#%%
freq = '1W'
nodal_price_day = nodal_price.resample(rule=freq).sum()
load_day = load.resample(rule=freq).sum()
wind_day = wind.resample(rule=freq).sum() / 1000
solar_day = solar.resample(rule=freq).sum() / 1000
hydro_day = hydro.resample(rule=freq).sum() / 1000
#%%

fig1, ax1 = plt.subplots()
color = 'red'
ax1.set_xlabel('Date (sample interval: ' + freq + ')')
ax1.set_ylabel('Avg. Nodal price $[ \dfrac{€}{MWh} ]$', color=color)
lns1 = ax1.plot(nodal_price_day.index, nodal_price_day.mean(axis=1), color='red', label='Nodal price')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(bbox_to_anchor = (1.12,1))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'black'
ax2.set_ylabel('Avg. Nodal energy generation [GWh]', color=color)  # we already handled the x-label with ax1
lns2 = ax2.plot(wind_day.index, wind_day.mean(axis=1), color='skyblue', linestyle='--', label='Wind generation')
lns3 = ax2.plot(solar_day.index, solar_day.mean(axis=1), color='gold', linestyle='--', label='Solar generation')
lns4 = ax2.plot(hydro_day.index, hydro_day.mean(axis=1), color='aqua', linestyle='--', label='ROR generation')
ax2.tick_params(axis='y', labelcolor=color)

# Combine axes to get all labels
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor = (1.14,1))

plt.title(filename)
plt.grid(axis='y')
plt.show()

#%%
##############################################################################
#Plot country figure
fig = plt.figure(figsize=(9, 9))
ax = plt.axes(projection=cartopy.crs.TransverseMercator(20))
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
ax.coastlines(resolution='10m')
ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.6,0.8,1), alpha=0.30)
ax.set_extent ((-9.5, 32, 35, 71), cartopy.crs.PlateCarree())
ax.gridlines()


# List of european countries not included in the data
europe_not_included = {'AD', 'AL','AX','BY', 'FO', 'GG', 'GI', 'IM', 'IS', 
                       'JE', 'LI', 'MC', 'MD', 'ME', 'MK', 'MT', 'RU', 'SM', 
                       'UA', 'VA', 'XK'}

# Create shapereader file name
shpfilename = shpreader.natural_earth(resolution='10m',
                                      category='cultural',
                                      name='admin_0_countries')

# Read the shapereader file
reader = shpreader.Reader(shpfilename)

# Record the reader
countries = reader.records()

# Print keys() used to 'index' variable
#print(country.attributes.keys())

# Determine name_loop variable
name_loop = 'start'

# PC number showed (1 to 30)
PC_NO = 2

# Start for-loop
for country in countries:
    
    #If the countrie is in the list of the european countries not to include, color it gray
    if country.attributes['ISO_A2'] in europe_not_included:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                          facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                          edgecolor="black", label=country.attributes['ADM0_A3'])
    
    elif country.attributes['REGION_UN'] == 'Europe':
        if country.attributes['NAME'] == 'Norway':
            name_loop = 'NO'
            
        elif country.attributes['NAME'] == 'France':
            name_loop = 'FR'
            
        else:
            name_loop = country.attributes['ISO_A2']
        
        #print(name_loop)
        for country_PSA in VT.index.values:
            if country_PSA == name_loop:
                #print("Match!")
                color_value = VT.loc[country_PSA][PC_NO-1]
                #print(color_value)
                if color_value <= 0:
                    #Farv rød
                    color_value = np.absolute(color_value)*1.5
                    ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                          facecolor=(1, 0, 0), alpha=(np.min([color_value, 1])), linewidth=0.15, 
                          edgecolor="black", label=country.attributes['ADM0_A3'])
                    
                    
                else:
                    # Farv grøn
                    color_value = np.absolute(color_value)*1.5
                    ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                          facecolor=(0, 1, 0), alpha=(np.min([color_value, 1])), linewidth=0.15, 
                          edgecolor="black", label=country.attributes['ADM0_A3'])
            
            
    else:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                          facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                          edgecolor="black", label=country.attributes['ADM0_A3'])
                

title = ()
    
plt.title("Colormap of Principle Component for Electricity Nodal Prices")
plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%'], loc='upper left')
test = np.zeros([30,30])
test[0,0]=-1
test[0,29]=1

cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0,0),(1,0.333,0.333),(1,0.666,0.666), 'white',(0.666,1,0.666),(0.333,1,0.333),(0,1,0),(0,1,0)])

cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
im = ax.imshow(test,cmap=cmap)                
plt.colorbar(im,cax=cax)
plt.suptitle(filename,fontsize=20,x=.51,y=0.938)


plt.show()











#%%
def annuity(n,r):
    # n = number of periods
    # r = changes per period
    # Example
    # Calculate the annuity factor for an asset with lifetime n years and
    # discount rate of r with a total capital cost of 100€
    # e.g. annuity(20,0.05)*100€ = 8.024 € / Year
    # Over 20 years that result in a total cost of 8.024[€]*20[years] = 160.49[€]

    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n

original_costs = pd.read_csv('C:/Users/jense/.spyder-py3/data_files/costs_2030.csv', index_col=(0,1))
cost = pd.DataFrame(index=network.buses.index)

# Onwind and offwind have both capital and marginal expenses, solar has only capital investments.
onwind = (network.generators.filter(like='onwind',axis=0).p_nom_opt * network.generators.filter(like='onwind',axis=0).capital_cost).rename(lambda x : x[:2])
onwind = onwind.groupby(onwind.index).sum()  # group DE0+DE1+DE2, etc...
onwind_marg =(network.generators_t.p.sum(axis=0).filter(like='onwind',axis=0) * network.generators.filter(like='onwind',axis=0).marginal_cost).rename(lambda x : x[:2])
onwind_marg = onwind_marg.groupby(onwind_marg.index).sum()  # group DE0+DE1+DE2, etc.

offwind = (network.generators.filter(like='offwind',axis=0).p_nom_opt * network.generators.filter(like='offwind',axis=0).capital_cost).rename(lambda x : x[:2])
offwind = offwind.reindex(onwind.index,fill_value=0)  # fill empty values
offwind_marg = (network.generators_t.p.sum(axis=0).filter(like='offwind',axis=0) * network.generators.filter(like='offwind',axis=0).marginal_cost).rename(lambda x : x[:2])
offwind_marg = offwind_marg.reindex(onwind_marg.index,fill_value=0)  # fill empty values

solar = (network.generators.filter(like='solar',axis=0).p_nom_opt * network.generators.filter(like='solar',axis=0).capital_cost).rename(lambda x : x[:2])

cost['onwind'] = onwind + onwind_marg
cost['offwind'] = offwind + offwind_marg
cost['solar'] = solar

cost['hydro'] = (original_costs.loc['hydro','FOM'].value/1e2 + annuity(original_costs.loc['hydro','lifetime'].value, 0.04)) *\
				   (original_costs.loc['hydro','investment'].value*1e3) *\
 				   network.storage_units.loc[(data_names + ' hydro'),"p_nom"].rename(lambda x : x[:2])
                    
cost['ror'] = (original_costs.loc['ror','FOM'].value/1e2 + annuity(original_costs.loc['hydro','lifetime'].value, 0.04)) *\
 				 (original_costs.loc['ror','investment'].value*1e3) *\
 				 network.generators.loc[(data_names + ' ror'),"p_nom"].rename(lambda x : x[:2])
                  
cost['phs'] = (original_costs.loc['PHS','FOM'].value/1e2 + annuity(original_costs.loc['hydro','lifetime'].value, 0.04)) *\
 				 (original_costs.loc['PHS','investment'].value*1e3) *\
 				 network.storage_units.loc[(data_names + ' PHS'),"p_nom"].rename(lambda x : x[:2])

battery_links = (network.links.filter(like='battery charger',axis=0).p_nom_opt * network.links.filter(like='battery charger',axis=0).capital_cost).rename(lambda x : x[:2])
battery_stores = (network.stores.filter(like='battery',axis=0).e_nom_opt * network.stores.filter(like='battery',axis=0).capital_cost).rename(lambda x : x[:2])
cost['battery'] = battery_links + battery_stores

hydrogen_links_1 = (network.links.filter(like='H2 Electrolysis',axis=0).p_nom_opt * network.links.filter(like='H2 Electrolysis',axis=0).capital_cost).rename(lambda x : x[:2])
hydrogen_links_2 = (network.links.filter(like='H2 Fuel Cell',axis=0).p_nom_opt * network.links.filter(like='H2 Fuel Cell',axis=0).capital_cost).rename(lambda x : x[:2])
hydrogen_stores_a = (network.stores.filter(like='H2 Store tank',axis=0).e_nom_opt * network.stores.filter(like='H2 Store tank',axis=0).capital_cost).rename(lambda x : x[:2])
hydrogen_stores_b = (network.stores.filter(like='H2 Store underground',axis=0).e_nom_opt * network.stores.filter(like='H2 Store underground',axis=0).capital_cost).rename(lambda x : x[:2])
cost['hydrogen storage'] = hydrogen_links_1 + hydrogen_links_2 + hydrogen_stores_a + hydrogen_stores_b

gas_M_1 = (network.stores_t.p.sum(axis=0).filter(like='gas',axis=0) * network.stores.filter(like='gas',axis=0).marginal_cost).rename(lambda x : x[:2])
gas_M_2 = (network.links_t.p0.sum(axis=0).filter(like='OCGT',axis=0) * network.links.filter(like='OCGT',axis=0).marginal_cost).rename(lambda x : x[:2])
gas_C = (network.links.filter(like='OCGT',axis=0).p_nom_opt * network.links.filter(like='OCGT',axis=0).capital_cost).rename(lambda x : x[:2])
cost['gas'] = gas_M_1 + gas_M_2 + gas_C

## Transmission.
# Need to be split onto each country which can be done in many ways. Example: 50/50 on the two connected countries. Not fair for transition countries like CH.
cost['transmission'] = (network.links[network.links.p_min_pu == -1].p_nom_opt * network.links[network.links.p_min_pu == -1].capital_cost).sum() *\
						  (network.loads_t.p.sum(axis=0) / network.loads_t.p.sum().sum())  # transmission cost split equally by load. 

cost = cost.fillna(0)
cost = cost.stack()































