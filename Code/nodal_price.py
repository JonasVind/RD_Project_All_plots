# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:10:13 2021

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

#%% Data Import

# Filename:
#filename = "Data\postnetwork-elec_only_0.125_0.6.h5" # 40% CO2
filename = "Data\postnetwork-elec_only_0.125_0.05.h5" # 95% CO2 constrain
# Import network
network = pypsa.Network(filename)

print("\nFile loaded: ", filename)

#%%%

price = network.buses_t.marginal_price

plt.figure()
plt.plot(price['DK'][0:1000])


#%%


original_costs = pd.read_csv('data/costs_2030.csv', index_col=(0,1))
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

# Calculated as FOM (in percent) times investment cost (in EUR per MW) times capacity. Hence, capital investment cost is ignored. 
cost['hydro'] = (original_costs.loc['hydro','FOM'].value/1e2 + annuity(original_costs.loc['hydro','lifetime'].value, 0.04)) *\
				   (original_costs.loc['hydro','investment'].value*1e3) *\
				   network.storage_units.loc[network.storage_units.group=="hydro","p_nom"].rename(lambda x : x[:2])
cost['ror'] = (original_costs.loc['ror','FOM'].value/1e2 + annuity(original_costs.loc['hydro','lifetime'].value, 0.04)) *\
				 (original_costs.loc['ror','investment'].value*1e3) *\
				 network.generators.loc[network.generators.group=="ror","p_nom"].rename(lambda x : x[:2])
cost['phs'] = (original_costs.loc['PHS','FOM'].value/1e2 + annuity(original_costs.loc['hydro','lifetime'].value, 0.04)) *\
				 (original_costs.loc['PHS','investment'].value*1e3) *\
				 network.storage_units.loc[network.storage_units.group=="PHS","p_nom"].rename(lambda x : x[:2])

battery_links = (network.links.filter(like='battery charger',axis=0).p_nom_opt * network.links.filter(like='battery charger',axis=0).capital_cost).rename(lambda x : x[:2])
battery_stores = (network.stores.filter(like='battery',axis=0).e_nom_opt * network.stores.filter(like='battery',axis=0).capital_cost).rename(lambda x : x[:2])
cost['battery'] = battery_links + battery_stores

hydrogen_links_1 = (network.links.filter(like='H2 Electrolysis',axis=0).p_nom_opt * network.links.filter(like='H2 Electrolysis',axis=0).capital_cost).rename(lambda x : x[:2])
hydrogen_links_2 = (network.links.filter(like='H2 Fuel Cell',axis=0).p_nom_opt * network.links.filter(like='H2 Fuel Cell',axis=0).capital_cost).rename(lambda x : x[:2])
hydrogen_stores_a = (network.stores.filter(like='H2 Store tank',axis=0).e_nom_opt * network.stores.filter(like='H2 Store tank',axis=0).capital_cost).rename(lambda x : x[:2])
hydrogen_stores_b = (network.stores.filter(like='H2 Store underground',axis=0).e_nom_opt * network.stores.filter(like='H2 Store underground',axis=0).capital_cost).rename(lambda x : x[:2])
cost['hydrogen storage'] = hydrogen_links_1 + hydrogen_links_2 + hydrogen_stores_a + hydrogen_stores_b

cost['methanation'] = (network.links.filter(like='Sabatier',axis=0).p_nom_opt * network.links.filter(like='Sabatier',axis=0).capital_cost).rename(lambda x : x[:2])

gas_M_1 = (network.stores_t.p.sum(axis=0).filter(like='gas',axis=0) * network.stores.filter(like='gas',axis=0).marginal_cost).rename(lambda x : x[:2])
gas_M_2 = (network.links_t.p0.sum(axis=0).filter(like='OCGT',axis=0) * network.links.filter(like='OCGT',axis=0).marginal_cost).rename(lambda x : x[:2])
gas_C = (network.links.filter(like='OCGT',axis=0).p_nom_opt * network.links.filter(like='OCGT',axis=0).capital_cost).rename(lambda x : x[:2])
cost['gas'] = gas_M_1 + gas_M_2 + gas_C

coal_M_1 = (network.stores_t.p.sum(axis=0).filter(like='coal',axis=0) * network.stores.filter(like='coal',axis=0).marginal_cost).rename(lambda x : x[:2])
coal_M_2 = (network.links_t.p0.sum(axis=0).filter(like='coal',axis=0) * network.links.filter(like='coal',axis=0).marginal_cost).rename(lambda x : x[:2])
coal_C = (network.links.filter(like='coal',axis=0).p_nom_opt * network.links.filter(like='coal',axis=0).capital_cost).rename(lambda x : x[:2])
cost['coal'] = coal_M_1 + coal_M_2 + coal_C

lignite_M_1 = (network.stores_t.p.sum(axis=0).filter(like='lignite',axis=0) * network.stores.filter(like='lignite',axis=0).marginal_cost).rename(lambda x : x[:2])
lignite_M_2 = (network.links_t.p0.sum(axis=0).filter(like='lignite',axis=0) * network.links.filter(like='lignite',axis=0).marginal_cost).rename(lambda x : x[:2])
lignite_C = (network.links.filter(like='lignite',axis=0).p_nom_opt * network.links.filter(like='lignite',axis=0).capital_cost).rename(lambda x : x[:2])
cost['lignite'] = lignite_M_1 + lignite_M_2 + lignite_C

nuclear_M_1 = (network.stores.filter(like='nuclear',axis=0).e_nom_opt * network.stores.filter(like='nuclear',axis=0).marginal_cost).rename(lambda x : x[:2])
nuclear_M_2 = (network.links_t.p0.sum(axis=0).filter(like='nuclear',axis=0) * network.links.filter(like='nuclear',axis=0).marginal_cost).rename(lambda x : x[:2])
nuclear_C = (network.links.filter(like='nuclear',axis=0).p_nom_opt * network.links.filter(like='nuclear',axis=0).capital_cost).rename(lambda x : x[:2])
cost['nuclear'] = nuclear_M_1 + nuclear_M_2 + nuclear_C

## Transmission.
# Need to be split onto each country which can be done in many ways. Example: 50/50 on the two connected countries. Not fair for transition countries like CH.
cost['transmission'] = (network.links[network.links.p_min_pu == -1].p_nom_opt * network.links[network.links.p_min_pu == -1].capital_cost).sum() *\
						  (network.loads_t.p.sum(axis=0) / network.loads_t.p.sum().sum())  # transmission cost split equally by load. 

cost = cost.fillna(0)
cost = cost.stack()