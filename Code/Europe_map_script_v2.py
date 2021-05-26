import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import pypsa
import tables

# Folder name of data files
directory = "Data\\"


# Name of file (must be in correct folder location)
filename = ["postnetwork-elec_only_0.125_0.05.h5",
             "postnetwork-elec_only_0.125_0.1.h5",
             "postnetwork-elec_only_0.125_0.2.h5",
             "postnetwork-elec_only_0.125_0.3.h5",
             "postnetwork-elec_only_0.125_0.4.h5",
             "postnetwork-elec_only_0.125_0.5.h5",
             "postnetwork-elec_only_0.125_0.6.h5"]

network = pypsa.Network(directory+filename[0])

# Array of 30 points load
load = network.loads_t.p_set
    
# Array of 30 point summed generation
generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
    
# Array of mismatch for all 30 points
mismatch = generation - load

# Transpose the data to get 30-by-30 instead of 8760-by-8760
mismatch = np.transpose(mismatch)

# Mean of mismatch (axis tells if its the average of the rows or columns)
mismatch_avg = np.mean(mismatch, axis=1)

# Subtract the average for a mean centered distribution 
#B = mismatch - np.tile(mismatch_avg,(30,1)).T
B = mismatch - np.tile(mismatch_avg,(8760,1)).T

# Covariance matrix
C = np.cov(B)

# Eigen vector and values
eigen_values, eigen_vectors = np.linalg.eig(C)

# Webside whos describing the process
# https://towardsdatascience.com/pca-with-numpy-58917c1d0391

# Creating array to describe variance explained by each of the eigen values
variance_explained = []

for i in eigen_values:
     variance_explained.append((i/sum(eigen_values))*100)

# Cumulative variance explained
variance_explained_cumulative = np.cumsum(variance_explained)

# Get the names of the data
data_names = network.loads_t.p.columns

# Define the eigen vectors in a new variable with names
VT = pd.DataFrame(data=eigen_vectors, index=data_names)

# Swap sign infront of values
#VT = VT * (-1)


#%%
##############################################################################
#Plot figure
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
PC_NO = 1

# Start for-loop
for country in countries:
    
    #If the countrie is in the list of the european countries not to include, color it gray
    if country.attributes['ISO_A2'] in europe_not_included:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                          facecolor=(0.8, 0.8, 0.8), alpha=1.00, linewidth=0.15, 
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
                          facecolor=(0.8, 0.8, 0.8), alpha=1.00, linewidth=0.15, 
                          edgecolor="black", label=country.attributes['ADM0_A3'])
                
title = ()
    
plt.title("Colormap of Principle Component")
plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%'], loc='upper left')
test = np.zeros([30,30])
test[0,0]=-1
test[0,29]=1

cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0,0),(1,0.333,0.333),(1,0.666,0.666), 'white',(0.666,1,0.666),(0.333,1,0.333),(0,1,0),(0,1,0)])

cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
im = ax.imshow(test,cmap=cmap)                
plt.colorbar(im,cax=cax)
plt.suptitle(filename[0],fontsize=20,x=.51,y=0.938)


plt.show()

