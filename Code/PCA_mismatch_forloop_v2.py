# -*- coding: utf-8 -*-
"""
Created on...

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
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import math
from sklearn.preprocessing import Normalizer

#%% Timer

t0 = time.time() # Start a timer

#%% Disable future warnings

" Disable future warnings "

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#%% What to plot

PCA_dis_plot = True # Plot showcasing distribution of PCA
hour_plot = True # hour and seasonal plot
PCA_bar_plot = True # 3 first PCA as bar plot
gen_plot = True # network plot with renewable generators
FFT_plot = True # Fast Fourier Transform plot
map_plot = False # Map for PCA

#%% Data Import

# Filename:
# Name of file (must be in same folder)
filename = ["postnetwork-elec_only_0.375_0.05.h5",
            "postnetwork-elec_only_0.25_0.05.h5",
            "postnetwork-elec_only_0.0625_0.05.h5",
            "postnetwork-elec_only_0_0.05.h5",
            "postnetwork-elec_only_0.125_0.05.h5",
            "postnetwork-elec_only_0.125_0.1.h5",
            "postnetwork-elec_only_0.125_0.2.h5",
            "postnetwork-elec_only_0.125_0.3.h5",
            "postnetwork-elec_only_0.125_0.4.h5",
            "postnetwork-elec_only_0.125_0.5.h5",
            "postnetwork-elec_only_0.125_0.6.h5"]
dic = 'Data\\' # Location of files

print("Files loaded")

#%% Starting for loop

# Counter for number of PCA components (integer)
PCA_mismatch_counter = np.zeros(len(filename)).astype(int)
PCA_price_counter = np.zeros(len(filename)).astype(int)

# Creates a for-loop of all the files
for i in np.arange(0,len(filename)):
    # User info:
    print("\nCalculating for: ",filename[i])
    # Create network from previous file
    n = pypsa.Network(dic+filename[i])
    
    #%%% Calculating mismatch

    # Defining dispatched electricity generation
    generation = n.generators_t.p.groupby(n.generators.bus, axis=1).sum()

    # Defining load 
    load = n.loads_t.p_set

    # Calculate mismatch
    #mismatch = p_t - load # Using available electricity generation
    mismatch = generation - load # Using dispatched electricity generation

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
    
    # Plot S out to see how much data can be descriped with PCA
    if PCA_dis_plot == True:
        # Check what PC is needed for 0.95
        PC_count = 0
        j = 0
        while PC_count <= 0.95:
            j += 1
            PC_count = sum(eig_val[0:j])
            PC_count = round(PC_count,3)
        # Plotting:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax1.semilogy(eig_val,'-o',color='k')
        ax2 = fig1.add_subplot(122)
        ax2.plot(np.cumsum(eig_val)/np.sum(eig_val),'-o',color='k')
        plt.suptitle(filename[i],fontsize=16,x=.51,y=0.95) #,x=.51,y=1.07
        text = "PC1-PC"+str(j)+" = "+str(PC_count)
        plt.text(4,0.8,text)

    #%%% U dataframe and hour average
    
    # Save time index
    data_index = load.index
    # Save U as a dataframe with time series
    T = pd.DataFrame(data=T,index=data_index)

    # Average hour and day
    T_avg_hour = T.groupby(data_index.hour).mean() # Hour
    T_avg_day = T.groupby([data_index.month,data_index.day]).mean() # Day


    #%%% Plotting different PC
    if hour_plot == True:
        plt.figure(figsize=(16,10))
        plt.subplot(2,2,1)
        plt.plot(T_avg_hour[0],label='k=1')
        plt.plot(T_avg_hour[1],label='k=2')
        plt.plot(T_avg_hour[2],label='k=3')
        #plt.plot(U_avg_hour[3],label='k=4')
        #plt.plot(U_avg_hour[4],label='k=5')
        #plt.ylim(-0.02,0.02)
        plt.xticks(ticks=range(0,24,2))
        plt.legend(loc='upper right',bbox_to_anchor=(1,1))
        plt.xlabel("Hours")
        plt.ylabel("a_k interday")
        plt.title("Hourly average for k-values for 2015 ")
        
        # X for year plot
        x_ax = range(len(T_avg_day[0]))
        
        plt.subplot(2,2,2)
        plt.plot(x_ax,T_avg_day[0]+1.0,label='k=1\n shifted +1.0')
        plt.plot(x_ax,T_avg_day[1],label='k=2')
        plt.plot(x_ax,T_avg_day[2]-1.0,label='k=3\n shifted -1.0')
        #plt.plot(x_ax,U_avg_day[3],label='k=4')
        #plt.plot(x_ax,U_avg_day[4],label='k=5')
        #plt.ylim(-0.13,0.025)
        plt.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.xlabel("day")
        plt.ylabel("a_k seasonal")
        plt.title("daily average for k-values for 2015 ")
        
        plt.subplot(2,2,3)
        plt.plot(T_avg_hour[3],label='k=4',color="c")
        plt.plot(T_avg_hour[4],label='k=5',color="m")
        plt.plot(T_avg_hour[5],label='k=6',color="y")
        #plt.plot(U_avg_hour[3],label='k=4')
        #plt.plot(U_avg_hour[4],label='k=5')
        #plt.ylim(-0.02,0.02)
        plt.xticks(ticks=range(0,24,2))
        plt.legend(loc='upper right',bbox_to_anchor=(1,1))
        plt.xlabel("Hours")
        plt.ylabel("a_k interday")
        plt.title("Hourly average for k-values for 2015 ")
        
        # X for year plot
        x_ax = range(len(T_avg_day[0]))
        
        plt.subplot(2,2,4)
        plt.plot(x_ax,T_avg_day[3]+0.5,label='k=4\n shifted +0.5',color="c")
        plt.plot(x_ax,T_avg_day[4],label='k=5',color="m")
        plt.plot(x_ax,T_avg_day[5]-0.5,label='k=6\n shifted -0.5',color="y")
        #plt.plot(x_ax,U_avg_day[3],label='k=4')
        #plt.plot(x_ax,U_avg_day[4],label='k=5')
        #plt.ylim(-0.13,0.025)
        plt.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.xlabel("day")
        plt.ylabel("a_k seasonal")
        plt.title("daily average for k-values for 2015 ")
        
    
        title_name = 'Data from: ',filename[i]
        plt.suptitle(filename[i],fontsize=20,x=.51,y=0.932) #,x=.51,y=1.07
        
        plt.show(all)

    #%%% Plotting different principle components effect on countries
    
    if PCA_bar_plot == True:
        countries = n.loads_t.p_set.columns
        eig_vec_0 = eig_vec[:,0]
        eig_vec_1 = eig_vec[:,1]
        eig_vec_2 = eig_vec[:,2]
        
        T_data = pd.DataFrame(data=eig_vec,index=countries)
        
        plt.figure(figsize=(14,10))
        plt.subplot(3,1,1)
        plt.bar(countries,eig_vec_0)
        plt.ylim(-1,1)
        #plt.xlabel("Countries")
        plt.ylabel("Principle complonent value")
        plt.title("principle component 1")
        
        plt.subplot(3,1,2)
        plt.bar(countries,eig_vec_1)
        plt.ylim(-1,1)
        #plt.xlabel("Countries")
        plt.ylabel("Principle complonent value")
        plt.title("principle component 2")
        
        plt.subplot(3,1,3)
        plt.bar(countries,eig_vec_2)
        plt.ylim(-1,1)
        #plt.xlabel("Countries")
        plt.ylabel("Principle complonent value")
        plt.title("principle component 3")
        
        plt.subplots_adjust(wspace=0, hspace=0.35)
        title_name = 'Data from: ',filename[i]
        plt.suptitle(filename[i],fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07
    
        plt.show(all)

    #%% Plotting tech map
    # Code from Leon
    if gen_plot == True:
        n.buses_copy = n.buses.copy()
    
        n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    
        techs = ["onwind", "solar", "ror"]
        
        fig, axes = plt.subplots(nrows=1, ncols=len(techs), subplot_kw={"projection":ccrs.PlateCarree()})
        
        size = 10
        
        fig.set_size_inches(size*len(techs), size)
        
        
        
        for j,tech in enumerate(techs):
        
            ax = axes[j]
        
            gens = n.generators[n.generators.carrier == tech]
        
            gen_distribution = gens.groupby("bus").sum()["p_nom_opt"].reindex(n.buses.index, fill_value=0.)
        
            n.plot(ax=ax, bus_sizes=0.02*gen_distribution)
        
            ax.set_title(tech)
        plt.suptitle(filename[i],fontsize=20,x=.51,y=0.81) #,x=.51,y=1.07
        plt.show(all)

    #%% FFT
    if FFT_plot == True:
        
        plt.figure(figsize=(18,12))
        plt.subplot(3,2,1)
        freq=np.fft.fftfreq(len(T[0]))  
        FFT=np.fft.fft(T[0])
        FFT[0]=0
        FFT=abs(FFT)/max(abs(FFT))
        plt.plot(1/freq,FFT)
        plt.xscale('log')
        plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        #plt.legend(loc='upper right')
        plt.text(10,0.9,"1/2 Day",ha='right')
        plt.text(22,0.9,"Day",ha='right')
        plt.text(22*7,0.9,"Week",ha='right')
        plt.text(22*7*4,0.9,"Month",ha='right')
        plt.text(22*365,0.9,"Year",ha='right')
        plt.xlabel('Hours')
        plt.title('Fourier Power Spectra for PC1')
        
        plt.subplot(3,2,2)
        freq=np.fft.fftfreq(len(T[1]))  
        FFT=np.fft.fft(T[1])
        FFT[0]=0
        FFT=abs(FFT)/max(abs(FFT))
        plt.plot(1/freq,FFT)
        plt.xscale('log')
        plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        #plt.legend(loc='upper right')
        plt.text(10,0.9,"1/2 Day",ha='right')
        plt.text(22,0.9,"Day",ha='right')
        plt.text(22*7,0.9,"Week",ha='right')
        plt.text(22*7*4,0.9,"Month",ha='right')
        plt.text(22*365,0.9,"Year",ha='right')
        plt.xlabel('Hours')
        plt.title('Fourier Power Spectra for PC2')
        
        plt.subplot(3,2,3)
        freq=np.fft.fftfreq(len(T[2]))  
        FFT=np.fft.fft(T[2])
        FFT[0]=0
        FFT=abs(FFT)/max(abs(FFT))
        plt.plot(1/freq,FFT)
        plt.xscale('log')
        plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        #plt.legend(loc='upper right')
        plt.text(10,0.9,"1/2 Day",ha='right')
        plt.text(22,0.9,"Day",ha='right')
        plt.text(22*7,0.9,"Week",ha='right')
        plt.text(22*7*4,0.9,"Month",ha='right')
        plt.text(22*365,0.9,"Year",ha='right')
        plt.xlabel('Hours')
        plt.title('Fourier Power Spectra for PC3')
        
        plt.subplot(3,2,4)
        freq=np.fft.fftfreq(len(T[3]))  
        FFT=np.fft.fft(T[3])
        FFT[0]=0
        FFT=abs(FFT)/max(abs(FFT))
        plt.plot(1/freq,FFT)
        plt.xscale('log')
        plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        #plt.legend(loc='upper right')
        plt.text(10,0.9,"1/2 Day",ha='right')
        plt.text(22,0.9,"Day",ha='right')
        plt.text(22*7,0.9,"Week",ha='right')
        plt.text(22*7*4,0.9,"Month",ha='right')
        plt.text(22*365,0.9,"Year",ha='right')
        plt.xlabel('Hours')
        plt.title('Fourier Power Spectra for PC4')
        
        plt.subplot(3,2,5)
        freq=np.fft.fftfreq(len(T[4]))  
        FFT=np.fft.fft(T[2])
        FFT[0]=0
        FFT=abs(FFT)/max(abs(FFT))
        plt.plot(1/freq,FFT)
        plt.xscale('log')
        plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        #plt.legend(loc='upper right')
        plt.text(10,0.9,"1/2 Day",ha='right')
        plt.text(22,0.9,"Day",ha='right')
        plt.text(22*7,0.9,"Week",ha='right')
        plt.text(22*7*4,0.9,"Month",ha='right')
        plt.text(22*365,0.9,"Year",ha='right')
        plt.xlabel('Hours')
        plt.title('Fourier Power Spectra for PC5')
        
        plt.subplot(3,2,6)
        freq=np.fft.fftfreq(len(T[5]))  
        FFT=np.fft.fft(T[3])
        FFT[0]=0
        FFT=abs(FFT)/max(abs(FFT))
        plt.plot(1/freq,FFT)
        plt.xscale('log')
        plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
        #plt.legend(loc='upper right')
        plt.text(10,0.9,"1/2 Day",ha='right')
        plt.text(22,0.9,"Day",ha='right')
        plt.text(22*7,0.9,"Week",ha='right')
        plt.text(22*7*4,0.9,"Month",ha='right')
        plt.text(22*365,0.9,"Year",ha='right')
        plt.xlabel('Hours')
        plt.title('Fourier Power Spectra for PC6')        
        
        plt.subplots_adjust(wspace=0, hspace=0.28)
        title_name = 'Data from: ',filename
    
        plt.suptitle(filename[i],fontsize=20,x=.51,y=0.93) #,x=.51,y=1.07
    
    plt.show(all)
    
    
    #%%% Plotting colormap
    
    if map_plot == True:
        # Eigen vector and values
        eigen_values = eig_val
        eigen_vectors = eig_vec
        
        # Webside whos describing the process
        # https://towardsdatascience.com/pca-with-numpy-58917c1d0391
        
        # Creating array to describe variance explained by each of the eigen values
        variance_explained = []
        
        for j in eigen_values:
             variance_explained.append((j/sum(eigen_values))*100)
        
        # Cumulative variance explained
        variance_explained_cumulative = np.cumsum(variance_explained)
        
        # Get the names of the data
        data_names = n.loads_t.p.columns
        
        # Define the eigen vectors in a new variable with names
        VT = pd.DataFrame(data=eigen_vectors, index=data_names)
        
        for k in range(6):
        
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
            
            # Determine name_loop variable
            name_loop = 'start'
            
            # PC number showed (1 to 30)
            PC_NO = k+1
            
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
            plt.suptitle(filename[i],fontsize=20,x=.51,y=0.938)
    
        plt.show(all)

#%%% Done
print("\nDone Calculating")

#%% Finish

t1 = time.time() # End timer

total_time = round(t1-t0)
total_time_min = math.floor(total_time/60)
total_time_sec = round(total_time-(total_time_min*60))

print("\n \nThe code is now done running. It took %s min and %s sec." %(total_time_min,total_time_sec))


