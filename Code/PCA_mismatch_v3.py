# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 08:48:54 2021

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
n = pypsa.Network(filename)

print("\nFile loaded: ", filename)


#%% Calculating mismatch

# Defining dispatched electricity generation
generation = n.generators_t.p.groupby(n.generators.bus, axis=1).sum()

# Defining load 
load = n.loads_t.p_set

# Calculate mismatch
mismatch = generation - load # Using available electricity generation
#mismatch = generation - load # Using dispatched electricity generation

# Collecting mismatch terms
gen_grouped = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum().values
mismatch_terms = pd.DataFrame({'wind': gen_grouped[:,0]+gen_grouped[:,1],
                                'ror': gen_grouped[:,2],
                                'solar': gen_grouped[:,3],
                                'load': load.sum(axis=1).values},index=mismatch.index)

# Plotting mismatch input:
x_fill = load.index[3000:3336]
y1_fill = 0
y2 = mismatch['ES'][3000:3336]
y2_fill = y2.values
# plotting
plt.figure(figsize=(12,4))
plt.plot(y2,label="mismatch",color='k',alpha=1,linewidth=1)
plt.fill_between(x_fill, y1_fill, y2_fill,
                 label='Positive\nmismatch',
                 where= y2_fill >= y1_fill,
                 color='g')
plt.fill_between(x_fill, y1_fill, y2_fill,
                 label='Negative\nmismatch',
                 where= y2_fill <= y1_fill,
                 color='r')
plt.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.grid(axis='y',alpha=0.5)
plt.title("Mismatch for Spain")

#%%% Steve Brunton
# Link: https://www.youtube.com/watch?v=fkf4IBRSeEc&ab_channel=SteveBrunton

X = mismatch

X_mean = np.mean(X,axis=0) # axis=0, mean at each colume 
X_mean = np.array(X_mean.values).reshape(30,1)

X_cent = np.subtract(X,X_mean.T)

c = 1/np.sqrt(np.sum(np.mean((X_cent.values)**2,axis=0)))

B = c*(X_cent.values)

#C_new = np.dot(B.T,B)

# Convariance
C = np.cov(B.T,bias=True)

assert np.size(C) <= 900, "C is too big"

eig_val, eig_vec = np.linalg.eig(C)

#T = B*eig_val
T = np.dot(B,eig_vec)

# Plot S out to see how much data can be descriped with PCA
if_plot = True
if if_plot == True:
    
    # Check what PC is needed for 0.95
    PC_count = 0
    j = 0
    while PC_count <= 0.95:
        j += 1
        PC_count = sum(eig_val[0:j])
        PC_count = round(PC_count,3)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.semilogy(eig_val,'-o',color='k')
    ax2 = fig1.add_subplot(122)
    ax2.plot(np.cumsum(eig_val)/np.sum(eig_val),'-o',color='k')
    plt.suptitle(filename,fontsize=16,x=.51,y=0.95) #,x=.51,y=1.07
    text = "PC1-PC"+str(j)+" = "+str(PC_count)
    plt.text(4,0.8,text)

#%% eigen values contribution

# Get the names of the data
data_names = n.loads_t.p.columns

# Array of 30 country load
load = n.loads_t.p_set

# Combine the load at every timestep for all countries
load_EU = np.sum(load, axis=1)

# Dataframe (array) for different generator technologies
generator_wind = pd.DataFrame(np.zeros([8760, 30]), columns=(data_names))
generator_solar = pd.DataFrame(np.zeros([8760, 30]), columns=(data_names))
generator_hydro = pd.DataFrame(np.zeros([8760, 30]), columns=(data_names))

# Counter for positioning in generator data
counter = 0
for i in n.generators.index:
    
    # Current value to insert into correct array and position
    value = np.array(n.generators_t.p)[:,counter]
    
    # Check for wind, solar and hydro
    if (i[-4:] == "wind"):
        generator_wind[i[0:2]] = generator_wind[i[0:2]] + value
    
    elif (i[-5:] == "solar"):
        generator_solar[i[0:2]] = generator_solar[i[0:2]] + value
    
    elif (i[-3:] == "ror"):
        generator_hydro[i[0:2]] = generator_hydro[i[0:2]] + value
    
    # Increase value of counter by 1
    counter +=1

# Combine the generation for each technology at every timestep for all countries
generator_wind_EU = np.sum(generator_wind, axis=1)
generator_solar_EU = np.sum(generator_solar, axis=1)
generator_hydro_EU = np.sum(generator_hydro, axis=1)
generation_EU = pd.DataFrame({'wind' : generator_wind_EU, 'solar' : generator_solar_EU, 'hydro' : generator_hydro_EU})

# Combined generation for each country at each timestep (8760 x 30)
generation = generator_wind + generator_solar + generator_hydro


# Wind
wind_term = np.array(mismatch_terms['wind'].T).reshape(8760,1)
wind_con = np.dot(generator_wind,eig_vec)
lambda_wind = (c**2)*(np.mean((wind_con**2),axis=0))

# ror
ror_term = np.array(mismatch_terms['ror'].T).reshape(8760,1)
ror_con = np.dot(generator_hydro,eig_vec)
lambda_ror = (c**2)*(np.mean((ror_con**2),axis=0))

# solar
solar_term = np.array(mismatch_terms['solar'].T).reshape(8760,1)
solar_con = np.dot(generator_solar,eig_vec)
lambda_solar = (c**2)*(np.mean((solar_con**2),axis=0))

# load
load_term = np.array(mismatch_terms['load'].T).reshape(8760,1)
load_con = np.dot(load.values,eig_vec)
lambda_load = (c**2)*(np.mean((load_con**2),axis=0))

# Collecting terms
lambda_gen = pd.DataFrame({'wind': lambda_wind,
                           'ror': lambda_ror,
                           'solar': lambda_solar,
                           'load': lambda_load})



#%% PCA dataframe and hour average

# Save time index
data_index = load.index
# Save U as a dataframe with time series
T = pd.DataFrame(data=T,index=data_index)

# Average hour and day
T_avg_hour = T.groupby(data_index.hour).mean() # Hour
T_avg_day = T.groupby([data_index.month,data_index.day]).mean() # Day


#%% Plotting PCA

plt.figure(figsize=(13,4))
plt.subplot(1,2,1)
plt.plot(T_avg_hour[0],label='k=1')
plt.plot(T_avg_hour[1],label='k=2')
plt.plot(T_avg_hour[2],label='k=3')
#plt.plot(T_avg_hour[3],label='k=4')
#plt.plot(T_avg_hour[4],label='k=5')
#plt.ylim(-0.02,0.02)
plt.xticks(ticks=range(0,24,2))
plt.legend(loc='upper right',bbox_to_anchor=(1,1))
plt.xlabel("Hours")
plt.ylabel("a_k interday")
plt.title("Hourly average for k-values for 2015 ")

# X for year plot
x_ax = range(len(T_avg_day[0]))

plt.subplot(1,2,2)
plt.plot(x_ax,T_avg_day[0],label='k=1')
plt.plot(x_ax,T_avg_day[1],label='k=2\n(shifted -0.05)')
plt.plot(x_ax,T_avg_day[2],label='k=3\n(shifted -0.10)')
#plt.plot(x_ax,T_avg_day[3],label='k=4')
#plt.plot(x_ax,T_avg_day[4],label='k=5')
plt.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.xlabel("day")
plt.ylabel("a_k seasonal")
plt.title("daily average for k-values for 2015 ")


#%%

countries = n.loads_t.p_set.columns
eig_vec_0 = eig_vec[:,0]
eig_vec_1 = eig_vec[:,1]
eig_vec_2 = eig_vec[:,2]

plt.figure(figsize=(14,10))
plt.subplot(3,1,1)
plt.bar(countries,eig_vec_0)
plt.ylim(-0.75,0.75)
#plt.xlabel("Countries")
plt.ylabel("Principle complonent value")
plt.title("principle component 1")

plt.subplot(3,1,2)
plt.bar(countries,eig_vec_1)
plt.ylim(-0.75,0.75)
#plt.xlabel("Countries")
plt.ylabel("Principle complonent value")
plt.title("principle component 2")

plt.subplot(3,1,3)
plt.bar(countries,eig_vec_2)
plt.ylim(-0.75,0.75)
#plt.xlabel("Countries")
plt.ylabel("Principle complonent value")
plt.title("principle component 3")

plt.subplots_adjust(wspace=0, hspace=0.35)
title_name = 'Data from: ',filename
plt.suptitle(filename,fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07


#%% FFT

plt.figure(figsize=(15,12))
plt.subplot(2,2,1)
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

plt.subplot(2,2,2)
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

plt.subplot(2,2,3)
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

plt.subplot(2,2,4)
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

title_name = 'Data from: ',filename
plt.suptitle(filename,fontsize=20,x=.51,y=0.93) #,x=.51,y=1.07

#%%

individual_plots = True

if individual_plots==True:
    for n in range(6):
        plt.figure()
        freq=np.fft.fftfreq(len(T[n]))  
        FFT=np.fft.fft(T[n])
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
        plt.title('FFT for '+'$\lambda_{'+str(n+1)+'}$: '+str(round(eig_val[n]*100,1))+'%')
    
        title = "PC"+str(n+1)
        plt.savefig("Figures/overleaf/mismatch/FFT/FFT_"+title+".png", bbox_inches='tight')

if individual_plots==True:
    for n in range(6):
        
        alpha_standard = 0.2
        alpha1 = alpha_standard
        alpha2 = alpha_standard
        alpha3 = alpha_standard
        alpha4 = alpha_standard
        alpha5 = alpha_standard
        alpha6 = alpha_standard
        
        if n==0:
            alpha1=1
        elif n==1:
            alpha2=1
        elif n==2:
            alpha3=1
        elif n==3:
            alpha4=1
        elif n==4:
            alpha5=1
        elif n==5:
            alpha6=1
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(T_avg_hour[0],label='$\lambda_1$',marker='.',alpha=alpha1)
        plt.plot(T_avg_hour[1],label='$\lambda_2$',marker='.',alpha=alpha2)
        plt.plot(T_avg_hour[2],label='$\lambda_3$',marker='.',alpha=alpha3)
        plt.plot(T_avg_hour[3],label='$\lambda_4$',marker='.',alpha=alpha4)
        plt.plot(T_avg_hour[4],label='$\lambda$=5',marker='.',alpha=alpha5)
        plt.plot(T_avg_hour[5],label='$\lambda$=6',marker='.',alpha=alpha6)
        #plt.ylim(-0.02,0.02)
        plt.xticks(ticks=range(0,24,2))
        #plt.legend(loc='upper right',bbox_to_anchor=(1,1))
        plt.ylim([-1.25,0.8])
        plt.xlabel("Hours")
        plt.ylabel("a_k interday")
        plt.title("Hourly average for k-values for 2015 ")
        
        # X for year plot
        x_ax = range(len(T_avg_day[0]))
        
        plt.subplot(1,2,2)
        plt.plot(x_ax,T_avg_day[0],label='$\lambda_1$',alpha=alpha1)
        plt.plot(x_ax,T_avg_day[1],label='$\lambda_2$',alpha=alpha2)
        plt.plot(x_ax,T_avg_day[2],label='$\lambda_3$',alpha=alpha3)
        plt.plot(x_ax,T_avg_day[3],label='$\lambda_4$',alpha=alpha4)
        plt.plot(x_ax,T_avg_day[4],label='$\lambda_5$',alpha=alpha5)
        plt.plot(x_ax,T_avg_day[5],label='$\lambda_6$',alpha=alpha6)
        plt.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.ylim([-1,1])
        plt.xlabel("day")
        plt.ylabel("a_k seasonal")
        plt.title("daily average for k-values for 2015 ")
    
        
        #title = 'Season plot for'+'$\lambda_{'+str(n+1)+'}$: '+str(round(eig_val[n]*100,1))+'%'
        #plt.suptitle(title,fontsize=18,x=.5,y=1) #,x=.51,y=1.07)
        
        title = "PC"+str(n+1)
        plt.savefig("Figures/overleaf/mismatch/season/season_"+title+".png", bbox_inches='tight')

