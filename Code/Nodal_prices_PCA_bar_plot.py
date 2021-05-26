import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
from RD_func import BAR, PCA, season_plot, FFT_plot

# Load data - CO2 constraint
# Folder name of data files
directory = 'Data\\'

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]

# Variable for principal components for plotting later
PC1_con = np.zeros((7,11))
PC2_con = np.zeros((7,11))
PC3_con = np.zeros((7,11))
PC4_con = np.zeros((7,11))
PC5_con = np.zeros((7,11))
PC6_con = np.zeros((7,11))
PC_con = []


# Variable for principal components for plotting later
PCA_bar_CO2 = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename_CO2)):
    network = pypsa.Network(directory+filename_CO2[i])
    
    # Get the names of the data
    data_names = network.loads_t.p.columns
    
    # Get time stamps
    time_index = network.loads_t.p.index
    
    # List of prices
    prices = network.buses_t.marginal_price
    
    # List of nodal prices for each country
    country_price = prices[data_names] # [€/MWh]
    country_price_gas = prices[(data_names + ' gas')] # [€/MWh]
    country_price_H2 = prices[(data_names + ' H2')] # [€/MWh]
    country_price_battery = prices[(data_names + ' battery')] # [€/MWh]
    
    # Sum up all the prices into one for every country
    nodal_price = country_price.values #+ country_price_gas.values + country_price_H2.values + country_price_battery.values
    nodal_price = pd.DataFrame(data=nodal_price, index=time_index, columns=data_names)
    
    
    # Limit
    nodal_price = np.clip(nodal_price, 0, 1000) 
    
    # PCA analysis of mismatch
    eigen_values, eigen_vectors, variance_explained, norm_const, T = PCA(nodal_price)
    
    T = pd.DataFrame(data=T,index=time_index)
    
    # Season Plot
    season_plot(T, time_index, filename_CO2[i])

    # FFT Plot
    FFT_plot(T, filename_CO2[i])

    PCA_bar_CO2.append(variance_explained)
    
    # Mean values
    country_price_mean = np.mean(country_price.values,axis=0)
    country_price_gas_mean = np.mean(country_price_gas.values,axis=0)
    country_price_H2_mean = np.mean(country_price_H2.values,axis=0)
    country_price_battery_mean = np.mean(country_price_battery.values,axis=0)
    
    # Centering data
    country_price_cent = np.subtract(country_price.values,country_price_mean.T)
    country_price_gas_cent = np.subtract(country_price_gas.values,country_price_gas_mean.T)
    country_price_H2_cent = np.subtract(country_price_H2.values,country_price_H2_mean.T)
    country_price_battery_cent = np.subtract(country_price_battery.values,country_price_battery_mean.T)
    
    # Contributions
    # Just price
    country_price_con = np.dot(country_price_cent,eigen_vectors)
    # Gas
    country_price_gas_con = np.dot(country_price_gas_cent,eigen_vectors)
    # H2
    country_price_H2_con = np.dot(country_price_H2_cent,eigen_vectors)
    # Battery
    country_price_battery_con = np.dot(country_price_battery_cent,eigen_vectors)
    
    # Check
    #Check = norm_const * (country_price_con + country_price_gas_con + country_price_H2_con + country_price_battery_con)
    
    # Eigen value contribution
    # Just price
    lambda_P = (norm_const**2) * (np.mean((country_price_con**2),axis=0))
    # Gas
    lambda_G = (norm_const**2) * (np.mean((country_price_gas_con**2),axis=0))
    # H2
    lambda_H = (norm_const**2) * (np.mean((country_price_H2_con**2),axis=0))
    # Battery
    lambda_B = (norm_const**2) * (np.mean((country_price_battery_con**2),axis=0))
    # price+gas
    lambda_PG = (norm_const**2)*2*(np.mean((country_price_con*country_price_gas_con),axis=0))
    # price+H2
    lambda_PH = (norm_const**2)*2*(np.mean((country_price_con*country_price_H2_con),axis=0))
    # price+battery
    lambda_PB = (norm_const**2)*2*(np.mean((country_price_con*country_price_battery_con),axis=0))
    # gas+H2
    lambda_GH = (norm_const**2)*2*(np.mean((country_price_gas_con*country_price_H2_con),axis=0))
    # gas+battery
    lambda_GB = (norm_const**2)*2*(np.mean((country_price_gas_con*country_price_battery_con),axis=0))
    # H2+battery
    lambda_HB = (norm_const**2)*2*(np.mean((country_price_H2_con*country_price_battery_con),axis=0))
    
    # Colleting terms
    lambda_collect = pd.DataFrame({'General':              lambda_P,
                                   'Gas':                  lambda_G,
                                   'H2':                   lambda_H,
                                   'Battery':              lambda_B,
                                   'General/\nGas':        lambda_PG,
                                   'General/\nH2':         lambda_PH,
                                   'General/\nBattery':    lambda_PB,
                                   'Gas/\nH2':             lambda_GH,
                                   'Gas/\nBattery':        lambda_GB,
                                   'H2/\nBattery':         lambda_HB,
                                   })
    lambda_tot = sum([+lambda_P,
                      +lambda_G,
                      +lambda_H,
                      +lambda_B,
                      +lambda_PG,
                      +lambda_PH,
                      +lambda_PB,
                      +lambda_GH,
                      +lambda_GB,
                      +lambda_HB
                      ])
    lambda_collect_all = pd.DataFrame({'General':              lambda_P,
                                       'Gas':                  lambda_G,
                                       'H2':                   lambda_H,
                                       'Battery':              lambda_B,
                                       'General/\nGas':        lambda_PG,
                                       'General/\nH2':         lambda_PH,
                                       'General/\nBattery':    lambda_PB,
                                       'Gas/\nH2':             lambda_GH,
                                       'Gas/\nBattery':        lambda_GB,
                                       'H2/\nBattery':         lambda_HB,
                                       'Total':                lambda_tot,
                                   })
    
    plt.figure(figsize=[16,16])
    for n in range(6):
        lambda_collect_procent = lambda_collect[n:n+1]/lambda_tot[n]*100 # percentage
        #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
        plt.subplot(3,2,n+1)
        plt.bar(lambda_collect.columns,lambda_collect_procent.values[0])
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
    plt.suptitle(filename_CO2[i],fontsize=20,x=.51,y=0.99) #,x=.51,y=1.07     
    plt.tight_layout()
    
    
    if i==5:
        individual_plots = True
        
        if individual_plots==True:
            for n in range(6):
                lambda_collect_procent = lambda_collect[n:n+1]/lambda_tot[n]*100 # percentage
                #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
                plt.figure(figsize=[9,5])
                plt.bar(lambda_collect.columns,lambda_collect_procent.values[0])
                plt.title('$\lambda_{'+str(n+1)+'}$: '+str(round(lambda_tot[n]*100,1))+'%')
                plt.ylabel('Influance [%]')
                plt.ylim([-50,125])
                plt.grid(axis='y',alpha=0.5)
                for k in range(10):
                    if lambda_collect_procent.values[:,k] < 0:
                        v = lambda_collect_procent.values[:,k] - 6.5
                    else:
                        v = lambda_collect_procent.values[:,k] + 2.5
                    plt.text(x=k,y=v,s=str(round(float(lambda_collect_procent.values[:,k]),2))+'%',ha='center',size='small')
                
                title = "PC"+str(n+1)
                plt.savefig("Figures/overleaf/nodal prices/contribution/contribution_"+title+"_nodal_prices"+".png", bbox_inches='tight')
            #plt.suptitle(filename,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07  
    
    
    plt.show(all)
    
    # Save PC1-PC6
    PC1_con[i] = lambda_collect_all[0:1].values
    PC2_con[i] = lambda_collect_all[1:2].values
    PC3_con[i] = lambda_collect_all[2:3].values
    PC4_con[i] = lambda_collect_all[3:4].values
    PC5_con[i] = lambda_collect_all[4:5].values
    PC6_con[i] = lambda_collect_all[5:6].values
    
    #%% season plot

    # Save time index
    data_index = T.index
    
    # Average hour and day
    T_avg_hour = T.groupby(data_index.hour).mean() # Hour
    T_avg_day = T.groupby([data_index.month,data_index.day]).mean() # Day
        
    
#%% Bar plot - CO2 constraint
matrix_CO2 = PCA_bar_CO2
PC_max_CO2 = 12
constraints_CO2 = ['40%', '50%', '60%', '70%', '80%', '90%', '95%']
title_CO2 = 'Variance for each PC as a function of $CO_{2}$ constraint with constant transmission size (2x current)'
xlabel_CO2 = '$CO_{2}$ Constraint'
suptitle_CO2 = 'Electricity Nodal Prices'

BAR(matrix_CO2, PC_max_CO2, filename_CO2, constraints_CO2, title_CO2, xlabel_CO2, suptitle_CO2)

#%%

individual_plots = True

if individual_plots==True:
    for n in range(6):
        plt.figure()
        freq=np.fft.fftfreq(len(T[n]))  
        FFT=np.fft.fft(T[n])
        FFT[n]=0
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
        plt.title('FFT for '+'$\lambda_{'+str(n+1)+'}$: '+str(round(eigen_values[n]*100,1))+'%')
    
        title = "PC"+str(n+1)
        plt.savefig("Figures/overleaf/nodal prices/FFT/FFT_"+title+"_nodal_prices"+".png", bbox_inches='tight')

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
        plt.ylim([-2,2])
        plt.xlabel("day")
        plt.ylabel("a_k seasonal")
        plt.title("daily average for k-values for 2015 ")
    
        
        #title = 'Season plot for'+'$\lambda_{'+str(n+1)+'}$: '+str(round(eig_val[n]*100,1))+'%'
        #plt.suptitle(title,fontsize=18,x=.5,y=1) #,x=.51,y=1.07)
        
        title = "PC"+str(n+1)
        plt.savefig("Figures/overleaf/nodal prices/season/season_"+title+"_nodal_prices"+".png", bbox_inches='tight')

plt.show(all)
#assert False, "Stop her"

#%% Contribution plot

# Collect
PC_con.append(PC1_con)
PC_con.append(PC2_con)
PC_con.append(PC3_con)
PC_con.append(PC4_con)
PC_con.append(PC5_con)
PC_con.append(PC6_con)

plt.figure(figsize=(14,16))#,dpi=500)

for i in range(6):
    # y functions comulated
    wind_con_data  = PC_con[i][:,:1].sum(axis=1)
    solar_con_data = PC_con[i][:,:2].sum(axis=1)
    hydro_con_data = PC_con[i][:,:3].sum(axis=1)
    load_con_data  = PC_con[i][:,:4].sum(axis=1)
    gen_cov_data   = PC_con[i][:,:7].sum(axis=1)
    load_cov_data  = PC_con[i][:,:10].sum(axis=1)
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
                     label='General',
                     color='cornflowerblue') # Because it is a beutiful color
    plt.fill_between(range(7), wind_con_data, solar_con_data,
                     label='Gas',
                     color='yellow')
    plt.fill_between(range(7), solar_con_data, hydro_con_data,
                     label='H2',
                     color='darkslateblue')
    plt.fill_between(range(7), hydro_con_data, load_con_data,
                     label='Battery',
                     color='slategray')
    plt.fill_between(range(7), load_con_data, gen_cov_data,
                     label='General\ncovariance',
                     color='brown',
                     alpha=0.5)
    plt.fill_between(range(7), gen_cov_data, load_cov_data,
                     label='Backup\ncovariance',
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
        
plt.suptitle("Nodel Price Contribution as a Function of CO2 Constrain",fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 

plt.show(all)


#%% Load data - Transmission constraint
# Folder name of data files
directory = 'Data\\'

# Name of file (starts from zero to 6-times-curren-size)
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]

# Variable for principal components for plotting later
PCA_bar_transmission = []

# Variable for principal components for plotting later
PC1_con = np.zeros((5,11))
PC2_con = np.zeros((5,11))
PC3_con = np.zeros((5,11))
PC4_con = np.zeros((5,11))
PC5_con = np.zeros((5,11))
PC6_con = np.zeros((5,11))
PC_con = []

# Creates a for-loop of all the files
for i in np.arange(0,len(filename_trans)):
    network = pypsa.Network(directory+filename_trans[i])
    
    # Get the names of the data
    data_names = network.loads_t.p.columns
    
    # Get time stamps
    time_index = network.loads_t.p.index
    
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
    
    # Limit
    nodal_price = np.clip(nodal_price, 0, 1000) 
    
    # PCA analysis of mismatch
    eigen_values, eigen_vectors, variance_explained, norm_const,T = PCA(nodal_price)
    
    T = pd.DataFrame(data=T,index=time_index)
    
    # Season Plot
    season_plot(T, time_index, filename_trans[i])

    # FFT Plot
    FFT_plot(T, filename_trans[i])

    PCA_bar_transmission.append(variance_explained)

    
    # Mean values
    country_price_mean = np.mean(country_price.values,axis=0)
    country_price_gas_mean = np.mean(country_price_gas.values,axis=0)
    country_price_H2_mean = np.mean(country_price_H2.values,axis=0)
    country_price_battery_mean = np.mean(country_price_battery.values,axis=0)
    
    # Centering data
    country_price_cent = np.subtract(country_price.values,country_price_mean.T)
    country_price_gas_cent = np.subtract(country_price_gas.values,country_price_gas_mean.T)
    country_price_H2_cent = np.subtract(country_price_H2.values,country_price_H2_mean.T)
    country_price_battery_cent = np.subtract(country_price_battery.values,country_price_battery_mean.T)
    
    # Contributions
    # Just price
    country_price_con = np.dot(country_price_cent,eigen_vectors)
    # Gas
    country_price_gas_con = np.dot(country_price_gas_cent,eigen_vectors)
    # H2
    country_price_H2_con = np.dot(country_price_H2_cent,eigen_vectors)
    # Battery
    country_price_battery_con = np.dot(country_price_battery_cent,eigen_vectors)
    
    # Check
    #Check = norm_const * (country_price_con + country_price_gas_con + country_price_H2_con + country_price_battery_con)
    
    # Eigen value contribution
    # Just price
    lambda_P = (norm_const**2) * (np.mean((country_price_con**2),axis=0))
    # Gas
    lambda_G = (norm_const**2) * (np.mean((country_price_gas_con**2),axis=0))
    # H2
    lambda_H = (norm_const**2) * (np.mean((country_price_H2_con**2),axis=0))
    # Battery
    lambda_B = (norm_const**2) * (np.mean((country_price_battery_con**2),axis=0))
    # price+gas
    lambda_PG = (norm_const**2)*2*(np.mean((country_price_con*country_price_gas_con),axis=0))
    # price+H2
    lambda_PH = (norm_const**2)*2*(np.mean((country_price_con*country_price_H2_con),axis=0))
    # price+battery
    lambda_PB = (norm_const**2)*2*(np.mean((country_price_con*country_price_battery_con),axis=0))
    # gas+H2
    lambda_GH = (norm_const**2)*2*(np.mean((country_price_gas_con*country_price_H2_con),axis=0))
    # gas+battery
    lambda_GB = (norm_const**2)*2*(np.mean((country_price_gas_con*country_price_battery_con),axis=0))
    # H2+battery
    lambda_HB = (norm_const**2)*2*(np.mean((country_price_H2_con*country_price_battery_con),axis=0))
    
    # Colleting terms
    lambda_collect = pd.DataFrame({'General':              lambda_P,
                                   'Gas':                  lambda_G,
                                   'H2':                   lambda_H,
                                   'Battery':              lambda_B,
                                   'General/\nGas':        lambda_PG,
                                   'General/\nH2':         lambda_PH,
                                   'General/\nBattery':    lambda_PB,
                                   'Gas/\nH2':             lambda_GH,
                                   'Gas/\nBattery':        lambda_GB,
                                   'H2/\nBattery':         lambda_HB,
                                   })
    lambda_tot = sum([+lambda_P,
                      +lambda_G,
                      +lambda_H,
                      +lambda_B,
                      +lambda_PG,
                      +lambda_PH,
                      +lambda_PB,
                      +lambda_GH,
                      +lambda_GB,
                      +lambda_HB
                      ])
    lambda_collect_all = pd.DataFrame({'General':              lambda_P,
                                       'Gas':                  lambda_G,
                                       'H2':                   lambda_H,
                                       'Battery':              lambda_B,
                                       'General/\nGas':        lambda_PG,
                                       'General/\nH2':         lambda_PH,
                                       'General/\nBattery':    lambda_PB,
                                       'Gas/\nH2':             lambda_GH,
                                       'Gas/\nBattery':        lambda_GB,
                                       'H2/\nBattery':         lambda_HB,
                                       'Total':                lambda_tot,
                                   })
    
    plt.figure(figsize=[18,16])
    for n in range(6):
        lambda_collect_procent = lambda_collect[n:n+1]/lambda_tot[n]*100 # percentage
        #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
        plt.subplot(3,2,n+1)
        plt.bar(lambda_collect.columns,lambda_collect_procent.values[0])
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
    plt.suptitle(filename_trans[i],fontsize=20,x=.51,y=0.99) #,x=.51,y=1.07     
    plt.tight_layout()
    
    plt.show(all)
    
    # Save PC1-PC6
    PC1_con[i] = lambda_collect_all[0:1].values
    PC2_con[i] = lambda_collect_all[1:2].values
    PC3_con[i] = lambda_collect_all[2:3].values
    PC4_con[i] = lambda_collect_all[3:4].values
    PC5_con[i] = lambda_collect_all[4:5].values
    PC6_con[i] = lambda_collect_all[5:6].values


#%% Bar plot - Transmission constraint
matrix_trans = PCA_bar_transmission
PC_max_trans = 12
constraints_trans = ['Zero', 'Current', '2x Current', '4x Current', '6x Current']
title_trans = 'Variance for each PC as a function of transmission size with constant $CO_{2}$ constraint (95%))'
xlabel_trans = 'Transmission size'
suptitle_trans = 'Electricity Nodal Prices'

BAR(matrix_trans, PC_max_trans, filename_trans, constraints_trans, title_trans, xlabel_trans, suptitle_trans)

#%% Contribution plot

# Collect
PC_con.append(PC1_con)
PC_con.append(PC2_con)
PC_con.append(PC3_con)
PC_con.append(PC4_con)
PC_con.append(PC5_con)
PC_con.append(PC6_con)

plt.figure(figsize=(14,16))#,dpi=500)

for i in range(6):
    # y functions comulated
    wind_con_data  = PC_con[i][:,:1].sum(axis=1)
    solar_con_data = PC_con[i][:,:2].sum(axis=1)
    hydro_con_data = PC_con[i][:,:3].sum(axis=1)
    load_con_data  = PC_con[i][:,:4].sum(axis=1)
    gen_cov_data   = PC_con[i][:,:7].sum(axis=1)
    load_cov_data  = PC_con[i][:,:10].sum(axis=1)
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
    plt.fill_between(range(5), np.zeros(5), wind_con_data,
                     label='General',
                     color='cornflowerblue') # Because it is a beutiful color
    plt.fill_between(range(5), wind_con_data, solar_con_data,
                     label='Gas',
                     color='yellow')
    plt.fill_between(range(5), solar_con_data, hydro_con_data,
                     label='H2',
                     color='darkslateblue')
    plt.fill_between(range(5), hydro_con_data, load_con_data,
                     label='Battery',
                     color='slategray')
    plt.fill_between(range(5), load_con_data, gen_cov_data,
                     label='General\ncovariance',
                     color='brown',
                     alpha=0.5)
    plt.fill_between(range(5), gen_cov_data, load_cov_data,
                     label='Backup\ncovariance',
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
        
plt.suptitle("Nodal Price Contribution as a Function of Transmission Link Sizes",fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07 

plt.show(all)