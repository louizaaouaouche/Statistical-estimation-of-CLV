# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pareto
import random


#The package contains a group of functions that are useful to generate data, compute confidence intervals or boostrapping.


#GLOBAL CONSTANTS
clv_theory = 10 
size= 10000 # size of each sample
n = 10000 # number of samples
DureeObs = 12*2 # duration of observation

######################################################################################
#                                                                                    #
#                                     GENERATORS                                     #
#                                                                                    #
######################################################################################

# EXPONENTIAL GENERATOR - without censored data 
def generate_nc_exponential(mean, size):
    """
    Generates random numbers following an exponential distribution

    Args:
        mean (float): The mean of the distribution.
        size (int): The number of random numbers to generate.

    Returns:
        An array of random numbers following an exponential distribution
        with the given mean.
    """
    return np.random.exponential(scale=mean, size=size)

def generate_exponential(mean, size , obs_duration):
    """
    Generates random numbers following an exponential distribution with some censored data

    Args:
        mean (float): The mean of the distribution.
        size (int): The number of random numbers to generate.
        obs_duration (int) : Duration of observation

    Returns:
        An array of random numbers following an exponential distribution
        with the given mean.
    """
    T = np.random.exponential(scale=mean, size=size)
    ancient = obs_duration * np.random.rand(size) # uniform distribution
    Y = T*(T<=ancient) +  ancient*(T>ancient)
    return Y , ancient

# PARETO GENERATOR
def generate_nc_pareto(clv_th, size):
    shape = 3

    T = pareto.rvs(shape, scale = 1, size=size) 

    return T

def generate_c_pareto(clv_th, size, obs_duration):
    shape = 3
    #T = (np.random.pareto(a=shape, size=size) + 1 ) * scale  # pareto 2 
    T = pareto.rvs(shape, scale = 1, size=size) 
    ancient = obs_duration* np.random.rand(size) +1   # uniform distribution
    Y = T * (T <= ancient) + ancient * (T > ancient)
    return Y, ancient

######################################################################################
#                                                                                    #
#                             CONFIDENCE INTERVALS                                   #
#                                                                                    #
######################################################################################

def confidence_lvl(lower_bound, upper_bound, estimators):
    """
    Calculates the percentage of estimates falling within a given confidence interval.

    Parameters:
    lower_bound (float): The lower limit of the confidence interval.
    upper_bound (float): The upper limit of the confidence interval.
    estimators (list[float]): A list of estimates.

    Returns:
    float: The percentage of estimates falling within the confidence interval.
    """
    in_IC=0
    for x in estimators : 
        if (x<upper_bound and x>lower_bound) : in_IC +=1
    return (in_IC / len(estimators))*100

def CI_plot(clv_values, lower, upper):
    """
    Plots confidence intervals 
    
    Args:
        clv_values : clv values related to the confidence intervals
        lower : The lower limit of the confidence interval.
        upper : The upper limit of the confidence interval.
    """
    
    # Define x-axis values
    x = range(len(clv_values))

    # Set figure size
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot the data
    ax.plot(x, clv_values, label='Computed Value', color='blue')
    #ax.plot(x, lower, color='red', label='lower')
    #ax.plot(x, upper, color='green', label='upper')
    ax.fill_between(x, lower, upper, alpha=0.2, label='CI', color='gray')
    plt.axhline(y=np.mean(lower), color='green', linestyle='--', label='Mean Lower')
    plt.axhline(y=np.mean(upper), color='red', linestyle='--', label='Mean Upper')

    # Add labels and legend
    ax.set_xlabel('Sample')
    ax.set_ylabel('CLV')
    ax.set_title('CLV Computation Results')
    ax.legend(loc='best')

    # Show the plot
    plt.show()
    
def CI_monoplot(clv_values, lower, upper):
    """
    Plots confidence intervals 
    
    Args:
        clv_values : clv values related to the confidence intervals
        lower : The lower limit of the confidence interval.
        upper : The upper limit of the confidence interval.
    """
    
    # Define x-axis values
    n = len(clv_values)
    x = range(n)

    # Set figure size
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot the data
    ax.plot(x, clv_values, label='CLV values', color='blue')
    ax.plot(x, lower, color='green', label='CI')
    ax.plot(x, upper, color='green')
    #ax.fill_between(x, lower, upper, alpha=0.2, label='CI', color='gray')

    # Add labels and legend
    ax.set_xlabel('Sample')
    ax.set_ylabel('CLV')
    ax.set_title('CLV Computation Results')
    ax.legend(loc='best')

    # Show the plot
    plt.show()

def CI_multiplot(clv, lower, upper):
    """
    Plots confidence intervals
    Plots lower, upper distributions
    Prints confidence level

    Args:
        clv : clv values related to the confidence intervals
        lower : The lower limit of the confidence interval.
        upper : The upper limit of the confidence interval.
    """

    # plot confidence interval
    CI_plot(clv, lower, upper)

    # plot of the distribution
    sns.histplot(lower, color='blue', kde=True, label='Lower CI')
    sns.histplot(upper, color='orange', kde=True, label='Upper CI')
    plt.axvline(x=np.mean(lower), color='green', linestyle='--', label='Mean Lower')
    plt.axvline(x=np.mean(upper), color='red', linestyle='--', label='Mean Upper')

    # Add titles and axis labels
    plt.title('Distribution of IC')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Add legend to the plot
    plt.legend()

    # Show the plot
    plt.show()

    # print confidence level
    print(f"Theoretical CI = [{np.mean(lower)}, {np.mean(upper)}].\nCONFIDENCE LEVEL: {confidence_lvl(np.mean(lower), np.mean(upper), clv):.2f}% of estimators are within this interval")


def CI_plot_95( values, alpha):
    # Calculate percentiles
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))

    # Plot the histogram with KDE and add vertical lines
    sns.histplot(values, kde=True)
    plt.axvline(lower, color='red', label='95% CI')
    plt.axvline(upper, color='red')
    plt.title("Distribution of CLV")
    plt.legend()
    plt.show()

    print("IC = [%f , %f ] " % (lower, upper))
    
    return lower, upper

    
def plot_bar_std(value1, value2, title):
    # Plot for the two first values
    labels = ['Not Censored','Censored']
    values = [value1, value2]
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Standard Deviation Value')
    plt.show()
    
    
######################################################################################
#                                                                                    #
#                                    BOOTSTRAPPING                                   #
#                                                                                    #
######################################################################################

def bootstrap(data1, data2): #
    """
    Performs bootstrap resampling on two datasets by randomly selecting data points with replacement.

    Parameters:
    data1 (list): The first dataset to be resampled.
    data2 (list): The second dataset to be resampled.

    Returns:
    tuple: A tuple of two lists, containing the resampled data from each dataset.
    """
    # Initialize two empty lists to hold the resampled data
    sample1 = []
    sample2 = []

    # Loop through each data point in the datasets
    for i in range(len(data1)):
        # Generate a random integer between 0 and the length of the dataset - 1
        rand_int = random.randint(0, len(data1) - 1)
        # Add the randomly selected data points to the resampled datasets
        sample1.append(data1[rand_int])
        sample2.append(data2[rand_int])

    # Return the resampled datasets as a tuple
    return sample1, sample2


def bootstrap_confidence_interval(data, censored, func, alpha=0.05, B=100):
    """
    Computes confidence interval based on bootstrapping
    
    Args:
        data : data points (lifetime of customers) for one sample
        censored ({0,1}) : 0 if data is not censored, 1 if it is 
        func : function calculating the CLV value according to data distribution
        alpha (0<= float <=1): significance level
        B : number of boostrap samples
    
    Returns:
    (float,float): the lower and upper bound of the confidence interval
    
    """
    n = len(data[0]) if censored else len(data)# Get the length of the original data
    
    bootstrap_statistics = []  # Initialize an empty list to store bootstrap statistics

    # Generate B bootstrap samples and compute the statistic of interest for each sample
    for b in range(B):
        random_indices = np.random.choice(range(n), size=n, replace=True)
        if censored:
            Y_bootstrap_sample = data[0][random_indices]  # Randomly sample n values with replacement
            A_bootstrap_sample = data[1][random_indices]
            bootstrap_statistic = func(Y_bootstrap_sample,A_bootstrap_sample)  
        else :
            bootstrap_sample = data[random_indices]  # Randomly sample n values with replacement
            bootstrap_statistic = func(bootstrap_sample)  # Compute the mean (replace with your desired statistic)
        bootstrap_statistics.append(bootstrap_statistic)  # Store the computed statistic for the bootstrap sample

    bootstrap_statistics.sort()  # Sort the bootstrap statistics in ascending order
    lower_percentile = 100 * alpha / 2  # Calculate the lower percentile of the confidence interval
    upper_percentile = 100 * (1 - alpha / 2)  # Calculate the upper percentile of the confidence interval
    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)  # Calculate the lower bound of the confidence interval
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)  # Calculate the upper bound of the confidence interval

    return lower_bound, upper_bound  # Return the lower and upper bounds of the confidence interval

def bootstrap_samples(data, censored, func, B=10000):
    """
    returns a list of func values of B samples of the data
    
    Args:
        data : data points (lifetime of customers) for one sample
        censored ({0,1}) : 0 if data is not censored, 1 if it is 
        func : function calculating the CLV value according to data distribution
        alpha (0<= float <=1): significance level
        B : number of boostrap samples
    
    Returns:
    list of samples, list of related func values
    """
    bootstrap_statistics = []  # Initialize an empty list to store bootstrap statistics
    if censored :
        Y_samples = []
        A_samples = []
        n = len(data[0])
    else :
        samples = []
        n = len(data)
    # Generate B bootstrap samples and compute the statistic of interest for each sample
    for b in range(B):
        random_indices = np.random.choice(range(n), size=n, replace=True)
        if censored:
            Y_bootstrap_sample = data[0][random_indices]  # Randomly sample n values with replacement
            Y_samples.append(Y_bootstrap_sample)
            A_bootstrap_sample = data[1][random_indices]
            A_samples.append(A_bootstrap_sample)
            bootstrap_statistic = func(Y_bootstrap_sample,A_bootstrap_sample)  
        else :
            bootstrap_sample = data[random_indices]  # Randomly sample n values with replacement
            samples.append(bootstrap_sample)
            bootstrap_statistic = func(bootstrap_sample)  # Compute the mean (replace with your desired statistic)
        bootstrap_statistics.append(bootstrap_statistic)  # Store the computed statistic for the bootstrap sample
    if censored:
        return Y_samples, A_samples, bootstrap_statistics
    
    else :
        return samples, bootstrap_statistics




def bootstrap_intervals(list_data, censored, func,  alpha=0.05, B=100):
    """
    Computes confidence interval based on bootstrapping
    
    Args:
        list_data : data points (lifetime of customers) for a list of samples
        censored ({0,1}) : 0 if data is not censored, 1 if it is 
        func : function calculating the CLV value according to data distribution
        alpha (0<= float <=1): significance level
        B : number of boostrap samples
    
    Returns:
    (float,float): the lower and upper bound of the confidence interval
    
    ================================================================================================================
    WARNING : the size of the data and value of B can affect drasticly the complexity of this function
    ================================================================================================================
    """
    
    
    lower_bounds = []
    upper_bounds = []
    
    for data in list_data:
        lower_bound,upper_bound = bootstrap_confidence_interval(data, censored, func, alpha, B)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    return lower_bounds,upper_bounds


######################################################################################
#                                                                                    #
#                                   CLV / ESTIMATOR                                  #
#                                                                                    #
######################################################################################
def print_conv_plot_distribution(size, n ,clv_theory, clv_real):
    """
    Plots the distribution of the errors of estimation between clv_thoery and clv_real
    Prints values about clv_real (mean, standard deviation)    
    """
    
    # clv_avg = clv_hat that averages all the estimated values of clv
    clv_avg = np.mean(clv_real)
    
    # standard deviation of clv
    clv_std = np.std(clv_real)
    
    # difference between clv predicted and the actual value
    clv_errors = np.array(clv_real) - clv_theory
    
    # print convergence values
    print("CONVERGENCE :\n \t size of each dataset =%d \n \t number of datasets =%d \n \t CLV thoery = %.2f \n \t CLV real = %f  \n \t Standard deviation CLV= %f" % (size, n,clv_theory,clv_avg,clv_std))
    
    # plot distribution of errors
    sns.histplot(clv_errors, kde = True).set_title("Distribution of CLV errors for all samples")
    
def monte_carlo(clv_thoery, size, n, censored, clv_func, generator_fun ):
    """
    Computes the clv value according to the func argument
    Args:
        clv_thoery : data points (lifetime of customers) for a list of samples
        size : size of each sample
        n : number of samples to generate
        func : function calculating the CLV value according to data distribution (geomtric, exponential)
    Returns:
        clv : avergae of clv values of all samples
        Y_n : list of samples    
    """
    clv = [] # list containing average of clv values for all samples
    Y_n = [] # list containing all the generated data

    for i in range(n):
        
        if censored:
            # generating data 
            Y_new, ancient_new = generator_fun(clv_thoery, size,DureeObs )
            # storing the sample
            Y_n.append((Y_new, ancient_new))
            # computing the avergae clv value of the sample
            clv.append(clv_func(Y_new, ancient_new))
        else:
            # generating data 
            Y_i = generator_fun(clv_thoery, size) 
            # storing the sample
            Y_n.append(Y_i)
            # computing the avergae clv value of the sample
            clv.append(clv_func(Y_i))
    
    return clv, Y_n

def clv_convergence(clv_values, clv_theory,n, title ):
    """
    Visualise the law of large numbers 
    """
    # calculate the mean of the CLV estimates for each iteration
    mean_estimates = [np.mean(clv_values[:i]) for i in range(1, n+1)]

    # plot the mean estimates over the number of iterations
    plt.plot(range(1, n+1), mean_estimates)

    # plot a horizontal line at the theoretical CLV value
    plt.axhline(y=clv_theory, color='r', linestyle='-', label='Theoretical CLV')

    # add labels and a title to the plot
    plt.xlabel('Number of Monte Carlo samples')
    plt.ylabel('Mean CLV Estimate')
    plt.title(title)

    # show the plot
    plt.legend()
    plt.show()

#Geomtric estimator - not censored

def geom_clv_estimator(T):
    """
    Function to calculate the geometric clv estimator
    
    Parameters:
    T (list or array): list or array of numerical values
    
    Returns:
    float: the geometric clv estimator of the values in T
    """
    return np.sum(T)/len(T)

#Geomtric estimator - censored

def geom_c_clv_estimator(Y, ancient):
    """
    Function to calculate the geometric clv estimator
    
    Parameters:
    Y (array): numerical values related to known lifetimes
    ancient (array): seniority values of customers since they entered the company 
    
    Returns:
    float: the geometric clv estimator values
    """
    A = np.where(Y!=ancient)[0]  
    N = np.where(Y==ancient)[0]
    return (np.sum(Y[A]) + np.sum(ancient[N]))/len(A)

#Exponential estimator - not censored

def exp_nc_clv_estimator(T):
    """
    Function to calculate the exponential clv estimator
    
    Parameters:
    T (list or array): list or array of numerical values
    
    Returns:
    float: the exponential clv estimator of the values in T
    """
    return np.sum(T)/len(T)

#Exponential estimator - censored

def exp_c_clv_estimator(Y , ancient):
    A = np.where(Y!=ancient)[0] # available data : T_i < a_i
    return np.sum(Y) /len(A)

def CI_exp_c_clv(a,Y,ancient):
    len_A = len(np.where(Y!=ancient)[0])
    lower = np.sum(Y)/(len_A + a * np.sqrt(len_A))
    upper = np.sum(Y)/(len_A + - a * np.sqrt(len_A))
    
    return lower, upper

# pareto estimator
def pareto_nc_clv_estimator(Y):
    denom = 1 - (np.sum(np.log(Y))/len(Y))
    return 1 / denom

def pareto_c_clv_estimator(Y, ancient):
    denom = 1 - (np.sum(np.log(Y))/ len(np.where(Y!=ancient)[0]))
    return 1 / denom

def CI_pareto_c_clv(a,Y,ancient):

    len_A = len(np.where(Y!=ancient)[0])
    lower = 1- (np.sum(np.log(Y)) / (len_A - 1.96* np.sqrt(len_A)))
    upper = 1- (np.sum(np.log(Y)) / (len_A + 1.96* np.sqrt(len_A)))
    
    return 1/upper, 1/lower

######################################################################################
#                                                                                    #
#                 ESTIMATORS TO TEST ON REAL DATA : LUNG CANCER                      #
#                                                                                    #
######################################################################################


#Geomtric estimator - censored

def geom_c_clv_estimator2(Y, status):
    """
    Function to calculate the geometric clv estimator
    
    Parameters:
    Y (array): numerical values related to known lifetimes
    status (array): true if dead, false otherwise
    
    Returns:
    float: the geometric clv estimator values
    """
    A = np.where(status==True)[0]  
    N = np.where(status==False)[0]
    return (np.sum(Y[A]) + np.sum(Y[N]))/len(A)


#Exponential estimator - censored

def exp_c_clv_estimator2(Y , status):
    
    return np.sum(Y) /np.count_nonzero(status)


# pareto estimator
def pareto_c_clv_estimator2(Y, status):
    denom = 1 - (np.sum(np.log(Y))/np.count_nonzero(status))
    return 1 / denom
