# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:30:12 2024

@author: anon
"""
import numpy as np
from scipy.special import gamma

def t_distribution_pdf(x, nu):
    coeff = gamma((nu+1)/2)/(np.sqrt(nu*np.pi)*gamma(nu/2))
    density = coeff*(1+x**2/nu)**(-0.5*(nu+1))
    return density

test_scores=[92.64,79.00,84.79,97.41,93.68,65.23,84.50,73.49,73.97,79.11]
average = 75
#compute mean and standard deviation of test scores
mean = sum(test_scores)/len(test_scores)
sd = np.std(test_scores)
#compute value for t0
t0 = (mean-average)/(sd/np.sqrt(len(test_scores)))
#compute valie for t_star
def find_t_star(prob, nu):
    x_start=0
    x_end=20
    num_points=10000
    x = np.linspace(x_start, x_end, num_points)
    y = t_distribution_pdf(x, nu)
    cdf = np.cumsum(y)*(x[1]-x[0])
    target_half_prob = prob/2
    index = np.where(cdf >= target_half_prob)[0][0]
    return x[index]
#determine if t0 is within the interval
def check_true(t0, find_t_star):
    if -find_t_star<=t0<=find_t_star:
        return True
    else:
        return False 

print(check_true(t0, find_t_star(0.95,len(test_scores)-1)))
print(t0)
#since the t0 value is positive, suggests that the new techniques are beneficial