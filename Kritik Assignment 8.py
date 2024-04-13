# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:58:39 2024

@author: scwkw
"""
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def gradient_descent(x0, y0, grad_f, alpha, n): 
    x = x0
    y = y0
    for i in range(n):
        grad_x, grad_y = grad_f(x, y)
        if grad_x < 0.00000001 and grad_y < 0.00000001:
            break
        else:
            x = x - alpha*grad_x
            y = y - alpha*grad_y
    return x, y

def fun_1(x, y): 
    return x**2+y**2

def grad_f_1(x, y):
    grad_x = 2*x
    grad_y = 2*y
    return grad_x, grad_y

def fun_2(x, y):
    return 1-np.exp(-x**2-(y-2)**2)-2*np.exp(-x**2-(y+2)**2)

def grad_f_2(x, y):
    grad_x = 2*x*np.exp(-x**2-(y-2)**2)+4*x*np.exp(-x**2-(y+2)**2)
    grad_y = np.exp(-x**2-(y-2)**2)*(-2*y+4)-2*np.exp(-x**2-(y+2)**2)*(-2*y-4)
    return grad_x, grad_y
            
print(gradient_descent(0.1, 0.1, grad_f_1, 0.1, 10))
print(gradient_descent(-1, 1, grad_f_1, 0.01, 100))   

print(gradient_descent(0, 1, grad_f_2, 0.01, 10000))
print(gradient_descent(0, -1, grad_f_2, 0.01, 10000))

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(X, Y)
z = fun_1(x, y)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(x, y, z, cmap = 'viridis', edgecolor = 'none')

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(X, Y)
z = fun_2(x, y)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(x, y, z, cmap = 'viridis', edgecolor = 'none')

