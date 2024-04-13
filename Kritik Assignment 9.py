# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:15:56 2024

@author: scwkw
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
x, y = sp.symbols('x y') 

#a) 
# Define function
f = sp.euler(x*sp.sin(y)) + y**3
# Take patial derivatives
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)
print("df_dx is", df_dx)
print("df_dy is", df_dy)

#b)
# Define function
g = (x**2)*y + x*y**2
# Take partial derivatives
dg_dx = sp.diff(g, x)
dg_dy = sp.diff(g, y) 
# Compute gradient vector at (1,1)
grad_vct_x = sp.lambdify([x,y],dg_dx)
grad_vct_y = sp.lambdify([x,y],dg_dy)
print("The gradient vector is (", dg_dx, ",", dg_dy, ")")
print("The magnitude of the gradient vector at (1,1) is", grad_vct_x(1,-1) + grad_vct_y(1,-1))

#c) 
# Define function
u = x**2 + y**2
h = sp.log(u)
# Take first partial derivatives
dh_dx = sp.diff(h, x)
dh_dy = sp.diff(h, y) 
# Take the second partial derivatives
dh_dxdx = sp.diff(dh_dx, x)
dh_dydy = sp.diff(dh_dy, y)
dh_dxdy = sp.diff(dh_dx, y)
print("dh_dxdx is", dh_dxdx)
print("dh_dydy is", dh_dydy)
print("dh_dxdy is", dh_dxdy)

#d)
# Define the function and derivative
j = x**3 - 3*x*y + y**3
j_func = sp.lambdify((x,y), j, 'numpy')
# Create the grid
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = j_func(X, Y)
# Plot the contour
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Contour plot of $j(x,y) = x^3 - 3xy + y^3$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

#e)
# Define the function and derivative
k = x**2 + y**2
k_func = sp.lambdify((x,y), k, 'numpy')
# Create the grid
x_vals = np.linspace(-10, 10, 400)
y_vals = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = k_func(X, Y)
# Plot the contour
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Contour plot of $k(x,y) = x^2 + y^2$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()





