"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 5
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 07-03-2022
-------------------------------------------------------------
"""

from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt

Nx=25 # size along x
Ny=25;  # size along y
radius=8 # radius of central lead
Niter=1500 # number of iterations to perform

if len(sys.argv) == 5 :
    for i in sys.argv:
        Nx = int(sys.argv[1])
        Ny = int(sys.argv[2])
        radius = int(sys.argv[3])
        Niter = int(sys.argv[4])

elif len(sys.argv) == 1:
    print("Alternately you could use the format: python <file.py> Nx Ny radius Niter")

else:
    sys.exit("Wrong number of arguments! Please use the format: python <file.py> Nx Ny radius Niter")

phi = zeros((Ny,Nx))

# We assume that the center of phi matrix is the coordiate point (0,0)
x = np.arange(Nx)
mid_value = 0
if Nx%2 == 1:
    mid_value = (Nx-1)/2
else:
    mid_value = Nx/2
y = np.arange(Ny)
x = x - mid_value   # Constructing coordinate vectors
y = y - mid_value
y = y[::-1]         # Reversing because the bottom row of the phi matrix must correspnd to -12

'''
Because the meshgrid function gives the coordinate vector for y starting from top to bottom

    y[0]    y[0]    y[0]    y[0]
    y[1]    y[1]    y[1]    y[1]
    y[2]    y[2]    y[2]    y[2]
    y[3]    y[3]    y[3]    y[3]    

'''

X,Y = np.meshgrid(x,y)  # Constructing coordinate matrix only to find the points of radius

ii = where(X*X + Y*Y < radius*radius)
phi[ii] = 1.0

# Finding new meshgrid so that in the plots, the values are from 0-Nx, 0-Ny
x = x + mid_value
y = y + mid_value
X,Y = np.meshgrid(x,y)

plt.figure(1)
plt.title("Initial Potential Contour Plot (in V)")
plt.contourf(X, Y, phi, levels = 50, cmap = plt.cm.get_cmap("plasma"))
plt.colorbar(label = r"$\phi$ values", orientation = "vertical")
plt.plot(X[ii],Y[ii],'ro')
plt.xlabel(r"$x \rightarrow$")
plt.ylabel(r"$y \rightarrow$")
plt.show()

oldphi = phi.copy()
errors = np.zeros(Niter)

# Perfoming the iterations
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1,1:-1] = (0.25)*(oldphi[1:-1,0:-2] + oldphi[1:-1,2:] + oldphi[0:-2,1:-1] + oldphi[2:,1:-1])
    phi[:,0] = phi[:,1]             # Making the first column equal to the second column so that do(phi)/do(x) = 0
    phi[:,-1] = phi[:,-2]           # Making the last column equal to the last second column so that do(phi)/do(x) = 0
    phi[0,:] = phi[1,:]             # Making the first row equal to the second row so that do(phi)/do(y) = 0
    phi[ii] = 1.0                   # Reasserting the boundary condition that the middle circle is at 1V
    errors[k] = abs(phi - oldphi).max()

plt.figure(2)
plt.grid(True)
plt.title("Semilog plot of Error")
plt.semilogy(range(Niter), errors, label = "Exact Error plot")
plt.plot(range(0,Niter,50), errors[range(0,Niter,50)], 'ro', label = "Every 50th iteration")
x = np.arange(Niter)
x.shape = (Niter,1)
A = np.c_[np.ones((Niter,1)),x]
b = np.log(errors)
fit1 = np.linalg.lstsq(A,b,rcond = None)[0]
fit1_val = np.dot(A,fit1)
plt.semilogy(range(Niter), np.exp(fit1_val), 'g', label = "Fit using all iterations")
fit2 = np.linalg.lstsq(A[500:],b[500:],rcond = None)[0]
fit2_val = np.dot(A[500:],fit2)
print(fit1); print(fit2)
plt.semilogy(range(500,Niter),np.exp(fit2_val), 'y', label = "Fit excluding first 500 iterations")
plt.xlabel(r"Number of Iterations $\rightarrow$")
plt.ylabel(r"Absolute Maximum Error $\rightarrow$")
plt.legend()
plt.show()

plt.figure(3)
plt.grid(True)
plt.title("Loglog plot of error")
plt.loglog(range(Niter), errors, label = "Exact error plot")
plt.loglog(range(0,Niter,50), errors[range(0,Niter,50)], 'ro', label = "Every 50th iteration")
plt.xlabel(r"Number of iterations $\rightarrow$")
plt.ylabel(r"Maximum absolute error $\rightarrow$")
plt.legend()
plt.show()

fig4 = plt.figure(4)
ax = p3.Axes3D(fig4, auto_add_to_figure=False)
fig4.add_axes(ax)
ax.set_title("3D potential plot (in V)")
surf = ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.jet)
plt.colorbar(surf,shrink = 0.5, label = r"$\phi$ values")
ax.set_xlabel(r"x $\rightarrow$")
ax.set_ylabel(r"y $\rightarrow$")
plt.show()

plt.figure(5)
plt.title("Final Potential Contour Plot (in V)")
plt.contourf(X, Y, phi, levels = 50, cmap = plt.cm.get_cmap("plasma"))
plt.colorbar(label = r"$\phi$ values", orientation = "vertical")
plt.plot(X[ii], Y[ii], 'ro', label = r"$\phi$ = 1V")
plt.xlabel(r"$x \rightarrow$")
plt.ylabel(r"$y \rightarrow$")
plt.legend()
plt.show()

# Calculation of Current Densities

Jx = (0.5)*(phi[1:-1,0:-2] - phi[1:-1,2:])
Jy = (0.5)*(phi[0:-2,1:-1] - phi[2:,1:-1])

Jx_temp = np.zeros((Ny,Nx))
Jy_temp = np.zeros((Ny,Nx))     #The currents arrays, which are 'Ny-2' by 'Nx-2' are padded with zero 
Jx_temp[1:Ny-1,1:Nx-1] = Jx
Jy_temp[1:Ny-1,1:Nx-1] = Jy       #currents so that they can be plotted on a 'Ny' by 'Nx' grid 

plt.figure(5)
plt.title("Vector plot of the current flow")

# print(phi); print(Jx_temp); print(Jy_temp)
plt.quiver(X, Y, Jx_temp, -Jy_temp, scale = 5, label = "Current vector")
plt.plot(X[ii], Y[ii], 'ro', label = r"$\phi$ = 1V")
plt.xlabel(r"$x \rightarrow$")
plt.ylabel(r"$y \rightarrow$")
plt.legend()
plt.show()

T = np.zeros((Ny,Nx)) + 300.0
heats = np.multiply(Jx,Jx) + np.multiply(Jy,Jy)

for k in range(Niter):
    T[1:Ny-1,1:Nx-1] = 0.25*( T[0:Ny-2,1:Nx-1] + T[2:Ny,1:Nx-1] + T[1:Ny-1,0:Nx-2] + T[1:Ny-1,2:Nx] + heats )
    T[0,:Nx] = 300.0
    T[:Ny,0] = T[:Ny,1]
    T[:Ny,Nx-1] = T[:Ny,Nx-2]
    T[Ny-1,:Nx] = T[Ny-2,:Nx]
    T[ii] = 300.0                                   

plt.figure(6)
plt.contourf(X, Y, T, levels=75, cmap=plt.cm.get_cmap("inferno"))
plt.colorbar(label="T values", orientation="vertical")
plt.plot(X[ii], Y[ii], 'ro', label = r"$\phi$ = 1V")
plt.title("Temperature plot (in K)")
plt.legend()
plt.xlabel(r"$x \rightarrow$")
plt.ylabel(r"$y \rightarrow$")
plt.show()