"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 4
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 25-02-2022
-------------------------------------------------------------
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy

pi = np.pi

# Functions to calculate exp(x) and cos(cos(x))
def exp_(val):
    return np.exp(val)

def cos_cos_(val):
    return np.cos(np.cos(val))

plt.figure(1)
plt.grid(True)
x = np.arange(-2*pi,4*pi,0.01)
x_2pi_periodic = x%(2*pi)
plt.xlim([-2*pi, 4*pi])
plt.semilogy(x, exp_(x), 'r', label = "Actual function")
plt.semilogy(x, exp_(x_2pi_periodic), 'c', label = "Expected Fourier function")
plt.ylabel(r"$e^{x} \rightarrow $", fontsize = 13)
plt.xlabel(r"$x \rightarrow $", fontsize = 13)
plt.title(r"Semi-Log Plot of $e^{x}$", fontsize = 15)
plt.legend(loc = 'upper right')
plt.show()

plt.figure(2)
plt.grid(True)
plt.xlim([-2*pi,4*pi])
plt.xlabel(r"$x \rightarrow $", fontsize = 13)
plt.ylabel(r"$\cos(\cos(x)) \rightarrow$", fontsize = 13)
plt.plot(x, cos_cos_(x_2pi_periodic), 'c', linewidth = 8, label = "Expected Fourier function")
plt.plot(x, cos_cos_(x), 'r', label = "Actual function")
plt.title(r"Plot of $\cos(\cos(x))$", fontsize = 15)
plt.legend(loc = 'upper right')
plt.show()

# Function to calculate the fourier coefficient by integration
def fourier_coeff(n,func):
    coeff = np.zeros(n)
    f_cos = lambda x,k : func(x)*np.cos(k*x)
    f_sin = lambda x,k : func(x)*np.sin(k*x)
    coeff[0] = (quad(func,0,2*pi))[0]/(2*pi)
    # For calculating the coefficients a1,a2,a3 ..., a1 - 1, a2 - 3, a3 - 5
    for i in range(1,n,2):
        coeff[i] = ((quad(f_cos,0,2*pi,args = (i//2 + 1)))[0])/pi
    for i in range(2,n,2):
        coeff[i] = ((quad(f_sin,0,2*pi,args = (i/2)))[0])/pi
    return coeff

exp_fourier_coeff = fourier_coeff(51,exp_)
plt.figure(3)
plt.grid(True)
plt.title(r"Semi-Log Plot of Fourier Series Coefficients for $e^{x}$", fontsize = 15)
plt.semilogy(range(51), abs(exp_fourier_coeff), 'ro')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

plt.figure(4)
plt.grid(True)
plt.title(r"Log-Log Plot of Fourier Series Coefficients for $e^{x}$", fontsize = 15)
plt.loglog(range(51), abs(exp_fourier_coeff), 'ro')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

cos_cos_fourier_coeff = fourier_coeff(51, cos_cos_)
plt.figure(5)
plt.grid(True)
plt.title(r"Semi-Log Plot of Fourier Series Coefficients for $\cos(\cos(x))$", fontsize = 15)
plt.semilogy(range(51), abs(cos_cos_fourier_coeff), 'ro')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

plt.figure(6)
plt.grid(True)
plt.title(r"Log-Log Plot of Fourier Series Coefficients for $\cos(\cos(x))$", fontsize = 15)
plt.loglog(range(51), abs(cos_cos_fourier_coeff), 'ro')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

def matrix_gen(num,x):
    # Used to generate A matrix in the question
    n = x.shape[0]
    mat = np.ones((n,1))
    for i in range(1,num+1):
        mat = np.c_[mat,np.cos(i*x)]
        mat = np.c_[mat,np.sin(i*x)]
    return mat

vec_x = np.linspace(0,2*pi,401)     # 401 because 0 is also included
vec_x = vec_x[:-1]              # Drop last term to have proper periodic integral
b = exp_(vec_x)             # exp has been written to take a vector

A = matrix_gen(25,vec_x)

exp_lstsq_coeff = scipy.linalg.lstsq(A,b)[0]      # The [0] is to pull the best fit vector, lstsq returns a list.

exp_coeff_diff = abs(exp_fourier_coeff - exp_lstsq_coeff)
print("The largest deviation between the fourier coefficients and least square coefficients of exp(x) is %f" %(np.amax(exp_coeff_diff)))
plt.figure(7)
plt.grid(True)
plt.title(r"Semi-Log Plot of Fourier Series Coefficients for $e^{x}$", fontsize = 15)
plt.semilogy(range(51),abs(exp_lstsq_coeff),'go')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

plt.figure(8)
plt.grid(True)
plt.title(r"Log-Log Plot of Fourier Series Coefficients for $e^{x}$", fontsize = 15)
plt.loglog(range(51), abs(exp_lstsq_coeff), 'go')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

b = cos_cos_(vec_x)          # cos(cos(x)) has been written to take a vector

cos_cos_lstsq_coeff = scipy.linalg.lstsq(A,b)[0]        # The [0] is to pull the best fit vector, lstsq returns a list.

cos_cos_coeff_diff = abs(cos_cos_fourier_coeff - cos_cos_lstsq_coeff)
print("The largest deviation between the fourier coefficients and least square coefficients of cos(cos(x)) is ",end = '')
print(np.amax(cos_cos_coeff_diff))

plt.figure(9)
plt.grid(True)
plt.title(r"Semi-Log Plot of Fourier Series Coefficients for $\cos(\cos(x))$", fontsize = 15)
plt.semilogy(range(51), abs(cos_cos_lstsq_coeff), 'go')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

plt.figure(10)
plt.grid(True)
plt.title(r"Log-Log Plot of Fourier Series Coefficients for $\cos(\cos(x))$", fontsize = 15)
plt.loglog(range(51), abs(cos_cos_lstsq_coeff), 'go')
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.show()

plt.figure(11)
plt.grid(True)
plt.title(r"Semi-Log Plot of Fourier Series Coefficients for $e^{x}$", fontsize = 15)
plt.semilogy(range(51), abs(exp_lstsq_coeff), 'go', label = "Least Squares Approach")
plt.semilogy(range(51), abs(exp_fourier_coeff), 'ro', label = "Integration Approach")
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.legend(loc = 'upper right')
plt.show()

plt.figure(12)
plt.grid(True)
plt.title(r"Log-Log Plot of Fourier Series Coefficients for $e^{x}$", fontsize = 15)
plt.loglog(range(51), abs(exp_lstsq_coeff), 'go', label = "Least Squares Approach")
plt.loglog(range(51), abs(exp_fourier_coeff), 'ro', label = "Integration Approach")
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.legend(loc = 'upper right')
plt.show()

plt.figure(13)
plt.grid(True)
plt.title(r"Semi-Log Plot of Fourier Series Coefficients for $\cos(\cos(x))$", fontsize = 15)
plt.semilogy(range(51), abs(cos_cos_lstsq_coeff), 'go', label = "Least Squares Approach")
plt.semilogy(range(51), abs(cos_cos_fourier_coeff), 'ro', label = "Integration Approach")
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.legend(loc = 'upper right')
plt.show()

plt.figure(14)
plt.grid(True)
plt.title(r"Log-Log Plot of Fourier Series Coefficients for $\cos(\cos(x))$", fontsize = 15)
plt.loglog(range(51), abs(cos_cos_lstsq_coeff), 'go', label = "Least Squares Approach")
plt.loglog(range(51), abs(cos_cos_fourier_coeff), 'ro', label = "Integration Approach")
plt.xlabel(r"$n \rightarrow$", fontsize = 13)
plt.ylabel(r"Magnitude of Fourier Coefficients $\rightarrow$", fontsize = 13)
plt.legend(loc = 'upper right')
plt.show()

x = np.arange(-2*pi,4*pi,0.05)
x_2pi_periodic = x%(2*pi)

plt.figure(15)
plt.grid(True)
plt.title(r"Expected Fourier plot and Fourier plot with Integration Coefficients of $e^{x}$", fontsize = 13)
plt.semilogy(x, np.dot(matrix_gen(25,x),exp_fourier_coeff), 'go', label = "Fourier with integration coefficients")
plt.semilogy(x, exp_(x_2pi_periodic), 'r', label = "Expected Fourier")
plt.xlabel(r"$x \rightarrow$", fontsize = 13)
plt.ylabel(r"$e^{x} \rightarrow$", fontsize = 13)
plt.ylim([1e-1,1e3])
plt.legend(loc = 'lower left')
plt.show()

plt.figure(16)
plt.grid(True)
plt.title(r"Expected Fourier plot and Fourier plot with Integration Coefficients of $\cos(\cos(x))$", fontsize = 13)
plt.plot(x, np.dot(matrix_gen(25,x),cos_cos_fourier_coeff), 'go', label = "Fourier with integration coefficients")
plt.plot(x, cos_cos_(x), 'r', label = "Expected Fourier")
plt.xlabel(r"$x \rightarrow$", fontsize = 13)
plt.ylabel(r"$\cos(\cos(x)) \rightarrow$", fontsize = 13)
plt.legend(loc = 'upper right')
plt.show()

plt.figure(17)
plt.grid(True)
plt.title(r"Expected Fourier plot and Fourier plot with Least Square Coefficients of $e^{x}$", fontsize = 13)
plt.semilogy(x, np.dot(matrix_gen(25,x),exp_lstsq_coeff), 'go', label = "Fourier with least square coefficients")
plt.semilogy(x, exp_(x_2pi_periodic), 'r', label = "Expected Fourier")
plt.xlabel(r"$x \rightarrow$", fontsize = 13)
plt.ylabel(r"$e^{x} \rightarrow$", fontsize = 13)
plt.ylim([1e-1,1e3])
plt.legend(loc = 'lower left')
plt.show()

plt.figure(18)
plt.grid(True)
plt.title(r"Expected Fourier plot and Fourier plot with Least Square Coefficients of $\cos(\cos(x))$", fontsize = 13)
plt.plot(x, np.dot(matrix_gen(25,x),cos_cos_lstsq_coeff), 'go', label = "Fourier with least square coefficients")
plt.plot(x, cos_cos_(x), 'r', label = "Expected Fourier")
plt.xlabel(r"$x \rightarrow$", fontsize = 13)
plt.ylabel(r"$\cos(\cos(x)) \rightarrow$", fontsize = 13)
plt.legend(loc = 'upper right')
plt.show()