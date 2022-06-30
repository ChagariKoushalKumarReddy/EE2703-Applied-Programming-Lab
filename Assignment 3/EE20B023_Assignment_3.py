"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 3
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 15-02-2022
-------------------------------------------------------------
"""

# Importing necessary modules

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy

# Checking if the fitting.dat file is present in the current directory
try:
    f = np.loadtxt('fitting.dat')

except:
    print("The file fitting.dat is not present in the current directory!")
    exit()

sigma = np.logspace(-1,-3,9)

# True values of A, B
A = 1.05
B = -0.105

# Function to calculate the true value of f(t)
def g(t,A,B):
    j2_t = sp.jv(2,t)
    return A*j2_t + B*t

time = [line[0] for line in f]
f_with_noise = []
for i in range(1,10):
    f_with_noise.append( [line[i] for line in f] )

labels = []         # labels for legends in the graph

for i in range(1,10):
    s = r"$\sigma_" + str(i) + "=" + str(round(sigma[i-1],3)) + "$"
    labels.append(s)

labels.append("True Value")

#First figure, which contains the plots of all the noisy datasets is generated (Also contains the true values plot)
print("Plotting the noisy function f(t) as compared to true value for different noise levels")
plt.figure(0)
plt.title("Q4: Data to be fitted to theory")
for i in range(0,9):
    plt.plot(time,f_with_noise[i])
true_value = [g(t,A,B) for t in time]
plt.plot(time,true_value,'k', linewidth = 3)
plt.legend(labels)
plt.grid(True)
plt.xlabel(r"$t\rightarrow$", size = 15)
plt.ylabel(r"$f(t) + noise\rightarrow$",size = 15)
plt.show()

#Second figure, which contains the errorbars of the noisy data with sigma=0.1 is generated (Also contains the true values plot) 
print("Plotting Error bar")
plt.figure(1)
plt.title("Q5: Data points for "+ r"$\sigma = 0.10$" + " along with the exact function")
plt.plot(time,true_value,'k', label = 'f(t)')
noise = []
for i in range(0,len(time)):
    noise.append(true_value[i] - f_with_noise[0][i])
stdev = np.std(noise)
plt.errorbar(time[::5],f_with_noise[0][::5], stdev, fmt = 'ro', label = 'Errorbar')
plt.grid(True)
plt.xlabel(r"$t\rightarrow$", size = 15)
plt.legend()
plt.show()


print("Please wait this may take some time")
a = np.linspace(0, 2, 21)
b = np.linspace(-0.2, 0, 21)
MSE = np.zeros((len(a), len(b)))

# Calculating the mean square error for different values of A, B
for i in range(len(a)):
    for j in range(len(b)):
        diff = np.zeros(0)      # Empty array
        for k in range(0, len(time)):
            g_value = np.array([g(t, a[i], b[j]) for t in time])
            diff = np.append(diff, (f_with_noise[0][k] - g_value[k])**2)    # f_with_noise[0][k] is the first column of data
        MSE[i][j] = diff.mean()

#Third figure, which contains the contours of the mean sqaured errors for different combinations of A, B is generated
print("Plotting Contour plot")
plt.figure(2)
contour_levels = np.linspace(0.025, 0.5, 20)
contour_set = plt.contour(a, b, MSE, contour_levels)
plt.clabel(contour_set, contour_levels[:4])
plt.title(r"Q8: Contour plot of $\epsilon_{ij}$")
plt.plot(1.05,-0.105,'ro')
plt.annotate('Exact Location',(1.05,-0.105))
plt.xlabel(r"$A\rightarrow$", size = 15)
plt.ylabel(r"$B\rightarrow$", size = 15)
plt.show()


first_column = [sp.jv(2,t) for t in time]
second_column = time
M = np.c_[first_column,second_column]

A_estimations = []
B_estimations = []
for i in range(9):
    x = scipy.linalg.lstsq(M,f_with_noise[i])   # Obtaining the least squares solution of all columns
    A_estimations.append(x[0][0])
    B_estimations.append(x[0][1])

A_error = [abs(i-1.05) for i in A_estimations]
B_error = [abs(i+0.105) for i in B_estimations]

#Fourth figure, which contains the linear plot of errors in estimate of A, B with respect to the noise is generated 
print("Plotting errors in estimated A, B with standard deviations of noise")
plt.figure(3)
plt.title("Q10: Variation of error with noise")
plt.plot(sigma, A_error, linestyle = '--', marker = 'o', color = 'r', label = '$Aerr$')
plt.plot(sigma, B_error, linestyle = '--', marker = 'o', color = 'g', label = '$Berr$')
plt.grid(True)
plt.xlabel(r"$Noise\ standard\ deviation \rightarrow$", size = 15)
plt.ylabel(r"$MS\ error \rightarrow$", size = 15)
plt.legend()
plt.show()

#Fifth figure, which contains the logarithmic plot of errors in estimate of A, B with respect to the noise is generated 
print("Plotting errors in estimated A, B on log log scale")
plt.figure(4)
plt.title("Q11: Variation of error with noise on log scale")
plt.loglog(sigma, A_error, linestyle = '', marker = 'o', color = 'r', label = '$Aerr$')
plt.loglog(sigma, B_error, linestyle = '', marker = 'o', color = 'g', label = '$Berr$')
plt.errorbar(sigma, A_error, A_error, fmt = 'ro')
plt.errorbar(sigma, B_error, B_error, fmt = 'go')
plt.grid(True)
plt.xlabel(r"$\sigma_n \rightarrow$", size = 15)
plt.ylabel(r"$MS\ error \rightarrow$", size = 15)
plt.legend()
plt.show()