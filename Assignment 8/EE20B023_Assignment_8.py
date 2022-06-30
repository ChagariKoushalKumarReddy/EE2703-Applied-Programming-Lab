"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 8
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 17-04-2022
-------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from pylab import *

arrow = r"$\rightarrow$"
pi = np.pi

x = linspace(0,2*np.pi,129)
x = x[:-1]
y = np.sin(5*x)
Y = fftshift(fft(y))/128.0
w = linspace(-64,63,128)
figure(1)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$" + arrow,size=12)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$" + arrow, size=12)
xlabel(r"$k$" + arrow, size=12)
grid(True)
show()

# Spectrum of sin(5t) with period 8pi and N = 1024 for better resolution
x = linspace(0,8*np.pi,1025)
x = x[:-1]
y = np.sin(5*x)
Y = fftshift(fft(y))/1024.0
w = linspace(-128,(1024-2)/8,1024)
figure(2)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$" + arrow,size=12)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$" + arrow, size=12)
xlabel(r"$k$" + arrow, size=12)
grid(True)
show()

# Wrong plot of AM modulated signal: (1+0.1cos(t))cos(10t)

t = linspace(0,2*np.pi,129)
t = t[:-1]
y = (1+0.1*np.cos(t))*np.cos(10*t)
Y = fftshift(fft(y))/128.0
w = linspace(-64,63,128)
figure(3)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$"+arrow,size=12)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$"+arrow,size=12)
xlabel(r"$\omega$"+arrow,size=12)
grid(True)
show()

# Correct plot using better resolution
t = linspace(-4*np.pi,4*np.pi,513)
t = t[:-1]
y = (1+0.1*np.cos(t))*np.cos(10*t)
Y = fftshift(fft(y))/512.0
w = linspace(-64,64,513)
w = w[:-1]
figure(4)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$"+arrow,size=12)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$"+arrow,size=12)
xlabel(r"$\omega$"+arrow,size=12)
grid(True)
show()

# sin^3(t)
t = linspace(-4*np.pi,4*np.pi,513)
t = t[:-1]
y = np.sin(t)**3
Y = fftshift(fft(y))/512.0
w = linspace(-64,64,513)
w = w[:-1]
figure(5)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$"+arrow,size=12)
title(r"Spectrum of $\sin^{3}(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$"+arrow,size=12)
xlabel(r"$\omega$"+arrow,size=12)
grid(True)
show()

# cos^3(t)
t = linspace(-4*np.pi,4*np.pi,513)
t = t[:-1]
y = np.cos(t)**3
Y = fftshift(fft(y))/512.0
w = linspace(-64,64,513)
w = w[:-1]
figure(6)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$"+arrow,size=12)
title(r"Spectrum of $\cos^{3}(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii = where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$"+arrow,size=12)
xlabel(r"$\omega$"+arrow,size=12)
grid(True)
show()

# Spectrum of frequency modulated signal

beta = 5
k = [i for i in range(-20,20)]
k = np.array(k)
bessels = (sp.jv(k,beta)/2)*(1j**k)

figure(7)
subplot(2,1,1)
grid(True)
stem(k,abs(bessels))
title("Fourier Coefficients Plot")
ylabel(r"$|a_{k}|$"+arrow,size=12)
subplot(2,1,2)
ii = where(abs(bessels)>1e-3)
plot(k[ii],angle(bessels[ii]),'go',lw=2)
xlim([-20,20])
ylabel(r'Phase of $a_{k}$' + arrow,size=12)
xlabel(r"$\omega$"+arrow,size=12)
grid(True)
show()

t = linspace(-16*np.pi,16*np.pi,4097)
t = t[:-1]
y = np.cos(20*t+5*np.cos(t))
Y = fftshift(fft(y))/4096.0
w = linspace(-128,128,4097)
w = w[:-1]
figure(8)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-35,35])
ylabel(r"$|Y|$"+arrow,size=12)
title(r"Spectrum of $cos(20t+5cos(t))$")
grid(True)
subplot(2,1,2)
ii = where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-35,35])
ylabel(r"Phase of $Y$"+arrow,size=12)
xlabel(r"$\omega$"+arrow,size=12)
grid(True)
show()

'''
Evaluating CTFT of Gaussian function e^(-t^2/2)
Time Period = 20, Wo = pi/10, N = 1024, Ws = 1024(pi/10)
Hence Ws is sufficiently large such that X(Ws/2) is small
'''
i = 8
def guassian_spectrum(period,N,flag=1):
        global i
        t = linspace(0,period,N+1)
        t = t[:-1]
        y = np.exp(-t*t/2)
        ii = where(t>=period/2)
        y[ii] = np.exp(-(period-t[ii])*(period-t[ii])/2)

        # Plotting the Gaussina in one time period from (0,Period)
        if i==8:
                figure(i+1)
                grid(True)
                plot(t,y)
                title("First period of Gaussian")
                xlabel("Time" + arrow, size = 12)
                ylabel("Function" + arrow)
                show()

        Y = fftshift(fft(y))*(period/N)                         # a[k]*Ts = X_fin(kWo)
        lim = N*pi/period
        w = linspace(-lim,lim-(2*pi/period),N)
        i += 1
        if flag==0:
                figure(i+1)
                subplot(2,1,1)
                plot(w,abs(Y),lw=2)
                xlim([-10,10])
                ylabel(r"$|Y|$"+arrow,size=12)
                title(r"Spectrum of $exp(-\frac{t^{2}}{2})$ for period = " + str(period) + r" and N = " + str(N))
                grid(True)
                subplot(2,1,2)
                plot(w,angle(Y),'go',lw=2)
                xlim([-10,10])
                ylabel(r"Phase of $Y$"+arrow,size=12)
                xlabel(r"$\omega$"+arrow,size=12)
                grid(True)
                show()

        Y_true = np.sqrt(2*pi)*np.exp(-w*w/2)
        error = np.max(abs(Y-Y_true))
        return error

guassian_spectrum(20,1024,0)

# Generating the spectrum for different values of period but with same N - Observe that resolution increases
for j in [8,16,24,32,40]:
        guassian_spectrum(j,1024,0)

errors = []
periods = []

for j in range(100):
        period = 1.5*(j+1)
        error = guassian_spectrum(period,1024)
        periods.append(period)
        errors.append(error)

figure(16)
plot(periods,errors)
xlabel("Period"+arrow)
ylabel("Max. error in DFT"+arrow)
title("Max. error vs Period plot")
grid(True) 
show()