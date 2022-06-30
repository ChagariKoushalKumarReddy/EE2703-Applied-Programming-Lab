"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 10
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 06-04-2022
-------------------------------------------------------------
"""

import scipy.signal as sig 
import matplotlib.pyplot as plt
import pylab as p

pi = p.pi

FIR_coefficients = p.zeros(12)
i = 0
with open('h.csv','r') as FIR:
        for line in FIR:
                FIR_coefficients[i] = float(line)
                i+=1

figure_no = 1
def plotter(x,y,xlabel = '',ylabel='',title=''):
        global figure_no
        plt.figure(figure_no)
        plt.grid(True)
        # plt.stem(x,y,linefmt = ':',basefmt= 'C2-')
        plt.plot(x,y)
        plt.xlabel(xlabel,size=12)
        plt.ylabel(ylabel,size=12)
        plt.title(title,size=15)
        # plt.show()
        figure_no += 1


# Magnitude and Phase response of FIR filter
w,h = sig.freqz(FIR_coefficients)

plt.figure(1)
plt.subplot(2,1,1)
plt.title("FIR filter",size =15)
plt.grid(True)
plt.plot(w,abs(h))
plt.ylabel("Magnitude",size=12)
plt.subplot(2,1,2)
plt.grid(True)
plt.plot(w,p.angle(h))
plt.ylabel("Phase",size=12)
plt.xlabel(r"$\omega$",size=12)
# plt.show()
figure_no += 1


n = p.arange(2**10)
x = p.cos(0.2*pi*n) + p.cos(0.85*pi*n)
plotter(n,x,'n','x','Input Signal')

# Linear Convolution
conv = p.convolve(x,FIR_coefficients)
n = p.arange(0,len(conv))
plotter(n,conv,'n','y','Linear Convolution result')

# Circular Convolution
n = p.arange(2**10)
y1 = p.ifft(p.fft(x)*p.fft(p.concatenate((FIR_coefficients,p.zeros(len(x)-len(FIR_coefficients))))))
plotter(n,p.real(y1),'n','y','Circular Convolution')

# Circular Convolution using Linear Convolution

P = len(FIR_coefficients)
m = p.ceil(p.log2(P))
print(P,m)
FIR_pad = p.concatenate((FIR_coefficients,p.zeros(int(2**m)-P)))
print(P, FIR_pad)
P = len(FIR_pad)
a = len(x)
b = int(p.ceil(a/2**m))
x_pad = p.concatenate((x,p.zeros(b*int(2**m)-a)))
L = len(x_pad)
print(b,L,x_pad)
y = p.zeros(L+P-1)
for i in range(b):
        temp = p.concatenate((x_pad[i*P:(i+1)*P],p.zeros(P-1)))
        y[i*P:(i+1)*P+P-1] += p.ifft(p.fft(temp) * p.fft( p.concatenate( (FIR_pad,p.zeros(len(temp)-len(FIR_pad))) ))).real

