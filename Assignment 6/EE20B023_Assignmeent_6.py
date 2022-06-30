"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 6
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 15-03-2022
-------------------------------------------------------------
"""

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

arrow = r"$\rightarrow$"

# For Q1, Q2 considering the form cos(at)e^(-bt)u(t)
def Q1_2(a,b):
    num = np.poly1d([1,b])
    den_temp = np.poly1d([1,2*b,b*b + a*a])
    diff_eqn = np.poly1d([1,0,2.25])
    den = np.polymul(den_temp,diff_eqn)
    H = sp.lti(num,den)                                 # The laplace transform of x(t) is generated
    t,x = sp.impulse(H,None,np.linspace(0,50,501))      # x(t) is found from X(s) using the sp.impulse function
    return t,x

plt.figure(1)
t1,x1 = Q1_2(a = 1.5, b = 0.5)
plt.grid(True)
plt.title("$x(t)$ vs $t$ for decay constant = $0.50$")
plt.plot(t1,x1)
plt.xlabel("$t$" + arrow)
plt.ylabel("$x(t)$" + arrow)
plt.show()

plt.figure(2)
t2,x2 = Q1_2(a = 1.5, b = 0.05)
plt.grid(True)
plt.title("$x(t)$ vs $t$ for decay constant = $0.05$")
plt.plot(t2,x2)
plt.xlabel("$t$" + arrow)
plt.ylabel("$x(t)$" + arrow)
plt.show()

# Q3

Sys_transfer_function = sp.lti([1],[1,0,2.25])          # X(s)/F(s)
t = np.linspace(0,100,1001)

# For loop to loop over the frequencies
for i in range(1,6):        # Total 5 graphs
    f_t = np.cos((1.4+0.05*(i-1))*t)*np.exp(-0.05*t)
    x_t = sp.lsim(Sys_transfer_function,f_t,t)[1]       # Output time response is obtained using sp.lsim function
    plt.figure(2+i)
    plt.grid(True)
    wd = 1.4 + 0.05*(i-1)
    plt.title("$x(t)$ vs $t$ for w = %.2f rad/s" %wd)
    plt.plot(t,x_t)
    plt.xlabel("$t$" + arrow)
    plt.ylabel("$x(t)$" + arrow)
    plt.show()

# Q4

X = sp.lti([1,0,2],[1,0,3,0])
t,x = sp.impulse(X,None,np.linspace(0,20,501))
Y = sp.lti([2],[1,0,3,0])
t,y = sp.impulse(Y,None,np.linspace(0,20,501))      # The responses x(t),y(t) are de-coupled and obtained for a given time vector
plt.figure(8)
plt.title("$x, y$ vs $t$")
plt.grid(True)
plt.plot(t,x,label = "$x(t)$")
plt.plot(t,y,label = "$y(t)$")
plt.xlabel("$t$" + arrow)
plt.ylabel("$x(t)$, $y(t)$")
plt.legend()
plt.show()

# Q5

R = 100
L = 1e-6
C = 1e-6
Steady_State_tf = sp.lti([1],[L*C,C*R,1])
w,mag,phase = Steady_State_tf.bode()            # Respective values of frequency, magnitude and phase are obtained
plt.figure(9)
plt.subplot(2,1,1)
plt.subplots_adjust()
plt.grid(True)
plt.title("Magnitude Response")
plt.semilogx(w,mag)
plt.xlabel("Frequency (in rad/s)" + arrow)
plt.ylabel("|H(s)| in dB" + arrow)
plt.subplot(2,1,2)
plt.grid(True)
plt.semilogx(w,phase)
plt.xlabel("Frequency (in rad/s)" + arrow)
plt.ylabel("Phase of H(s) in deg" + arrow)
plt.title("Phase Response")
plt.show()

# Q6
t1 = np.linspace(0,30*1e-6,301)
Vi_t = np.cos(1e3*t1) - np.cos(1e6*t1)
Vo_t = sp.lsim(Steady_State_tf,Vi_t,t1)[1]              # Output time response till 30 micro seconds is obtained
plt.figure(10)
plt.grid(True)
plt.title(r"Output Voltage Response (till 30$\mu$s)")
plt.plot(t1,Vo_t)
plt.xlabel("$t$" + arrow)
plt.ylabel("$V_{o}(t)$" + arrow)
plt.show()

t2 = np.linspace(0,0.01,100001)
Vi_t = np.cos(1e3*t2) - np.cos(1e6*t2)
Vo_t = sp.lsim(Steady_State_tf,Vi_t,t2)[1]              # Output time response till 10 milli seconds is obtained
plt.figure(11)
plt.grid(True)
plt.title(r"Output Voltage Response (till 10ms)")
plt.plot(t2,Vo_t)
plt.xlabel("$t$" + arrow)
plt.ylabel("$V_{o}(t)$" + arrow)
plt.show()