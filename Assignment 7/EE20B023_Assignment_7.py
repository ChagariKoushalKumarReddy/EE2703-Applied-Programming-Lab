"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 7
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 25-03-2022
-------------------------------------------------------------
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

arrow = r"$\rightarrow$"
pi = np.pi
R1 = 1e4
R2 = 1e4
R3 = 1e4
C1 = 1e-9
C2 = 1e-9
Vi = 1
G = 1.586
s = sp.symbols('s')

# Function to calculate the algebraic low pass tf
def lowpass(R1,R2,C1,C2,G,Vi):
    A = sp.Matrix([ [0,0,1,-1/G], [-1/(1+s*R2*C2),1,0,0], [0,-G,G,1], [-(1/R1)-(1/R2)-s*C1,1/R2,0,s*C1]])
    b = sp.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    Vo = V[3]
    return Vo

# Function to compute scipy.signal.lti tf from sympy algebraic expression of tf
def sympy_tf2sig_tf(tf):
    n,d = tf.as_numer_denom()
    num = n.as_poly(s).all_coeffs()
    den = d.as_poly(s).all_coeffs()
    num = [float(i) for i in num]
    den = [float(i) for i in den]
    TF = sig.lti(num,den)
    return TF

lowpass_tf = lowpass(R1,R2,C1,C2,G,Vi)
lowpass_TF = sympy_tf2sig_tf(lowpass_tf)
w = np.logspace(0,8,1001)
w,mag,phase = lowpass_TF.bode(w)
plt.figure(1)
plt.grid(True)
plt.title("Magnitude response of Low Pass Butterworth Filter")
plt.semilogx(w,mag)
plt.xlabel(r"$\omega$" + arrow)
plt.ylabel(r"$|H(j\omega)|$" + arrow)
plt.show()

t = np.linspace(0,0.001,1000)
t1,step_response = sig.step(lowpass_TF,None,T = t)
plt.figure(2)
plt.grid(True)
plt.title("Step Response of Low Pass Butterworth Filter")
plt.plot(t1,step_response)
plt.xlabel("t" + arrow)
plt.ylabel("Filter Response of $u(t)$" + arrow)
plt.show()
print(lowpass_TF)

# Q2

def sum_of_sinusoids_response(w1,w2,t,H):
    x = np.sin(w1*t) + np.cos(w2*t)
    Vo = sig.lsim(H,x,t)[1]
    return Vo

t = np.linspace(0, 0.01, 100000)
V_input = np.sin(2*pi*1e3*t) + np.cos(2*pi*1e6*t)
Vo = sum_of_sinusoids_response(2*pi*1e3,2*pi*1e6,t,lowpass_TF)
plt.figure(3)
plt.title("LPF response to a mixed frequency sinusoid")
plt.plot(t,V_input,label = "$V_{i}$")
plt.plot(t,Vo,label = "$V_{o}$")
plt.xlabel("$t$" + arrow)
plt.ylabel("$V(t)$" + arrow)
plt.legend()
plt.show()

# Function to calculate the algebraic high pass tf
def highpass(R1,R3,C1,C2,G,Vi):
    A = sp.Matrix([ [0,0,1,-1/G], [-s*C2,(1/R3)+s*C2,0,0], [0,-G,G,1], [s*C1+s*C2+(1/R1),-s*C2,0,-1/R1]])
    b = sp.Matrix([0,0,0,Vi*s*C1])
    V = A.inv()*b
    Vo = V[3]
    return Vo

high_pass_tf = highpass(R1,R3,C1,C2,G,Vi)
highpass_TF = sympy_tf2sig_tf(high_pass_tf)
w = np.logspace(0,10,1001)
w,mag,phase = highpass_TF.bode(w)
plt.figure(4)
plt.grid(True)
plt.title("Magnitude response of High Pass Butterworth Filter")
plt.semilogx(w,mag)
plt.xlabel(r"$\omega$" + arrow)
plt.ylabel(r"$|H(j\omega)|$" + arrow)
plt.show()

t = np.linspace(0,0.001,1000)
t1,step_response = sig.step(highpass_TF,None,T = t)
plt.figure(5)
plt.grid(True)
plt.title("Step Response of High Pass Butterworth Filter")
plt.plot(t1,step_response)
plt.xlabel("t" + arrow)
plt.ylabel("Filter Response of $u(t)$" + arrow)
plt.show()

t = np.linspace(0, 1e-5, 100000)
V_input = np.sin(2*pi*1e3*t) + np.cos(2*pi*1e6*t)
Vo = sum_of_sinusoids_response(2*pi*1e3,2*pi*1e6,t,highpass_TF)
plt.figure(6)
plt.grid(True)
plt.title("HPF response to a mixed frequency sinusoid")
plt.plot(t,V_input,label = "$V_{i}$")
plt.plot(t,Vo,label = "$V_{o}$")
plt.xlabel("$t$" + arrow)
plt.ylabel("$V(t)$" + arrow)
plt.legend()
plt.show()

# Damped Sinusoid Responses - e^(-At)sin(Bt)u(t)
def damped_sinusoid_response(H,A,B,t):
    V_in = np.exp(-A*t)*np.cos(B*t)
    V_o = sig.lsim(H,V_in,t)[1]
    return V_o

# We will find the response for A = 200, w = 1000 Hz and w = 1MHz of both LPF and HPF

t = np.linspace(0,0.01,501)
V_lpf1 = damped_sinusoid_response(lowpass_TF,200,2*pi*1000,t)
plt.figure(7)
plt.grid(True)
plt.title("LPF Damped sinusoid response with low frequency")
plt.plot(t,V_lpf1)
plt.xlabel("$t$" + arrow)
plt.ylabel("$V_{o}(t)$" + arrow)
plt.show()

V_hpf1 = damped_sinusoid_response(highpass_TF,200,2*pi*1000,t)
plt.figure(8)
plt.grid(True)
plt.title("HPF Damped sinusoid response with low frequency")
plt.plot(t,V_hpf1)
plt.xlabel("$t$" + arrow)
plt.ylabel("$V_{o}(t)$" + arrow)
plt.show()

t = np.linspace(0,0.01,100000)
V_lpf2 = damped_sinusoid_response(lowpass_TF,200,2*pi*1e6,t)
plt.figure(9)
plt.grid(True)
plt.title("LPF Damped sinusoid response with high frequency")
plt.plot(t,V_lpf2)
plt.xlabel("$t$" + arrow)
plt.ylabel("$V_{o}(t)$" + arrow)
plt.show()

V_hpf2 = damped_sinusoid_response(highpass_TF,200,2*pi*1e6,t)
plt.figure(10)
plt.grid(True)
plt.title("HPF Damped sinusoid response with high frequency")
plt.plot(t,V_hpf2)
plt.xlabel("$t$" + arrow)
plt.ylabel("$V_{o}(t)$" + arrow)
plt.show()

# Plotting the damped sinusoid signals for which the response is calculated above
plt.figure(11)
plt.grid(True)
plt.title("Damped sinusoid with low frequency")
plt.plot(t,np.exp(-200*t)*np.cos(2*pi*1000*t))
plt.xlabel("$t$" + arrow)
plt.ylabel("Signal")
plt.show()

plt.figure(12)
plt.grid(True)
plt.title("Damped sinusoid with high frequency")
plt.plot(t,np.exp(-200*t)*np.cos(2*pi*1e6*t))
plt.xlabel("$t$" + arrow)
plt.ylabel("Signal")
plt.show()
