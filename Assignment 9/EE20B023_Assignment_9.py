"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 9
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 24-04-2022
-------------------------------------------------------------
"""

import pylab as p
pi = p.pi

# Plotting the actual function for which we want the Spectrum
t1 = p.linspace(-pi,pi,65);t1=t1[:-1]
t2 = p.linspace(-3*pi,-pi,65);t2=t2[:-1]
t3 = p.linspace(pi,3*pi,65);t3=t3[:-1]
# y=sin(sqrt(2)*t)
p.figure(1)
p.plot(t1,p.sin(p.sqrt(2)*t1),'b',lw=2)
p.plot(t2,p.sin(p.sqrt(2)*t2),'r',lw=2)
p.plot(t3,p.sin(p.sqrt(2)*t3),'r',lw=2)
p.ylabel(r"$y$",size=16)
p.xlabel(r"$t$",size=16)
p.title(r"$\sin\left(\sqrt{2}t\right)$")
p.grid(True)
p.show()

''' 
Plotting the function for which the program was calculating the Spectrum 
It was taking the periodic extension of the function in the given interval
'''
y= p.sin(p.sqrt(2)*t1)
p.figure(2)
p.plot(t1,y,'b',lw=2)
p.plot(t2,y,'r',lw=2)
p.plot(t3,y,'r',lw=2)
p.ylabel(r"$y$",size=16)
p.xlabel(r"$t$",size=16)
p.title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
p.grid(True)
p.show()

# DFT spectrum of sin(root(2)t) without windowing
t = p.linspace(-pi,pi,65);t=t[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = p.sin(p.sqrt(2)*t)
y[0] = 0 # the sample corresponding to -tmax should be set zeroo
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/64.0
w = p.linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
p.figure(3)
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-10,10])
p.ylabel(r"$|Y|$",size=12)
p.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-10,10])
p.ylabel(r"Phase of $Y$",size=12)
p.xlabel(r"$\omega$",size=12)
p.grid(True)
p.show()

# Spectrum of the ramp function
t = p.linspace(-pi,pi,65);t=t[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = t
y[0] = 0 # the sample corresponding to -tmax should be set zeroo
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/64.0
w = p.linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
p.figure(4)
p.semilogx(abs(w),20*p.log10(abs(Y)),lw=2)
p.xlim([1,10])
p.ylim([-20,0])
p.xticks([1,2,5,10],["1","2","5","10"],size=10)
p.ylabel(r"$|Y|$ (dB)",size=12)
p.title(r"Spectrum of a digital ramp")
p.xlabel(r"$\omega$",size=12)
p.grid(True)
p.show()

# Graph showing periodic extension of windowed sin(sqrt(2)t)
n = p.arange(64)
wnd = p.fftshift(0.54+0.46*p.cos(2*pi*n/63))
y = p.sin(p.sqrt(2)*t1)*wnd
p.figure(5)
p.plot(t1,y,'b',lw=2)
p.plot(t2,y,'r',lw=2)
p.plot(t3,y,'r',lw=2)
p.ylabel(r"$y$",size=12)
p.xlabel(r"$t$",size=12)
p.title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
p.grid(True)
p.show()


# Graph of windowed 
t = p.linspace(-pi,pi,65);t=t[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = p.arange(64)
wnd = p.fftshift(0.54+0.46*p.cos(2*pi*n/63))
y = p.sin(p.sqrt(2)*t)*wnd
y[0] = 0 # the sample corresponding to -tmax should be set zeroo
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/64.0
w = p.linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
p.figure(6)
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-8,8])
p.ylabel(r"$|Y|$",size=12)
p.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-8,8])
p.ylabel(r"Phase of $Y$",size=12)
p.xlabel(r"$\omega$",size=12)
p.grid(True)
p.show()

# Periodic repetition graph of windowed sin(root2*t) signal
t1 = p.linspace(-4*pi,4*pi,257); t1 = t1[:-1]
t2 = p.linspace(4*pi,12*pi,257); t2 = t2[:-1]
t3 = p.linspace(-12*pi,-4*pi,257); t3 = t3[:-1]
n = p.arange(256)
wnd = p.fftshift(0.54+0.46*p.cos(2*pi*n/255))
y = p.sin(p.sqrt(2)*t1)*wnd
p.figure(7)
p.grid(True)
p.plot(t1,y,'b',lw = 2)
p.plot(t2,y,'r',lw=2)
p.plot(t3,y,'r',lw=2)
p.title(r"Periodic repetition of $\sin(\sqrt{2}t)x w(t)$ with period $8\pi$ and $N=256$")
p.xlabel("t",size=12)
p.ylabel("y",size=12)
p.show()

# Spectrum of windowed sin signal with increased resolution
t = p.linspace(-4*pi,4*pi,257);t=t[:-1]
dt = t[1]-t[0]
fmax = 1/dt # fmax is the sampling frequency
n = p.arange(256)
wnd = p.fftshift(0.54+0.46*p.cos(2*pi*n/256))
y = p.sin(p.sqrt(2)*t)
y = y*wnd
y[0] = 0 # the sample corresponding to -tmax should be set zeroo
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/256.0
w = p.linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
p.figure(8)
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
p.xlim([-4,4])
p.ylabel(r"$|Y|$",size=12)
p.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-4,4])
p.ylabel(r"Phase of $Y$",size=12)
p.xlabel(r"$\omega$",size=12)
p.grid(True)
p.show()


# Specturm of cos^3(0.86t)
t = p.linspace(-4*pi,4*pi,257);t=t[:-1]
dt = t[1]-t[0]
fmax = 1/dt # fmax is the sampling frequency
wo = 0.86
y = p.cos(wo*t)**3
y[0] = 0 # the sample corresponding to -tmax should be set zero
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/256.0
w = p.linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
p.figure(9)
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',lw=2)
p.xlim([-8,8])
p.ylabel(r"$|Y|$",size=12)
p.title(r"Spectrum of $\cos^{3}(0.86t)$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-8,8])
p.ylabel(r"Phase of $Y$",size=12)
p.xlabel(r"$\omega$",size=12)
p.grid(True)
p.show()

# Specturm of cos^3(0.86t) with Hamming Window
t = p.linspace(-4*pi,4*pi,257);t=t[:-1]
dt = t[1]-t[0]
fmax = 1/dt # fmax is the sampling frequency
wo = 0.86
n = p.arange(256)
wnd = p.fftshift(0.54+0.46*p.cos(2*pi*n/255))
y = (p.cos(wo*t)**3)*wnd
y[0] = 0 # the sample corresponding to -tmax should be set zero
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/256.0
w = p.linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
p.figure(10)
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',lw=2)
p.xlim([-8,8])
p.ylabel(r"$|Y|$",size=12)
p.title(r"Spectrum of $\cos^{3}(0.86t)$ with Hamming Window")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-8,8])
p.ylabel(r"Phase of $Y$",size=12)
p.xlabel(r"$\omega$",size=12)
p.grid(True)
p.show()

# Estimation of frequency and initial phase of a sinusoid using DFT
omega = 0.5
delta = pi
pow = 3.3

t = p.linspace(-4*pi,4*pi,1025); t = t[:-1]
fmax = 1/(t[1]-t[0])
w = p.linspace(-pi*fmax,pi*fmax,1025); w = w[:-1]
y1 = p.cos(omega*t + delta)
n = p.arange(1024)
wnd = p.fftshift(0.54+0.46*p.cos(2*pi*n/1023))
y = y1*wnd
y[0] = 0
y = p.fftshift(y)
Y = p.fftshift(p.fft(y))/1024
ii = p.where(w>0)
omega_o = p.sum((abs(Y[ii])**pow)*w[ii])/p.sum(abs(Y[ii])**pow)
delta_o = p.angle(Y[:: -1][p.argmax(abs(Y [:: -1]))])
print(omega_o,delta_o)
p.figure(11)
p.subplot(2,1,1)
p.xlim([-3,3])
p.ylabel("|Y|",size=12)
p.plot(w,abs(Y),'b',lw = 2)
p.grid(True)
p.title(r"Spectrum of $\cos(%f t + %f)$ with Hamming Window" %(omega,delta))
p.subplot(2,1,2)
p.grid(True)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'ro',lw=2)
p.xlabel("t",size=12)
p.ylabel("Phase of Y",size=12)
p.xlim([-3,3])
p.show()

# Estimation of frequency and initial phase of a sinusoid using DFT with white gaussian noise
omega = 0.5
delta = pi
pow = 3.3

t = p.linspace(-4*pi,4*pi,1025); t = t[:-1]
fmax = 1/(t[1]-t[0])
w = p.linspace(-pi*fmax,pi*fmax,1025); w = w[:-1]
y1 = p.cos(omega*t + delta) + 0.1*p.randn(1024)
n = p.arange(1024)
wnd = p.fftshift(0.54+0.46*p.cos(2*pi*n/1023))
y = y1*wnd
y[0] = 0
y = p.fftshift(y)
Y = p.fftshift(p.fft(y))/1024
ii = p.where(w>0)
omega_o = p.sum((abs(Y[ii])**pow)*w[ii])/p.sum(abs(Y[ii])**pow)
delta_o = p.angle(Y[:: -1][p.argmax(abs(Y[:: -1]))])
print(omega_o,delta_o)
p.figure(12)
p.subplot(2,1,1)
p.xlim([-3,3])
p.ylabel("|Y|",size=12)
p.plot(w,abs(Y),'b',lw = 2)
p.grid(True)
p.title(r"Spectrum of noisy $\cos(%f t + %f)$ with Hamming Window" %(omega,delta))
p.subplot(2,1,2)
p.grid(True)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'ro',lw=2)
p.xlabel("t",size=12)
p.ylabel("Phase of Y",size=12)
p.xlim([-3,3])
p.show()


# Plot of chirped signal
t = p.linspace(-pi,pi,1025); t = t[:-1]
y = p.cos(16*t*(1.5+(t/(2*pi))))
p.figure(13)
p.grid(True)
p.title(r"$\cos(16t(1.5+\frac{t}{2\pi}))$")
p.xlabel("t")
p.ylabel("y")
p.plot(t,y,'b',lw=2)
p.show()

# DFT of chirped signal
t = p.linspace(-pi,pi,1025);t=t[:-1]
dt = t[1]-t[0]
fmax = 1/dt # fmax is the sampling frequency
y = p.cos(16*t*(1.5+(t/(2*pi))))
y[0] = 0 # the sample corresponding to -tmax should be set zero
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/1024.0
w = p.linspace(-pi*fmax,pi*fmax,1025);w=w[:-1]
p.figure(14)
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',lw=2)
p.xlim([-50,50])
p.ylabel(r"$|Y|$",size=12)
p.title(r"Spectrum of chirped signal $\cos(16t(1.5+\frac{t}{2\pi}))$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-50,50])
p.ylabel(r"Phase of $Y$",size=12)
p.xlabel(r"$\omega$",size=12)
p.grid(True)
p.show()

# Chirped Signal Surface Plot showing evolution of DFT with time
t = p.linspace(-pi,pi,1025);t=t[:-1]
y = p.cos(16*t*(1.5+(t/(2*pi))))
y = y.reshape(16,64)
DFT_mat = p.zeros((16,64))
for i in range(16):
    DFT_mat[i,:] = abs(p.fftshift(p.fft(y[i,:]))/64)

t_axis = t.reshape(16,64)
t_axis = p.mean(t_axis, axis=1)
w_axis = p.linspace(-512,512,65); w_axis = w_axis[:-1]
ii = p.where(abs(w_axis)<=150)[0]
wv, tv = p.meshgrid(w_axis[ii],t_axis)

fig2 = p.figure(15)
ax = fig2.add_subplot(111, projection='3d')
surf = ax.plot_surface(wv, tv, DFT_mat[:,ii], cmap=p.cm.get_cmap("plasma"))
p.colorbar(surf,shrink=0.5,label="|Y| values")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$t$")
ax.set_zlabel(r"$|Y|$")
ax.set_title("Surface plot of DFTs of various portions of the chirped signal")
p.show()