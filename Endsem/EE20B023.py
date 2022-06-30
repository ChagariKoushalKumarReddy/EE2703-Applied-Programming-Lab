"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        END SEMESTER EXAM

        Topic : Finding Currents in a Half Wave Dipole Antenna
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date of completion: 12-04-2022
-------------------------------------------------------------
"""

import pylab as p

# Initializing the parameters
pi = p.pi
l = 0.5 
c = 2.9979e8
mu0 = 4e-7*pi
N = 4
Im = 1              # Maximum current
a = 0.01            # Radius of the wire

lamda = 4*l
f = c/lamda
k = 2*pi/lamda
dz = l/N            # Spacing of current samples

i = p.linspace(-N,N,2*N+1)
z = i*dz
I = p.zeros((2*N+1),dtype = complex)
I[0] = 0 ; I [2*N] = 0 ; I[N] = Im

# 2N - 2 locations of unknown currents
u = p.delete(z,(0,N,2*N))           # Deleting the first, middle and last entries
J = p.zeros(2*N-2)
print("z : ", z)
print("u : ", u)
print("I : ", I)
print("J : ", J)


# Question 2
def M(n):
    # Matrix M of order 2*N-2 by 2*N-2
    return (1/(2*pi*a))*p.identity(2*n-2)

print("M : ",(M(4)).round(2))

# 3rd Question
# Meshgrid generates 2D matrices which are used to find the 
# distance between 2 points zi, zj as i,j go from 0 to 2N+1

zj,zi = p.meshgrid(z,z)
zij = p.zeros((2*N+1,2*N+1))
zij = (zi - zj)**2
Rz = p.zeros((2*N+1,2*N+1))
# Since radius of the wire, a = r
Rz = p.sqrt(a**2 + zij)               
print("Rz : ", (Rz).round(2))

# Here we are calculating distances for unknown current locations

zj,zi = p.meshgrid(u,u)
zij = p.zeros((2*N-2,2*N-2))
zij = (zi - zj)**2
Ru = p.zeros((2*N-2,2*N-2))
# Since radius of the wire, a = r
Ru = p.sqrt(a**2 + zij)               
print("Ru : ", (Ru).round(2))

# Calculation of matrices P and Pb

j = complex(0,1)         # Defining j       
P = p.zeros((2*N-2,2*N-2), dtype = complex)
P = mu0*p.exp(-j*k*Ru)*dz/(4*pi*Ru)
print("P*1e8 : ", (P*1e8).round(2))

Pb = p.zeros((2*N-2),dtype = complex)
# From Rz[:N], we don't need distances to z = 0, z = N, z = 2N
Rz_del = p.delete(Rz[:,N],(0,N,2*N))
Pb = mu0*p.exp(-j*k*Rz_del)*dz/(4*pi*Rz_del)
print("Pb*1e8 : ", (Pb*1e8).round(2))

# 4th Question

# Q corresponds to contribution of unknown currents to H
Q = p.zeros((2*N-2,2*N-2),dtype = complex)
Q[:] = P[:]*a*(1/mu0)*((j*k/Ru[:]) + (1/(Ru[:]**2)))
print("Q : ",(Q).round(3))

# Qb corresponds to contribution of In to H
Qb = p.zeros((2*N-2,1), dtype = complex)
Qb[:,0] = Pb[:]*a*(1/mu0)*((j*k/Rz_del) + 1/(Rz_del**2))
print("Qb : ", (Qb).round(3))

# 5th Question
J = p.dot(p.inv(M(N)-Q),Qb)*Im
print("J : ", J)

# Finding current vector with boundary conditions
I[1:N] = J[:N-1,0]
I[N+1:-1] = J[N-1:,0]
print("I : ", I)

# Finding assumed current in the dipole antenna

I_assumed = p.zeros((2*N+1))
I_assumed[:] = Im*p.sin(k*(l-z[:]))

p.figure(1)
p.grid(True)
p.plot(z,abs(I), label = 'Actual Current')
p.plot(z,abs(I_assumed), label = 'Assumed Current')
p.ylabel('Current',size = 12)
p.xlabel('z', size = 12)
p.title('Plot of Current vs z for N = 4', size = 15)
p.legend(loc = 'upper right')
p.show()