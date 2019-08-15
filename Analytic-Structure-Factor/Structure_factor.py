import math
import numpy as np
import cmath as cm
import pylab
from scipy.integrate import quad

pi=np.pi
##eta=0.3 #initial value
k=2. # k=kappa*sigma
T=298
epsilon=78.3
epsilon0=8.85*10**(-12)
sigma=5*10**(-9)
##psi0=72*10**(-3) # initial
##psi0=z*1.6*10**(-19)/(pi*epsilon0*epsilon*sigma*(2.+k))
beta=1/(1.38064852*10**(-23)*T)
sin=math.sin
cos=math.cos
sinh=math.sinh
cosh=math.cosh
exp=math.exp
rad=math.degrees

def S(K, eta, z):
        psi0=z*1.6*10**(-19.)/(pi*epsilon0*epsilon*sigma*(2.+k))
        gamma=beta*pi*epsilon0*epsilon*sigma*psi0**2/exp(-k)

        delta=1-eta

        alpha1=-(2*eta+1)*delta/k
        alpha2=(14*eta**2-4*eta-1)/k**2
        alpha3=36*eta**2/k**4

        beta1=-(eta**2+7*eta+1)*delta/k
        beta2=9*eta*(eta**2+4*eta-2)/k**2
        beta3=12*eta*(2*eta**2+8*eta-1)/k**4

        nu1=-(eta**3+3*eta**2+45*eta+5)*delta/k
        nu2=(2*eta**3+3*eta**2+42*eta-20)/k**2
        nu3=(2*eta**3+30*eta-5)/k**4
        nu4=nu1+24*eta*k*nu3
        nu5=6*eta*(nu2+4*nu3)

        phi1=6*eta/k
        phi2=delta-12*eta/k**2

        tau1=(eta+5)/(5*k)
        tau2=(eta+2)/k**2
        tau3=-12*eta*gamma*exp(-k)*(tau1+tau2)
        tau4=3*eta*k**2*(tau1**2-tau2**2)
        tau5=3*eta*(eta+8)/10-2*(2*eta+1)**2/k**2

        a1=(24*eta*gamma*exp(-k)*(alpha1+alpha2+(1+k)*alpha3)-(2*eta+1)**2)/delta**4
        a2=24*eta*(alpha3*(sinh(k)-k*cosh(k))+alpha2*sinh(k)-alpha1*cosh(k))/delta**4
        a3=24*eta*((2*eta+1)**2/k**2-delta**2/2+alpha3*(cosh(k)-1-k*sinh(k))-alpha1*sinh(k)+alpha2*cosh(k))/delta**4

        b1=(3*eta*(eta+2)**2/2-12*eta*gamma*exp(-k)*(beta1+beta2+(1+k)*beta3))/delta**4
        b2=12*eta*(beta3*(k*cosh(k)-sinh(k))-beta2*sinh(k)+beta1*cosh(k))/delta**4
        b3=12*eta*(delta**2*(eta+2)/2-3*eta*(eta+2)**2/k**2-beta3*(cosh(k)-1-k*sinh(k))+beta1*sinh(k)-beta2*cosh(k))/delta**4

        v1=((2*eta+1)*(eta**2-2*eta+10)/4-gamma*exp(-k)*(nu4+nu5))/(5*delta**4)
        v2=(nu4*cosh(k)-nu5*sinh(k))/(5*delta**4)
        v3=((eta**3-6*eta**2+5)*delta-6*eta*(2*eta**3-3*eta**2+18*eta+10)/k**2+24*eta*nu3+nu4*sinh(k)-nu5*cosh(k))/(5*delta**4)

        p1=(gamma*exp(-k)*(phi1-phi2)**2-(eta+2)/2)/delta**2
        p2=((phi1**2+phi2**2)*sinh(k)+2*phi1*phi2*cosh(k))/delta**2
        p3=((phi1**2+phi2**2)*cosh(k)+2*phi1*phi2*sinh(k)+phi1**2-phi2**2)/delta**2

        t1=tau3+tau4*a1+tau5*b1
        t2=tau4*a2+tau5*b2+12*eta*(tau1*cosh(k)-tau2*sinh(k))
        t3=tau4*a3+tau5*b3+12*eta*(tau1*sinh(k)-tau2*(cosh(k)-1))-2*eta*(eta+10)/5-1

        mu1=t2*a2-12*eta*v2**2
        mu2=t1*a2+t2*a1-24*eta*v1*v2
        mu3=t2*a3+t3*a2-24*eta*v2*v3
        mu4=t1*a1-12*eta*v1**2
        mu5=t1*a3+t3*a1-24*eta*v1*v3
        mu6=t3*a3-12*eta*v3**2

        lambda1=12*eta*p2**2
        lambda2=24*eta*p1*p2-2*b2
        lambda3=24*eta*p2*p3
        lambda4=12*eta*p1**2-2*b1
        lambda5=24*eta*p1*p3-2*b3-k**2
        lambda6=12*eta*p3**2

        omega16=mu1*lambda6-mu6*lambda1
        omega13=mu1*lambda3-mu3*lambda1
        omega36=mu3*lambda6-mu6*lambda3
        omega15=mu1*lambda5-mu5*lambda1
        omega35=mu3*lambda5-mu5*lambda3
        omega26=mu2*lambda6-mu6*lambda2
        omega12=mu1*lambda2-mu2*lambda1
        omega14=mu1*lambda4-mu4*lambda1
        omega34=mu3*lambda4-mu4*lambda3
        omega25=mu2*lambda5-mu5*lambda2
        omega24=mu2*lambda4-mu4*lambda2

        omega4=omega16**2-omega13*omega36
        omega3=2*omega16*omega15-omega13*(omega35+omega26)-omega12*omega36
        omega2=omega15**2+2*omega16*omega14-omega13*(omega34+omega25)-omega12*(omega35+omega26)
        omega1=2*omega15*omega14-omega13*omega24-omega12*(omega34+omega25)
        omega0=omega14**2-omega12*omega24

        O=[omega0, omega1, omega2, omega3, omega4]

        def Roots(Pol):
                a, b, c, d, e=Pol[0], Pol[1], Pol[2], Pol[3], Pol[4]
                Delta1=2*c**3-9*b*c*d+27*b**2*e+27*a*d**2-72*a*c*e
                Delta0=c**2-3*b*d+12*a*e
                p=(8*a*c-3*b**2)/(8*a**2)
                q=(b**3-4*a*b*c+8*a**2*d)/(8*a**3)
                Q=((Delta1+cm.sqrt(Delta1**2-4*Delta0**3))/2.)**(1/3.)
                S=0.5*cm.sqrt(-2.*p/3.+(Q+Delta0/Q)/(3*a))
                x1=-b/(4*a)-S+0.5*cm.sqrt(-4*S**2-2*p+q/S)
                x2=-b/(4*a)-S-0.5*cm.sqrt(-4*S**2-2*p+q/S)
                x3=-b/(4*a)+S+0.5*cm.sqrt(-4*S**2-2*p-q/S)
                x4=-b/(4*a)+S-0.5*cm.sqrt(-4*S**2-2*p-q/S)
                return[x1, x2, x3, x4]

        R=Roots(O[::-1])
        F=R[2:]
        F1=F[1]
        C=-(omega16*F1**2+omega15*F1+omega14)/(omega13*F1+omega12)
        B=b1+b2*C+b3*F1
        A=a1+a2*C+a3*F1
        def a(K): return A*(sin(K)-K*cos(K))/K**3+B*((2/K**2-1)*K*cos(K)+2*sin(K)-2/K)/K**3+eta*A*(24/K**3+4*(1-6/K**2)*sin(K)-(1-12/K**2+24/K**4)*K*cos(K))/(2*K**3)+C*(k*cosh(k)*sin(K)-K*sinh(k)*cos(K))/(K*(K**2+k**2))+F1*(k*sinh(k)*sin(K)-K*(cosh(k)*cos(K)-1))/(K*(K**2+k**2))+F1*(cos(K)-1)/K**2-gamma*exp(-k)*(k*sin(K)+K*cos(K))/(K*(K**2+k**2))
        return 1./(1-24*eta*a(K))

##Z=[0, 10, 20, 50, 100, 200]
##col=['--','k-', 'r-', '-', 'g-', 'y-']
##for i in range(len(Z)):        
##        X=np.arange(0.01, 20, 0.01)
##        Y=[S(x, 0.05, Z[i]) for x in X]
##        pylab.plot(X, Y, col[i])
##pylab.xlabel("Q$\sigma$")
##pylab.ylabel("S(Q$\sigma$)")
##pylab.title("Structure Factor")
##pylab.show()

##import pylab
##
##def F(q, R): return (4*np.pi*R**3)*(np.sin(q*R)-q*R*np.cos(q*R))/(q*R)**3
##def P(q, R): return F(q, R)**2

##R=1.
##X=np.linspace(0.1, 20, 1000)*R
##Y=[np.log(P(q, R)) for q in X]
##pylab.plot(X, Y)
##pylab.show()

R=1. 
def F(q): return (4*pi*R**3)*(sin(q*R)-q*R*cos(q*R))/(q*R)**3
def P(q): return F(q)**2
def I(q): return P(q)*S(q, 0.05, 20)

##X=np.arange(0.01, 20, 0.01)
##Y=[I(x) for x in X]
##pylab.plot(X, Y, 'k-')

##pylab.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Eta=np.linspace(0.01, 0.3, 100)
X=np.linspace(3, 7, 100)
fig=plt.figure()
ax=plt.axes(projection='3d')
for eta in Eta:
        Z=[S(X[i], eta, 50) for i in range(len(Eta))]
        ax.plot3D([eta]*100, X, Z, 'red')

ax.set_ylabel('Q$\sigma$')
ax.set_xlabel('$\eta$')
ax.set_zlabel('S(Q$\sigma$)')
plt.title('Isometric plot, $\psi_0$=180 mV, k=2, z=50')
plt.show()
