import random
import numpy as np
import pylab

L=64
T=2.2
N=1000
h=0

S=np.zeros((L, L), dtype=int)
for i in range(L):
        for j in range(L):
                r=random.uniform(0, 1)
                if r<=0.5: S[i, j]=1
                else: S[i, j]=-1


p=np.arange(L, dtype=int)+1
p[-1]=0
m=np.arange(L, dtype=int)-1
m[0]=L-1

for I in range(N):
        for M in range(L**2):
                i=random.choice(range(L))
                j=random.choice(range(L))
                deltaE=2*S[i, j]*(S[p[i], j]+S[i, p[j]]+S[m[i], j]+S[i,m[j]])+2*h*S[i, j]
                W=np.exp(-deltaE/T)
                r=random.random()
                if r<W: S[i, j]=-S[i, j]


##for i in range(L):
##        for j in range(L):
##                if S[i, j]==1: pylab.plot(i, j, 'k->', ms=10)
##                else: pylab.plot(i, j, 'r<-', ms=10)

##pylab.show()

        
import pygame, random, sys
from pygame.locals import *



white=(255, 255, 255)
green=(51, 255, 51)
red=(255, 0, 0)
yellow=(255, 255, 51)
blue=(0, 0, 255)
black=(0, 0, 0)
pink=(255, 0, 255)

DISPLAYSURF=pygame.display.set_mode((L*10, L*10))
pygame.display.set_caption('Glauber spin flip')

DISPLAYSURF.fill(white)
clock=pygame.time.Clock()
Moves=0

while True:
        for event in pygame.event.get():
                if event.type==QUIT:
                        pygame.image.save(DISPLAYSURF, 'direct-disks.png')
                        pygame.quit()
                        sys.exit()

        DISPLAYSURF.fill(white)
        for M in range(L**2):
                i=random.choice(range(L))
                j=random.choice(range(L))
                deltaE=2*S[i, j]*(S[p[i], j]+S[i, p[j]]+S[m[i], j]+S[i,m[j]])+2*h*S[i, j]
                W=np.exp(-deltaE/T)
                r=random.random()
                if r<W: S[i, j]=-S[i, j]
        

        for i in range(L):
                for j in range(L):
                        if S[i, j]==1: pygame.draw.circle(DISPLAYSURF, red, (int(i*10), int(j*10)), int(5), 0)
                        else: pygame.draw.circle(DISPLAYSURF, white, (int(i*10), int(j*10)), int(5), 0)


##        clock.tick(40)
        Moves+=1
        print(Moves)

        pygame.display.update()  