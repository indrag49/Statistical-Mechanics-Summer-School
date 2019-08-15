import random
import numpy as np
import pylab
import pygame, random, sys
from pygame.locals import *

L=64
##T=2.2
T=4
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

white=(255, 255, 255)
green=(51, 255, 51)
red=(255, 0, 0)
yellow=(255, 255, 51)
blue=(0, 0, 255)
black=(0, 0, 0)
pink=(255, 0, 255)

DISPLAYSURF=pygame.display.set_mode((L*10, L*10))
pygame.display.set_caption('Kawasaki spin exchange')

DISPLAYSURF.fill(white)
clock=pygame.time.Clock()
Moves=0

while True:
        for event in pygame.event.get():
                if event.type==QUIT:
                        pygame.image.save(DISPLAYSURF, 'kawasaki.png')
                        pygame.quit()
                        sys.exit()

        DISPLAYSURF.fill(white)
        for M in range(L**2):
                i=random.choice(range(L))
                j=random.choice(range(L))
                neb=random.choice([[0, 1], [0, -1], [-1, 0], [1, 0]])
                ineb=i+neb[0]
                jneb=j+neb[1]
                if ineb>L-1:ineb=0
                if jneb>L-1:jneb=0
                if ineb<0:ineb=L-1
                if jneb<0:jneb=L-1
                deltaE1=(S[i, j]-S[ineb, jneb])*(S[p[i], j]+S[i, p[j]]+S[m[i], j]+S[i, m[j]]-S[ineb, jneb])
                deltaE2=(S[ineb,jneb]-S[i,j])*(S[p[ineb], jneb]+S[ineb,p[jneb]]+S[m[ineb], jneb]+S[ineb, m[jneb]]-S[i,j])
                deltaE=deltaE1+deltaE2
                W=np.exp(-deltaE/T)
                r=random.random()
                if r<W:
                        itemp=np.copy(S[i,j])
                        S[i,j]=S[ineb,jneb]
                        S[ineb,jneb]=itemp
        

        for i in range(L):
                for j in range(L):
                        if S[i, j]==1: pygame.draw.circle(DISPLAYSURF, black, (int(i*10), int(j*10)), int(5), 0)
                        else: pygame.draw.circle(DISPLAYSURF, white, (int(i*10), int(j*10)), int(5), 0)


##        clock.tick(40)
        Moves+=1
        print(Moves)

        pygame.display.update()  
