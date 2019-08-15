
import random
import numpy
import pylab

def stabilise_board(Board):
        t=0
        while np.count_nonzero(Board>=4)!=0:
                if Board[0, 0]>=4:
                        t+=1
                        Board[0, 1]+=1
                        Board[1, 0]+=1
                        Board[0, 0]-=4

                if Board[0, L-1]>=4:
                        t+=1
                        Board[0, L-2]+=1
                        Board[1, L-1]+=1
                        Board[0, L-1]-=4

                if Board[L-1, 0]>=4:
                        t+=1
                        Board[L-2, 0]+=1
                        Board[L-1, 1]+=1
                        Board[L-1, 0]-=4

                if Board[L-1, L-1]>=4:
                        t+=1
                        Board[L-2, L-1]+=1
                        Board[L-1, L-2]+=1
                        Board[L-1, L-1]-=4

                for j in range(1, len(Board)-1):
                        if Board[0, j]>=4:
                                t+=1
                                Board[0, j+1]+=1
                                Board[0, j-1]+=1
                                Board[1, j]+=1
                                Board[0, j]-=4

                        if Board[L-1, j]>=4:
                                t+=1
                                Board[L-1, j+1]+=1
                                Board[L-1, j-1]+=1
                                Board[L-2, j]+=1
                                Board[L-1, j]-=4

                        if Board[j, 0]>=4:
                                t+=1
                                Board[j-1, 0]+=1
                                Board[j+1, 0]+=1
                                Board[j, 1]+=1
                                Board[j, 0]-=4

                        if Board[j, L-1]>=4:
                                t+=1
                                Board[j-1, L-1]+=1
                                Board[j+1, L-1]+=1
                                Board[j, L-2]+=1
                                Board[j, L-1]-=4
                        
                
                
                for i in range(1, len(Board)-1):
                        for j in range(1, len(Board)-1):
                                if Board[i, j]>=4:
                                        t+=1
                                        Board[i-1, j]+=1
                                        Board[i+1, j]+=1
                                        Board[i, j+1]+=1
                                        Board[i, j-1]+=1
                                        Board[i, j]-=4


        return Board

L=100
board=np.ones((L, L), dtype=int)*3

def sandpile2(board):
        N=L**2
        min_val, max_val=0, L

##        for i in range(1, n+1):
        while True:
                i=random.choice(range(L))
                j=random.choice(range(L))
                board[i, j]+=1
                board=stabilise_board(board)
                pylab.matshow(board, cmap=plt.cm.Blues)
##                for i in xrange(L):
##                        for j in xrange(L):
##                                c = board[j,i]
##                                pylab.text(i, j, str(" "), va='center', ha='center')

                pylab.draw()
                pylab.pause(0.0001)
                pylab.clf()

sandpile2(board)