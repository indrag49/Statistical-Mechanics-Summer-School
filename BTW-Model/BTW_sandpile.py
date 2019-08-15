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


##        return [Board, t]
        return [Board, sum(np.sum(Board, axis=0))]
        
def sandpile1(n, board):
        N=L**2
        Y=[]

        for i in range(1, n+1):
                I=random.choice(range(L))
                J=random.choice(range(L))
                board[I, J]+=1
##                [board, topplings]=stabilise_board(board)
                [board, no]=stabilise_board(board)
##                Y+=[topplings, ]
                Y+=[no*1./N,]

        return Y

n=20000 # no. of sand grains added
L=64 # length of board
board1=np.ones((L, L), dtype=int)*3
board2=np.random.randint(4, size=(L, L))
board3=np.zeros((L, L))
pylab.plot(range(1, n+1), sandpile1(n, board1), '-')
pylab.plot(range(1, n+1), sandpile1(n, board2), 'g-')
pylab.plot(range(1, n+1), sandpile1(n, board3), 'r-')
pylab.xlabel("Average no. of sands in the system")
pylab.ylabel("No. of sand grains added to the system")
pylab.title("BTW sandpile model")
pylab.ylim(0, 5)
pylab.show()