import math
import sys
import numpy as np
from numpy.linalg.linalg import LinAlgError

def f1(x,y):
    return math.sin(x*y)+x-y

def f2(x,y):
    return y*math.cos(x*y)+1

def df1dx(x,y):
    return y*math.cos(x*y)+1

def df1dy(x,y):
    return x*math.cos(x*y)-1

def df2dx(x,y):
    return -1*y*y*math.sin(x*y)

def df2dy(x,y):
    return math.cos(x*y)-(x*y*math.sin(x*y))

# Solving the linear eqn for h1, h2
def solvelin(x,y,f1,f2,df1dx,df1dy,df2dx,df2dy):
    J = np.array([[df1dx(x,y),df1dy(x,y)],[df2dx(x,y),df2dy(x,y)]])
    f = np.array([f1(x,y),f2(x,y)])
    try:
        h = np.linalg.solve(J, -f)
    except LinAlgError:
        print("Error! - Jacobian is Singular for x = {0}, y = {1}".format(x,y),"No solution found")
        sys.exit(1)
    return h[0],h[-1]

# Finding solution
def solution(x,y,f1,f2,df1dx,df1dy,df2dx,df2dy,solvelin, max_itr = 1000, eps=1.0e-5, printxy=False):
    itr = 0
    h1,h2 = solvelin(x,y,f1,f2,df1dx,df1dy,df2dx,df2dy)
    while abs(h1)+abs(h2) > eps and itr < max_itr:
        x += h1
        y += h2
        if(printxy):
            print("Root after {0} iteration is ({1:.5f},{2:.5f})".format(itr+1,x,y))
        h1,h2 = solvelin(x,y,f1,f2,df1dx,df1dy,df2dx,df2dy)
        itr += 1

    # if abs(h1)+abs(h2) > eps:
    #     itr = -1
    return x, y, itr


print("------- Question 2 solution -------")
x0 = 1
y0 = 2
tol = 1.0e-3
x, y, itr = solution(x0,y0,f1,f2,df1dx,df1dy,df2dx,df2dy,solvelin, eps=tol, printxy=True)
print("Final Solution: (",x,y,")")