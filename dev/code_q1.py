import math
import sys

def newton(f, dfdx, x, max_itr = 100, eps=1.0e-5, printx=False):
    itr = 0
    h = float(f(x))/dfdx(x)
    e = []
    while abs(h) > eps and itr < max_itr:
        e.append(h)
        try:
            x = x - h
        except ZeroDivisionError:
            print("Error! - Derivative is zero for x = ", x,"No solution found")
            sys.exit(1)     # Aborting
        if(printx):
            print("Root after iteration {0} is {1:.5f}".format(itr+1,x))
        h = float(f(x))/dfdx(x)
        itr += 1

    # if abs(h) > eps:
    #     itr = -1
    return x, itr, e

def bisection(f,a,b,max_itr = 100, eps=1.0e-5):
    if(f(a)*f(b)>=0):
        print("Try with better intial guess of a & b")
        return
    c = a
    itr = 0
    while((b-a)>=eps and itr<max_itr):
        c = (a+b)/2
        if(f(c)==0.0):
            break
        # print(a,b)
        if(f(c)*f(a) < 0.0):
            b = c
        else:
            a = c
        itr += 1
        
    return c,itr

def ordconv(e):
    l = len(e)
    p = []
    for i in range(l-2):
        o = math.log(abs(e[i+2]/e[i+1]))/math.log(abs(e[i+1]/e[i]))
        p.append(o)
    return sum(p)/len(p)
        
def f(x):
    return math.exp(-x)-math.sin(x)
    
def dfdx(x):
    return -math.exp(-x)-math.cos(x)

print("------- Question 1 solution -------")
# Part A
print("------- Part A solution -------")
a0 = -2.0
b0 = 3.0
print("Searching for root in the interval [{0}, {1}]".format(a0,b0))
int_x, itr = bisection(f, a = a0, b = b0,max_itr = 3, eps=1.0e-5)
print("Intial Guess after 3 iterations: {0:.5f}".format(int_x))

# Part B & C
print("------- Part B and C solution -------")
x0 = int_x
print("Searching for root starting from x = {0}".format(x0))
x,itr,e = newton(f, dfdx, x0, eps=1.0e-5, printx=True)
print("Final root :{0:.5f}".format(x))
print("Number of iterations:",itr)

# Part D
print("------- Part D solution -------")
p = ordconv(e)
print("Order of Convergence of NRM: {0:.5f}".format(p))