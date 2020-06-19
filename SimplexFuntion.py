# This file encodes basic simplex methods to solve the LP
# with two pivot rule to choose: the steepest rule and Dantzig rule


# This is specifies the initialization for the MCF problem
# Find a basic feasible solution (BFS) for this problem

import numpy as np


def initial_MCF(c,A,b,P1,L1,k):
    S = A[(2*L1):, 0:P1]
    R = A[0:L1, 0:P1]
    print('intially', S.shape)
    d = b[(2*L1):]
    u = b[0:L1]
    n = P1+2*L1+1
    ls = []
    for i in range(k):
        ind = np.where(S[i, :] == 1)[0]
        R0 = R[:,ind]
        s = ind.size
        J = np.array([])
        # for j in range(s):
        #     indu = np.where(R0[:,j]!=0)[0]
        #     u0 = sum(sum(u[indu]))
        #     J = np.append(J, u0)
        # MaxInd = J.argmax()
        ls.append(ind[0])
    x = np.zeros((P1, 1))
    x[ls, :] = d
    y = np.dot(R, x)
    lam = u - y
    if any(lam < 0):
        print('error')
        return np.array([]), 0
    t = max(y/u)
    j = (y/u).argmax()
    mu = t*u-y
    K = np.arange(0,2*L1)+P1
    lis = np.hstack((np.array(ls),K))
    lis = np.delete(lis, k+j)
    lis = np.append(lis, n-1)
    return np.vstack((x,mu,lam,t)), lis


# simplex method for MCF problem
def simplex(c,A,b,piv,P1,L1,k):
    X, indb = initial_MCF(c,A,b,P1,L1,k)
    #ind = np.where(X[:,0]!=0)[0]
    if X.size == 0:
        raise Exception('Intialization failure')
    N = np.arange(0,P1+2*L1+1)
    indn = np.delete(N,indb)
    print("initial basis:",indb)
    B = A[:,indb]
    print("B:", B.shape)
    B_inv = np.linalg.inv(B)
    N = A[:,indn]
    iter = 0        # determine the max iteration
    count = 0
    while True:
        iter += 1
        y = np.dot(B_inv.transpose(), c[indb,:])
        s = c[indn,:] - N.transpose().dot(y)
        if all(s>=0):
            print("it's done")
            t = X[-1,0]
            solution = X
                #list(X[0:P1,0])+[t]
            return solution, t
        if piv == 'Danzig':
            q1, q2 = DanzigSelect(s, indn)
        if piv == 'steepest-edge':
            A_re = B_inv.dot(N)
            A_n = np.linalg.norm(A_re,axis=0,keepdims=True)
            q1, q2 = SteepSelect(A_n, s, indn)
        if piv == 'Bland':
            q1, q2 = Bland(s, indn)
        v = A[:, q1:(q1+1)]
        u = B_inv.dot(v)    # a column vector
        if all(u<=0):
            raise Exception('Linear program is unbounded.')
        inds1 = np.where(u[:,0]>0)[0]
        inds2 = indb[inds1]
        ratio = X[inds2,:]/u[inds1,:]
        x_q = min(ratio)        # a number
        p1 = inds2[ratio.argmin()]
        p2 = inds1[ratio.argmin()]
        t_old = X[-1,0]
        X[indb,:] = X[indb,:]-x_q*u
        t = X[-1,0]
        if t == t_old:
            count += 1
        else:
            count = 0
        print("enter basis: ", q1, "quit basis: ", p1, 'optval:, ', X[-1,0], 'iter: ', iter)
        indb = np.delete(indb,p2)
        indb = np.append(indb,q1)
        indn = np.delete(indn,q2)
        indn = np.append(indn,p1)
        B = A[:, indb]
        # B_inv = update_B_inv(B_inv,u,p2)     # this can be improved using linalg technique
        B_inv = np.linalg.pinv(B)
        N = A[:, indn]
        # if iter == 1000:
        #     state = 'run out of time'
        #     break
        if count == 10:
            state = 'converge'
            break
    t = X[-1, 0]
    solution = X
        #list(X[0:P1, 0]) + [t]
    print('state: ', state)
    # print('iter: iter')
    return solution, t

def Bland(s,indn):
    ind = np.where(s[:,0]<0)[0]
    q = ind[0]
    return indn[q], q

def DanzigSelect(s,indn):
    q = s.argmin()
    return indn[q], q

def SteepSelect(A_n,s,indn):
    ratio = s/A_n.transpose()
    q = ratio.argmin()
    return indn[q], q

# def update_B_inv(B_inv,u, p):
#     m = u.shape[0]
#     u1 = -u/u[p]
#     u1[p] = 1/u[p]
#     u1 = np.floor(u1*1000000)/(1000000)
#     P = np.identity(m)
#     P[:,p:(p+1)] = u1
#     #V = np.hstack((B_inv,u))
#     W = P.dot(B_inv)
#     e = P.dot(u)
#     e = np.floor(e*1000000)/(1000000)
#     print('where:, ', np.where(e[:,0]!=0)[0])
#     return W