
from SimplexFuntion import *
import numpy as np

class MCFLinearProgram:
    def __init__(self,L1,P1,k,p,seed):
        self.A = np.zeros((2*L1+k,P1+2*L1+1))
        self.b = np.zeros((2*L1+k,1))
        self.c = np.zeros((P1+2*L1+1,1))
        self.p = p
        self.seed = seed
        self.k = k
        self.L = L1
        self.P = P1
        self.X = 0
        self.indb = 0
        self.indn = 0
        self.iter = 0
        self.count = 0
        self.maxnum = 50
        self.weight = 0.08


    def coef_specific(self, c, A, b):
        self.A = A
        self.b = b
        self.c = c
        self.maxnum = 10000



    def coef_initalize(self):
        np.random.seed(self.seed)
        R = np.random.binomial(1, self.p, [self.L, self.P])
        d = np.random.gamma(2, 2, [self.k, 1]) * 1
        u = np.random.gamma(2, 2, [self.L, 1]) * 3 + 20
        s = self.P - self.k
        ind = np.random.randint(0, self.k, s)
        S = np.zeros((self.k, s))
        for i in range(s):
            j = ind[i]
            S[j, i] = 1
        S = np.hstack((np.identity(self.k), S))
        I = np.identity(self.L)
        self.c[-1,0] = 1
        S1 = np.hstack((R, np.zeros((self.L, self.L)), np.identity(self.L), np.zeros((self.L, 1))))
        S2 = np.hstack((np.zeros((self.L, self.P)), -np.identity(self.L), np.identity(self.L), u))
        S3 = np.hstack((S, np.zeros((self.k, (2 * self.L + 1)))))
        self.A = np.vstack((S1, S2, S3))
        self.b = np.vstack((u, u, d))

    def env_intialize(self):
        self.X, self.indb = initial_MCF(self.c,self.A,self.b,self.P,self.L,self.k)
        if self.X.size == 0:
            print('Intialization failure')
            self.seed += 1
            self.__init__(self.L,self.P,self.k,self.p,self.seed)
            self.coef_initalize()
            return self.env_intialize()
        N = np.arange(0, self.P + 2 * self.L + 1)
        self.indn = np.delete(N, self.indb)
        B = self.A[:, self.indb]
        B_inv = np.linalg.inv(B)
        N = self.A[:, self.indn]
        return B_inv, N



    def iterate(self, input_action, B_inv, N):
        reward = 0
        terminal = False

        if (input_action!=0)&(input_action!=1):
            raise ValueError('Wrong actions!')

        # input_actions == 0: do Danzig pivot
        # input_actions == 1: choose steepest-edge pivot

        self.iter += 1
        if self.iter > self.maxnum:
            reward = 0
            terminal =True
            self.seed += 1
            self.__init__(self.L,self.P,self.k,self.p,self.seed)
            self.coef_initalize()
            B_inv, N = self.env_intialize()
            return B_inv, N, reward, terminal


        y = np.dot(B_inv.transpose(), self.c[self.indb, :])
        try:
            s = self.c[self.indn, :] - N.transpose().dot(y)
        except:
            return np.array([]), np.array([]), reward, terminal
        if all(s >= 0):
            print("it's done")
            reward = 0
            terminal = True
            self.seed += 1
            self.__init__(self.L,self.P,self.k,self.p,self.seed)
            self.coef_initalize()
            B_inv, N = self.env_intialize()
            return B_inv, N, reward, terminal

        if input_action==0:
            q1, q2 = DanzigSelect(s, self.indn)
        if input_action==1:
            A_re = B_inv.dot(N)
            A_n = np.linalg.norm(A_re, axis=0, keepdims=True)
            q1, q2 = SteepSelect(A_n, s, self.indn)
        v = self.A[:, q1:(q1 + 1)]
        u = B_inv.dot(v)  # a column vector
        if all(u <= 0):
            raise Exception('Linear program is unbounded.')
        inds1 = np.where(u[:, 0] > 0)[0]
        inds2 = self.indb[inds1]
        ratio = self.X[inds2, :] / u[inds1, :]
        x_q = min(ratio)  # a number
        p1 = inds2[ratio.argmin()]
        p2 = inds1[ratio.argmin()]
        t_old = self.X[:, 0].dot(self.c[:,0])
        self.X[self.indb, :] = self.X[self.indb, :] - x_q * u
        t = self.X[:, 0].dot(self.c[:,0])
        if t == t_old:
            self.count += 1
        else:
            self.count = 0
        print("enter basis: ", q1, "quit basis: ", p1, 'optval:, ', self.X[-1, 0], 'iter: ', self.iter)
        self.indb = np.delete(self.indb, p2)
        self.indb = np.append(self.indb, q1)
        self.indn = np.delete(self.indn, q2)
        self.indn = np.append(self.indn, p1)
        B = self.A[:, self.indb]
        # B_inv = update_B_inv(B_inv,u,p2)     # this can be improved using linalg technique
        try:
            B_inv = np.linalg.pinv(B)
        except:
            return np.array([]), np.array([]), reward, terminal
        N = self.A[:, self.indn]
        # if iter == 1000:
        #     state = 'run out of time'
        #     break
        if self.count == 10:
            print('converge')
            if input_action == 0:
                reward = 1-1/self.maxnum
            else:
                reward = 1 - (1+self.weight)/self.maxnum
            terminal = True
            self.seed += 1
            self.__init__(self.L, self.P, self.k, self.p, self.seed)
            self.coef_initalize()
            B_inv, N = self.env_intialize()
            return B_inv, N, reward, terminal
        if input_action == 0:
            reward = -1/self.maxnum
        else:
            reward = (-1-self.weight)/self.maxnum
        return B_inv, N, reward, terminal










