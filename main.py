# from data import *
import time
from SimplexFuntion import *
from DeepSimDQN import *
from misc import *


R = np.load('R141.npy')
S = np.load('S141.npy')
u = np.load('u141.npy')
d = np.load('d141.npy')

u = u*24

L1 = u.shape[0]
k = d.shape[0]
P1 = R.shape[1]
print(P1,L1,k)
print(S.shape)
print(sum(sum(R))/L1/P1)

# after import and preprocessing the data
# this file solves the problem as a LP
# using simplex method with Dantzig's rule
# or the steepest rule as pivot

# first we add slack variables to convert it into a
# standard linear programming: c*x s.t. AX = b, X>=0

# X = (x,mu,lam,t)', mu, lam is the slack variable for
# inquality concerning merely u and t,u

# note the dimension x is 364-dim (with slackness)
# A is 196*364-dim, b is 196-dm, c is also 364-dim with the last one 1

# assume A is m*n-dim
n = P1+2*L1+1       #364
m = 2*L1+k          #196

# construct the vector c, only the last element is 1
c = np.zeros((n,1))
c[-1,0] = 1

print(c.shape, c[-1,0])

# construct the matrix A
S1 = np.hstack((R,np.zeros((L1,L1)),np.identity(L1),np.zeros((L1,1))))
S2 = np.hstack((np.zeros((L1,P1)),-np.identity(L1),np.identity(L1),u))
S3 = np.hstack((S,np.zeros((k,(2*L1+1)))))
A = np.vstack((S1,S2,S3))
print(A.shape)

# construct the vector b

b = np.vstack((u,u,d))
print(b.shape)

# now we can apply simplex method
# piv = 'steepest-edge'
# t0 = time.time()
# solution, optimValue = simplex(c,A,b,piv,P1,L1,k)
# t1 = time.time()
# print(abs(A.dot(np.array(solution))-b).sum(axis=0)[0]/abs(b).sum(axis=0)[0])
# if abs(A.dot(np.array(solution))-b).sum(axis=0)[0]<1e-4:
#     print('feasible')
# else:
#     print('not true')
# print("cpu time is: ", t1-t0)
# print("solution is: %s, the optimal value is: %.6f" %(solution,optimValue))
#
# # # # Bland
# piv = 'Bland'
# t0 = time.time()
# solution, optimValue = simplex(c,A,b,piv,P1,L1,k)
# t1 = time.time()
# print(abs(A.dot(np.array(solution))-b).sum(axis=0)[0]/abs(b).sum(axis=0)[0])
# if abs(A.dot(np.array(solution))-b).sum(axis=0)[0]<1e-4:
#     print('feasible')
# else:
#     print('not true')
# print("cpu time is: ", t1-t0)
# print("solution is: %s, the optimal value is: %.6f" %(solution,optimValue))
# #
# # # Danzig
# #
# piv = 'Danzig'
# t0 = time.time()
# solution, optimValue = simplex(c,A,b,piv,P1,L1,k)
# t1 = time.time()
# if all(A.dot(np.array(solution))==b):
#     print('feasible')
# else:
#     print('not true')
# print(abs(A.dot(np.array(solution))-b).sum(axis=0)[0]/abs(b).sum(axis=0)[0])
# print("cpu time is: ", t1-t0)
# print("solution is: %s, the optimal value is: %.6f" %(solution,optimValue))


# deepsimplex
model_file_name = 'modelreal\model141\checkpoint-episode-100.pth.tar'
p = 0.02 # p is not important in this case and can be set any value
dim = str(L1) + ',' + str(P1) + ',' + str(k) + ',' + str(p)
print('load pretrained model file: ' + model_file_name)
model = DeepSimDQN(epsilon=0., mem_size=0, dim=dim)   # we don't need epsilon and mem_size in testing
episode, epsilon, _ = load_checkpoint(model_file_name, model)
print("episode: ", episode)
model.time_step = 0
problem = Simplex.MCFLinearProgram(model.L, model.P, model.k, model.p, 0)
problem.coef_specific(c, A, b)
B_inv, N = problem.env_intialize()
y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
s = problem.c[problem.indn, :] - N.transpose().dot(y)
initial_state = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
model.set_initial_state(initial_state)
t0 = time.time()
while True:
    action = model.get_optim_action()
    B_inv_old = B_inv
    B_inv, N, r, terminal = problem.iterate(action, B_inv, N)
    if B_inv.size == 0:
        B_inv = B_inv_old
        continue
    if terminal:
        break
    y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
    s = problem.c[problem.indn, :] - N.transpose().dot(y)
    o_next = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
    model.current_state = o_next
    model.increase_time_step()
t1 = time.time()
ave_cost = model.current_state[-1]
print(abs(A.dot(np.array(problem.X))-b).sum(axis=0)[0]/abs(b).sum(axis=0)[0])
print("cpu time is: ", t1-t0)
print("the optimal value is: %.6f" %(ave_cost))
