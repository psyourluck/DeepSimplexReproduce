import time
from SimplexFuntion import *
from DeepSimDQN import *
from misc import *


L1 = 512
P1 = 1024
k = 50
p = 0.02

np.random.seed(1)

# seed 1 with 512,1024, Danzig beats steepest
# seed 1 with 256, 512 steepest beats Danzig

R = np.random.binomial(1,p,[L1,P1])
d = np.random.gamma(2,2,[k,1])*1
u = np.random.gamma(2,2,[L1,1])*3+20
s = P1 - k
ind = np.random.randint(0,k,s)
S = np.zeros((k,s))
for i in range(s):
    j = ind[i]
    S[j,i] = 1
S = np.hstack((np.identity(k),S))

print(P1,L1,k)
print(S.shape)



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



# Bland quite slow in this case
# piv = 'Bland'
# t0 = time.time()
# solution, optimValue = simplex(c,A,b,piv,P1,L1,k)
# t1 = time.time()
# print(solution.shape)
# if all(A.dot(solution)==b):
#     print('feasible')
# else:
#     print('not true')
# print("cpu time is: ", t1-t0)
# print("solution is: %s, the optimal value is: %.6f" %(solution,optimValue))





# piv = 'Danzig'
# t0 = time.time()
# solution, optimValue = simplex(c,A,b,piv,P1,L1,k)
# t1 = time.time()
# print(solution.shape)
# if all(A.dot(solution)==b):
#     print('feasible')
# else:
#     print('not true')
# print("cpu time is: ", t1-t0)
# print("solution is: %s, the optimal value is: %.6f" %(solution,optimValue))
#
#
#
#
#
#
#
# piv = 'steepest-edge'
# t0 = time.time()
# solution, optimValue = simplex(c,A,b,piv,P1,L1,k)
# t1 = time.time()
# print(solution.shape)
# if all(A.dot(solution)==b):
#     print('feasible')
# else:
#     print('not true')
# print("cpu time is: ", t1-t0)
# print("solution is: %s, the optimal value is: %.6f" %(solution,optimValue))


# Now we try to implement deepsimplex

model_file_name = 'checkpoint-episode-200.pth.tar'
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
print("cpu time is: ", t1-t0)
print("solution is: %s, the optimal value is: %.6f" %(solution,ave_cost))
