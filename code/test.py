# import argparse
# def fun(i):
# 	i = int(i)+1
# 	print(i)
#
# parser = argparse.ArgumentParser(description='........')
# #nagrs="+" 至少给-l 选项传递1个参数，
# parser.add_argument("-l","--dim",dest="list",nargs='+',help="list")
# args = parser.parse_args()
#
#
# for i in args.list:
# 	#处理每个参数
# 	fun(i)

import wrapsim as Simplex
from DeepSimDQN import *
#
# model = DeepSimDQN(epsilon=1, mem_size=800, dim='256,512,50,0.02')
# best_cost = 0.
#
#         #load_checkpoint(options.weight, model)
# problem = Simplex.MCFLinearProgram(model.L,model.P,model.k,model.p,10)
# problem.coef_initalize()
# B_inv, N = problem.env_intialize()
#
#
# y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
# s = problem.c[problem.indn, :] - N.transpose().dot(y)
# initial_state = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
# model.set_initial_state(initial_state)
#
#
# for i in range(100):
#     action = model.get_action_randomly()
#     B_inv, N, r, terminal = problem.iterate(action, B_inv, N)
#     y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
#     s = problem.c[problem.indn, :] - N.transpose().dot(y)
#     o_next = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
#     model.store_transition(o_next, action, r, terminal)
import sys
import argparse
from DeepSimDQN import *
from misc import *

compare('checkpoint-episode-200.pth.tar', dim='512,1024,50,0.02')