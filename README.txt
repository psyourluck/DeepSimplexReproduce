This file demonstrates all the files in the package:
data:
R.npy,S.npy,d.npy,u.npy	encodes the data transformed from JLF023
R141.npy,S141.npy,d141.npy,u141.npy	encodes the data from JLF141
			the transformation can be seen in data.py, data2.py, where mut file need to be properly 
			delt with, we use excel to make it into two column with numeric type

DeepSimDQN.py	encodes the construction of the DNN in deep Q learning
main.py 		solves the real data problem
main2.py 		solves the special case problem
maindeep.py	solves the deep Q learning problem, this file should be run in the cmd
		with --train 1 to train the model and --weight in the train session will load the existing model
		--weight without train will just test the existing model
misc.py		encodes all the necessary fuctions used in the DeepSimplex
SimplexFunction.py 	encodes all the function used to implement simplex method
		with Dantzig, steepest edge or Bland as the pivot rule
wrapsim.py	encodes how the environment in RL is formulated from solving LP using simplex

modelreal, modelSimulation	just saves the trained model 0,200,....800,1000
			note here we save the model at 400 and train it again 
			so the model in 600,800,1000 will have episodes 200 400 600 and will do observation again
			at the beginning