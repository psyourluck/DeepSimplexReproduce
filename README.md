# finalproject
 + These codes reproduces a project rejected by ICML using RL to learn a pivot in simplex method, it's a final project for the Algorithms in Large Data in PKU
 
The.npy files are the data saved by numpy using Python, they are the real data to test the model  
The data file can be preprocessed by data.py and data2.py, according to the how the data is saved  

SimplexFuntion.py provides all the functions needed to implement the simplex method  
3 pivot rule: Dantzig, Steepest edge and the Bland's rules are available  

main.py tests the methods on the real data  
main2.py tests the methods on one special simulated case where Dantzig ourperforms Steepest edge  

misc.py DeepSimDQN wrapsim are the files needed for training RL   

misc.py provides all the necessary functions needed    
DeepSimDQN.py provides the network parameter and class  
wrapsim.py provides how the environment is organized in this case  

to train the model using maindeep.py in cmd  
running with --train 1 will train the model,   
--weight can load existing model to train again    
if no --train is provided, the file will test the existing model loaded  
some other parameters included epsilon greedy, exploration times,... can also be revised and viewed  

