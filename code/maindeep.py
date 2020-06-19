import sys
import argparse
from DeepSimDQN import *
from misc import *


parser = argparse.ArgumentParser(description='DeepSimplex Demo Using RL solving linear programming')

parser.add_argument('--train', default=False,
        help='If set true, train the model; otherwise, solve LP with pretrained model')
# parser.add_argument('--cuda', action='store_true', default=False,
#         help='If set true, with cuda enabled; otherwise, with CPU only')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--gamma', type=float,
        help='discount rate', default=1)
parser.add_argument('--batch_size', type=int,
        help='batch size', default=8)
parser.add_argument('--memory_size', type=int,
         help='memory size for experience replay', default=800)
parser.add_argument('--init_e', type=float,
        help='initial epsilon for epsilon-greedy exploration',
        default=0.99)
parser.add_argument('--final_e', type=float,
        help='final epsilon for epsilon-greedy exploration',
        default=0.01)
parser.add_argument('--observation', type=int,
        help='random observation number in the beginning before training',
        default=10)
parser.add_argument('--exploration', type=int,
         help='number of exploration using epsilon-greedy policy',
         default=100)
parser.add_argument('--max_episode', type=int,
        help='maximum episode of training',
        default=1000)
parser.add_argument('--weight', type=str,
        help='weight file name for finetunig(Optional)', default='')
parser.add_argument('--save_checkpoint_freq', type=int,
        help='episode interval to save checkpoint', default=200)
parser.add_argument('--dim',
        help='dimension of the MCF problem for random genearation, L1, P1, k, p',
        default='512,1024,50,0.02')



if __name__ == '__main__':
    args = parser.parse_args()

    if not args.train and args.weight == '':
        print('When test, a pretrained weight model file should be given')
        sys.exit(1)
    if args.train:
        # print(args.list)
        model = DeepSimDQN(epsilon=args.init_e, mem_size=args.memory_size, dim=args.dim)
        resume = False
        args.weight = 'model_best.pth.tar'
        train_dqn(model, args, resume)
    else:
        compare(args.weight, dim=args.dim)
        absolute(0, dim = args.dim)
        absolute(1, dim=args.dim)


# lr = 0.0001
# gamma = 0.99
# batch_size = 32
# mem_size = 5000
# initial_epsilon = 1.0
# final_epsilon = 0.1
# observation = 100
# exploration = 50000
# max_epsilon = 10000