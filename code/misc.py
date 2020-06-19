from DeepSimDQN import *
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import wrapsim as Simplex

dimesion = [256,512,50,0.02]



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint model to disk

        state -- checkpoint state: model weight and other info
                 binding by user
        is_best -- if the checkpoint is the best. If it is, then
                   save as a best model
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(filename, model):
    """Load previous checkpoint model

       filename -- model file name
       model -- DQN model
    """
    try:
        checkpoint = torch.load(filename)
    except:
        # load weight saved on gpy device to cpu device
        # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(filename, map_location=lambda storage, loc:storage)
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    print('pretrained episode = {}'.format(episode))
    print('pretrained epsilon = {}'.format(epsilon))
    model.load_state_dict(checkpoint['state_dict'])
    ave_cost = checkpoint.get('best_cost', None)
    if ave_cost is None:
        ave_cost = checkpoint.get('ave_cost')
    print('pretrained ave cost = {}'.format(ave_cost))
    return episode, epsilon, ave_cost

def train_dqn(model, options, resume):
    """Train DQN

       model -- DQN model
       lr -- learning rate
       max_episode -- maximum episode
       resume -- resume previous model
       model_name -- checkpoint file name
    """

    best_cost = 1
    if resume:
        if options.weight is None:
            print('when resume, you should give weight file name.')
            return
        print('load previous model weight: {}'.format(options.weight))
        _, _, best_cost = load_checkpoint(options.weight, model)
        #load_checkpoint(options.weight, model)
    problem = Simplex.MCFLinearProgram(model.L,model.P,model.k,model.p,10)
    problem.coef_initalize()
    B_inv, N = problem.env_intialize()
    optimizer = optim.Adam(model.parameters(), lr=options.lr)
    criterion = nn.MSELoss()

    y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
    s = problem.c[problem.indn, :] - N.transpose().dot(y)
    initial_state = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
    model.set_initial_state(initial_state)

    # action = 1
    # B_inv, N, r, terminal = problem.iterate(action,B_inv,N)
    for i in range(options.observation):
        action = model.get_action_randomly()
        B_inv, N, r, terminal = problem.iterate(action, B_inv, N)
        y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
        s = problem.c[problem.indn, :] - N.transpose().dot(y)
        o_next = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
        model.store_transition(o_next, action, r, terminal)



    Reward = []
    for episode in range(options.max_episode):
        model.time_step = 0
        model.set_train()
        total_reward = 0.
        #begin an episode
        while True:
            print('episode: ', episode, end=' ')
            optimizer.zero_grad()
            # first play the game and put the transition pair into the pool
            action = model.get_optim_action()
            B_inv_old = B_inv
            B_inv, N, r, terminal = problem.iterate(action,B_inv,N)
            if B_inv.size == 0:
                B_inv = B_inv_old
                continue
            total_reward += options.gamma**model.time_step * r
            y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
            s = problem.c[problem.indn, :] - N.transpose().dot(y)
            o_next = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
            model.store_transition(o_next, action, r, terminal)
            model.increase_time_step()
            # Step 1: obtain random minibatch from replay memory
            try:
                minibatch = random.sample(model.replay_memory, options.batch_size)  # sample without replacement
                state_batch = np.array([data[0] for data in minibatch])
                # state_batch = np.concatenate(state_batch, axis=0)
                # state_batch = state_batch.astype(np.float32)
                action_batch = np.array([data[1] for data in minibatch])
                reward_batch = np.array([data[2] for data in minibatch])
                next_state_batch = np.array([data[3] for data in minibatch])
                # next_state_batch = np.concatenate(next_state_batch, axis=0)
                # next_state_batch = next_state_batch.astype(np.float32)
                state_batch_var = torch.from_numpy(state_batch)
                state_batch_var = state_batch_var.float()
                with torch.no_grad():
                    next_state_batch_var = torch.from_numpy(next_state_batch)
                    next_state_batch_var = next_state_batch_var.float()
            # Step 2: calculate y
                    q_value_next = model.forward(next_state_batch_var)
                q_value = model.forward(state_batch_var)
            except:
                continue

            y = reward_batch.astype(np.float32)
            max_q, _ = torch.max(q_value_next, dim=1)
            y = torch.from_numpy(y)
            action_set = np.zeros((options.batch_size,2))
            for i in range(options.batch_size):
                if not minibatch[i][4]:
                    y[i] += options.gamma*max_q[i]
                ind = action_batch[i]
                action_set[i,ind] = 1

            y = y.double()
            action_batch_var = torch.from_numpy(action_set)
            q_value = q_value.double()
            q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)

            loss = criterion(q_value, y)
            loss.backward()

            optimizer.step()
            # when the problem is solved the process ends
            if terminal:
                break

        print('episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
                episode, model.epsilon, model.time_step, total_reward))
        Reward.append(total_reward)

        # decrease the epsilon
        if model.epsilon > options.final_e:
            delta = (options.init_e - options.final_e)/options.exploration
            model.epsilon -= delta
        #
        # if episode % 10 == 0:
        #     ave_cost = test_dqn(model, episode)
        #     save_checkpoint({
        #         'episode': episode,
        #         'epsilon': model.epsilon,
        #         'state_dict': model.state_dict(),
        #         'best_cost': best_cost,
        #          }, True, 'checkpoint-episode-%d.pth.tar' %episode)
        # if episode % options.save_checkpoint_freq == 0:
        #     save_checkpoint({
        #         'episode:': episode,
        #         'epsilon': model.epsilon,
        #         'state_dict': model.state_dict(),
        #         'time_step': ave_cost,
        #          }, False, 'checkpoint-episode-%d.pth.tar' %episode)
        # else:
        #     continue
        # print('save checkpoint, episode={}'.format(
        #          episode))

        if episode % 50 == 0:
            ave_cost = test_dqn(model, episode)

        if ave_cost < best_cost:
            best_cost = ave_cost
            save_checkpoint({
                'episode': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'best_cost': best_cost,
                'reward': Reward,
                 }, True, 'checkpoint-episode-%d.pth.tar' %episode)
        elif episode % options.save_checkpoint_freq == 0:
            save_checkpoint({
                'episode': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'ave_cost': ave_cost,
                'reward': Reward,
                 }, False, 'checkpoint-episode-%d.pth.tar' %episode)
        else:
            continue
        print('save checkpoint, episode={}, ave time step={:.2f}'.format(
                 episode, ave_cost))










def test_dqn(model, episode):
    """Test the behavor of dqn when training

       model -- dqn model
       episode -- current training episode



       problem = Simplex.MCFLinearProgram(model.L,model.P,model.k,model.p,1)
    problem.coef_initalize()
    B_inv, N = problem.env_intialize()
    optimizer = optim.Adam(model.parameters(), lr=options.lr)
    criterion = nn.MSELoss()

    y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
    s = problem.c[problem.indn, :] - N.transpose().dot(y)
    initial_state = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
    model.set_initial_state(initial_state)

    """
    model.set_eval()
    ave_cost = 0.
    for test_case in range(5):
        model.time_step = 0
        problem = Simplex.MCFLinearProgram(model.L, model.P, model.k, model.p, 10000)
        problem.coef_initalize()
        B_inv, N = problem.env_intialize()
        y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
        s = problem.c[problem.indn, :] - N.transpose().dot(y)
        initial_state = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
        model.set_initial_state(initial_state)
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
            o_next = np.append(s[:,0], problem.X[:,0].dot(problem.c[:,0]))
            model.current_state = o_next
            model.increase_time_step()
        ave_cost += problem.X[:,0].dot(problem.c[:,0])
    ave_cost /= 5
    print('testing: episode: {}, average time: {}'.format(episode, ave_cost))
    return ave_cost


def compare(model_file_name, dim):
    """
       compare the trained model with the original steepest and Danzig
       weight -- model file name containing weight of dqn

    """
    print('load pretrained model file: ' + model_file_name)
    model = DeepSimDQN(epsilon=0., mem_size=0, dim=dim)   # we don't need epsilon and mem_size in testing
    episode, epsilon, _ = load_checkpoint(model_file_name, model)
    print("episode: ", episode)
    model.epsilon = epsilon
    ave_cost = 0
    for test_case in range(10):
        print("test case: ", test_case, end=' ')
        model.time_step = 0
        problem = Simplex.MCFLinearProgram(model.L, model.P, model.k, model.p, 10000+test_case)
        problem.coef_initalize()
        B_inv, N = problem.env_intialize()
        y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
        s = problem.c[problem.indn, :] - N.transpose().dot(y)
        initial_state = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
        model.set_initial_state(initial_state)
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
        ave_cost += model.current_state[-1]
    ave_cost /= 10
    print('testing: episode: {}, average cost: {}'.format(episode, ave_cost))
    return ave_cost

def absolute(action, dim):
    act = ['Danzig', 'Steepest-edge']
    print('Use '+ act[action] + 'to solve LP')
    model = DeepSimDQN(epsilon=0., mem_size=0, dim=dim)  # we don't need epsilon and mem_size in testing
    ave_cost = 0
    for test_case in range(10):
        print("test case: ", test_case, end=' ')
        time_step = 0
        problem = Simplex.MCFLinearProgram(model.L, model.P, model.k, model.p, 10000 + test_case)
        problem.coef_initalize()
        B_inv, N = problem.env_intialize()
        y = np.dot(B_inv.transpose(), problem.c[problem.indb, :])
        s = problem.c[problem.indn, :] - N.transpose().dot(y)
        current_state = np.append(s[:, 0], problem.X[:, 0].dot(problem.c[:, 0]))
        while True:
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
            current_state = o_next
            time_step += 1
        ave_cost += current_state[-1]
    ave_cost /= 10
    print('testing: method: {}, average cost: {}'.format(act[action], ave_cost))
    return ave_cost