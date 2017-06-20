import gym
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/tmp/frozenlake0',force=True)
init_q_val = .2
discount = .6
learning_rate = .4
use_softmax = 1
# Get q values for a state s in [0, 16), action a in [0, 4) with q_vals[s * 16 + a]
q_vals = init_q_val * np.ones((16, 4))

success = 0
total = 0

n_explore_episodes = 9000
n_exploit_episodes = 1000
n_episodes = n_explore_episodes + n_exploit_episodes
# For each episode, reset the environment and get initial observation
# Run a loop for a max number of timesteps, render the environment, print the observation.
# Choose an action.
# Observation: (x, v, theta, w)
#env: step, reset, render, close seed
p_start = 1
rand_action_probs = []
for i_episode in range(n_episodes):
    observation = env.reset()
    p_random_action = p_start * (1 - min(max(0, float(i_episode) / n_explore_episodes), 1))
    rand_action_probs.append(p_random_action)
    for t in range(100):
        #is_eval_run = i_episode > n_explore_episodes
        is_eval_run = False
        s_start = observation
        
        '''if is_eval_run:
            action = np.argmax(q_vals[s_start, :])
        else:
            unnorm_dist = np.exp(q_vals[s_start, :]) if use_softmax else q_vals[s_start, :]
            dist = unnorm_dist / np.sum(unnorm_dist)
            action = np.random.choice(4, 1, p=dist)[0]
            if i_episode > 8000:
                print i_episode, " ", observation, " ",  action, " ", dist'''

        if np.random.rand() > p_random_action:
            action = np.argmax(q_vals[s_start, :])
        else:
            action = np.random.choice(4,1, p=[.25, .25, .25, .25])[0]
        observation, reward, done, info = env.step(action)
        s_next = observation
        #IMPORTANT: DONE DOESNT ALWAYS HAPPEN IN A GOAL STATE!!
        if s_next is not s_start:
            q_next = reward + discount * max(q_vals[s_next, :])
            next_q_val = (1 - learning_rate) * q_vals[s_start, action] + learning_rate * (q_next)
            if next_q_val < 1e-8 and s_start is not 3:
                print s_start, "->", s_next, ". action: ", action, ", lr: ", learning_rate, ", qv old: ", q_vals[s_start, action], ", qnext: ", q_next, ", reward: ", reward, ", disc: ", discount, ", qtakingmaxof ", q_vals[s_next, :]
                print q_vals
                assert 0
                if next_q_val > 1:
                    print "lr: ", learning_rate, ", qv old: ", q_vals[s_start, action], ", qnext: ", q_next, ", reward: ", reward, ", disc: ", discount, ", qtakingmaxof ", q_vals[s_next, :]
                    i_episode = n_episodes
                    assert next_q_val <= 1
                    break
            q_vals[s_start, action] = min(max(0,next_q_val),1)
        assert q_vals[1][1] > 1e-8
        if s_next in [5, 7, 11, 12, 15]:
            if is_eval_run:
                success += reward
                total += 1
        if done: 
            break
        assert (t is not 99)
plt.plot(np.arange(0, len(rand_action_probs)), rand_action_probs)
plt.show()
print "q matrix"
print q_vals
print "max over state qvals"
print np.reshape(np.max(q_vals, 1), (4, 4))
print np.reshape(np.argmax(q_vals, 1), (4, 4))
if total is not 0:
    print success, "/", total, "=",success/float(total)
print info
print "0:Left, 1:Down, 2:Right, 3:Up"
env.close()

'''
env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/tmp/frozenlake0',force=True)
for i_episode in range(n_episodes):
    observation = env.reset()
    for t in range(100):
        action = np.argmax(q_vals[observation, :])
        observation, reward, done, info = env.step(action)
        if done:
            break'''
env.close()
gym.upload('/tmp/frozenlake0', api_key='sk_7zik3dAjTqKFAnINWxiEA')
