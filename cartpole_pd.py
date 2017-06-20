import gym
from time import sleep

def main():
  env = gym.make('CartPole-v0')
  env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
  for i_episode in range(200):
    observation = env.reset()
    for t in range(1000):
      #env.render()
      action = calc_action(observation)
      #print observation, " ", action
      observation, reward, done, info = env.step(action)
      if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
  env.close()
  gym.upload('/tmp/cartpole-experiment-1', api_key='sk_7zik3dAjTqKFAnINWxiEA')

k = [1, 5.8, -1000, -100]
def calc_action(observation):
  angle = observation[2]
  angle_dot = observation[3]
  k_angle = k[2]
  k_ang_vel = k[3]
  thrust = -(k_angle * angle + k_ang_vel * angle_dot)
  if thrust > 0:
    action = 1
  else:
    action = 0
  return action

main()
