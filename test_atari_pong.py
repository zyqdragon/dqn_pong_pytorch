import gymnasium as gym
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

## 创建一个游戏环境
#env = gym.make("CartPole-v1",render_mode="rgb_array")
env = gym.make("CartPole-v1",render_mode="human")
env = gym.make("Pong-v4",render_mode="human")
print("------env.action_space=",env.action_space)
print("------env.observation_space=",env.observation_space)
mean_action=env.unwrapped.get_action_meanings()
print("-----------mean_action=",mean_action)
## 训练20个epoch
for i in range(200):
    ## 每次训练前都初始化环境
    env.reset()
    ## 与环境的最大交互次数为100
    for step in range(800):
        ## 渲染环境
        env.render()
        ## 从 action 空间中获取一个 action
        action = env.action_space.sample()
        print("-----------action=",action)
        ## 根据 action 与环境进运行一次交互
        observation,reward,done,info,_ = env.step(action)
        step_result=env.step(action)
        print("-----------len of step_result=",len(step_result))
        print("-----------step_result_0=",len(step_result[0]))
        print("observation=",observation.shape)
        data=observation[1:2,1:3,:]
        print("----**************----data=",data.shape)
        ## 如果返回当前游戏结束，推出当前游戏继续下一次训练
        cv2.imwrite('./img_space/img'+str(step)+'.png',observation)
        time.sleep(1)
        if(done):
            break
