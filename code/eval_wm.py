import torch
import numpy as np
import gym
import grid_gym
from grid_gym.envs.grid_world import *
from modules.world_models.world_model import EncodedWorldModel, WorldModel
import os
import matplotlib.pyplot as plt
from time import time
plt.ioff()
np.set_printoptions(linewidth=400)

ACTIONS = (torch.tensor((-1.0, 0.0)),  # Left
           torch.tensor((0.0, -1.0)),  # Down
           torch.tensor((1.0, 0.0)),  # Right
           torch.tensor((0.0, 1.0)),  # Up
           torch.tensor((0.0, 0.0))  # Stay
           )


def draw_heat_map(array, path, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(array, cmap='jet', vmin=0.0)
    ax.set_title(f'Heat map of {name}')
    fig.colorbar(ax=ax, mappable=im, orientation='vertical')
    plt.savefig(path)
    plt.close()


def size_from_env_name(env_name: str) -> tuple:
    i = 0
    for i in range(len(env_name) - 1, -1, -1):
        if env_name[i] == 'x':
            break
    return (int(env_name[i+1-3:-len('-v0')-3]), int(env_name[i+1:-len('-v0')]))


def get_env_instance(env_name):
    if env_name == 'GridWorldBox11x11-v0':
        env = GridWorldBox11x11()
    elif env_name == 'GridWorldSpiral28x28-v0':
        env = GridWorldSpiral28x28()
    elif env_name == 'GridWorldSpiral52x50-v0':
        env = GridWorldSpiral52x50()
    elif env_name == 'GridWorld10x10-v0':
        env = GridWorld10x10()
    elif env_name == 'GridWorld25x25-v0':
        env = GridWorld25x25()
    elif env_name == 'GridWorld42x42-v0':
        env = GridWorld42x42()
    elif env_name == 'GridWorldRandFeatures42x42-v0':
        env = GridWorldRandFeatures42x42()
    elif env_name == 'GridWorldContinuousAction42x42-v0':
        env = GridWorldContinuousAction42x42()
    else:
        raise ValueError('Wrong env_name.')
    return env


def eval_wm(policy, wm, env_name):
    start_time = time()
    size = size_from_env_name(env_name)
    env = get_env_instance(env_name)
    # Map where Max Q-values will be plotted into
    q_grid = np.zeros(size)
    pe_grid = np.zeros(size)
    walls_map = env.map if hasattr(env, 'map') else None
    for i in range(size[0]):
        for j in range(size[1]):
            if walls_map is not None and walls_map[i, j] == 1:
                continue
            s = torch.zeros(size)
            s[i, j] = 1
            # Finding best action from policy
            env.pos = [i, j]
            a = policy.predict(s.reshape((-1,)))
            ns = torch.from_numpy(env.step(a.numpy())[0])
            if isinstance(wm, EncodedWorldModel):
                ns = wm.encoder(ns)
            pns = wm.predict(s.reshape((-1,)), a)
            q_grid[i, j] = (ns - pns).abs().sum()
            # Finding prediction error for each cardinal direction
            for a in ACTIONS:
                env.pos = [i, j]
                ns = torch.from_numpy(env.step(a.numpy())[0])
                if isinstance(wm, EncodedWorldModel):
                    ns = wm.encoder(ns)
                pns = wm.predict(s.reshape((-1,)), a)
                pe_grid[i, j] += (ns - pns).abs().sum()/len(ACTIONS)  # Take the average PE across all actions above
    print('Eval time:', time() - start_time)
    return q_grid, pe_grid, walls_map
