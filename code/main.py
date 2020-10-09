import torch
import gym
from gym.spaces import Discrete, Box
import grid_gym
from param_parse import parse_args
import numpy as np
import random
import shutil
import os
from utils.visualise import Visualise
from modules.trainer import Trainer, TrainerEpisodic
from modules.world_models.world_model import WorldModel, EncodedWorldModel
from modules.policies.policy import Policy, PolicyStochastic
from modules.encoders.encoders import Encoder_1D, Encoder_2D
from eval_wm import eval_wm
from datetime import datetime
from time import time
import matplotlib.pyplot as plt


def draw_heat_map(visitation_count, t, folder_name):
    folder_name = folder_name + "/heat_maps/"
    os.makedirs(folder_name, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.power(visitation_count, 1 / 1), cmap='jet', vmin=0.0, vmax=0.01)
    ax.set_title(f'Heat map of visitation counts at t={t}')
    fig.colorbar(ax=ax, mappable=im, orientation='vertical')
    plt.savefig(f'{folder_name}{t}.png')
    plt.close()


def check_input_type(env):
    if isinstance(env.action_space, Box):
        return tuple(env.action_space.sample().shape), False
    elif isinstance(env.action_space, Discrete):
        return (env.action_space.n,), True
    else:
        raise ValueError('What is this action space?')


def main(env, visualise, folder_name, **kwargs):
    shutil.copyfile(os.path.abspath(__file__), folder_name + 'main.py')
    shutil.copyfile(os.getcwd() +'/modules/trainer.py', folder_name + 'trainer.py')
    obs_dim = tuple(env.observation_space.sample().shape)
    assert len(obs_dim) == 1 or len(obs_dim) == 3, f'States should be 1D or 3D vector. Received: {obs_dim}'
    a_dim, discrete_action = check_input_type(env)
    print('Observation space:', obs_dim)
    print('Action space:', a_dim, '(Discrete)' if discrete_action else '(Continuous)')
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    policy = Policy(obs_dim, a_dim, sigma=kwargs['exploration_noise'], discrete_action=discrete_action, device=device, **kwargs)
    # policy = PolicyStochastic(obs_dim, a_dim, sigma=kwargs['exploration_noise'], discrete_action=discrete_action, device=device, **kwargs)
    policy.save(folder_name)
    if kwargs['encoder_type'] == 'none':
        encoder = None
    elif kwargs['encoder_type'] == 'random':
        pass
    elif kwargs['encoder_type'] == 'vae':
        pass
    elif kwargs['encoder_type'] == 'idf':
        pass
    elif kwargs['encoder_type'] == 'cont':
        pass
    if encoder is None:
        wm = WorldModel(obs_dim, a_dim, **kwargs)
    else:
        wm = EncodedWorldModel(obs_dim, a_dim, kwargs['z_dim'], encoder, device=device, **kwargs)

    trainer = TrainerEpisodic(x_dim=obs_dim, a_dim=a_dim, policy=policy, wm=wm, encoder=encoder, **kwargs)

    scores = {'train': [0.0], 'eval': [0.0]}
    start_time = datetime.now()
    # start_time = time()
    while trainer.train_steps < kwargs['train_steps']:
        done = False
        s_t = torch.from_numpy(env.reset()).to(dtype=torch.float32, device=device)
        # env.render()
        s_ts, a_ts, noises = [s_t], [], []
        score = 0
        info = None
        while not done:
            for i in range(kwargs['batch_size']):
                a_t, noise = policy.act(s_t)
                a_ts.append(a_t)
                if discrete_action:
                    a = a_t.argmax().item()
                else:
                    a = a_t
                s_tp1, r_t, done, info = env.step(a)
                s_tp1 = torch.from_numpy(s_tp1).to(dtype=torch.float32, device=device)
                s_ts.append(s_tp1)
                noises.append(noise)
                # env.render()
                score += r_t
                s_t = s_tp1
                if done:
                    break
            if trainer.train_steps < kwargs['train_steps']:
                trainer.train_step(torch.stack(s_ts), torch.stack(a_ts), torch.stack(noises))
                s_ts = [s_t]
                a_ts = []
                noises = []
            if trainer.train_steps % kwargs['export_interval'] == 0:
                visualise.storage.train.policy_loss = float(np.mean(trainer.losses['policy'][-5:]))
                visualise.storage.train.wm_loss = float(np.mean(trainer.losses['wm'][-5:]))
                visualise.storage.train.unique_states = float(info['unique_states'])
                visualise.storage.train.ext_rewards = float(np.mean(scores['train'][-5:]))
                visualise.train_iteration_update()

            if trainer.train_steps % kwargs['eval_interval'] == 0:
                print(trainer.train_steps, datetime.now() - start_time)
                if kwargs['env_name'][:9] == 'GridWorld':
                    q_map, pe_map, walls_map = eval_wm(policy, wm, kwargs['env_name'])
                    visualise.storage.eval.q_map = q_map
                    visualise.storage.eval.pe_map = pe_map
                    visualise.storage.eval.walls_map = walls_map
                    visualise.storage.eval.int_rewards = float(q_map.sum())
                    visualise.storage.eval.wm_loss = float(pe_map.sum())
                    visualise.storage.eval.density_map = info['density']
                visualise.eval_iteration_update()
                # trainer.save_models(folder_name + 'saved_objects/')
        scores['train'].append(score)


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    run_name = f"{args['save_dir']}{args['env_name']}/{args['time_stamp']}_-_{args['name']}_{args['encoder_type']}_{args['seed']}/"
    print(run_name)
    environment = gym.make(args['env_name'])
    visualise = Visualise(run_name, **args)
    try:
        main(environment, visualise, run_name, **args)
    finally:
        environment.close()
        visualise.close()
