import torch
import gym
import grid_gym
from param_parse import parse_args
import numpy as np
import random
import shutil
import os
from utils.visualise import Visualise
from modules.trainer import Trainer
from modules.world_models.world_model import WorldModel, EncodedWorldModel
from modules.policies.policy import Policy
from modules.encoders.encoders import Encoder_1D, Encoder_2D
from datetime import datetime
from modules.replay_buffers.replay_buffer_torch import DynamicsReplayBuffer
from modules.trainer import Trainer
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


def main(env, visualise, folder_name, **kwargs):
    shutil.copyfile(os.path.abspath(__file__), folder_name + 'main.py')
    obs_dim = tuple(env.observation_space.sample().shape)
    assert len(obs_dim) == 1 or len(obs_dim) == 3, f'States should be 1D or 3D vector. Received: {obs_dim}'
    a_dim = tuple(env.action_space.sample().shape)
    print('Observation space:', obs_dim)
    print('Action space:', a_dim)
    if len(obs_dim) == 1:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = Policy(obs_dim, a_dim, sigma=kwargs['exploration_noise'], device=device, **kwargs)
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

    trainer = Trainer(x_dim=obs_dim, a_dim=a_dim, policy=policy, wm=wm, encoder=encoder, **kwargs)

    scores = {'train': [], 'eval': []}
    start_time = datetime.now()
    buffer = DynamicsReplayBuffer(kwargs['buffer_size'], device)
    while True:
        done = False
        s_t = env.reset()
        env.render()
        while not done:
            a_t = policy.act(torch.from_numpy(s_t).to(dtype=torch.float32, device=device)).numpy()
            s_tp1, r_t, done, info = env.step(a_t)
            env.render()
            scores['train'].append(r_t)
            buffer.add(s_t, a_t, s_tp1, done)
            if trainer.train_steps < kwargs['train_steps']:
                xs_t, as_t, xs_tp1, dones = buffer.sample(kwargs['batch_size'])
                trainer.train_step(xs_t, as_t, xs_tp1)
            if trainer.train_steps % kwargs['eval_interval'] == 0:
                print(trainer.train_steps)
                print(trainer.losses)
                visualise.train_iteration_update(**{k: np.mean(i[-kwargs['eval_interval']:])
                                                    for k, i in trainer.losses.items() if i != []},
                                                 ext=np.mean(scores['train'][-kwargs['eval_interval']:]))
                trainer.save_models(folder_name + 'saved_objects/')
            s_t = s_tp1


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
