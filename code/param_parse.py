import argparse
from utils.utils import CONV_LAYERS2015
from datetime import datetime
from ast import literal_eval as make_tuple


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--export_interval', type=int, default=25)
    parser.add_argument('--eval_interval', type=int, default=int(500))
    parser.add_argument('--buffer_size', type=int, default=int(2e3))
    parser.add_argument('--train_steps', type=int, default=int(5e4))
    parser.add_argument('--exploration_noise', help='Sigma of exploration gaussian noise',
                        type=float, default=.01)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--policy_lr', type=float, default=1e-3)
    parser.add_argument('--policy_target_net_steps', type=int, default=0)
    parser.add_argument('--policy_soft_target', type=bool, default=False)
    parser.add_argument('--policy_tau', type=float, default=0.01)
    parser.add_argument('--z_dim', type=str, default='(32,)')
    parser.add_argument('--wm_h_dim', type=str, default='(32,)')
    parser.add_argument('--wm_target_net_steps', type=int, default=0)
    parser.add_argument('--wm_lr', type=float, default=1e-3)
    parser.add_argument('--wm_soft_target', type=bool, default=False)
    parser.add_argument('--wm_tau', type=float, default=0.01)
    parser.add_argument('--wm_enc_lr', type=float, default=1e-3)
    parser.add_argument('--enc_target_net_steps', type=int, default=0)
    parser.add_argument('--enc_soft_target', type=bool, default=False)
    parser.add_argument('--enc_tau', type=float, default=0.01)
    parser.add_argument('--encoder_type', type=str, default="none",
                        choices=['none', 'random', 'cont', 'idf', 'vae'])
    parser.add_argument('--encoder_load_path', type=str, default='')
    parser.add_argument('--resize_dim', type=str, default='(84, 84)')
    parser.add_argument('--grayscale', type=bool, default=True)
    parser.add_argument('--frame_stack', type=int, default=4)
    parser.add_argument('--conv_layers', type=tuple, default=CONV_LAYERS2015)

    parser.add_argument('--neg_samples', type=int, default=10)
    parser.add_argument('--hinge_value', type=float, default=0.1)
    parser.add_argument('--idf_inverse_hdim', type=str, default='(64,)')

    args = parser.parse_args().__dict__
    args['time_stamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    # Convert to tuples
    args['z_dim'] = make_tuple(args['z_dim'])
    args['wm_h_dim'] = make_tuple(args['wm_h_dim'])
    args['resize_dim'] = make_tuple(args['resize_dim'])
    args['idf_inverse_hdim'] = make_tuple(args['idf_inverse_hdim'])
    return args
