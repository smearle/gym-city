'''Do inference on a trained agent, for qualitative analysis or interactive play.'''
import argparse
import datetime

import torch


def str2bool(val):
    '''Interpret relevant strings as boolean values. For argparser.'''

    if isinstance(val, bool):
        return val

    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    if val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    ''' For training.'''
    parser = get_parser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not torch.cuda.is_available():
        print('CUDA not available')

    if args.experiment_name == '':
        args.experiment_name += '{}'.format(datetime.datetime.now())
  # args.experiment_name = "{}_{}".format(args.experiment_name, datetime.datetime.now())
    model_name = args.model

    if args.model == 'FractalNet':
        if args.rule != 'extend':
            model_name += '-{}'.format(args.rule)
       #model_name += '-{}recs'.format(args.n_recs)
        if args.intra_shr:
            model_name += '_intra'

        if args.inter_shr:
            model_name += '_inter'

        if args.drop_path:
            model_name += '_drop'

    if args.load_dir:
        args.save_dir = args.load_dir
        args.log_dir = args.load_dir
    else:
        args.save_dir = "trained_models/{}_{}/{}_w{}_{}s_{}".format(
            args.algo,
            model_name, args.env_name, args.map_width,
            args.max_step,
            args.experiment_name)
    # otherwise we might cut eval graph short by reloading too much
   #args.save_interval = args.eval_interval
    return args

def get_parser():
    '''The basic set of arguments pertaining to gym-city.'''
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=12,
                        help='how many training CPU processes to use (default: 12)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log', type=str2bool, default=True)
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='MicropolisEnv-v0',
                        help='environment to train on (default: MicropolisEnv-v0)')
#   parser.add_argument('--log-dir', default='trained_models',
#                       help='directory to save agent logs (default: /tmp/gym)')
#   parser.add_argument('--save-dir', default='./trained_models',
#                       help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--render', action='store_true', default=False,
                        help="render gui of single agent during training")
    parser.add_argument('--print-map', action='store_true', default=False)
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--vis', type=str2bool, default=True,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--map-width', type=int, default=16,
                        help="width of micropolis map")
    parser.add_argument('--model', default='FractalNet')
    parser.add_argument('--curiosity', action='store_true', default=False)
    parser.add_argument('--no-reward', action='store_true', default=False)
    parser.add_argument('--experiment_name', default='', help='a title for the experiment log')
    parser.add_argument('--overwrite', action='store_true', help='overwrite log files and saved model, optimizer')
    parser.add_argument('--max-step', type=int, default=200)

    ######## Fractal Net ########

#   parser.add_argument('--squeeze', action='store_true',
#           help= 'squeeze outward columns of fractal by recurrent up and down convolution')
#   parser.add_argument('--n-conv-recs', default=2,
#           help='number of recurrences of convolution at base level of fractal net')
    parser.add_argument('--load-dir', default=None,
            help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--record', default=False, action='store_true',
            help='film videos of inference')
########################################### Fractal Nets
    parser.add_argument('--drop-path', action='store_true', help='enable global and local drop path on fractal model (ignored otherwise)')
    parser.add_argument('--inter-shr', action='store_true',
            help='layers shared between columns')
    parser.add_argument('--intra-shr', action='store_true',
            help='layers shared within columns')
    parser.add_argument('--auto-expand', default=False, action = 'store_true',
            help='increment fractal recursion of loaded network')
    parser.add_argument('--rule', default='extend',
            help='which fractal expansion rule to apply if using a fractal network architecture')
    parser.add_argument('--n-recs', default=3, type=int,
            help='number of times the expansion rule is applied in the construction of a fractal net')
########################################### Micropolis
    parser.add_argument('--power-puzzle', action='store_true',
            help='a minigame: the agent uses wire to efficiently connect zones.')
   #parser.add_argument('--simple-reward', action='store_true',
   #        help='reward only for overall population according to game')
   #parser.add_argument('--traffic-only', action='store_true',
   #        help='reward only for overall traffic')
    parser.add_argument('--random-builds', type=str2bool, default=True,
            help='episode begins with random, potentially static (unbulldozable) builds on the map')
    parser.add_argument('--random-terrain', default=True, type=str2bool,
            help='episode begins on randomly generated micropolis terrain map')
    parser.add_argument('--n-chan', type=int, default=64)
    parser.add_argument('--val-kern', default=3)
    parser.add_argument(
        '--prebuild', default=False, help='GoL mini-game \
        encouraging blossoming structures')
    parser.add_argument(
        '--extinction-prob', type=float, default=0.0, help='probability of extinction event')
    parser.add_argument(
        '--extinction-type', type=str, default=None,
        help='type of extinction event')
    parser.add_argument('--im-render', action='store_true',
            help='Render micropolis as a simplistic image')
########################################### Game of Life
    parser.add_argument(
        '--prob-life', type=int, default=20,
        help='percent chance each tile is alive on reset')
########################################### ICM
    parser.add_argument(
        '--eta',
        type=float,
        default=0.01,
        metavar='LR',
        help='scaling factor for intrinsic reward')
    parser.add_argument(
        '--beta',
        type=float,
        default=0.2,
        metavar='LR',
        help='balance between inverse & forward')
    parser.add_argument(
        '--lmbda',
        type=float,
        default=0.1,
        metavar='LR',
        help='lambda : balance between A2C & icm')
    parser.add_argument(
        '--poet',
        type=str2bool,
        default=False,
        help='set targets for environment, replaces fixed reward function')

    return parser
