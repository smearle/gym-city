# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np

from poet_distributed.es import initialize_worker  # noqa
from poet_distributed.poet_algo import MultiESOptimizer  # noqa

parser = ArgumentParser()
parser.add_argument('log_file')
parser.add_argument('--init', default='random')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=0.9999)
parser.add_argument('--lr_limit', type=float, default=0.001)
parser.add_argument('--noise_std', type=float, default=0.1)
parser.add_argument('--noise_decay', type=float, default=0.999)
parser.add_argument('--noise_limit', type=float, default=0.01)
parser.add_argument('--l2_coeff', type=float, default=0.01)
parser.add_argument('--batches_per_chunk', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--eval_batches_per_step', type=int, default=50)
parser.add_argument('--n_iterations', type=int, default=200)
parser.add_argument('--steps_before_transfer', type=int, default=25)
parser.add_argument('--master_seed', type=int, default=111)
parser.add_argument('--repro_threshold', type=int, default=100)
parser.add_argument('--mc_lower', type=int, default=25)
parser.add_argument('--mc_upper', type=int, default=340)
parser.add_argument('--max_num_envs', type=int, default=100)
parser.add_argument('--adjust_interval', type=int, default=6)
parser.add_argument('--normalize_grads_by_noise_std', action='store_true', default=False)
parser.add_argument('--propose_with_adam', action='store_true', default=False)
parser.add_argument('--checkpointing', action='store_true', default=False)
parser.add_argument('--returns_normalization', default='normal')
parser.add_argument('--stochastic', action='store_true', default=False)
parser.add_argument('--envs', nargs='+')
parser.add_argument('--start_from', default=None)  # Json file to start from

parser.add_argument('--env-name', default='MicropolisEnv-v0',
        help='environment to train on (default: MicropolisEnv-v0)')
parser.add_argument('--map-width', type=int, default=20,
                    help="width of micropolis map")
parser.add_argument('--max-step', type=int, default=200)
########################################### Fractal Nets
parser.add_argument('--model', default='FractalNet')
parser.add_argument('--drop-path', action='store_true', help='enable global and local drop path on fractal model (ignored otherwise)')
parser.add_argument('--inter-shr', action='store_true',
  help='layers shared between columns')
parser.add_argument('--intra-shr', action='store_true',
        help='layers shared within columns')
parser.add_argument('--auto-expand', default=False, action = 'store_true',
        help='increment fractal recursion of loaded network')
parser.add_argument('--rule', default = 'extend',
  help='which fractal expansion rule to apply if using a fractal network architecture')
parser.add_argument('--n-recs', default=3, type=int,
  help='number of times the expansion rule is applied in the construction of a fractal net')
parser########################## Micropolis
parser.add_argument('--power-puzzle', action='store_true',
  help='a minigame: the agent uses wire to efficiently connect zones.')
parser.add_argument('--simple-reward', action='store_true',
  help='reward only for overall population according to game')
parser.add_argument('--traffic-only', action='store_true',
  help='reward only for overall traffic')
parser.add_argument('--random-builds', action='store_true',
  help='episode begins with random static (unbulldozable) builds on the map')
parser.add_argument('--random-terrain', action='store_true',
            help='episode begins on randomly generated micropolis terrain map')
parser.add_argument('--render', action='store_true', default=False,
                    help="render gui of single agent during training")

args = parser.parse_args()
print(args)
logger.info(args)

experiment = args.log_file
import os
try:
    os.mkdir('./ipp'.format(experiment))
    os.mkdir('./logs'.format(experiment))
except FileExistsError:
    pass
try:
    os.mkdir('./ipp/{}'.format(experiment))
    os.mkdir('./logs/{}'.format(experiment))
except FileExistsError:
    pass
open('./logs/{}.log'.format(experiment), 'a').close()
open('./ipp/{}.log'.format(experiment), 'a').close()
args.log_file = './logs/{}'.format(experiment)

def run_main(args):
    import ipyparallel as ipp

    client = ipp.Client(
           #debug=False
            )
    engines = client[:]
    r_engine = client[0]
    print(dir(engines))
    engines.block = True
    scheduler = client.load_balanced_view()
    engines.apply(initialize_worker)

    #set master_seed
    np.random.seed(args.master_seed)

    optimizer_zoo = MultiESOptimizer(args=args, engines=engines, 
            scheduler=scheduler, client=client,
            r_engine=r_engine)

    optimizer_zoo.optimize(iterations=args.n_iterations,
                       propose_with_adam=args.propose_with_adam,
                       reset_optimizer=True,
                       checkpointing=args.checkpointing,
                       steps_before_transfer=args.steps_before_transfer)
    #client.shutdown()

run_main(args)
