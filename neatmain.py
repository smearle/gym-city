# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import multiprocessing
import os

import click
import gym
import neat

import numpy as np


from PyTorch_NEAT.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from PyTorch_NEAT.pytorch_neat.activations import relu_activation
from PyTorch_NEAT.pytorch_neat.neat_reporter import LogReporter
from PyTorch_NEAT.pytorch_neat.adaptive_net import AdaptiveNet

import gym_city
max_env_steps = 200

batch_size = 1
DEBUG = False

def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(9, render_gui=True)
    return env


def make_net(genome, config, bs):
    input_coords = [[-k, i - 4, j - 4] for j in range(9) for i in range(9) for k in range(32)]
    hidden_coords = [[-k, i - 1, j - 1]  for j in range(3) for i in range(3) for k in range(32)]
    output_coords = [[-k, i - 4, j - 4] for j in range(9) for i in range(9) for k in range(19)]
    return AdaptiveNet.create(
        genome,
        config,
        input_coords=input_coords,
        output_coords=output_coords,
        hidden_coords=hidden_coords,
        weight_threshold=0.4,
        batch_size=batch_size,
        activation=relu_activation,
        device="cuda:0",
    )



def activate_net(net, states, debug=False, step_num=0):
    if debug and step_num == 1:
        print("\n" + "=" * 20 + " DEBUG " + "=" * 20)
        print(net.delta_w_node)
        print("W_i init: ", net.input_to_hidden[0])
        print("W_o init: ", net.hidden_to_output[0])
    outputs = net.activate(states).cpu().numpy()
    if debug and (step_num - 1) % 100 == 0:
        print("\nStep {}".format(step_num - 1))
        print("Outputs: ", outputs[0])
        print("Delta W: ", net.delta_w_node)
        print("W_i: ", net.input_to_hidden[0])
        print("W_o: ", net.hidden_to_output[0])
    return np.argmax(outputs, axis=1)



@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
def run(n_generations, n_processes):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                    evaluator.eval_genome, ((genome, config) for _, genome in genomes)
                    )

            for (_, genome) in genomes:
                genome.fitness = evaluator.eval_genome(genome, config)
    else:
        def eval_genomes(genomes, config):
            for i, (_, genome) in enumerate(genomes):
                try:
                    genome.fitness = evaluator.eval_genome(
                        genome, config, debug=DEBUG and i % 100 == 0
                    )
                except Exception as e:
                    print(genome)
                    raise e


    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("log.json", evaluator.eval_genome)
    pop.add_reporter(logger)



    winner = pop.run(eval_genomes, n_generations)

    print(winner)
    final_performance = evaluator.eval_genome(winner, config)
    print("Final performance: {}".format(final_performance))
    generations = reporter.generation + 1
    return generations



if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
