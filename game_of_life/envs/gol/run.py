import argparse
import itertools

import cv2
import numpy as np
import torch
from torch import ByteTensor, Tensor
from torch.nn import Conv2d, Parameter
from torch.nn.init import zeros_

import utils


def step(state, get_neighbors):
    # Get neighbor counts of cells
    neighbors = get_neighbors(state)[0, ...]

    # Alive cell with less than two neighbors should die
    rule1 = (neighbors < 2).type(Tensor)
    
    print(rule1, state)
    rule1 = rule1.cuda()

    mask1 = (rule1 * state[0, ...]).type(ByteTensor)

    # Alive cell with more than two neighbors should die
    rule2 = (neighbors > 3).type(Tensor)
    rule2 = rule2.cuda()
    mask2 = (rule2 * state[0, ...]).type(ByteTensor)

    # Dead cell with exactly three neighbors should spawn
    rule3 = (neighbors == 3).type(Tensor)
    rule3 = rule3.cuda()
    mask3 = (rule3 * (1 - state[0, ...])).type(ByteTensor)

    # Update state
    state[0, mask1] = 0
    state[0, mask2] = 0
    state[0, mask3] = 1
    return state


def run_world(opts, device):
    step_count = 0
    size = opts.size
    prob = opts.prob
    tick_ratio = opts.tick_ratio
    record_entropy = opts.record_entropy

    entropies = []
    with torch.no_grad():
        combinations = 2 ** (3 * 3)
        channels = 1
        state = init_world(size, channels, prob).to(device)
        get_neighbors = get_neighbors_map(channels).to(device)
        structure_similarity = get_structure_similarity(combinations, channels)
        structure_similarity.to(device)
        i = 0
        cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
        while True:
            if should_step(step_count, tick_ratio):
                cv2.imshow("Game of Life", state.cpu().numpy())
                state.to(device)
                print(state)
                state = step(image2state(state), get_neighbors)

                if record_entropy:
                    entropy = get_entropy(
                        state, structure_similarity, combinations)
                    entropies.append(entropy)
                state = state2image(state)
            step_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1
    cv2.destroyAllWindows()

    if record_entropy:
        entropies = np.array(entropies)
        utils.plot_entropies(entropies)


def init_world(size, channels, prob):
    return torch.distributions.Bernoulli(
        Tensor([prob])).sample(torch.Size([size, size, channels])).squeeze(-1)


def image2state(image):
    return image.permute(2, 0, 1).unsqueeze(0)


def state2image(state):
    return state.squeeze(0).permute(1, 2, 0)


def get_neighbors_map(channels):
    neighbors_filter = Conv2d(channels, channels, 3, padding=1)
    neighbors_filter.weight = Parameter(Tensor([[[[1, 1, 1],
                                                  [1, 0, 1],
                                                  [1, 1, 1]]]]),
                                        requires_grad=False)
    neighbors_filter.bias = zeros_(neighbors_filter.bias)
    return neighbors_filter


def get_structure_similarity(combinations, channels):
    tensors = torch.zeros([combinations, 1, 3, 3])
    elems = list(map(Tensor, itertools.product([0, 1], repeat=9)))
    for i, elem in enumerate(elems):
        tensors[i] = elem.view(1, channels, 3, 3)
    structure_similarity = Conv2d(
        channels, combinations, 3, stride=3, groups=channels)
    structure_similarity.weight = Parameter(tensors, requires_grad=False)
    structure_similarity.bias = zeros_(structure_similarity.bias)
    return structure_similarity


def get_entropy(state, structure_similarity, combinations):
    configs = structure_similarity(state)
    match_weights = structure_similarity.weight.view(combinations, -1).sum(-1)
    distribution = torch.zeros([combinations])

    # Smooth distribution incase configuration doesn't exist
    distribution.fill_(utils.EPSILON)
    for i, weight in enumerate(match_weights):
        config = configs[0][i]
        mask = config == weight
        distribution[i] += config[mask].shape[0]
    distribution /= distribution.sum()

    entropy = utils.entropy(distribution, 2)
    info = "Max Event Probability: {} | Entropy: {}".format(
        distribution.max(), entropy)
    print(info)
    return entropy


def should_step(step_count, tick_ratio):
    return step_count % tick_ratio == 0


def main():
    opts = argparse.ArgumentParser(description='Game of Life')
    opts.add_argument(
        '-s',
        '--size',
        help='Size of world grid',
        default=700)
    opts.add_argument(
        '-p',
        '--prob',
        help='Probability of life in the initial seed',
        default=.05)
    opts.add_argument(
        '-tr',
        '--tick_ratio',
        help='Ticks needed to update on time step in game',
        default=1)
    opts.add_argument(
        '-re',
        '--record_entropy',
        help='Should record entropy of configurations',
        default=True)
    opts = opts.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_world(opts, device)


if __name__ == "__main__":
    main()
