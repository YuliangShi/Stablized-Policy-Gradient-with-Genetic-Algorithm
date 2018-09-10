#!/usr/bin/env python

import torch
import torch.distributions as distr
import torch.nn as nn
import torch.optim as optim

from Net import Net
from utils import *

import gym
import gym.spaces

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from copy import deepcopy


def train_PGGA(
        # exp_name,
        env_name="CartPole-v0",
        max_path_length=400,
        learning_rate=1e-3,
        animate=False,
        sigma=1000,
        # theta_c=0,
        # normalized=True,
        seed=0,
        gamma=0.9,
        ave_weight=0.9,
        n_iter=1000,
        n_layers=2,
        min_timesteps_per_batch=100,
        n_gen=50,
        size=32,
        n_workers=200,
        T_e=10,
        T=50
):
    traing_log = []
    env = gym.make(env_name)

    max_path_length = max_path_length or env.spec.max_episode_steps

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]
    obv_dim = env.observation_space.shape[0]

    net = Net(obv_dim, [size for _ in range(n_layers)], act_dim, [nn.ReLU() for _ in range(n_layers)],
              discrete=discrete, deterministic=False)

    net_optim = optim.Adam(net.parameters(), lr=learning_rate)

    # state_dict = net.state_dict()
    # obj = evaluate_obj(net, state_dict, max_path_length, env)
    # imp_ave = 0

    for i in range(n_iter):
        # print("LINE 65:", gamma)
        print("***************** Iter %d ******************" % i)
        # print("At the very beginning of iter %d:" % i, list(net.parameters()))
        new_obj, loss = train_PG(net, net_optim,
                                      env, max_path_length,
                                      min_timesteps_per_batch, gamma,
                                      i,
                                      animate=animate, discrete=discrete)
        # imp = new_obj - obj
        # print(new_obj)
        # print(imp)
        # print(- abs(imp_ave) * 0.5)

        # print("Outside of GA:", list(net.parameters()))

        # if imp < - abs(imp_ave) * 2:
        #
        #     performance = new_obj
        #     population = init_population_with_fit(n_workers, net, sigma, max_path_length, net.state_dict(), env)
        #     elite = init_elite(population, T_e, net, max_path_length, env)
        #
        #     print("After GA initialization:", list(net.parameters()))
        #
        #     for g in range(n_gen):
        #         print("     ****************** GEN %d ******************" % g)
        #
        #         population = get_new_population_with_fit(population, T, n_workers, sigma, net, max_path_length, env)
        #
        #         elite = get_elite(population, T_e, elite, net, max_path_length, env)
        #
        #         all_fitness = np.array([each[1] for each in population])
        #         performance = elite[1]
        #
        #         # print("After gen %d of GA:" % g, list(net.parameters()))
        #
        #         print("     accumulated reward of the elite: %d" % performance)
        #         print("     ave accumulated reward of offsprings: %d" % np.mean(all_fitness))
        #         print("     ********************************************")
        #
        #     net.load_state_dict(elite[0])
        #     # print("After loading the elite of GA:", list(net.parameters()))
        #     imp = performance - obj
        #     new_obj = performance

        # imp_ave = ave_weight * imp_ave + (1 - ave_weight) * imp
        #
        net_optim.zero_grad()
        loss.backward()
        net_optim.step()
        obj = new_obj
        #
        traing_log.append(obj)

        print("accumulated reward after Iter %d: %d" % (i, obj))
        print()

    # print(traing_log)

    return traing_log


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=list, default=['PG'], nargs="*")

    parser.add_argument("--env_name", type=str, default="HalfCheetah-v2")
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--max_trjc_len', '-ml', type=int, default=150)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)

    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-2)
    parser.add_argument('--batch_size', '-b', type=int, default=2000)
    parser.add_argument('--gamma', '-gm', type=float, default=0.95)

    parser.add_argument('--n_gen', '-g', type=int, default=1)
    parser.add_argument('--ave_weight', '-aw', type=float, default=1.)
    parser.add_argument('--sigma', '-si', type=float, default=500.)
    parser.add_argument('--num_workers', '-w', type=int, default=200)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--T_e', type=int, default=10)

    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--n_layers', '-l', type=int, default=2)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--render', '-ren', action='store_true')

    args = parser.parse_args()

    max_path_length = args.max_trjc_len if args.max_trjc_len > 0 else None

    training_log_all = []

    for i in range(len(args.exp_name)):

        training_log = []

        for e in range(args.n_experiments):
            seed = args.seed + 5 * e
            print('Running experiment with seed %d' % seed)

            def train_func():
                return train_PGGA(
                    # exp_name=args.exp_name,
                    env_name=args.env_name,
                    max_path_length=max_path_length,
                    learning_rate=args.learning_rate,
                    # animate=args.render,
                    sigma=args.sigma,
                    # theta_c=args.theta_c,
                    # normalized=args.normalized,
                    seed=seed,
                    gamma=args.gamma,
                    ave_weight=args.ave_weight,
                    n_iter=args.n_iter,
                    n_layers=args.n_layers,
                    min_timesteps_per_batch=args.batch_size,
                    n_gen=args.n_gen,
                    size=args.size,
                    n_workers=args.num_workers,
                    T_e=args.T_e,
                    T=args.T
                )

            training_log.append(train_func())

        training_log_all.append(training_log)

    name = args.exp_name[0] + '_' + args.env_name + "_Net" + str(args.n_layers) + "_" + str(args.size) + \
        "_lr" + str(args.learning_rate) + \
        "_bz" + str(args.batch_size) + \
        "_gm" + str(args.gamma) + \
        "_Net" + str(args.n_layers) + "_" + str(args.size) + \
        "_s" + str(args.seed)

    np.savetxt(name, training_log_all[0])  # save to disk


if __name__ == "__main__":
    main()
