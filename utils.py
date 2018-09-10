import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
from torch.optim import Adam

import random

import gym

import numpy as np
from copy import deepcopy

from Net import *
# from model import PolicyNetwork, ValueNetwork

import time


# Evolution Strategy Utility Functions
def generate_off_spring(tc, n_workers, sigma):
    """
    generate n offsprings and epsilons
    :param tc: state dict
    :param n_workers: a number
    :param sigma: std
    :return: off_spring, epsilons
    """
    off_springs = []
    epsilons = []
    for _ in range(n_workers):
        epsilon_cur = deepcopy(tc)
        off_spring_cur = deepcopy(tc)
        for param in tc:
            distribution = distr.Normal(torch.zeros(tc[param].shape),
                                        torch.zeros(tc[param].shape) + torch.Tensor([sigma]))
            epsilon_cur[param] = distribution.sample()
            off_spring_cur[param] = tc[param] + epsilon_cur[param]
        epsilons.append(epsilon_cur)
        off_springs.append(off_spring_cur)

    return off_springs, epsilons


def evaluate_obj(net, state_dict, max_path_length, env, num_episode=1):
    """
    compute fitness for a given state of the network
    :param net: Net
    :param state_dict:
    :param max_path_length:
    :param env: the environment
    :param num_episode: integer
    :return: averaged fitness: float
    """
    net.load_state_dict(state_dict)
    fitness = 0

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    for i in range(num_episode):
        obv = torch.Tensor([env.reset()])
        for t in range(max_path_length):
            # env.render()
            # try:
            #     action = net(obv).item()
            # except:
            #     action = net(obv).numpy()
            action, _ = net(obv)
            # print(list(net.parameters()))
            # print(action)
            if discrete:
                obv, rew, done, info = env.step(action.item())
            else:
                obv, rew, done, info = env.step(action.numpy())
            obv = torch.Tensor([obv])
            fitness += rew
            if done:
                break
    return fitness / num_episode


def compute_fitness(net, state_dict, max_path_length, env, num_episode=1):
    """
    compute fitness for a given state of the network
    :param net: Net
    :param state_dict:
    :param max_path_length:
    :param env:
    :param num_episode:
    :return: averaged fitness
    """
    net.load_state_dict(state_dict)
    fitness = 0

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    for i in range(num_episode):
        obv = torch.Tensor([env.reset()])
        # print("Check Line 101")
        for t in range(max_path_length):
            action, _ = net(obv)

            if discrete:
                obv, rew, done, info = env.step(action.item())
            else:
                obv, rew, done, info = env.step(action.numpy())
            obv = torch.Tensor([obv])
            fitness += rew
            if done:
                break
    return fitness / num_episode


def compute_new_tc(old_tc, all_fitness, epsilons, sigma, lr):
    """
    update old theta center to a new one
    :param old_tc: state_dict
    :param all_fitness: list of real numbers
    :param epsilons: list of state_dict
    :param sigma: std
    :param lr: learning_rate
    :return: new_tc
    """
    new_tc = old_tc

    all_fitness = normalization(all_fitness)

    for param in old_tc:
        acc = torch.Tensor([0])
        for i in range(len(all_fitness)):
            acc = all_fitness[i] * epsilons[i][param] + acc
        new_tc[param] = old_tc[param] + lr * (acc / (len(all_fitness) * sigma))

    return new_tc


def normalization(all_data):
    """
    :param all_data: list
    :return: normalized all_data
    """
    all_data = np.array(all_data)
    mean = np.mean(all_data)
    std = np.std(all_data)
    for i in range(len(all_data)):
        all_data[i] = (all_data[i] - mean) / (std + 1e-8)
    return all_data.tolist()


# Genetic Algorithm Utilities
def init_population_with_fit(population_size, net, sigma, max_path_length, theta_0, env, deterministic=False):
    """
    Initialize population by random or deterministic rule
    :param population_size: int
    :param net: DNN
    :param max_path_length: a number
    :param env: environment
    :param deterministic: Boolean
    :param init_num: None or int
    :return: population (a sorted nested list in the form [[state_dict, fitness],...])
    """
    assert init_num is not None if deterministic else True, \
        "You should provide the initial number for initialize the parameters."
    state_dict = net.state_dict()
    population = [[None, None] for _ in range(population_size)]
    if deterministic:
        for i in range(population_size):
            cur_child = deepcopy(state_dict)
            for param in cur_child:
                cur_child[param] = torch.zeros(cur_child[param].shape) + theta_0[param]
            population[i][0] = cur_child
    else:
        for i in range(population_size):
            cur_child = deepcopy(state_dict)
            for param in cur_child:
                distribution = distr.Normal(torch.zeros(cur_child[param].shape),
                                            2 * sigma)
                cur_child[param] = distribution.sample() + theta_0[param]
            population[i][0] = cur_child
    for each in population:
        fitness = compute_fitness(net, each[0], max_path_length, env)
        each[1] = fitness
    sort_population(population)
    return population


def init_elite(population, T_e, net, max_path_length, env):
    """

    :param population: sorted nested list
    :param T: an integer (the number of top ones that we are going to select)
    :return: the best in the first population
    """
    assert T_e > 1, "The number of top ones should be strictly greater than 1."
    assert T_e < len(population) + 1, "The number of top ones should less than the size of population."
    candidates = population[:T_e]
    elite = compute_elite(candidates, net, max_path_length, env)
    return elite


def get_elite(population, T_e, elite, net, max_path_length, env):
    """"""
    assert T_e > 2, "The number of top ones should be strictly greater than 2."
    assert T_e < len(population) + 1, "The number of top ones should less than the size of population."
    candidates = population[:T_e - 1]
    new_candidate = compute_elite(candidates, net, max_path_length, env)
    if new_candidate[1] > elite[1]:
        return new_candidate
    else:
        return elite


def compute_elite(candidates, net, max_path_length, env):
    """"""
    for each in candidates:
        acc = 0
        for _ in range(10):
            acc += compute_fitness(net, each[0], max_path_length, env)
        acc /= 10
        each[1] = acc
    sort_population(candidates)
    return candidates[0]


def sort_population(population_with_fit):
    """
    sort nested list in descending order
    :param population_with_fit: a nested list in the form [[state_dict, fitness],...]
    :return: sorted nested list with descending order with respect to fitness
    """
    population_with_fit.sort(key=lambda x: x[1], reverse=True)


def mutate_parent(parent, sigma, net):
    """
    Generate one offspring from one parent.
    :param parent: state_dict
    :param sigma: std
    :param net: DNN
    :return: child (state_dict)
    """

    state_dict = net.state_dict()

    assert [state_dict[param].shape for param in state_dict] == [parent[param].shape for param in parent], \
        "The shape of parent should meet the shape of state_dict of the your network."

    child = deepcopy(state_dict)

    for param in child:
        distribution = distr.Normal(torch.zeros(child[param].shape), 2 * sigma)
        child[param] = parent[param] + distribution.sample()

    return child


def get_new_population_with_fit(old_population, T, population_size, sigma, net, max_path_length, env):
    """
    get new sored and nested list with state_dicts and fitness
    :param old_population: a sorted nested list in the form [[state_dict, fitness],...] from last generation
    :param T: a integer (top T parents that we are going to generate new population from)
    :param net: DNN
    :param max_path_length: a integer
    :param env: environment
    :return: new_population (a new sorted nested list in the form [[state_dict, fitness],...])
    """
    assert T > 1, "The number of top ones should be strictly greater than 1."
    new_population = [[None, None] for _ in old_population]
    choices = [i for i in range(1, T)]
    for i in range(population_size):
        t = random.choice(choices)
        new_population[i][0] = mutate_parent(old_population[t][0], sigma, net)
        new_population[i][1] = compute_fitness(net, new_population[i][0], max_path_length, env)
    sort_population(new_population)
    return new_population


def policy_gradient_loss(log_prob, adv, num_path):
    # print("num_path: %d" % num_path)
    # print("log_prob: %s" % log_prob)
    # print("adv: %s" % adv)
    return - (log_prob.view(-1, 1) * adv).sum() / num_path


def pathlength(path):
    return len(path["reward"])


def train_PG(policy, policy_optimizer,
             env, max_path_length, min_timesteps_per_batch, gamma, itr,
             baseline_prediction=None, baseline_optimizer=None,
             reward_to_go=True, nn_baseline=False, normalize_advantages=True,
             animate=False, discrete=False):
    policy_loss = policy_gradient_loss  # Loss function that we'll differentiate to get the policy gradient.

    if nn_baseline:
        baseline_loss = nn.MSELoss()

    # Collect paths until we have enough timesteps
    timesteps_this_batch = 0
    paths = []
    while True:
        ob_ = env.reset()
        obs, acs, rewards, log_probs = [], [], [], []
        animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.05)
            ob = torch.from_numpy(ob_).float().unsqueeze(0)
            obs.append(ob)
            ac_, log_prob = policy(ob)
            acs.append(ac_)
            log_probs.append(log_prob)
            if discrete:
                ac = int(ac_)
            else:
                ac = ac_.squeeze(0).numpy()
            ob_, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > max_path_length:
                break
        path = {"observation": torch.cat(obs, 0),
                "reward": torch.Tensor(rewards),
                "action": torch.cat(acs, 0),
                "log_prob": torch.cat(log_probs, 0)}
        paths.append(path)
        timesteps_this_batch += pathlength(path)
        if timesteps_this_batch > min_timesteps_per_batch:
            break

    # Build arrays for observation, action for the policy gradient update by concatenating
    # across paths
    ob_no = torch.cat([path["observation"] for path in paths], 0)

    q_n_ = []
    for path in paths:
        rewards = path['reward']
        num_steps = pathlength(path)
        if reward_to_go:
            q_n_.append(torch.cat([(torch.pow(gamma, torch.arange(num_steps - t)) * rewards[t:]).sum().view(-1, 1)
                                   for t in range(num_steps)]))
        else:
            q_n_.append((torch.pow(gamma, torch.arange(num_steps)) * rewards).sum() * torch.ones(num_steps, 1))
    q_n = torch.cat(q_n_, 0)

    if nn_baseline:
        b_n = baseline_prediction(ob_no)
        q_n_std = q_n.std()
        q_n_mean = q_n.mean()
        b_n_scaled = b_n * q_n_std + q_n_mean
        adv_n = (q_n - b_n_scaled).detach()
    else:
        adv_n = q_n

    if normalize_advantages:
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + np.finfo(np.float32).eps.item())

    if nn_baseline:
        target = (q_n - q_n_mean) / (q_n_std + np.finfo(np.float32).eps.item())
        baseline_optimizer.zero_grad()
        b_loss = baseline_loss(b_n, target)
        b_loss.backward()
        baseline_optimizer.step()

    log_probs = torch.cat([path["log_prob"] for path in paths], 0)
    loss = policy_loss(log_probs, adv_n, len(paths))

    # policy_optimizer.zero_grad()
    # loss.backward()
    # policy_optimizer.step()

    returns = [path["reward"].sum() for path in paths]
    ave_return = np.mean(returns)

    # gradients = dict()
    #
    # state_dict = policy.state_dict()
    # i = 0
    # for param in state_dict:
    #     gradients[param] = list(policy.parameters())[i].grad
    #     i += 1

    return ave_return, loss


def policy_gradient(net, lr, env, max_path_length, num_episode=1):
    estimator = []
    gradients = dict()
    optimizer = optim.Adam(net.parameters(),
                           lr=lr)
    optimizer.zero_grad()

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    for i in range(num_episode):

        rewards_all_t = []
        log_prob_all_t = []

        obv = torch.Tensor([env.reset()])
        for t in range(max_path_length):
            action, log_prob = net(obv)
            if discrete:
                obv, rew, done, info = env.step(action.item())
            else:
                obv, rew, done, info = env.step(action.numpy())

            obv = torch.Tensor([obv])

            log_prob_all_t.append(log_prob)
            rewards_all_t.append(rew)

            if done:
                break

        rtg_all_t = rewards_to_rtg(rewards_all_t, norm=True)

        log_prob_all_t = torch.stack(log_prob_all_t, dim=1).squeeze()

        estimator.extend(torch.Tensor(rtg_all_t) * log_prob_all_t)

    estimator = sum(estimator)
    estimator /= num_episode

    estimator.backward()

    optimizer.step()

    state_dict = net.state_dict()
    i = 0
    for param in state_dict:
        gradients[param] = list(net.parameters())[i].grad
        i += 1

    new_objective = evaluate_obj(net, net.state_dict(), max_path_length, env)

    return new_objective, gradients


def rewards_to_rtg(rewards_all_t, norm=True):
    rtg = [None for _ in range(len(rewards_all_t))]
    for i in range(len(rtg)):
        rtg[i] = sum(rewards_all_t[i:])
    rtg = np.array(rtg)
    if norm:
        rtg -= np.mean(rtg)
        rtg /= np.std(rtg)
    return rtg.tolist()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.reset()
    act_dim = env.action_space.n
    obv_dim = env.observation_space.shape[0]
    net = Net(obv_dim, [1], act_dim, [nn.ReLU()])
    old = init_population_with_fit(5, net, 1, 150, env)
    new = get_new_population_with_fit(old, 3, 5, 1, net, 150, env)
    for each in new:
        print(each[1])
