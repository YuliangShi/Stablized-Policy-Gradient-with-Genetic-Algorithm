import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist


class Net(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_sizes, hidden_atvs, out_atv=None, deterministic=True, discrete=True,
                 act_range=None):

        super(Net, self).__init__()

        # assert act_range if not discrete else act_range is None, \
        #     "The range (2 * act_dim tensor) of actions space is required. / range is not required."
        assert len(hidden_sizes) == len(hidden_atvs), \
            "The number of hidden layers is not equal to the number of hidden atvs."
        assert isinstance(hidden_sizes, (tuple, list)) and isinstance(hidden_atvs, (tuple, list)), \
            "hidden_sizes and hidden_atvs should be list or tuple."

        self.deter = deterministic
        self.discrete = discrete

        if not self.discrete:
            self.act_range = act_range
            self.out_size = out_sizes

        layers = list()

        layers.append(nn.Linear(in_size, hidden_sizes[0]))
        if hidden_atvs[0]:
            layers.append(hidden_atvs[0])

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if hidden_atvs[i]:
                layers.append(hidden_atvs[i])

        layers.append(nn.Linear(hidden_sizes[-1], out_sizes))
        if out_atv:
            layers.append(out_atv)

        if not discrete:
            mean = [nn.Linear(out_sizes, 16), nn.Linear(16, out_sizes)]
            self.mean = nn.Sequential(*mean)
            std = [nn.Linear(out_sizes, 16), nn.Linear(16, out_sizes)]
            self.std = nn.Sequential(*std)

        self.sequence = nn.Sequential(*layers)

    def forward(self, obv):
        if self.discrete:
            # return deterministic action
            if self.deter:
                x = self.sequence(obv)
                probs = F.softmax(x, dim=1)
                return torch.argmax(probs, dim=1)
            # have randomness in sampling action
            else:
                # print(obv)
                x = self.sequence(obv)
                # print(x)
                logits = F.log_softmax(x, dim=1)
                # print(logits.exp())
                # print(logits)
                distribution = dist.Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
                return action.detach(), log_prob
        else:
            # return deterministic action
            if self.deter:
                action = F.tanh(self.sequence(obv))
                return action
            # have randomness in action
            else:
                x = self.sequence(obv)
                mean = F.tanh(self.mean(x))  # params that are not going to be updated
                log_std = F.tanh(self.std(x))

                # cov_mtx = torch.Tensor([each.diag().tolist() for each in std.exp()])
                # construct a (multi-)normal distribution and sample from it
                # print(mean, cov_mtx, sep="\n")
                distribution = dist.Normal(loc=mean, scale=log_std.exp())
                # distribution = dist.MultivariateNormal(mean, cov_mtx)
                action = distribution.sample()
                log_prob = distribution.log_prob(action).sum(1).view(-1, 1)

                return torch.clamp(action.detach(), min=-3., max=3.), log_prob


class ValueNetwork(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size, activations, out_activation=None):
        super(ValueNetwork, self).__init__()

        assert len(hidden_sizes) > 0, 'No hidden layer exists.'
        assert len(hidden_sizes) == len(activations), 'Num of hidden layers does not match that of activations.'

        fc = [nn.Linear(in_size, hidden_sizes[0]), activations[0]]
        for i in range(len(hidden_sizes) - 1):
            fc.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            fc.append(activations[i + 1])
        fc.append(nn.Linear(hidden_sizes[-1], out_size))
        if out_activation is not None:
            fc.append(out_activation())
        self.fc = nn.Sequential(*fc)

    def forward(self, o):
        return self.fc(o)


if __name__ == "__main__":
    net = Net(2, [1, 2], 3, [nn.ReLU(), nn.ReLU()], deterministic=False, discrete=False)
    print(net)

    optimizer = optim.Adam(net.parameters())
    for _ in range(3):
        _, out = net(torch.Tensor([[1, 2]]))
        out = torch.mean(out)

        optimizer.zero_grad()

        out.backward()
        optimizer.step()

        state_dict = net.state_dict()
        new = dict()
        i = 0
        for param in state_dict:
            new[param] = list(net.parameters())[i].grad
            i += 1

        print(new)
    # print(state_dict)
    # print(list(net.parameters()))
    # for param in net.parameters():
    #     print(param, param.grad)
    # for param in state_dict:
    #     print(param)
    # state_dict = net.state_dict()
    # print(state_dict)

    # print(net.load_state_dict(state_dict))
    # # print(net(torch.Tensor([[1., 3.], [2., 3.]])))

    # print()
    # for value in state_dict.values():
    #     print(value)
    #     break
    # print()
    # torch.manual_seed(0)
    #
    # distribution1 = dist.Normal(torch.Tensor([0, 0]), torch.Tensor([1]))
    # print(distribution1.sample())
    #
    # distribution2 = dist.Normal(torch.Tensor([0, 0]), torch.Tensor([1]))
    # print(distribution2.sample())
    #
    # distribution3 = dist.Normal(torch.Tensor([0, 0]), torch.Tensor([1]))
    # print(distribution3.sample())

    # for key in state_dict.keys():
    #     state_dict[key] = torch.Tensor([0])
    # net.load_state_dict(state_dict)
    # for value in net.state_dict().values():
    #     print(value)

    # for k in state_dict:
    #     distribution = dist.Normal(torch.zeros(state_dict[k].shape), torch.zeros(state_dict[k].shape) + torch.Tensor([1]))
    #     print(distribution.sample())
