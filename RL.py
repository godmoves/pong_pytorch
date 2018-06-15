# queue data structure. fast appends. and pops. replay memory
import argparse
from collections import deque
import os
import random  # random

import cv2  # read in pixel data
import numpy as np  # math
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pong  # our class


# hyper params
ACTIONS = 3  # up,down, stay
# define our learning rate
GAMMA = 0.99
# for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
# how many frames to anneal epsilon
EXPLORE = 500000
ADDITIONAL_OB = 50000
# store our experiences, the size of it
REPLAY_MEMORY = 500000
# batch size to train on
BATCH = 100


class CNN(nn.Module):
    # CNN only
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear_1 = nn.Linear(3136, 784)
        self.linear_2 = nn.Linear(784, ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.linear_1(x.view(x.size(0), -1)))
        x = self.linear_2(x)
        return x


class CNN_bn(nn.Module):
    # CNN with batch normalization
    def __init__(self):
        super(CNN_bn, self).__init__()
        self.conv_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.linear_1 = nn.Linear(3136, 784)
        self.linear_2 = nn.Linear(784, ACTIONS)

    def forward(self, x):
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.linear_1(x.view(x.size(0), -1)))
        x = self.linear_2(x)
        return x


class CNN_pool(nn.Module):
    # CNN with pooling layers
    def __init__(self):
        super(CNN_pool, self).__init__()
        self.conv_1 = nn.Conv2d(4, 32, kernel_size=6, stride=2)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear_1 = nn.Linear(3136, 784)
        self.linear_2 = nn.Linear(784, ACTIONS)

    def forward(self, x):
        x = F.relu(self.pool_1(self.conv_1(x)))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.linear_1(x.view(x.size(0), -1)))
        x = self.linear_2(x)
        return x


def to_tensor(x):
    # convert numpy array to tensor
    x = np.array(x)
    x = x.reshape(-1, 84, 84, 4)
    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.from_numpy(x).float().cuda()
    return x


def constructNet(arg):
    if arg == 'CNN':
        net = CNN
    elif arg == 'CNN_bn':
        net = CNN_bn
    elif arg == 'CNN_pool':
        net = CNN_pool
    else:
        net = arg
        raise NotImplementedError('{} is not implemented.'.format(net))
    return net


# deep q network. feed in pixel data to graph session
def trainAgent(net):

    # initialize our game
    game = pong.PongGame()

    # create a queue for experience replay to store policies
    # and set the maxlength equals to the size of replay memory
    D = deque(maxlen=REPLAY_MEMORY)

    # intial frame
    frame = game.getPresentFrame()
    # convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    # binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    Network = net().cuda()

    optimizer = optim.Adam(Network.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir='./logs')

    if os.path.exists('./params.pkl'):
        print('Restore from exists model')
        Network.load_state_dict(torch.load('./params.pkl'))
        # TODO: record steps
        steps = 0
    else:
        steps = 0

    expected_epsilon = INITIAL_EPSILON - steps * (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    if expected_epsilon > FINAL_EPSILON:
        epsilon = expected_epsilon
    else:
        epsilon = FINAL_EPSILON
    total_observe = steps + ADDITIONAL_OB

    # training time
    while(1):
        out_t = Network(to_tensor(inp_t))
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            maxIndex = random.randrange(ACTIONS)
        else:
            _, maxIndex = torch.max(out_t, 1)
            maxIndex = maxIndex.cpu().numpy()[0]
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward tensor if score is positive
        reward_t, frame, hit_rate, hit_rate_100 = game.getNextFrame(argmax_t)
        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        # new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)

        # add our input tensor, argmax tensor, reward and updated input tensor tos tack of experiences
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # training iteration
        if steps > total_observe:

            # get values from our replay memory
            minibatch = random.sample(D, BATCH)
            # minibatch = np.array(minibatch).transpose()

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            # out_batch = out.eval(feed_dict={inp: inp_t1_batch})
            out_prev_batch = Network(to_tensor(inp_batch))
            out_batch = Network(to_tensor(inp_t1_batch))

            # add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch.data.cpu().numpy()[i]))

            # action = np.mean(np.multiply(argmax_batch, out_prev_batch.data.cpu().numpy()), axis=1)
            action = torch.sum(out_prev_batch.mul(torch.FloatTensor(argmax_batch).cuda()), dim=1)
            gt_batch = torch.FloatTensor(reward_batch).cuda() + GAMMA * out_batch.max(1)[0]
            gt_batch = torch.autograd.Variable(gt_batch, requires_grad=False)

            loss = criterion(action, gt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update our input tensor the the next frame
        inp_t = inp_t1
        steps += 1

        # record the agent's performance every 100 steps
        if steps % 100 == 0:
            writer.add_scalars('', {'hit_rate': hit_rate,
                                    'hit_rate_100': hit_rate_100}, steps)

        # print our where we are after saving where we are
        if steps % 10000 == 0:
            torch.save(Network.state_dict(), './params.pkl')

        print("TIMESTEP", steps, "/ EPSILON %7.5f" % epsilon, "/ ACTION", maxIndex,
              "/ REWARD", reward_t, "/ Q_MAX %e" % torch.max(out_t))

        # stop traing after 1M steps
        if steps > 1000000:
            break


def main():
    parser = argparse.ArgumentParser(description='Choose network architecture.')
    parser.add_argument('--net', default='CNN', help='The kind of net you use, default is CNN')
    args = parser.parse_args()
    net = constructNet(args.net)
    # train our agent
    trainAgent(net)


if __name__ == "__main__":
    main()
