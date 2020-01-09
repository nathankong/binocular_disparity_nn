import numpy as np

import torch
import torch.nn as nn

class BinocularNetwork(nn.Module):
    def __init__(self, n_filters=28, k_size=19, input_size=30, init_gabors=False):
        super(BinocularNetwork, self).__init__()
        assert k_size % 2 == 1, "Kernel/filter size must be odd!"

        self.n_filters = n_filters
        self.k_size = k_size
        self.input_size = input_size
        self.num_in_channels = 2

        self.simple_unit = nn.Sequential(
            nn.Conv2d(
                self.num_in_channels,
                n_filters,
                kernel_size=k_size,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2, return_indices=False)
        )
        n_units = n_filters*(30-k_size+1)*(30-k_size+1) / 4
        self.complex_unit = nn.Sequential(
            nn.Linear(n_units, 2, bias=True)
        )
        #self.log_softmax = torch.nn.LogSoftmax(dim=1)

        if not init_gabors:
            print "Random initialization for simple units."
            self.simple_unit.apply(self._init_weights)
        else:
            print "Initialize Gabors for simple units."
            self.simple_unit.apply(self._init_gabors)

        print "Initialize complex units with zeros."
        self.complex_unit.apply(self._init_zeros)

    def _init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            print m
            nn.init.xavier_uniform_(m.weight)
            np.save("initial_kernels_xavier.npy", m.weight.data.numpy())

    def _init_zeros(self, m):
        if type(m) == nn.Linear:
            print m
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)

    def _init_gabors(self, m):
        if type(m) == nn.Conv2d:
            print m
            # Initialize weights as Gabor RFs with zero-disparity preference, as per Goncalves and Welchman.
            theta = 0.0
            f = 1
            sd = 0.3
            phase = np.pi * np.linspace(0., 1.5, self.n_filters)

            k = np.zeros((self.n_filters, self.num_in_channels, self.k_size, self.k_size))
            klim = 3*sd
            xm, ym = np.meshgrid(np.linspace(-klim, klim, self.k_size), np.linspace(-klim, klim, self.k_size))
            xyt = xm * np.cos(theta) + ym * np.sin(theta)
            i = 0
            for phi in phase:
                k[i, 0, :, :] = np.exp(-((xm**2)+(ym**2))/(2*sd**2)) * np.cos(xyt*f*2*np.pi + phi)
                k[i, 0, :, :] = k[i, 0, :, :] / np.sum(np.fabs(k[i, 0, :, :]))
                k[i, 1, :, :] = k[i, 0, :, :]
                i += 1
            np.save("initial_kernels_gabor.npy", k)

            m.weight = torch.nn.Parameter(torch.from_numpy(k).float(), requires_grad=True)
            m.bias.data.fill_(0.)

    def forward(self, x):
        x = self.simple_unit(x)
        x = torch.flatten(x, start_dim=1) 
        x = self.complex_unit(x)
        return x

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print "Device:", device

    m = BinocularNetwork().to(device)
    t = torch.rand(5,2,30,30)
    q = m(t.to(device))
    print q.size()
    print q

    print m.simple_unit[0].weight

    from torch.autograd import Variable
    a = torch.rand(5,2, requires_grad=True).to(device)
    print a
    a = Variable(torch.rand(5,2).to(device), requires_grad=True)
    print a
    a = torch.rand(5,2, device=device, requires_grad=True)
    print a
    
