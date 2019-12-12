import torch
import torch.nn as nn

class BinocularNetwork(nn.Module):
    def __init__(self, n_filters=28, k_size=19, input_size=30):
        super(BinocularNetwork, self).__init__()
        self.simple_unit = nn.Sequential(
            nn.Conv2d(
                2,
                n_filters,
                kernel_size=k_size,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=False)
        )
        n_units = n_filters*(30-k_size+1)*(30-k_size+1) / 4
        self.complex_unit = nn.Sequential(
            nn.Linear(n_units, 2, bias=True)
        )
        #self.softmax = torch.nn.Softmax(dim=1)

        self.simple_unit.apply(self.init_weights)
        self.complex_unit.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.simple_unit(x)
        x = torch.flatten(x, start_dim=1) 
        x = self.complex_unit(x)
        #x = self.softmax(x)
        return x

if __name__ == "__main__":
    m = BinocularNetwork()
    t = torch.rand(5,2,30,30)
    q = m(t)
    print q.size()
    print q


