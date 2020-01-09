import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(0)

def get_cross_section(filter_pair, filter_idx, mid_idx):
    # filter_pairs: dimension (2, filter_height, filter_width)
    filter_left = filter_pair[0,mid_idx,:]
    filter_right = filter_pair[1,mid_idx,:]
    
    return filter_left, filter_right

def compute_gabor_in_range(xvals, parameters):
    # xvals: array of integers over which to compute the Gabor function
    # Rest of the arguments are torch autograd variables
    A, f, x0, phi, sigma = parameters

    cos_vals = torch.cos(2. * np.pi * f * (xvals-x0) + phi)
    coef = A * torch.exp(-1. * torch.pow(xvals-x0, 2) / (2. * torch.pow(sigma, 2)))

    return torch.mul(coef, cos_vals)

def get_x_vals(curve):
    width = curve.shape[0]
    assert width > 1
    xmin = 0. - ((width-1) / 2)
    xmax = -1. * xmin
    xvals = np.linspace(xmin, xmax, width)
    return xvals

def fit_1d_gabor(curve, num_steps=100000):
    # curve: 1d np.array of the filter mid section
    # length of curve defines the range we are operating in
    xvals = get_x_vals(curve)
    
    A = Variable(torch.rand(1), requires_grad=True)
    f = Variable(torch.rand(1), requires_grad=True)
    x0 = Variable(torch.rand(1), requires_grad=True)
    phi = Variable(torch.rand(1)*2.*np.pi-np.pi, requires_grad=True)
    sigma = Variable(torch.rand(1), requires_grad=True)

    parameter_list = [A, f, x0, phi, sigma]
    print("Initial parameters:", parameter_list)
    losses = optimize(curve, num_steps, parameter_list)
    print("Final parameters:", parameter_list)

    _parameters = list()
    for p in parameter_list:
        _parameters.append(p.data.numpy()[0])

    return _parameters, losses, xvals

def optimize(curve, num_steps, parameter_list):
    xvals = Variable(torch.from_numpy(get_x_vals(curve)), requires_grad=False)
    target = Variable(torch.from_numpy(curve).type(torch.DoubleTensor), requires_grad=False)

    loss_func = nn.MSELoss(reduction="sum")
    optimizer = optim.SGD(parameter_list, lr=0.05)
    
    prev_loss = np.Inf
    losses = list()
    for i in range(num_steps):
        # Reset the gradients
        optimizer.zero_grad()

        # Compute fit
        fits = compute_gabor_in_range(xvals, parameter_list)
        loss = loss_func(fits, target)
        curr_loss = loss.data.cpu().numpy()
        
        if np.abs(prev_loss - curr_loss) < 1e-11:
            break

        # Compute gradients and update parameters
        loss.backward()
        optimizer.step()

        prev_loss = curr_loss
        losses.append(prev_loss)
        if (i+1) % 5000 == 0:
            print("Step {}, Loss:".format(i+1), curr_loss)

    return losses

def fit_filter_pair(filter_left, filter_right, filter_idx):
    right_params, right_losses, right_xvals = fit_1d_gabor(filter_right, num_steps=100000)
    left_params, left_losses, left_xvals = fit_1d_gabor(filter_left, num_steps=100000)

    side = ["left", "right"]
    params = [left_params, right_params]
    losses = [left_losses, right_losses]
    xvals = [left_xvals, right_xvals]
    targets = [filter_left, filter_right]

    filter_name = "filter_{}".format(filter_idx)

    fits = dict()
    fits[filter_name] = dict()
    for i in range(len(side)):
        fits[filter_name][side[i]] = dict()

    for i in range(len(side)): # Left/Right
        fits[filter_name][side[i]]["params"] = params[i]
        fits[filter_name][side[i]]["losses"] = losses[i]
        fits[filter_name][side[i]]["xvals"] = xvals[i]
        fits[filter_name][side[i]]["target"] = targets[i]

    return fits

def main(filters):
    # filters: array of dimensions (num_filters, 2, filter_height, filter_width)
    num_filters = filters.shape[0]
    height = filters.shape[3]
    assert height % 2 == 1

    mid_idx = (height-1) / 2
    all_fits = list()
    for i in range(num_filters):
        filter_left, filter_right = get_cross_section(filters[i], i, mid_idx)
        fits = fit_filter_pair(filter_left, filter_right, i)
        all_fits.append(fits)

    fits_fname = "fits/fits.pkl".format(exp_id)
    pickle.dump(all_fits, open(fits_fname, "wb"))

if __name__ == "__main__":
    exp_id = "gabor"
    results = "results/results_{}.pkl".format(exp_id)
    results = pickle.load(open(results, "rb"))
    epochs_filter = np.array(results["simple_unit_weights"])
    filters = epochs_filter[-1] # Get filter pairs for last epoch

    # Test the function for one filter pair
    filters = np.expand_dims(filters[13], axis=0)

    print "Filters dimensions:", filters.shape
    print "Fitting filters..."
    main(filters)



