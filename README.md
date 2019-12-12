## Reproducing Results from Goncalves and Welchman, 2017

### First, setup virtual environment and install requirements after the repository is cloned
```
cd binocular_disparity_nn
virtualenv disp
source disp/bin/activate
pip install -r requirements.txt
```

### Run training with default parameters
`python code/train_bnn.py`

------
#### TODOs
- [ ] Change `code/utils.compute_accuracy(...)` function name to something more appropriate
- [ ] Add code for saving model checkpoints
- [ ] Add code for saving losses and accuracies
