## Reproducing Results from Goncalves and Welchman, 2017

### First, setup virtual environment and install requirements
```
virtualenv disp
source disp/bin/activate
pip install -r requirements.txt
```

### Image Directory
The images must be organized in two sub-directories, where each directory is a specific class.
Look at code in `code/make_dataset_dir.py` to see how the directory structure is organized.

### Run training with default parameters
`python code/train_bnn.py --imagedir PATH/TO/IMAGES`

------
#### TODOs
- [ ] Change `code/utils.compute_accuracy(...)` function name to something more appropriate
- [ ] Add code for saving model checkpoints
- [ ] Add code for saving losses and accuracies
