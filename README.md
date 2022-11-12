Graph-Structured Policy Learning for Multi-Goal Manipulation Tasks
---------------------------------------------------------------------
[Paper](https://arxiv.org/abs/2207.11313) | [Project Page](https://dmklee.github.io/graph-structured-manip/)

---------------------------------------------------------------------
This is the code for the paper "Graph-Structured Policy Learning for Multi-Goal
Manipulation Tasks" published in IROS'22.

## Installation
The code was tested using Python 3.8, and the neural networks are instantiated with PyTorch.
```
pip install -r requirements.txt
```

## Training
To train the method on block structures with different maximum heights, run:
```
python -m src.train --folder=./results --max_height=1 --num_env_steps=50000 --method=Ours
```
We recommend using 50k env steps per structure height (e.g. 250k for max height of 5).
You can also run the baselines by replacing the method argument with one of the following:
`Ours, HER, UVFA, Shaped, NeighborReplay, Curriculum`.
