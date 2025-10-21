# RLX — Offline Reinforcement Learning in JAX

**RLX** is a JAX/Flax-based codebase for **Offline Reinforcement Learning (ORL)**.  
Experiments target **OGBench** and **D4RL** datasets, with consistent training, logging, and evaluation pipelines.

> Built for clarity, reproducibility, and easy extension.


## Installation

RLX requires Python 3.9+ and is based on JAX. The main dependencies are `jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`. To install the full dependencies, simply run:

```shell
pip install -r requirements.txt
```

## Supported algorithms

- Model-free
    - [Behavior Cloning (BC)](./agents/bc.py)
    - [Implicit Q-Learning (IQL)](./agents/iql.py)
    - [Conservative Q-Learning (CQL)](./agents/cql.py)




## Acknowledgments

This codebase is built on top of [fql](https://github.com/seohongpark/fql)