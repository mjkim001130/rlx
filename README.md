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


### Caution

- For quicker and more stable D4RL reproduction, we recommend using **(256, 256)** MLPs instead of (512, 512, 512, 512). 

    ```python
    #actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
    #value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
    actor_hidden_dims=(256, 256),  # Actor network hidden dimensions.
    value_hidden_dims=(256, 256),  # Value network hidden dimensions.
    ```

- **CQL on AntMaze (medium/large)**
    > Multiple reports note that CQL often **fails to match the original paper’s scores** on AntMaze *medium*/*large*, sometimes performing significantly worse.



## Acknowledgments

This codebase is built on top of [fql](https://github.com/seohongpark/fql)