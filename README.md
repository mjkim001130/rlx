# RLX — Offline Reinforcement Learning in JAX

**RLX** is a JAX/Flax-based codebase for **Offline Reinforcement Learning**.  
Experiments target **OGBench** and **D4RL** datasets, with consistent training, logging, and evaluation pipelines.

## Installation

RLX requires Python 3.9+ and is based on JAX. The main dependencies are `jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`. To install the full dependencies, simply run:

```shell
pip install -r requirements.txt
```

> [!NOTE]
> To use D4RL environments, you need to additionally set up MuJoCo 2.1.0.


## Supported algorithms

- Model-free
    - [Behavior Cloning (BC)](./agents/bc.py)
    - [Implicit Q-Learning (IQL)](https://arxiv.org/pdf/2110.06169)
    - [Conservative Q-Learning (CQL)](https://arxiv.org/pdf/2006.04779)
    - [TD3+BC](https://arxiv.org/pdf/2106.06860)

## Usage

To reproduce results on **D4RL AntMaze** with 5 seeds (0–4), simply run:
```bash
./d4rl.sh
```



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