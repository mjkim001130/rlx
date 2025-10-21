# RLX: Offline reinforcement learning implements based on JAX.




## Installation

RLX equires Python 3.9+ and is based on JAX. The main dependencies are `jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`. To install the full dependencies, simply run:

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