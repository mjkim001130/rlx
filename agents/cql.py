import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class CQLAgent(flax.struct.PyTreeNode):
    """Conservative Q-learning (CQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def critic_loss(self, batch, grad_params):
        """Compute the CQL loss."""
        # TODO : implement critic_loss
        pass

    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        # TODO

    def target_update(self, network, module_name):
        """Update the target network."""
        # TODO

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # TODO

    def sample_actions(
            self,
            observations,
            seed=None,
            temperature=1.0,
    ):
        """Sample actions from the actor."""
        # TODO

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        # TODO

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cql',  # Agent name.
        )
    )
    # TODO
    return config