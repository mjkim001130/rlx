import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import ml_collections
import optax

from agents.sac import SACAgent
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, LogParam, Value


class CQLAgent(SACAgent):
    """Conservative Q-learning (CQL) agent."""

    def critic_loss(self, batch, grad_params, rng):
        """Compute the CQL critic loss."""

        # ======== TD target ========
        rng, td_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=td_rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        next_q = next_qs.min(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        if self.config['backup_entropy']:
            target_q = (target_q - self.config['discount'] * batch['masks'] * next_log_probs * self.network.select('alpha')())

        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        td_loss = jnp.square(q - target_q).mean()


        # ======== CQL(H) penalty ========
        K = int(self.config['cql_num_actions'])
        tau = jnp.maximum(self.config['cql_temperature'], 1e-6)
        B, A = batch['actions'].shape

        rng, rng_rand, rng_cur, rng_ns = jax.random.split(rng, 4)

        a_rand = jax.random.uniform(rng_rand, (B, K, A), minval=-1, maxval=1)

        cur_dist = self.network.select('actor')(batch['observations'])
        a_cur = cur_dist.sample(seed=rng_cur, sample_shape=(K,)).transpose((1, 0, 2))
        a_cur = jnp.clip(a_cur, -1, 1)
        
        ns_dist = self.network.select('actor')(batch['next_observations'])
        a_ns = ns_dist.sample(seed=rng_ns, sample_shape=(K,)).transpose((1, 0, 2))
        a_ns = jnp.clip(a_ns, -1, 1)

        a_cat = jnp.concatenate([a_rand, a_cur, a_ns], axis=1)
        BK = a_cat.shape[1]

        obs_rep = jnp.repeat(batch['observations'], BK, axis=0)           # (B*3K, obs)
        act_rep = a_cat.reshape(B * BK, A)                                # (B*3K, A)
        q_all = self.network.select('critic')(obs_rep, act_rep, params=grad_params)  # (E, B*3K)
        q_all = q_all.reshape(q_all.shape[0], B, BK)  # (E, B, 3K)


        cql_ood = tau * logsumexp(q_all / tau, axis=2) # (E, B)

        q_data = jnp.min(q, axis=0)
        cql_penalty = (jnp.mean(cql_ood, axis=0) - q_data).mean()

        critic_loss = td_loss + self.config['cql_alpha'] * cql_penalty

        return critic_loss, {
            'critic_loss': critic_loss,
            'td_loss': td_loss,
            'cql_penalty': cql_penalty,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            #actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_hidden_dims=(256, 256, 256),  # Actor network hidden dimensions.
            #value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_hidden_dims=(256, 256, 256),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            backup_entropy=False, # Whether to back up entropy in the critic loss.
            cql_alpha=10.0,
            cql_temperature=1.0,
            cql_num_actions=10,
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
