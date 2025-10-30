import copy
from typing import Any

import flax
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class TD3BCAgent(flax.struct.PyTreeNode):
    """Twin Delayed DDPG + Behavior Cloning (TD3+BC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    max_action: float = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the TD3+BC critic loss."""
        rng, sample_rng = jax.random.split(rng)

        next_dist = self.network.select('target_actor')(batch['next_observations'])
        next_mode = next_dist.mode()
        if not self.config['tanh_squash']:
            next_mode = jnp.tanh(next_mode)

        sigma = self.config['policy_noise']
        noise_clip = self.config['noise_clip']
        noise = jnp.clip(jax.random.normal(sample_rng, next_mode.shape) * sigma, -noise_clip, noise_clip)

        next_actions = jnp.clip((next_mode + noise) * self.max_action, -self.max_action, self.max_action)

        target_q1, target_q2 = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        target_q = jnp.minimum(target_q1, target_q2)
        q_backup = batch['rewards'] + self.config['discount'] * batch['masks'] * target_q

        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = ((q1 - q_backup) ** 2 + (q2 - q_backup) ** 2).mean()

        q_values = jnp.stack([q1, q2], axis=0)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_values.mean(),
            'q_max': q_values.max(),
            'q_min': q_values.min(),
            'target_q_mean': q_backup.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the TD3+BC actor loss."""
        del rng

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        mode = dist.mode()
        if not self.config['tanh_squash']:
            mode = jnp.tanh(mode)

        actions = jnp.clip(mode * self.max_action, -self.max_action, self.max_action)

        q = self.network.select('critic')(batch['observations'], actions=actions)
        lam = self.config['alpha'] / (jnp.abs(q).mean() + 1e-6)
        lam = jax.lax.stop_gradient(lam)

        bc_loss = jnp.mean((actions - batch['actions']) ** 2)
        actor_loss = -lam * q.mean() + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'lambda': lam,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, actor_weight=1.0):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        actor_weight = jnp.asarray(actor_weight, dtype=jnp.float32)
        loss = critic_loss + actor_weight * actor_loss
        info['actor/weight'] = actor_weight
        return loss, info

    def target_update(self, network, module_name):
        """Update the Polyak-averaged target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        params = unfreeze(network.params)
        params[f'modules_target_{module_name}'] = new_target_params
        return network.replace(params=freeze(params))

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        step = self.network.step
        do_actor = jnp.equal(step % self.config['policy_delay'], 0)
        actor_weight = jnp.where(do_actor, 1.0, 0.0)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, actor_weight=actor_weight)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network, 'critic')
        new_network = jax.lax.cond(
            do_actor,
            lambda net: self.target_update(net, 'actor'),
            lambda net: net,
            new_network,
        )

        info['actor/do_update'] = actor_weight

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample deterministic actions from the actor."""
        del seed, temperature

        dist = self.network.select('actor')(observations)
        actions = dist.mode()
        if not self.config['tanh_squash']:
            actions = jnp.tanh(actions)

        actions = jnp.clip(actions * self.max_action, -self.max_action, self.max_action)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new TD3+BC agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        max_action = float(config.get('max_action', 1.0))

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['critic_hidden_dims'],
            layer_norm=config['critic_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=False,
            const_std=config['const_std'],
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        params = unfreeze(network_params)
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']
        network = TrainState.create(network_def, freeze(params), tx=network_tx)

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            max_action=max_action,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='td3_bc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(256, 256),  # Actor network hidden dimensions.
            critic_hidden_dims=(256, 256),  # Critic network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            critic_layer_norm=False,  # Whether to use layer normalization for the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            policy_noise=0.2,  # Stddev of target policy smoothing noise.
            noise_clip=0.5,  # Clipping range for target policy noise.
            policy_delay=2,  # Policy update frequency.
            alpha=2.5,  # Behavior cloning coefficient.
            tanh_squash=True,  # Whether to squash actions with tanh inside the actor.
            const_std=True,  # Whether to use constant actor standard deviation.
            actor_fc_scale=0.01,  # Final layer initialization scale for the actor.
            max_action=1.0,  # Maximum action magnitude.
            encoder=ml_collections.config_dict.placeholder(str),  # Optional visual encoder name.
        )
    )
    return config
