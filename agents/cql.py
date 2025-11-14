import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import ml_collections

from agents.sac import SACAgent


class CQLAgent(SACAgent):
    """Conservative Q-learning (CQL) agent."""

    def critic_loss(self, batch, grad_params, rng):
        """Compute the CQL critic loss."""

        # ======== TD target ========
        rng, td_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=td_rng)

        # target critic Q(s', a')
        next_q1, next_q2 = self.network.select('target_critic')(
            batch['next_observations'], next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2)

        # deterministic backup / entropy backup
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        if self.config['backup_entropy']:
            alpha = self.network.select('alpha')()
            target_q = target_q - self.config['discount'] * batch['masks'] * alpha * next_log_probs

        # current critic Q(s, a_data)
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['actions'], params=grad_params
        )

        td_loss1 = jnp.square(q1 - target_q).mean()
        td_loss2 = jnp.square(q2 - target_q).mean()
        td_loss = 0.5 * (td_loss1 + td_loss2)

        # =========== CQL(H) penalty ================
        K = int(self.config['cql_num_actions'])           # num_repeat_actions
        tau = jnp.maximum(self.config['cql_temperature'], 1e-6)
        B, A = batch['actions'].shape                     # B=batch size, A=action dim

        rng, rng_rand, rng_cur, rng_ns = jax.random.split(rng, 4)

        # =========== Random actions ===========
        a_rand = jax.random.uniform(
            rng_rand, (B, K, A), minval=-1.0, maxval=1.0
        )  # (B, K, A)

        obs_rand = jnp.repeat(batch['observations'], K, axis=0)   # (B*K, obs_dim)
        act_rand = a_rand.reshape(B * K, A)                       # (B*K, A)

        q1_rand, q2_rand = self.network.select('critic')(
            obs_rand, act_rand, params=grad_params
        )  # (B*K,), (B*K,)

        q1_rand = q1_rand.reshape(B, K)
        q2_rand = q2_rand.reshape(B, K)

        logp_rand = A * jnp.log(0.5)
        q1_rand = q1_rand - logp_rand
        q2_rand = q2_rand - logp_rand

        # =========== Current policy actions π(a|s) ===========
        cur_dist = self.network.select('actor')(batch['observations'])
        a_cur, logp_cur = cur_dist.sample_and_log_prob(
            seed=rng_cur, sample_shape=(K,)
        )   # a_cur: (K, B, A), logp_cur: (K, B)
        a_cur = jnp.transpose(a_cur, (1, 0, 2))          # (B, K, A)
        logp_cur = jnp.transpose(logp_cur, (1, 0))       # (B, K)

        obs_cur = jnp.repeat(batch['observations'], K, axis=0)  # (B*K, obs_dim)
        act_cur = a_cur.reshape(B * K, A)                       # (B*K, A)

        q1_cur, q2_cur = self.network.select('critic')(
            obs_cur, act_cur, params=grad_params
        )
        q1_cur = q1_cur.reshape(B, K)
        q2_cur = q2_cur.reshape(B, K)

        logp_cur_sg = jax.lax.stop_gradient(logp_cur)
        q1_cur = q1_cur - logp_cur_sg
        q2_cur = q2_cur - logp_cur_sg

        # =========== Next state policy actions π(a|s') ===========
        ns_dist = self.network.select('actor')(batch['next_observations'])
        a_ns, logp_ns = ns_dist.sample_and_log_prob(
            seed=rng_ns, sample_shape=(K,)
        )   # (K, B, A), (K, B)
        a_ns = jnp.transpose(a_ns, (1, 0, 2))            # (B, K, A)
        logp_ns = jnp.transpose(logp_ns, (1, 0))         # (B, K)

        obs_ns = jnp.repeat(batch['next_observations'], K, axis=0)  # (B*K, obs_dim)
        act_ns = a_ns.reshape(B * K, A)                             # (B*K, A)

        q1_ns, q2_ns = self.network.select('critic')(
            obs_ns, act_ns, params=grad_params
        )
        q1_ns = q1_ns.reshape(B, K)
        q2_ns = q2_ns.reshape(B, K)

        logp_ns_sg = jax.lax.stop_gradient(logp_ns)
        q1_ns = q1_ns - logp_ns_sg
        q2_ns = q2_ns - logp_ns_sg


        # (B, 3K)
        cat_q1 = jnp.concatenate([q1_rand, q1_cur, q1_ns], axis=1)
        cat_q2 = jnp.concatenate([q2_rand, q2_cur, q2_ns], axis=1)

        # CQL penalties
        conservative1 = tau * logsumexp(cat_q1 / tau, axis=1).mean() - q1.mean()
        conservative2 = tau * logsumexp(cat_q2 / tau, axis=1).mean() - q2.mean()

        cql_penalty = 0.5 * (conservative1 + conservative2)

        critic_loss = td_loss + self.config['cql_alpha'] * cql_penalty

        q = jnp.stack([q1, q2], axis=0)   

        return critic_loss, {
            'critic_loss': critic_loss,
            'td_loss': td_loss,
            'cql_penalty': cql_penalty,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

"""
AntMaze requires five hidden layers for the critic networks,
while other tasksperformance suffers with this number of layers.
(See https://arxiv.org/pdf/2210.07105 Observatiion 3)
"""

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cql',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256, 256),
            #actor_hidden_dims=(512, 512, 512, 512),
            
            value_hidden_dims=(256, 256, 256, 256, 256),
            #value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            target_entropy=ml_collections.config_dict.placeholder(float),
            target_entropy_multiplier=0.5,
            tanh_squash=True,
            state_dependent_std=True,
            actor_fc_scale=0.01,
            backup_entropy=False,  #  _deterministic_backup=True
            cql_alpha=5.0,
            cql_temperature=1.0,
            cql_num_actions=10,
            const_std=True,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
