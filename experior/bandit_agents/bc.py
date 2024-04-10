import jax.numpy as jnp
import jax
import chex

from experior.envs import Environment


def make_multi_armed_bc(env: Environment, num_envs: int, total_steps: int):
    def train(rng, expert_fraction: chex.Array, meta_params: chex.Array = None):
        # init env
        rng, rng_ = jax.random.split(rng)
        env_params = jax.vmap(env.init_env, in_axes=(0, None, None))(
            jax.random.split(rng_, num_envs), env.default_params, meta_params
        )
        rng, rng_ = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(
            jax.random.split(rng_, num_envs), env_params
        )
        n_actions = env.num_actions

        def _env_step(runner_state, i):
            obs, env_state, rng = runner_state

            # select (UCB)
            rng, rng_ = jax.random.split(rng)
            action = jax.random.choice(rng_, n_actions, p=expert_fraction)
            action = jnp.repeat(action, num_envs)

            # step env
            rng, rng_ = jax.random.split(rng)
            new_obs, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(rng_, num_envs), env_state, action, env_params
            )

            runner_state = (new_obs, env_state, rng)
            return (
                runner_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    **info,
                },
            )

        runner_state = (obs, env_state, rng)
        runner_state, metrics = jax.lax.scan(
            _env_step, runner_state, jnp.arange(total_steps)
        )

        return runner_state, metrics

    return train
