import jax
from model import UNet
import jax.numpy as jnp
import optax
from functools import partial
from flax.training import train_state
from diffusion import forward_diffusion_sample, add_noise_to_context

def compute_loss(pred_noise, true_noise):
    loss = jnp.mean((pred_noise - true_noise) ** 2)
    return loss

def train_step(state, batch, key):
    def loss_fn(params):
        key_t, key_noise, key_context = jax.random.split(key, 3)
        batch_size = batch['x'].shape[0]
        t = jax.random.randint(key_t, (batch_size,), 0, 1000)
        x_t, noise = forward_diffusion_sample(batch['x'], t, key_noise)

        context_frames = batch['x']
        context_noisy = add_noise_to_context(context_frames, key_context)

        pred_noise = state.apply_fn({'params': params}, x_t, t, context_noisy)

        loss = compute_loss(pred_noise, noise)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def create_train_state(rng, learning_rate):
    model = UNet()
    # Initialize with x having 1 channel and context_noisy having 1 channel
    params = model.init(
        rng,
        jnp.ones([1, 28, 28, 1]),              # x: (1, 28, 28, 1)
        jnp.ones([1], dtype=jnp.int32),        # t: (1,)
        jnp.ones([1, 28, 28, 1])               # context_noisy: (1, 28, 28, 1)
    )['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
