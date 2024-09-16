import jax
import jax.numpy as jnp
from diffusion import (
    alphas,
    betas,
    alphas_cumprod,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    add_noise_to_context
)

def sample(model, params, rng, context_frames, num_steps):

    x_shape = context_frames.shape
    rng, rng_init = jax.random.split(rng)
    x = jax.random.normal(rng_init, x_shape) 

    for i in reversed(range(0, num_steps)):
        rng, step_rng = jax.random.split(rng)
        key_noise, key_context = jax.random.split(step_rng)
        
        batch_size = x_shape[0]
        t = jnp.full((batch_size,), i, dtype=jnp.int32) 
        
        context_noisy = add_noise_to_context(
            context_frames, key_context, alpha_min=0.0, alpha_max=0.0
        ) 

        pred_noise = model.apply({'params': params}, x, t, context_noisy)

        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        if i > 0:
            alpha_bar_prev = alphas_cumprod[i - 1]
        else:
            alpha_bar_prev = 1.0

        beta = betas[i]

        # Compute posterior mean
        coef1 = (
            beta * sqrt_alphas_cumprod[i - 1] / (1 - alphas_cumprod[i])
            if i > 0 else 0.0
        )
        coef2 = (
            (1 - alphas_cumprod[i - 1]) * jnp.sqrt(alphas[i]) / (1 - alphas_cumprod[i])
            if i > 0 else 0.0
        )
        posterior_mean = coef1 * x + coef2 * pred_noise

        if i > 0:
            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.normal(noise_rng, x_shape)
            x = posterior_mean + jnp.sqrt(beta) * noise
        else:
            x = posterior_mean

    return x
