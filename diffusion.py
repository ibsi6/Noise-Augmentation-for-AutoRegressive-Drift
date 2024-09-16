import jax
import jax.numpy as jnp
import numpy as np

# Diffusion hyperparameters
T = 1000  # Number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod)

betas = jnp.array(betas)
alphas_cumprod = jnp.array(alphas_cumprod)
sqrt_alphas_cumprod = jnp.array(sqrt_alphas_cumprod)
sqrt_one_minus_alphas_cumprod = jnp.array(sqrt_one_minus_alphas_cumprod)

def forward_diffusion_sample(x_0, t, key):
    key_noise = key  # Use key for noise
    noise = jax.random.normal(key_noise, x_0.shape)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.reshape((-1, 1, 1, 1))
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape((-1, 1, 1, 1))
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise


def add_noise_to_context(context, key, alpha_min=0.0, alpha_max=1.0):
    key_alpha, key_noise = jax.random.split(key)  # Split the key into two
    alpha = jax.random.uniform(
        key_alpha, 
        shape=(context.shape[0], 1, 1, 1), 
        minval=alpha_min, 
        maxval=alpha_max
    )
    noise = jax.random.normal(key_noise, context.shape)
    context_noisy = alpha * context + (1 - alpha) * noise
    return context_noisy

