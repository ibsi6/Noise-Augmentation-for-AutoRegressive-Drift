from flax import linen as nn
import jax
import jax.numpy as jnp

class DownSample(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        return x

class UpSample(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = jax.image.resize(
            x, 
            shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), 
            method='nearest'
        )
        x = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        return x

class UNet(nn.Module):
    @nn.compact
    def __call__(self, x, t, context_noisy):
        x = jnp.concatenate([x, context_noisy], axis=-1)  # Shape: (batch_size, 28, 28, 2)
        print(f"After concatenation: {x.shape}")  # Debug statement

        # Downsampling path
        h1 = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)  # Shape: (batch_size, 28, 28, 64)
        h1 = nn.relu(h1)
        h2 = DownSample(128)(h1)  # Shape: (batch_size, 14, 14, 128)
        h3 = DownSample(256)(h2)  # Shape: (batch_size, 7, 7, 256)

        # Embedding for time step t
        t_embedding = nn.Embed(num_embeddings=1000, features=32)(t)  # Shape: (batch_size, 32)
        t_embedding = t_embedding[:, None, None, :]  # Shape: (batch_size, 1, 1, 32)
        t_embedding = nn.Dense(256)(t_embedding)  # Shape: (batch_size, 1, 1, 256)
        t_embedding = jnp.broadcast_to(t_embedding, (h3.shape[0], h3.shape[1], h3.shape[2], 256))  # Shape: (batch_size, 7, 7, 256)

        # Bottleneck
        bottleneck = nn.Conv(512, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(h3 + t_embedding)  # Shape: (batch_size, 7, 7, 512)
        bottleneck = nn.relu(bottleneck)
        print(f"bottleneck shape: {bottleneck.shape}")  # Debug statement

        # Upsampling path
        h4 = UpSample(256)(bottleneck)  # Shape: (batch_size, 14, 14, 256)
        h4 = jnp.concatenate([h4, h2], axis=-1)  # Shape: (batch_size, 14, 14, 384)

        h5 = UpSample(128)(h4)  # Shape: (batch_size, 28, 28, 128)
        h5 = jnp.concatenate([h5, h1], axis=-1)  # Shape: (batch_size, 28, 28, 192)

        # Final convolution
        out = nn.Conv(1, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(h5)  # Shape: (batch_size, 28, 28, 1)
        return out
