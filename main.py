import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from data import load_dataset, data_loader
from train import create_train_state, train_step
from sample import sample
from model import UNet

def visualize_samples(untrained_samples, trained_samples, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    for i in range(num_samples):
        axes[0, i].imshow(untrained_samples[i, :, :, 0], cmap='gray')
        axes[0, i].set_title("Untrained")
        axes[0, i].axis('off')
        axes[1, i].imshow(trained_samples[i, :, :, 0], cmap='gray')
        axes[1, i].set_title("Trained")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    key = jax.random.PRNGKey(0)
    dataset = load_dataset()

    # Hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-3

    state = create_train_state(key, learning_rate)

    model = UNet()
    num_samples = 5
    num_steps = 100
    context_frames = jnp.zeros((num_samples, 28, 28, 1))

    print("Generating samples with the untrained model.")
    rng_untrained = jax.random.PRNGKey(42)
    untrained_samples = sample(model, state.params, rng_untrained, context_frames, num_steps=num_steps)
    untrained_samples = np.array(untrained_samples)
    untrained_samples = (untrained_samples + 1) / 2 

    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        dataloader = data_loader(dataset, batch_size)
        for batch in dataloader:
            key, subkey = jax.random.split(key)
            state, loss = train_step(state, batch, subkey)
            epoch_loss += loss
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("Generating samples with the trained model...")
    rng_trained = jax.random.PRNGKey(123)
    trained_samples = sample(model, state.params, rng_trained, context_frames, num_steps=num_steps)
    trained_samples = np.array(trained_samples)
    trained_samples = (trained_samples + 1) / 2

    print("Visualizing samples before and after training...")
    visualize_samples(untrained_samples, trained_samples, num_samples=num_samples)

if __name__ == "__main__":
    main()
