import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from utils import load_data, preprocess_data
from config import Config
# Main training script


# Hyperparameters
input_size = Config.INPUT_SIZE
hidden_size = Config.HIDDEN_SIZE
output_size = Config.OUTPUT_SIZE
batch_size = Config.BATCH_SIZE
learning_rate = Config.LEARNING_RATE
num_epochs = Config.NUM_EPOCHS


generator = Generator(input_size, output_size, hidden_size)
discriminator = Discriminator(output_size, hidden_size, 1)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Data loading
train_loader = load_data(batch_size)

# Training loop
#num_epochs = 50

for epoch in range(num_epochs):
    for real_data, _ in train_loader:
        # Real data
        real_data = preprocess_data(real_data)
        real_labels = torch.ones(batch_size, 1)

        # Fake data
        noise = torch.randn(batch_size, input_size)
        generated_data = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)

        # Discriminator training
        optimizer_d.zero_grad()
        real_output = discriminator(real_data)
        fake_output = discriminator(generated_data.detach())
        loss_real = criterion(real_output, real_labels)
        loss_fake = criterion(fake_output, fake_labels)
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Generator training
        optimizer_g.zero_grad()
        fake_output = discriminator(generated_data)
        loss_g = criterion(fake_output, real_labels)
        loss_g.backward()
        optimizer_g.step()

    # Print loss for every epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {loss_g.item():.4f}, Discriminator Loss: {loss_d.item():.4f}")

# Save the trained models

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
