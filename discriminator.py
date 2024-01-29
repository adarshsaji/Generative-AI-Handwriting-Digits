import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        # Define the architecture of the discriminator
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),  # Using LeakyReLU for better gradient flow
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Using Sigmoid activation for binary classification
        )
        

    def forward(self, data):
        # Forward pass logic
        discriminator_output = self.model(data)
        return discriminator_output

# Example instantiation of the discriminator
# input_size: Size of the input data vector
# hidden_size: Size of the hidden layer
# output_size: Size of the output (binary classification)
discriminator = Discriminator(input_size=784, hidden_size=256, output_size=1)

# Example forward pass with generated data (from the generator)
generated_data = torch.randn(1, 784)  # Example generated data
discriminator_output = discriminator(generated_data)
print("Discriminator Output:", discriminator_output.item())
