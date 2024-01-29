import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Generator, self).__init__()

        # Define the architecture of the generator
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Using Tanh activation for normalized output in the range [-1, 1]
        )
        

    def forward(self, noise):
        # Forward pass logic
        generated_data = self.model(noise)
        return generated_data

# Example instantiation of the generator
# input_size: Size of the input noise vector
# output_size: Size of the generated output
# hidden_size: Size of the hidden layer
generator = Generator(input_size=100, output_size=784, hidden_size=256)

# Example forward pass with random noise
random_noise = torch.randn(1, 100)  # Example random noise vector
generated_output = generator(random_noise)
print("Generated Output Shape:", generated_output.shape)
