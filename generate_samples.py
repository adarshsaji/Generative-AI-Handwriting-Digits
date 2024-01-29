import torch
from torch import nn
from torchvision.utils import save_image
import os

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Using Tanh activation for normalized output in the range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

def create_generator(input_size, hidden_size, output_size):
    return Generator(input_size, hidden_size, output_size)


def generate_samples(generator, num_samples, output_dir='generated_images'):
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        random_noise = torch.randn(num_samples, 100)  # Adjust noise dimension if needed
        generated_samples = generator(random_noise)

    for i, sample in enumerate(generated_samples):
        save_image(sample.view(1, 28, 28), os.path.join(output_dir, f'generated_{i+1}.png'))

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Instantiate the generator
    generator = create_generator(input_size=100, hidden_size=256, output_size=784)

    # Load the trained generator model
    generator.load_state_dict(torch.load('generator.pth'))
    generator.eval()  # Set the generator to evaluation mode

    # Generate and save new samples
    num_samples = 10  # Adjust as needed
    generate_samples(generator, num_samples)
