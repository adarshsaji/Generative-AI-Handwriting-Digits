# Generative AI Handwriting Digits Project

## Overview
This repository contains the code for a Conditional Generative Adversarial Network (GAN) designed to generate diverse and realistic handwritten digits conditioned on specific attributes such as style and thickness.

## Project Structure

- `generator.py`: Implementation of the generator network.
- `discriminator.py`: Implementation of the discriminator network.
- `train.py`: Training script for the Conditional GAN.
- `generate_samples.py`: Generate sample images to test GAN performance.
- `utils.py`: Utility functions for data loading and preprocessing.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch

### Installation
1. Clone the repository: `git clone https://github.com/adarshsaji/Generative-AI-Handwriting-Digits.git`
2. Navigate to the project directory: `cd Generative-AI-Handwriting-Digits`
3. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Modify the configuration in `config.py` to suit your requirements.
2. Run the training script: `python train.py`

## Results
After training, the model will be able to generate diverse and realistic handwritten digits based on specified attributes.

## Acknowledgments
- This project is inspired by the application of GANs in generative tasks.

## Contributing
Contributions are welcome! Feel free to submit pull requests or open issues.

## License
This project is licensed under the [MIT License](LICENSE).
