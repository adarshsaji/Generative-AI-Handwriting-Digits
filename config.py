class Config:
    # Model parameters
    INPUT_SIZE = 100  # Size of the input noise vector for the generator
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 784  # Size of the generated output (flattened image for MNIST)

    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0002
    NUM_EPOCHS = 100

  
