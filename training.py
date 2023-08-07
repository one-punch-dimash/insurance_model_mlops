import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def training(linreg, X_train, y_train, input_size, num_epochs = 10000):
    
    model = linreg(input_size = input_size)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.MSELoss()

    train_losses = []

    for i in range(num_epochs):
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
    
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Save the training loss for plotting
        train_losses.append(loss.item())
    
    # Plot the training loss
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Time')
    
    output_directory = 'outputs/'
    output_file_name = 'training_loss_plot.png'
    output_path = output_directory + output_file_name
    plt.savefig(output_path)
    
    return model