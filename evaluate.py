import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def evaluate(model, X_test, y_test, scaler):
    #Model evaluation

    with torch.no_grad():
        model.eval()
        y_pred_test = model(X_test)
    
        # Inverse transform the predicted y to get original range results
        y_pred_test = scaler.inverse_transform(y_pred_test.numpy())
        y_test = scaler.inverse_transform(y_test.numpy())
    
        # Calculate evaluation metrics (e.g., mean squared error)
        mse = nn.MSELoss()(torch.tensor(y_pred_test), torch.tensor(y_test))
        #print("Mean Squared Error on Test Set:", mse.item())
        
        mse_value = mse.item()
        
        mse_output_directory = 'outputs/'
        mse_output_file_name = 'mse_value.txt'
        mse_output_path = mse_output_directory + mse_output_file_name
        with open(mse_output_path, 'w') as file:
            file.write('MSE value is ' + str(mse_value))
    
        # Create scatter plot to compare predicted vs. true values
        plt.scatter(y_pred_test, y_test, color='b', label='Predicted vs. True')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.title('Predicted vs. True Values')
        plt.legend()
        test_scatter_output_directory = 'outputs/'
        test_scatter_output_file_name = 'test_scatter_plot.png'
        test_scatter_output_path = test_scatter_output_directory + test_scatter_output_file_name
        plt.savefig(test_scatter_output_path)
        
        # Calculate residuals
        residuals = y_test - y_pred_test
    
        # Create a histogram of residuals
        plt.hist(residuals, bins=50, color='b', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        test_residual_output_directory = 'outputs/'
        test_residual_output_file_name = 'test_residual_plot.png'
        test_residual_output_path = test_residual_output_directory + test_residual_output_file_name
        plt.savefig(test_residual_output_path)