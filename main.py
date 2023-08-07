from models.linreg import LinearRegressionModel
from data_preprocessing import data_load, data_preprocessing
from training import training
from models.linreg import LinearRegressionModel
from evaluate import evaluate

data_path = 'data/insurance.csv'

if __name__ == "__main__":
    # Run the data load step
    print("Running data load...")
    df = data_load(raw_data_path = data_path)
    print("Data load completed.")

    # Run the data preprocessing step
    print("Running data preprocessing...")
    X_train, X_test, y_train, y_test, scaler, input_size = data_preprocessing(df = df)
    print("Data preprocessing completed.")
    
    # Run the model training step
    print("Running model training...")
    model = training(LinearRegressionModel, X_train, y_train, input_size, num_epochs = 1000)
    print("Model training completed.")

    # Run the evaluation and print the metrics
    print("Running model evaluation...")
    evaluate(model, X_test, y_test, scaler)
    print("Model evaluation completed.")