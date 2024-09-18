"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from AIcandy_LSTM_model_hdbhkibl import PricePredictionModel, create_sequences
import warnings
import os

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


# Command:
# python AIcandy_LSTM_train_emabpupv.py --data_file history_price.csv --num_epochs 100 --batch_size 32 --checkpoint_path aicandy_lstm_checkpoint_xpisdedn.pth


def main(data_file, num_epochs, batch_size, checkpoint_path):
    # Hyperparameters
    SEQUENCE_LENGTH = 20
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    BATCH_SIZE = batch_size
    LEARNING_RATE = 0.001
    NUM_EPOCHS = num_epochs
    PATIENCE = 20

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    df = pd.read_csv(data_file)
    print(df.head(10))  
    prices = df['CloseFixed'].values.reshape(-1, 1)
    plt.figure(figsize=(10, 6))  
    plt.plot(prices, label='Close prices')  
    plt.title('AIcandy.vn - Close prices over time')  
    plt.xlabel('Time')  
    plt.ylabel('Price')  
    plt.legend()  
    plt.grid(True)  
    plt.savefig('price_actual.png') 
    plt.close()
    
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices).flatten()
    print(prices_scaled[:10])

    # Create sequences
    sequences, targets = create_sequences(prices_scaled, SEQUENCE_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)
    print(X_train[0])

    # Convert to PyTorch tensors and move to device
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model, loss function, and optimizer
    model = PricePredictionModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        best_val_loss = float('inf')

    # Early stopping
    patience_counter = 0

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(-1))
            loss = criterion(outputs, batch_y.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x.unsqueeze(-1))
                loss = criterion(outputs, batch_y.unsqueeze(-1))
                val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print("Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    # Load best model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(sequences).unsqueeze(-1).to(device)).cpu().squeeze().numpy()

    # Inverse transform predictions and actual prices
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(targets.reshape(-1, 1))

    # Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Price')
    plt.plot(predictions_rescaled, label='Predicted Price')
    plt.title('AIcandy.vn - Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('price_actual_predict.png')
    plt.close()

    # Predict next day's price
    last_sequence = torch.FloatTensor(sequences[-1]).unsqueeze(0).unsqueeze(-1).to(device)
    next_day_prediction = model(last_sequence).cpu().item()
    next_day_price = scaler.inverse_transform([[next_day_prediction]])[0][0]

    print(f"Predicted price for the next day: {next_day_price:.2f}")



if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--data_file', type=str, required=True, help='Path to file data')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint_path', type=str, default='aicandy_lstm_checkpoint_xpisdedn.pth', help='Path to save the checkpoint')


    args = parser.parse_args()
    main(args.data_file, args.num_epochs, args.batch_size, args.checkpoint_path)