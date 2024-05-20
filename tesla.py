import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np

# Download historical data for Tesla's stock price
ticker = 'TSLA'
data = yf.download(ticker, start="2010-01-01", end="2024-01-01")

# Preprocess the data
data['Close'] = data['Close'].fillna(method='ffill')  # Fill any missing values with the last available value
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Convert data to NumPy array
scaled_data = np.array(scaled_data)

# Create sequences of windowed data
window_size = 3
X = np.array([scaled_data[i:i+window_size] for i in range(len(scaled_data) - window_size)])
y = scaled_data[window_size:]

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define a simple neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
input_size = window_size
hidden_size = 64
output_size = 1
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into training and validation sets (last two years for validation)
train_size = int(0.8 * len(X))
train_X, valid_X = X[:train_size], X[train_size:]
train_y, valid_y = y[:train_size], y[train_size:]

# Convert train_X and valid_X to PyTorch tensors
train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
valid_X_tensor = torch.tensor(valid_X, dtype=torch.float32)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_X_tensor)
    loss = criterion(outputs, torch.tensor(train_y, dtype=torch.float32))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Validate the model
model.eval()
with torch.no_grad():
    valid_outputs = model(valid_X_tensor)
    valid_loss = criterion(valid_outputs, torch.tensor(valid_y, dtype=torch.float32))
    print(f'Validation Loss: {valid_loss.item():.4f}')
