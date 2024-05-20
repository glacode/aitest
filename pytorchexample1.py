import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 2 input features, 64 output features
        self.fc2 = nn.Linear(64, 1)  # 64 input features, 1 output feature

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = 3*X[:,0] - 2*X[:,1] + 1 + 0.1*np.random.randn(100)  # y = 3*x1 - 2*x2 + 1 + noise

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y to match model output shape

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the trained model for inference

X_new = np.random.rand(10, 2)  # 10 new samples, 2 features

# Convert new data to PyTorch tensor
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

# Put the model in evaluation mode
model.eval()

# Perform inference
with torch.no_grad():  # No need to compute gradients during inference
    predictions = model(X_new_tensor)

# Convert predictions tensor to numpy array
predictions_np = predictions.numpy()

# Compute actual values using the formula
y_actual = 3*X_new[:,0] - 2*X_new[:,1] + 1

# Display predictions and actual values
print("Predictions:")
print(predictions_np.squeeze())
print("Actual Values (Computed using the Formula):")
print(y_actual)
