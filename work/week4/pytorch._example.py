'''
pytorch._example.py
This code implements a simple PyTorch model for regression tasks,
using a dataset of input-output pairs. The model consists of a single hidden layer with ReLU activation, and it is trained using stochastic gradient descent. The code includes a Trainer class for managing the training process, including data loading, loss calculation, and model prediction.
It also includes a simple dataset class and a main block to run the training and prediction.

'''

import torch # is the main PyTorch library for tensor operations and deep learning; 
import torch.nn as nn   # is the neural network module in PyTorch, providing classes for building neural networks
from torch.utils.data import Dataset, DataLoader # is used to create custom datasets and data loaders for batching and shuffling data during training


class SimpleDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"]
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=config["learning_rate"]
        )
        self.dataset = None
        self.train_loader = None
        
    def _setup(self):
        self.model.to(self.device)
        
    def load_data(self):
        x_data = [[(i - 50) / 50.0] for i in range(100)]
        y_data = [[3 * x[0] + 2] for x in x_data]
        
        self.dataset = SimpleDataset(x_data, y_data)
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=self.config["batch_size"],
            shuffle=True
        )
        
    def calc_loss_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        return self.loss_fn(outputs, targets)
    
    def train_model(self, epochs=None):
        self.model.to(self.device)

        if epochs is None:
            epochs = self.config["epochs"]
            
        i = 0
        for epoch in range(epochs):
            self.model.train()
            
            for inputs, targets in self.train_loader:
                i+=1
                loss = self.calc_loss_batch(inputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        print(i)
    
    def predict(self, input_value):
        self.model.eval()
        test_input = torch.tensor([[float(input_value)]]).to(self.device)
        with torch.no_grad():
            prediction = self.model(test_input)
        return prediction.item()


if __name__ == "__main__":
    config = {
        "input_size": 1,
        "hidden_size": 10,
        "output_size": 1,
        "batch_size": 20,
        "learning_rate": 0.01,
        "epochs": 500
    }
    
    trainer = Trainer(config)
    trainer._setup()
    trainer.load_data()
    trainer.train_model()
    
    test_value = 0.2
    prediction = trainer.predict(test_value)
    print(f"Model prediction for input {test_value}: {prediction:.4f}")
    print(f"Expected output (3*{test_value}+2): {3*test_value + 2:.4f}")
    import matplotlib.pyplot as plt

    x_vals = torch.linspace(-1, 1, 100).view(-1, 1)
    x_vals_np = x_vals.numpy()
    y_true = 3 * x_vals_np + 2

    trainer.model.eval()
    with torch.no_grad():
        y_pred = trainer.model(x_vals.to(trainer.device)).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals_np, y_true, label='True Values (y = 3x + 2)', alpha=0.5)
    plt.scatter(x_vals_np, y_pred, label='Model Predictions', alpha=0.5)
    plt.legend()
    plt.title('Model Predictions vs True Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()