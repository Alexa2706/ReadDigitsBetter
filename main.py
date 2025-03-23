import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import load_dataset
from plot import plot_data
import pandas as pd
"""
    Hyperparameters:
"""
np.random.seed(0)
val_size = 5000
learning_rate = 1e-4
batch_size = 256
epochs = 20

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def eval(model, data):
    model.eval()
    x = torch.tensor(data.drop(columns = ['label']).values, dtype=torch.float32)
    y = torch.tensor(data['label'].values, dtype=torch.long)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    val_losses = []
    with torch.no_grad():
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print(f"Got {num_correct} / {num_samples} correct {num_correct / num_samples * 100:.2f}%")

def test(model, data):
    data = torch.tensor(data.values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, predictions = out.max(1)
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),  # 1-indexed IDs as per Kaggle format
        'Label': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission file created with {len(predictions)} predictions.")

def train(model, data, optimizer, epochs = 10, batch_size = 32):
    x = torch.tensor(data.drop(columns = ['label']).values, dtype=torch.float32)
    y = torch.tensor(data['label'].values, dtype=torch.long)
    for e in range(epochs):
        running_loss = 0.0
        batch_count = 0
        model.train()
        indx = np.random.permutation(len(x)) #random permutacija
        for start in range(0, len(x), batch_size):
            batch_indx = indx[start:start+batch_size]
            batch_x = x[batch_indx]
            batch_y = y[batch_indx]
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.cross_entropy(out, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        print(f'Epoch {e+1}/{epochs}, Loss: {running_loss/batch_count:.4f}')
def main():
    train_data, test_data = load_dataset()
    #split the data
    val_indices = np.random.choice(train_data.index, size=val_size, replace=False)
    val_data = train_data.loc[val_indices].copy()
    train_data = train_data.drop(val_indices)

    model = nn.Sequential( 
        Flatten(),
        nn.BatchNorm1d(784),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-4)
    train(model, train_data, optimizer, epochs, batch_size)
    eval(model, val_data)
    test(model, test_data)
    
if __name__ == '__main__':
    main()
    