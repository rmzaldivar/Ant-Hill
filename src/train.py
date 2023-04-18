import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from ant_hill import AntHill
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.optim import LBFGS

def train_ant_hill(ant_hill, train_data, epochs, batch_size, device):
    ant_hill.to(device)
    ant_hill.train()
    
    # Use DataLoader for batching input data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize likelihood and optimizer
    likelihood = ant_hill.likelihood.to(device)
    mll = ExactMarginalLogLikelihood(likelihood, ant_hill)
    optimizer = LBFGS(ant_hill.parameters(), lr=1e-1)
    
    # Automatic Mixed Precision
    scaler = GradScaler()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            def closure():
                optimizer.zero_grad()
                with autocast():
                    output = ant_hill(x)
                    loss = -mll(output, y)
                scaler.scale(loss).backward()
                return loss

            loss = optimizer.step(closure)
            epoch_loss += loss.item()

        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

def train_ant_hill_online(model, data_loader, criterion, optimizer, scaler, device):
    model.train()

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Train with autocast and gradient scaling
        with autocast():
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()

        # Backpropagation with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
