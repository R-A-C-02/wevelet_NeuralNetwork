import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Dati: x ∈ [-2π, 2π], y = sin(x)
x = torch.linspace(-2*np.pi, 2*np.pi, 200).unsqueeze(1)
y = torch.sin(x)

# Rete lineare (senza ReLU)
model_linear = nn.Sequential(
    nn.Linear(1, 10),
    nn.Linear(10, 1)
)

# Rete con ReLU
model_relu = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Funzione per allenare un modello
def train(model, x, y, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return model

# Allena entrambe le reti
train(model_linear, x, y)
train(model_relu, x, y)

# Predizione
y_pred_linear = model_linear(x).detach().numpy()
y_pred_relu = model_relu(x).detach().numpy()

# Plot
plt.figure(figsize=(10,5))
plt.plot(x.numpy(), y.numpy(), label="y = sin(x)", color='black', linewidth=2)
plt.plot(x.numpy(), y_pred_linear, label="Solo Lineare", linestyle='--')
plt.plot(x.numpy(), y_pred_relu, label="Con ReLU", linestyle='-')
plt.legend()
plt.title("Confronto tra rete lineare e con ReLU")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
