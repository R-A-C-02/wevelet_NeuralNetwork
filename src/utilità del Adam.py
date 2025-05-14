import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Dati: y = sin(x)
x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1)
y = torch.sin(x)

# Modello base
def build_model():
    return nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

# Training generico
def train(model, optimizer, x, y, label):
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# Modelli e ottimizzatori
model_adam = build_model()
model_sgd = build_model()

optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)

# Allena entrambi
losses_adam = train(model_adam, optimizer_adam, x, y, "Adam")
losses_sgd = train(model_sgd, optimizer_sgd, x, y, "SGD")

# Confronta le curve di loss
plt.plot(losses_adam, label='Adam')
plt.plot(losses_sgd, label='SGD')
plt.title("Convergenza Adam vs SGD")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
