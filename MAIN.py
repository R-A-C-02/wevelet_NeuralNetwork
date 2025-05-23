'''MODIFICA LE SEGUENTE COSA PROSSIMO GIRO:
    aggiorna la rete deep learning conun secondo-terzo gruppo di neuroni
    crea un modello che runni su tutta la serie storica 
        quindi crea un ciclo for dove runni il modello per ogni osservazione delle onde 
    Insrisci i dati:
        crea un collegamento al file che contiene le wevelets
    inserisci grafici:
        1- dove si vede l'andamento della combinazione dei wevelet pass rispetto alla wave madre (S)
        2- fail il grafico 1 ma come una sommatoria (così da creare una fake-serie storica)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Gumbel-Softmax utilities
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = sample_gumbel(logits.shape)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    y_soft = gumbel_softmax_sample(logits, temperature)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft

# GumbelSelector con pesi
class GumbelSelectorWeighted(nn.Module):
    def __init__(self, input_size=6, k=3):
        super().__init__()
        self.k = k
        self.logits = nn.Parameter(torch.randn(input_size))  # per la selezione
        self.output_weights = nn.Parameter(torch.rand(k))    # pesi appresi

    def forward(self, x, temperature=0.5):
        # 1. Gumbel softmax sampling
        probs = gumbel_softmax(self.logits.unsqueeze(0), temperature=temperature, hard=False)
        _, topk_indices = torch.topk(probs, self.k, dim=1)

        # 2. Estrai solo i k input selezionati
        selected_inputs = x[:, topk_indices[0]]  # shape: (batch_size, k)

        # 3. Applica i pesi solo ai selezionati
        weighted_sum = (selected_inputs * self.output_weights).sum(dim=1, keepdim=True)

        return weighted_sum, topk_indices, self.output_weights
    



# Input
# Definisci i mini-indici con valori random
a1 = round(np.random.uniform(1, 100), 2)
a2 = round(np.random.uniform(1, 100), 2)
a3 = round(np.random.uniform(1, 100), 2)
a4 = round(np.random.uniform(1, 100), 2)
a5 = round(np.random.uniform(1, 100), 2)
a6 = round(np.random.uniform(1, 100), 2)
gigino = torch.tensor([[a1, a2, a3, a4, a5, a6]])

# Definisci il valore S con un valore random
S_true = torch.tensor([[365.]])


model = GumbelSelectorWeighted()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()


# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    pred, indices, weights = model(gigino, temperature=0.5)
    loss = loss_fn(pred, S_true)
    loss.backward()
    optimizer.step()
    abs_loss = math.sqrt(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE: {loss.item():.4f}, Loss (absolute): {abs_loss:.4f}, Prediction: {pred.item():.2f}")

# Output finale
final_pred, final_indices, final_weights = model(gigino, temperature=0.1)

# Stampa i valori di tutti gli indici
print("Valori degli indici:")
print("S:", S_true)
print("a1:", a1)
print("a2:", a2)
print("a3:", a3)
print("a4:", a4)
print("a5:", a5)
print("a6:", a6)

print("\nFinal prediction:", final_pred.item())
print("Selected indices:", final_indices.tolist())
print("Selected weights:", final_weights.detach().numpy())

# Mostra i 3 elementi scelti da gigino e i loro pesi
selected_values = gigino[0, final_indices[0]]
weighted_values = selected_values * final_weights
print("Selected values:", selected_values.detach().numpy())
print("Weighted contribution:", weighted_values.detach().numpy())
print("Final sum:", weighted_values.sum().item())
