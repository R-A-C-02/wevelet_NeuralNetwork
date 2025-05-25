#MODIFICA LE SEGUENTE COSA PROSSIMO GIRO:
# ❌ sistema tutti i file di spegaizone in dettaglio
# ❌ aggiungi un EARLY STOPPING \ limite massimo (1000 epoche)
#    sostituisci i valori della lista gigino conle wevelets


#CREA UN 3° DELLA RETE NEURALE 
#   aggiorna la rete deep learning con un secondo-terzo gruppo di neuroni
#   crea un modello che runni su tutta la serie storica 
#    quindi crea un ciclo for dove runni il modello per ogni osservazione delle onde 

#inserisci grafici:
# ❌  1- crea il grafico delle perfomance della rete neruale nella creazione della migliori combinazione di wevelets 
#   2-dove si vede l'andamento della combinazione dei wevelet pass rispetto alla wave madre (S)
#   2- fail il grafico 1 ma come una sommatoria (così da creare una fake-serie storica)
 
# Unifica i codici :
#         sia il main di alex che quelli di Ivan
#     Pubblica su github:
#         su github pubblica solo la rete neurale che hai creato, mentre tieni una repository privata con tutto il progetto fatto insieme

# Cose da fare prima dell'esposizione fuinale:
#   perchè il gumbell nois fa un doppio log ??
#   fai degli esempi pratici della softmax (tipo: le preferenze di ricerca dei libri che ti piacciono)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np


# Gumbel-Softmax utilities
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = sample_gumbel(logits.shape)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):  #è una temperatura rigida ma non troppo (forse)
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
a1 = 18.98
a2 = 20.02
a3 = 33.02
a4 = 34.02
a5 = 26.12
a6 = 55.45

gigino = torch.tensor([[a1, a2, a3, a4, a5, a6]])

# Definisci il valore S con un valore random
S_true = torch.tensor([[365.]])


model = GumbelSelectorWeighted()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

# Salva gli andamenti durante il training
losses = []
predictions = []

# Parametri
early_stop_tolerance = 0.025  # percentuale di tolleranza per un lato (sarà raddopiato)
max_epochs = 1000
best_loss = float('inf')
best_pred = None
best_indices = None
best_weights = None

# Training loop con early stopping
for epoch in range(max_epochs):
    optimizer.zero_grad()
    pred, indices, weights = model(gigino, temperature=0.5)
    loss = loss_fn(pred, S_true)
    loss.backward()
    optimizer.step()
    abs_loss = math.sqrt(loss.item())

    losses.append(loss.item())
    predictions.append(pred.item())

    # Aggiorna il miglior risultato trovato
    if abs_loss < best_loss:
        best_loss = abs_loss
        best_pred = pred.item()
        best_indices = indices.clone()
        best_weights = weights.clone()

    # # Stampa progressi
    # if epoch % 10 == 0 or epoch == max_epochs - 1:
    #     print(f"\nEpoch {epoch}, MSE: {loss.item():.4f}, Loss (absolute): {abs_loss:.4f}, Prediction: {pred.item():.2f}")
    #     print(f"Selected indices: {indices}, Selected weights: {weights}")

    # Condizione di early stopping (entro ± early_stop_tolerance% di S_true)
    target = S_true.item()
    lower_bound = target * (1 - early_stop_tolerance)
    upper_bound = target * (1 + early_stop_tolerance)
    if lower_bound <= pred.item() <= upper_bound:
        print(f"\n✅ Early stopping at epoch {epoch}: prediction {pred.item():.2f} within {early_stop_tolerance * 100 * 2:.2f}% of target {target}")
        best_pred = pred.item()
        best_indices = indices.clone()
        best_weights = weights.clone()
        break



############################################################################################################################



# Stampa i valori di tutti gli indici
print("\nValori degli indici:")
print("S:", S_true)
print("a1:", a1)
print("a2:", a2)
print("a3:", a3)
print("a4:", a4)
print("a5:", a5)
print("a6:", a6)

# Output finale
print("\n──────────── Risultato finale ────────────")
print(f"Best prediction: {best_pred:.2f}")
print(f"Best selected indices: {best_indices}")
print(f"Best selected weights: {best_weights.detach().numpy()}")

# Mostra i valori e contributi finali
selected_values = gigino[0, best_indices[0]]
weighted_values = selected_values * best_weights
print("Selected values:", selected_values.detach().numpy())
print("Weighted contribution:", weighted_values.detach().numpy())
print("Final sum:", weighted_values.sum().item())


# Prepara la figura con più subplot
fig, axs = plt.subplots(2, 2,  figsize=(14, 10))
fig.suptitle('Dashboard delle prestazioni del modello', fontsize=16)

# --- GRAFICO 1: Andamento della Loss ---
axs[0, 0].plot(losses, label='MSE Loss', color='blue')
axs[0, 0].set_title('Andamento della Loss')
axs[0, 0].set_xlabel('Epoche')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
axs[0, 0].grid(True)

# --- GRAFICO 2: Andamento della Prediction ---
axs[0, 1].plot(predictions, label='Prediction', color='green')
axs[0, 1].axhline(S_true.item(), color='red', linestyle='--', label='S_true')
axs[0, 1].set_title('Andamento della Prediction')
axs[0, 1].set_xlabel('Epoche')
axs[0, 1].set_ylabel('Prediction')
axs[0, 1].legend()
axs[0, 1].grid(True)

# --- GRAFICO 3: Logits e Probabilità finali ---
final_logits = model.logits.detach().numpy()
final_probs = F.softmax(model.logits, dim=0).detach().numpy()
indices_labels = [f'a{i+1}' for i in range(6)]

axs[1, 0].bar(indices_labels, final_logits, alpha=0.6, label='Logits')
axs[1, 0].bar(indices_labels, final_probs, alpha=0.6, label='Softmax Probabilities')
axs[1, 0].set_title('Logits e Probabilità finali')
axs[1, 0].set_ylabel('Valore')
axs[1, 0].legend()
axs[1, 0].grid(True)


# --- TABELLA RIASSUNTIVA ---
selected_indices = best_indices[0].tolist()
selected_values = gigino[0, best_indices[0]].detach().numpy()
selected_weights = best_weights.detach().numpy()
weighted_contributions = selected_values * selected_weights

axs[1, 1].axis('off')  # Spegni l’asse per fare spazio alla tabella
summary_text = (
    f"valori riassuntivi: \n"
    f"Early stopping at epoch {epoch}: prediction {pred.item():.2f} within {early_stop_tolerance * 100 * 2:.2f}% of target {target}"
    f"\n──────── Valori degli indici ──────── \n"
    f"S_true: {S_true.item()}\n"
    f"a1: {a1}\n"
    f"a2: {a2}\n"
    f"a3: {a3}\n"
    f"a4: {a4}\n"
    f"a5: {a5}\n"
    f"a6: {a6}\n\n"
    f"──────── Risultato finale ────────\n"
    f"Best prediction: {best_pred:.2f}\n"
    f"Best indices: {selected_indices}\n"
    f"Selected values: {np.round(selected_values, 2)}\n"
    f"Best weights: {np.round(selected_weights, 2)}\n"
    f"Weighted contrib.: {np.round(weighted_contributions, 2)}\n"
    f"Final sum: {weighted_contributions.sum():.2f}"
)
axs[1, 1].set_title('Tabella riassuntiva risultati')
axs[1, 1].text(0.5, 0.5, summary_text, fontsize=10, va='center', ha='center', wrap=True)


plt.tight_layout(rect=[0, 0, 1, 0.95])  # Lascia spazio al titolo generale
plt.show()