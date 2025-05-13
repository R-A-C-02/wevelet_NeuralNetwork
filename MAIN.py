import torch                    # libreria per ML e DL
import torch.nn as nn           # moduli per la rete neurale
import torch.optim as optim     # moduli per l'ottimizzazione

class WaveletWeight(nn.Module): # Rete neurale con 6 input e 3 output, ogni input è un livello di wevelts
    def __init__(self):         # costruzione della rete neurale con 3 layer e 4 neuroni (1 input, 2 hidden, 1 output)
        super(WaveletWeight, self).__init__()   # fc= fully connected 
        self.fc1 = nn.Linear(6, 64)     # 6 input = 64 neuroni (disolito si inizia con questa quantità, è modificabile)
        self.fc2 = nn.Linear(64, 32)    # i 32 gli ho scleti io, convenzione di scrittura
        self.fc3 = nn.Linear(32, 3)     # 3 output = 3 mix  di livelli di wevelet ponderati in modo diverso
        self.relu = nn.ReLU()           # funzione di attivazione - unità linerare rettificata (per modellare relazioni complesse)
                                        # le RN (Rete Neurale) usano modelli linerari ma bisogna "rompere" questa linearità altrimenti divanta una semplice somma di mod lineari e quindi, PER ALTRE INFO VEDI FONDO PAGINA        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)        # Output raw, o eventualmente usa softmax per normalizzazione
        return out

# Esempio di input: batch di segnali filtrati (shape: batch_size x 6)
# Ogni riga è un'osservazione: [A1, A4, A5, A8, A11, ...]
batch_size = 32
X_dummy = torch.rand(batch_size, 6)

# Target dummy (può essere ad esempio il valore vero del segnale ricostruito o un obiettivo derivato)
# In questo esempio generico, usiamo un target casuale
y_dummy = torch.rand(batch_size, 3)

# Istanziamento del modello
model = WaveletWeight()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (semplificato)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_dummy)
    loss = criterion(outputs, y_dummy)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")




'''
la classe ReLU() prende in 
'''