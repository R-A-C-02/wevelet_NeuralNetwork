import torch                    # libreria per ML e DL
import torch.nn as nn           # moduli per la rete neurale
import torch.optim as optim     # moduli per l'ottimizzazione


# questo modello per un istante t del segnale madre (fai finta che il segnale sia lungo 2048 osservazioni, noi stiamo calcolando la combianzione di wevvelet-pass per t=1  ) quindi sarà da strutturare poi un modello LM ancor apiù grande per tutto il segnale. Per questo sto facnedo qauesta NN che da come oput put solo 3 wevelet pass e non 11 o 6, così da rendre a livvello computazionale\di calcolo fattibile il modello più grande (che è effettiavmente     quello che useremo in pratica) se non mi sono spiegato bene su che cosa fa questo NN guarda il seguente documento a pagine 4\15 e credo possa esser più comprensibile :  https://www.docenti.unina.it/webdocenti-be/allegati/materiale-didattico/89098



class WaveletWeight(nn.Module): # Rete neurale con 6 input e 3 output, ogni input è un livello di wevelts
    def __init__(self):         # costruzione della rete neurale con 3 layer e 4 neuroni (1 input, 2 hidden, 1 output)
        super(WaveletWeight, self).__init__()   # fc= fully connected 
        self.fc1 = nn.Linear(6, 64)     # 6 input = 64 neuroni (disolito si inizia con questa quantità, è modificabile)
        self.fc2 = nn.Linear(64, 32)    # i 32 gli ho scleti io, convenzione di scrittura
        self.fc3 = nn.Linear(32, 3)     # 3 output = 3 livelli di wevelet ponderati in modo diverso
        self.relu = nn.ReLU()           # funzione di attivazione - unità linerare rettificata (per modellare relazioni complesse), MAGGIORI INFO A PIE' DI PAGINA
     

    def forward(self, x):           # passaggio dei dati attraverso la rete
        x = self.relu(self.fc1(x))  #passaggio dati anche per il non lineare "relu"
        x = self.relu(self.fc2(x))
        out = self.fc3(x)           #lo stratto finale va senza attivvazine (relu)
        return out                  # 3 valori output MAGGIORI INFO A PIE' DI PAGINA


# ESEMPIO RANDOM
# Esempio di input: batch\lista di segnali in cui ogni riga è un'osservazione: [A1, A4, A5, A8, A11, ...] [[A1, A4, A5, A8, A11, ...] [A1, A4, A5, A8, A11, ...] [A1, A4, A5, A8, A11, ...] 
batch_size = 32
X_dummy = torch.rand(batch_size, 6) # numeri casualoitra 0 e 1

y_dummy = torch.rand(batch_size, 3) # Target dummy = target casuale


# Istanziamento del modello
model = WaveletWeight()
criterion = nn.MSELoss()            # è la funzione di costo (quanto lontane sono le sue previsioni dal target)
optimizer = optim.Adam(model.parameters(), lr=0.001)    # ottimizzatore MAGGIORI INFO A PIE' DI PAGINA

# Training loop (semplificato)
for epoch in range(100):            # è come un ciclo for che va da 0 a 100 (epoche indica i valori 0,1,2,...99) MAGGIORI INFO A PIE' DI PAGINA
    optimizer.zero_grad()
    outputs = model(X_dummy)
    loss = criterion(outputs, y_dummy)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")




'''
Le funzioni di attivazione (come ReLU) sono fondamentali nelle reti neurali
perché introducono non linearità tra i layer. Senza di esse, la rete sarebbe
una semplice combinazione di trasformazioni lineari, incapace di modellare
relazioni complesse tra variabili (es. curve, soglie, saturazioni)
ReLU (Rectified Linear Unit) è definita come:
    ReLU(x) = x se x ≥ 0, altrimenti 0
Guarda il file "utilità del ReLU().py"
'''

'''
Come 3 out possiamo mettere la combinazione ideale di wevvelts (pondeate) che si avinano il più possibile ad S
L'idea è di portare 11 wevelets-pass (A1, A2,A3,....A11) a 3 output, quindi 3 wevelets-pass che combinati ci diano un +90% di rappresentazione del S,  così da avvere un modello più leggero ma allos tesso pratico (e fare un pò gli sborroni alla giuria dell'enel ma soprattutto rendere i calcoli per i prossimi modelli di ML più fattibili )
'''

'''
OTTIMIZZATORE: Adam (Adaptive Moment Estimation) Prende i pesi appena creati e li modfica leggermente per minimizzare il loss
' model.parameters()' passa tutti i parametri  del modello all'ottimizzatore 
! lr= 0.001' è il learning rate, ovvero la velocità con cui l'ottimizzatore cerca di minimizzare il loss
Guarda il file "utilità del ADAM.py"
'''

'''
Il commadno Epoche serve perchè i NN non imaprano subito e quindi devi rifarli fare lo stesso calcolo più volt (in questo vcaso100 vvolte, ti ricordo che questo è solo per l'istante t=x)
'''