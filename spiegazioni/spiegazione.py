######################################################################################################################################
######################################################################################################################################

# questo file è per scpiegare tutti i moduli e le funzioni usate nel progetto
# molto spesso alcune cose sono basilari e futili (o lameno sono surclasate da altrimoduli\funct) 
# ma per avere una visione piu compelta della costruzione di NN sono stati riportati in questo file

######################################################################################################################################
######################################################################################################################################






######################################################################################################################################
#1° versione della RN -  ERRATA 
######################################################################################################################################
import torch                    # libreria per ML e DL
import torch.nn as nn           # moduli per la rete neurale
import torch.optim as optim     # moduli per l'ottimizzazione


# questo modello per un istante t del segnale madre (fai finta che il segnale sia lungo 2048 osservazioni, noi stiamo 
# calcolando la combianzione di wevvelet-pass per t=1  ) quindi sarà da strutturare poi un modello LM ancor apiù grande 
# per tutto il segnale. Per questo sto facnedo qauesta NN che da come oput put solo 3 wevelet pass e non 11 o 6, così da 
# rendre a livvello computazionale fattibile il modello più grande (che è effettiavmente quello che useremo in pratica) 
# se non mi sono spiegato bene su che cosa fa questo NN guarda il seguente documento a pagine 4\15 e credo possa esser 
# più comprensibile :  https://www.docenti.unina.it/webdocenti-be/allegati/materiale-didattico/89098



class WaveletWeight(nn.Module): # Rete neurale con 6 input e 3 output, ogni input è un livello di wevelts
    def __init__(self):         # costruzione della rete neurale con 3 layer e 4 neuroni (1 input, 2 hidden, 1 output)
        super(WaveletWeight, self).__init__()   # fc= fully connected 
        self.fc1 = nn.Linear(6, 64)     # 6 input = 64 neuroni (disolito si inizia con questa quantità, è modificabile)
        self.fc2 = nn.Linear(64, 32)    # i 32 gli ho scleti io, convenzione di scrittura
        self.fc3 = nn.Linear(32, 3)     # 3 output = 3 livelli di wevelet ponderati in modo diverso
        self.relu = nn.ReLU()           # funzione di attivazione - unità linerare rettificata (per modellare relazioni complesse), MAGGIORI INFO A PIE' DI PAGINA'1
     

    def forward(self, x):           # passaggio dei dati attraverso la rete
        x = self.relu(self.fc1(x))  #passaggio dati anche per il non lineare "relu"
        x = self.relu(self.fc2(x))
        out = self.fc3(x)           #lo stratto finale va senza attivvazine (relu)
        return out                  # 3 valori output MAGGIORI INFO A PIE' DI PAGINA'2


# ESEMPIO RANDOM
# Esempio di input: batch\lista di segnali in cui ogni riga è un'osservazione: [A1, A4, A5, A8, A11, ...] [[A1, A4, A5, A8, A11, ...] [A1, A4, A5, A8, A11, ...] [A1, A4, A5, A8, A11, ...] 
batch_size = 32
X_dummy = torch.rand(batch_size, 6) # numeri casualoitra 0 e 1

y_dummy = torch.rand(batch_size, 3) # Target dummy = target casuale


# Istanziamento del modello
model = WaveletWeight()
criterion = nn.MSELoss()            # è la funzione di costo (quanto lontane sono le sue previsioni dal target)
optimizer = optim.Adam(model.parameters(), lr=0.001)    # ottimizzatore MAGGIORI INFO A PIE' DI PAGINA'3

# Training loop (semplificato)
for epoch in range(100):            # è come un ciclo for che va da 0 a 100 (epoche indica i valori 0,1,2,...99) MAGGIORI INFO A PIE' DI PAGINA'4
    optimizer.zero_grad()           # azzerra i gradienti accumulati dal passaggio precedente  così da poter ripetere la stessa operazione, sono salvati di default 
    outputs = model(X_dummy)        # calcola il modello
    loss = criterion(outputs, y_dummy)  #calcola il loss 
    loss.backward()                 # calcola i gradienti (per ottenere il miglior modello)
    optimizer.step()                # aggiorna i parametri del modello
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")    #stampa la perdita ogni 10 epoche per monitorare l'andamento 


    
#OUT DA ESEMPIO DEL MODELLO: 
# Epoch 0, Loss: 0.4643
# Epoch 10, Loss: 0.3250
# Epoch 20, Loss: 0.2017
# Epoch 30, Loss: 0.1140
# Epoch 40, Loss: 0.0860
# Epoch 50, Loss: 0.0848
# Epoch 60, Loss: 0.0802
# Epoch 70, Loss: 0.0766
# Epoch 80, Loss: 0.0739
# Epoch 90, Loss: 0.0710




'''
1'Le funzioni di attivazione (come ReLU) sono fondamentali nelle reti neurali
perché introducono non linearità tra i layer. Senza di esse, la rete sarebbe
una semplice combinazione di trasformazioni lineari, incapace di modellare
relazioni complesse tra variabili (es. curve, soglie, saturazioni)
ReLU (Rectified Linear Unit) è definita come:
    ReLU(x) = x se x ≥ 0, altrimenti 0
Guarda il file "utilità del ReLU().py"
'''

'''
2'Come 3 out possiamo mettere la combinazione ideale di wevvelts (pondeate) che si avinano il più possibile ad S
L'idea è di portare 11 wevelets-pass (A1, A2,A3,....A11) a 3 output, quindi 3 wevelets-pass che combinati ci diano un 
+90% di rappresentazione del S,  così da avvere un modello più leggero ma allos tesso pratico (e fare un pò gli sborroni 
alla giuria dell'enel ma soprattutto rendere i calcoli per i prossimi modelli di ML più fattibili )
'''

'''
3'OTTIMIZZATORE: Adam (Adaptive Moment Estimation) Prende i pesi appena creati e li modfica leggermente per minimizzare il loss
' model.parameters()' passa tutti i parametri  del modello all'ottimizzatore 
! lr= 0.001' è il learning rate, ovvero la velocità con cui l'ottimizzatore cerca di minimizzare il loss
Guarda il file "utilità del ADAM.py"
'''

'''
4'Il commadno Epoche serve perchè i NN non imaprano subito e quindi devi rifarli fare lo stesso calcolo più volt (in 
questo vcaso100 vvolte, ti ricordo che questo è solo per l'istante t=x) così da far diminuire la loss
'''






######################################################################################################################################
#2° versione della RN
######################################################################################################################################

"""
─────────────────────────────────────────────────────────────────────────────
📘 SPIEGAZIONE DEL MODELLO GUMBELSELECTORWEIGHTED
─────────────────────────────────────────────────────────────────────────────

✅ COSA FA QUESTO CODICE?
────────────────────────────
Il modello `GumbelSelectorWeighted` è un blocco di selezione e ponderazione:
- Prende 6 input numerici (es. a1, a2, a3, a4, a5, a6).
- Sceglie automaticamente i 3 input più utili tramite Gumbel-Softmax.
- Applica loro dei pesi allenabili.
- Somma i contributi pesati per stimare un valore finale S.

Anche se non ha strati di neuroni o layer nascosti (come una classica RN), è comunque un modello
allenabile: ottimizza i parametri (logits e pesi) usando backpropagation.

────────────────────────────
⚙️ COMPONENTI PRINCIPALI:
────────────────────────────
- logits (parametri): determinano quali input scegliere.
- Gumbel-Softmax: introduce una scelta probabilistica, ma differenziabile.
- output_weights: pesi specifici per i 3 input selezionati.
- Somma finale: combina gli input pesati per fare la stima.
- MSE loss + Adam optimizer: aggiornano logits e pesi per ridurre l’errore.

                        Molto bella come spiegazione- da isnerire nel README
                        ────────────────────────────
                        🔍 PERCHÉ SEMBRA QUASI CASUALE?
                        ────────────────────────────
                        La selezione degli input usa Gumbel-Softmax, che aggiunge un elemento di
                        casualità (per esplorare). Tuttavia, questa casualità è controllata:
                        il modello impara progressivamente a preferire gli input più utili
                        grazie all’ottimizzazione.

                        Non è un semplice brute-force, ma un processo guidato
                        da ottimizzazione differenziabile.

                        ────────────────────────────
                        🚀 È UNA RETE NEURALE?
                        ────────────────────────────
                        - Non è una rete neurale classica: non ha strati di neuroni e attivazioni.
                        - Ma è un modello allenabile con PyTorch, con parametri ottimizzati
                        tramite gradient descent, come avviene nelle reti neurali.

────────────────────────────
🧠 FRASE CHIAVE DA RICORDARE
────────────────────────────
Questo modello non usa neuroni, ma è un sistema differenziabile
che impara selezione e ponderazione, ottimizzando direttamente
i parametri con backpropagation.
"""


import torch                            # Libreria per i tensori e le reti neurali
import torch.nn as nn                   # Modulo per creare i modelli neurali
import torch.nn.functional as F         # Funzioni matematiche avanzate come softmax
import math                             # Libreria per funzioni matematiche di base
import numpy as np                      # Libreria per gestire array e numeri casuali



# Gumbel-Softmax utilities, SPIEGAZIONE A PIE' DI PAGINA'1
def sample_gumbel(shape, eps=1e-20):                    #Funzione per generare rumore Gumbel
    U = torch.rand(shape)                               # Numeri casuali uniformi tra 0 e 1
    return -torch.log(-torch.log(U + eps) + eps)        # Trasforma questi numeri per ottenere rumore "Gumbel"



def gumbel_softmax_sample(logits, temperature):         # Anche questa funzione è presente a PIE' DI PAGINA'1
    gumbel_noise = sample_gumbel(logits.shape)          # Genera rumore casuale con forma uguale ai logits
    y = logits + gumbel_noise                           # Crea punteggi disturbati per fare scelte esplorative
    return F.softmax(y / temperature, dim=-1)           # Trasforma i punteggi in probabilità (addestrabili)


# Gumbel-Softmax, ON\OFF hard , SPIEGAZIONE A PIE' DI PAGINA'2
def gumbel_softmax(logits, temperature=1.0, hard=False):
    y_soft = gumbel_softmax_sample(logits, temperature)             # Risultato: y_soft è un vettore di probabilità morbide.[0.84, 0.11, 0.05] 
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]                 # Trova l'indice con probabilità massima
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)  # Crea un vettore con 1 solo in quella posizione
        return (y_hard - y_soft).detach() + y_soft                  # Torna un vettore “duro” ma con gradiente
    else:
        return y_soft                                               # Torna un vettore “morbido”



# GumbelSelector con pesi, SPIEGAZIONE A PIE' DI PAGINA'3 
class GumbelSelectorWeighted(nn.Module):
    def __init__(self, input_size=6, k=3):                   # quanti input inserire e quanti output ottenere
        super().__init__()                                   # serve per iniziallizare corretamente la classe
        self.k = k                                           # quanti input selezionare
        self.logits = nn.Parameter(torch.randn(input_size))  # per la selezione
        self.output_weights = nn.Parameter(torch.rand(k))    # pesi appresi



    def forward(self, x, temperature=0.5):
        # 1. Calcola probabilità con Gumbel-softmax
        probs = gumbel_softmax(self.logits.unsqueeze(0), temperature=temperature, hard=False)
        _, topk_indices = torch.topk(probs, self.k, dim=1)

        # 2. Estrai solo i k input selezionati
        selected_inputs = x[:, topk_indices[0]]  

        # 3. Moltiplica input scelti per pesi appres
        weighted_sum = (selected_inputs * self.output_weights).sum(dim=1, keepdim=True)

        return weighted_sum, topk_indices, self.output_weights



# Input
# Definisci i mini-indici con valori random
a1 = 18.98
a2 = 20.02
a3 = 33.02
a4 = 34.02
a5 = 26.12
a6 = 55.45
# Creazione della lista di tensori di input, SPIEGAZIONE A PIE' DI PAGINA'X
gigino = torch.tensor([[a1, a2, a3, a4, a5, a6]])

# Definisci il valore S con un valore random
S_true = torch.tensor([[110.]])


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


# Creazione del tensore di input, SPIEGAZIONE A PIE' DI PAGINA'1
X = torch.tensor([A1, A2, A3, A4, A5, A6])

# Inserimento del valore di S (valore a tempo T della onda madre)
S = float(input("Inserisci il valore di S: "))

# Creazione del tensore di output desiderato,  SPIEGAZIONE A PIE' DI PAGINA'2
y_desiderato = torch.tensor([S, 0, 0])






'''
1' Immagina di aggiungere rumore (come disturbo casuale) a una decisione\scelta\choice per rendere le scelte meno prevedibili. 
Il Gumbel noise è un tipo di rumore casuale usato per simulare scelte discrete (tipo "scegli il più grande") in modo probabilistico. 
Spiegazione dettagvliata su softmax_vs_gumbel_softmax.ipynb
'''



'''
2. obiettivo della funzione:
Simulare una scelta discreta (tipo argmax) ma in modo continuo e differenziabile.
    Usi la Gumbel-Softmax per fare una scelta probabilistica e addestrabile.
    Se hard=True, restituisce un one-hot vector (cioè una scelta netta).
    Ma lo fa in modo differenziabile, quindi può essere usato nel training di una rete.
Spiegazione dettalgiata su ONorOFF_hard_choice.ipynb
'''




'''
3. Questo codice definisce una classe PyTorch chiamata GumbelSelectorWeighted, che serve a selezionare automaticamente un 
sottoinsieme di input (es. 3 su 6) e assegnare loro dei pesi appresi per stimare una quantità come una somma ponderata.

Quando diciamo “classe PyTorch”, intendiamo una classe Python che estende torch.nn.Module, cioè che fa parte della 
struttura con cui PyTorch costruisce modelli neurali personalizzati.

Spiegazione riga per riga di qeusta classe la trovi in GumbelSelectorWeighted.py
'''




'''
X' Immagina di avere 6 scatole, una per ogni input (A1, A2, A3, A4, A5, A6). Ogni scatola contiene un numero.
Il codice X = torch.tensor([A1, A2, A3, A4, A5, A6]) crea un mega contenitore  chiamato "tensore" (X) che tiene dentro
tutte e 6 le scatole con i numeri. Poi, il computer può usare il contenitore (X) per fare calcoli e operazioni con 
i numeri dentro.
In questo caso, il contenitore (X) è utilizzato come input per la rete neurale, che significa che la rete neurale 
userà i numeri dentro il contenitore (X) per fare calcoli e produrre un output.
'''


######################################################################################################################################
#3° versione della RN
######################################################################################################################################


