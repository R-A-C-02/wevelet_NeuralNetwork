
"""
─────────────────────────────────────────────────────────────────────────────
 GUMBEL-SOFTMAX: COS'È E COME FUNZIONA
─────────────────────────────────────────────────────────────────────────────

La Gumbel-Softmax è una tecnica per simulare una selezione discreta (come argmax:
    argmax è una funzione che ti dice:  "Qual è la posizione del valore più grande in una lista?")
in modo continuo e differenziabile. Serve quando vogliamo selezionare un sottoinsieme
di input (es. 3 su 6) ma vogliamo comunque poter allenare la rete con backpropagation.

Funziona aggiungendo "rumore Gumbel" a ogni logit (punteggio) e poi applicando
la softmax:

    g_i = -log(-log(U_i))         # rumore Gumbel
    y_i = softmax((logits + g_i) / temperature)

Il rumore Gumbel segue una distribuzione Gumbell:
- È una distribuzione di probabilità asimmetrica.
- È usata per modellare i valori estremi (massimi o minimi).
- Perfetta per scenari in cui vogliamo campionare il massimo (come argmax).
Utile per modellare valori estremi (massimi). In pratica, questo rumore rende ogni scelta
"casualmente disturbata", ma sempre guidata dai punteggi originali.

EFFETTO DELLA TEMPERATURA (tau):
────────────────────────────────
- Alta tau (es. 5.0): scelte morbide → simili a una media
- Bassa tau (es. 0.1): scelte più nette → simili a argmax

Questo permette di iniziare con scelte esplorative (tau alta)
e finire con scelte decise (tau bassa), simulando un "raffreddamento".

Per esempio:
    logits = [3.0, 1.5, 0.5]
    gumbel_noise = [0.1, 2.3, 1.0]
    noisy_logits = [3.1, 3.8, 1.5]
    softmax → output ≈  1 [3.8] perchè sommando il valore logit con il noise esso è il più alto

In questo modo possiamo scegliere "quasi come l'argmax" ma
senza perdere la capacità di calcolare gradienti.

 Utile per:
- selezionare feature in reti neurali
- apprendere scelte discrete durante l'allenamento
- generare maschere binarie (approx.) per selezioni strutturate
"""



"""
────────────────────────────────────────────────────────────────────────────
SOFTMAX E GUMBEL-SOFTMAX — COSA SONO E COME FUNZIONANO
────────────────────────────────────────────────────────────────────────────

 1. SOFTMAX
────────────────────────────
La funzione softmax prende un vettore di valori reali (logits)
e li trasforma in una distribuzione di probabilità, dove:

- Tutti i valori sono tra 0 e 1
- La loro somma è esattamente 1

Formula matematica:
    softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)

Esempio pratico:
    logits = [3.0, 1.0, 0.0]
    softmax → [0.84, 0.11, 0.05]

Uso:
    - Classificazione
    - Output finale di una rete neurale
    - Differenziabile → adatto al training

    

 2. GUMBEL-SOFTMAX
────────────────────────────
Gumbel-Softmax è una variante "rumorosa" della softmax,
che permette di simulare una scelta tipo `argmax`,
ma in modo continuo e addestrabile (differenziabile).

Funziona così:
    1. Aggiunge rumore Gumbel ai logits
    2. Applica softmax con una temperatura τ

Formula:
    yᵢ = exp((logitsᵢ + gᵢ) / τ) / Σ exp((logitsⱼ + gⱼ) / τ)

Dove:
    - gᵢ = rumore Gumbel = -log(-log(Uᵢ)), Uᵢ ∼ Uniform(0,1)
    - τ = temperatura: controlla quanto la scelta è "netta"

Effetti della temperatura τ:
    - τ alta (es. 5.0): scelte morbide, più simili alla media
    - τ bassa (es. 0.1): scelte quasi discrete, simili a argmax

Vantaggi:
    - Permette selezioni quasi discrete, ma con gradienti
    - Utile per scegliere elementi (es. 3 su 6 feature)
    - Addestrabile via backpropagation
'''
    



'''
────────────────────────────────────────────────────────────────────────────
Confronto rapido:
────────────────────────────────────────────────────────────────────────────

| Funzione        | Output         | Differenziabile   | Scelte nette?   |
|---------------- |--------------- |------------------ |---------------- |
| softmax         | probabilità    | ✅ sì             | ❌ no          |
| gumbel_softmax  | probabilità    | ✅ sì             | ✅ quasi       |
| argmax          | indice intero  | ❌ no             | ✅ sì          |

FRASE DA RICORDARE
> softmax è una distribuzione, gumbel_softmax è una scelta.
"""




#softmax: distribuisce le probabilità in modo proporzionale.
#argmax: sceglie una sola opzione con certezza (non derivabile).
#gumbel_softmax: simula scelte discrete in modo "morbido" ma addestrabile (grazie al rumore Gumbel e alla temperatura).
