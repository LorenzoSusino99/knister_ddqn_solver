# Knister Game API

This repository contains a Python implementation of the Knister dice game, exposed through a simple and explicit API.

---

## The Knister Game

Knister is a turn based dice game played on a 5x5 grid.

Also known as *Würfel-Bingo*, it is such a notorious game that it only has a Wikipedia page in German. If you are curious (or brave), see [here](https://de.wikipedia.org/wiki/W%C3%BCrfel-Bingo).

### Dice mechanics

At each turn, two six sided dice are rolled and their sum is obtained.  
The possible values range from 2 to 12.

The probability distribution is not uniform:
- 7 is the most likely outcome
- 2 and 12 are the least likely outcomes

The rolled value must be placed into one empty cell of the grid.

### Game flow

1. The game starts with an empty 5x5 grid.
2. A dice roll is generated.
3. The player chooses one empty cell and places the roll value there.
4. The score is updated incrementally based on the new grid state.
5. Steps 2 to 4 repeat until all cells are filled.
6. The game ends when the grid is full.

---

## Scoring Rules

Scores are computed independently for:
- each row
- each column
- the two diagonals

Diagonal scores are multiplied by a constant factor.

Empty cells do not contribute to scoring.

### Combinations

For a given line (row, column, or diagonal):

| Combination                     | Score |
|---------------------------------|-------|
| Five of a kind                  | 10    |
| Four of a kind                  | 6     |
| Full house (3 + 2)              | 8     |
| Three of a kind                 | 3     |
| Two pairs                       | 3     |
| One pair                        | 1     |
| Straight of 5 values, no 7      | 12    |
| Straight of 5 values, with 7    | 8     |

A straight is valid only if:
- the line contains exactly five values
- all values are distinct
- values are consecutive

---

## Architettura dell IA
L agente utilizza una Dueling Double Deep Q-Network (D3QN) con un architettura neurale custom progettata per estrarre feature spaziali e combinatorie dalla griglia.

### 1. State Representation (Input)
L agente riceve una rappresentazione tensoriale avanzata dello stato di gioco:
* Grid State: Un tensore 3D di forma (13, 5, 5) dove ogni canale rappresenta un valore del dado (One-Hot encoding spaziale).
* Current Roll: Un vettore One-Hot di dimensione 13 che indica il valore del dado da piazzare nel turno corrente.

### 2. Neural Network Design
La rete KnisterQNet implementa diversi rami specializzati:
* Asymmetric Convolutions: Filtri (1, 5) e (5, 1) per analizzare specificamente l integrita di righe e colonne.
* Diagonal Branch: Un ramo dedicato all estrazione e analisi delle diagonali, data la loro importanza strategica.
* Dueling Head: Separazione della stima del valore dello stato (V) dal vantaggio di ogni singola azione (A) per migliorare la stabilita.
* Normalization: Uso di LayerNorm per stabilizzare l apprendimento contro le ampie variazioni di reward.

### 3. Advanced Learning Techniques
Per ottimizzare le prestazioni e puntare a punteggi elevati, sono state implementate tecniche avanzate:
* Double DQN: Per ridurre la sovrastima sistematica dei valori Q.
* Self-Imitation Learning (SIL): Un Elite Buffer memorizza le migliori partite di sempre per permettere all agente di ripassare le strategie vincenti.
* Advanced Reward Shaping: La funzione di reward e stata modellata per fornire feedback densi:
    * Delta Score: Incremento immediato di punteggio.
    * Potential Gain: Premia le mosse che preparano combinazioni future (es. tris o poker).
    * Flexibility: Incentiva il mantenimento di opzioni aperte in linee promettenti.
    * Dead Cells: Penalizza il posizionamento di numeri che rendono impossibili le scale.

## Struttura del Repository
* api.py: Logica core del gioco, gestione del posizionamento e calcolo dei punteggi.
* knister_ai.py: Implementazione della D3QN, della rete neurale e del loop di addestramento.
* play.py: Script CLI per testare manualmente il gioco e verificare le regole.
* README.md: Documentazione del progetto.

## Utilizzo

### Requisiti
* Python 3.10+
* PyTorch
* NumPy

### Addestramento
Per avviare l addestramento dell agente:
```bash
python knister_ai.py
```

### Test
Per testare il checkpoint generato avviare:
```bash
python test_ai.py
```
