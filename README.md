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

## AI Architecture
The agent utilizes a Dueling Double Deep Q-Network (D3QN) with a custom neural architecture designed to extract spatial and combinatorial features from the grid.

### 1. State Representation (Input)
The agent receives an advanced tensorial representation of the game state:
* Grid State: A 3D tensor of shape (13, 5, 5) where each channel represents a dice value (spatial One-Hot encoding).
* Current Roll: A One-Hot vector of size 13 indicating the dice value to be placed in the current turn.

### 2. Neural Network Design
The KnisterQNet network implements several specialized branches:
* Asymmetric Convolutions: (1, 5) and (5, 1) filters to specifically analyze the integrity of rows and columns.
* Diagonal Branch: A dedicated branch for extracting and analyzing diagonals, given their strategic importance.
* Dueling Head: Separation of the state value estimation (V) from the advantage of each single action (A) to improve stability.
* Normalization: Use of LayerNorm to stabilize learning against wide variations in rewards.

### 3. Advanced Learning Techniques
To optimize performance and target high scores, advanced techniques have been implemented:
* Double DQN: To reduce systematic overestimation of Q-values.
* Self-Imitation Learning (SIL): An Elite Buffer stores the best games ever played to allow the agent to revisit winning strategies.
* Advanced Reward Shaping: The reward function has been modeled to provide dense feedback:
    * Delta Score: Immediate score increase.
    * Potential Gain: Rewards moves that prepare future combinations such as three-of-a-kind or four-of-a-kind.
    * Flexibility: Encourages maintaining open options in promising lines.
    * Dead Cells: Penalizes the placement of numbers that make straights impossible.

## Repository Structure
* api.py: Core game logic, placement management, and score calculation.
* knister_ai.py: Implementation of the D3QN, neural network, and training loop.
* play.py: CLI script to manually test the game and verify rules.
* README.md: Project documentation.

## Usage

### Requirements
* Python 3.10+
* PyTorch
* NumPy

### Training
To start training the agent:
```bash
python knister_ai.py
```

### Test
To test the agent saved in 'checkpoint_knister.pth':
```bash
python test_ai.py
```
