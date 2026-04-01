import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from api import KnisterGame, InvalidAction
import os


# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DICE_VALUES = 13 
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
REWARD_SCALE = 10.0

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition):
        # Le nuove transizioni ricevono la massima priorità nota per essere campionate subito
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
            
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calcolo degli Importance Sampling Weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, td_errors, offset=0.1):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + offset

class KnisterQNet(nn.Module):
    def __init__(self):

        super(KnisterQNet, self).__init__()
        # Estrae relazioni locali e pattern parziali (es. dadi vicini)
        self.common_conv = nn.Sequential(
            nn.Conv2d(NUM_DICE_VALUES, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Estrae il valore dell'intera riga -> (128, 5, 1) e dell'intera colonna -> (128, 1, 5)
        self.conv_row = nn.Conv2d(64, 128, kernel_size=(1, 5)) 
        self.conv_col = nn.Conv2d(64, 128, kernel_size=(5, 1))
        
        # Gestione del dado
        self.dice_fc = nn.Sequential(
            nn.Linear(NUM_DICE_VALUES, 32),
            nn.ReLU()
        )

        self._init_flatten_size(NUM_DICE_VALUES)
        self.feature_norm = nn.LayerNorm(self.flat_size)
        
        # DUELING HEADS: Separano il Valore dello Stato dal Vantaggio dell'Azione
        self.value_head = nn.Sequential(
            nn.Linear(self.flat_size, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(self.flat_size, 256), 
            nn.ReLU(), 
            nn.Linear(256, 25)
        )

    def _init_flatten_size(self, num_dice_values):
        # Inizializzazione dinamica della dimensione del flattening
        with torch.no_grad():
            dummy_grid = torch.zeros(1, num_dice_values, 5, 5)
            dummy_dice = torch.zeros(1, num_dice_values)

            # Passaggio nel tronco comune
            x_common = self.common_conv(dummy_grid)
            
            # Scanner asimmetrici
            x_row = self.conv_row(x_common).view(1, -1)
            x_col = self.conv_col(x_common).view(1, -1)

            # Estraiamo le diagonali direttamente dalle feature profonde
            d1 = torch.diagonal(x_common, dim1=-2, dim2=-1) 
            d2 = torch.diagonal(torch.flip(x_common, [-1]), dim1=-2, dim2=-1)
            x_diag = torch.cat([d1, d2], dim=1).view(1, -1)
            
            x_dice = self.dice_fc(dummy_dice)
            
            combined = torch.cat([x_row, x_col, x_diag, x_dice], dim=1)
            self.flat_size = combined.size(1)

    def forward(self, grid_t, dice_t):
        # Propagazione reale
        x_common = self.common_conv(grid_t)
        
        x_row = F.relu(self.conv_row(x_common)).view(grid_t.size(0), -1)
        x_col = F.relu(self.conv_col(x_common)).view(grid_t.size(0), -1)
        
        d1 = torch.diagonal(x_common, dim1=-2, dim2=-1)
        d2 = torch.diagonal(torch.flip(x_common, [-1]), dim1=-2, dim2=-1)
        x_diag = torch.cat([d1, d2], dim=1).view(grid_t.size(0), -1)
        
        x_dice = self.dice_fc(dice_t)
        
        combined = torch.cat([x_row, x_col, x_diag, x_dice], dim=1)
        combined = self.feature_norm(combined)

        v = self.value_head(combined)
        a = self.advantage_head(combined)
        
        # Dueling formula standard
        return v + (a - a.mean(dim=1, keepdim=True))

class KnisterEnvironmentWrapper:
    def __init__(self):
        self.game = KnisterGame()

    def reset(self):
        self.game.new_game()
        return self._get_state_representation(), self._get_action_mask()

    def step(self, action):
        try:
            self.game.choose_action(action)
            # REWARD PURO: Usiamo solo il delta del gioco scalato linearmente
            raw_reward = self.game.get_last_reward() 
            reward = raw_reward / REWARD_SCALE
            done = self.game.has_finished()
        except InvalidAction:
            return self._get_state_representation(), -1.0, False, self._get_action_mask()
        
        return self._get_state_representation(), reward, done, self._get_action_mask()

    def _get_action_mask(self):
        mask = np.zeros(25, dtype=np.float32)
        mask[self.game.get_available_actions()] = 1.0
        return mask

    def _get_state_representation(self):
        grid = self.game.get_grid()
        roll = self.game.get_current_roll() or 0
        grid_3d = np.zeros((NUM_DICE_VALUES, 5, 5), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                grid_3d[grid[r, c], r, c] = 1.0
        roll_one_hot = np.eye(NUM_DICE_VALUES)[roll].astype(np.float32)
        return grid_3d, roll_one_hot

class KnisterDQN_Agent:
    def __init__(self, lr=1e-5):
        self.q_net = KnisterQNet().to(DEVICE)
        self.target_net = KnisterQNet().to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=100000)

    def get_masked_action(self, state, mask, epsilon):
        if random.random() < epsilon:
            return random.choice(np.where(mask == 1.0)[0])
        
        grid_t = torch.FloatTensor(state[0]).unsqueeze(0).to(DEVICE)
        dice_t = torch.FloatTensor(state[1]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals = self.q_net(grid_t, dice_t)
            q_vals[0, torch.from_numpy(mask).to(DEVICE) == 0] = -1e9
            return q_vals.argmax().item()

    def train_step(self, beta):
        if len(self.memory.buffer) < BATCH_SIZE * 10: 
            return None

        batch, indices, weights = self.memory.sample(BATCH_SIZE, beta)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        
        gs = torch.FloatTensor(np.array([s[0] for s in states])).to(DEVICE)
        ds = torch.FloatTensor(np.array([s[1] for s in states])).to(DEVICE)
        n_gs = torch.FloatTensor(np.array([s[0] for s in next_states])).to(DEVICE)
        n_ds = torch.FloatTensor(np.array([s[1] for s in next_states])).to(DEVICE)
        
        act_t = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rew_t = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        done_t = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
        mask_t = torch.FloatTensor(np.array(next_masks)).to(DEVICE)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(DEVICE)

        # Calcolo Q(s, a) corrente
        curr_q = self.q_net(gs, ds).gather(1, act_t)
        
        # DOUBLE DQN LOGIC
        with torch.no_grad():
            # A. Seleziona la migliore azione nel Next State usando la Rete Principale (q_net)
            n_q_main = self.q_net(n_gs, n_ds)
            n_q_main[mask_t == 0] = -1e9
            best_next_actions = n_q_main.argmax(dim=1, keepdim=True)
            
            # B. Valuta l'azione scelta usando la Rete Target (target_net)
            max_n_q = self.target_net(n_gs, n_ds).gather(1, best_next_actions)
            target_q = rew_t + (1 - done_t) * GAMMA * max_n_q

        # Calcolo del TD Error puro per aggiornare il Buffer
        td_errors = (target_q - curr_q).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights_t * F.mse_loss(curr_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0) # Previene l'esplosione dei gradienti
        self.optimizer.step()
        
        return loss.item()

    def soft_update_target_network(self):
        # Eseguito ad ogni step: aggiorna lentamente i pesi della Target Net
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

# VISUALIZZATORE E MAIN LOOP 
def print_grid_expert(grid_3d):
    visual_grid = np.zeros((5, 5), dtype=int)
    for v in range(2, 13):
        visual_grid[grid_3d[v] == 1] = v
    print("\n--- GRIGLIA RECORD ---")
    for row in visual_grid:
        print(" ".join([f"{v:2d}" if v != 0 else " ." for v in row]))

def get_symmetries(state, action, reward, next_state, done, next_mask):
    """
    Data Augmentation: Genera 8 transizioni equivalenti sfruttando le
    simmetrie spaziali della griglia 5x5 di Knister.
    """
    grid_3d, dice_1d = state
    n_grid_3d, n_dice_1d = next_state
    
    symmetries = []
    
    # Mappiamo l'azione (scalare 0-24) in una matrice 2D (5x5) per ruotarla facilmente
    action_2d = np.zeros((5, 5), dtype=np.float32)
    row, col = divmod(action, 5)
    action_2d[row, col] = 1.0
    
    # Rimodelliamo la maschera (array 1D) in 5x5
    mask_2d = next_mask.reshape(5, 5)
    
    # Il gruppo D4 ha 4 rotazioni di base
    for k in range(4):
        # Ruota la griglia
        g_rot = np.rot90(grid_3d, k=k, axes=(1, 2))
        ng_rot = np.rot90(n_grid_3d, k=k, axes=(1, 2))
        
        # Ruota l'azione e la maschera
        a_rot = np.rot90(action_2d, k=k)
        m_rot = np.rot90(mask_2d, k=k)
        
        # Ricostruiamo la transizione
        symmetries.append((
            (g_rot.copy(), dice_1d),
            int(np.argmax(a_rot)), 
            reward,
            (ng_rot.copy(), n_dice_1d),
            done,
            m_rot.flatten()
        ))
        
        # Per ogni rotazione aggiungiamo la sua specchiatura orizzontale
        g_flip = np.flip(g_rot, axis=2)
        ng_flip = np.flip(ng_rot, axis=2)
        a_flip = np.flip(a_rot, axis=1)
        m_flip = np.flip(m_rot, axis=1)
        
        symmetries.append((
            (g_flip.copy(), dice_1d),
            int(np.argmax(a_flip)),
            reward,
            (ng_flip.copy(), n_dice_1d),
            done,
            m_flip.flatten()
        ))
        
    return symmetries

env = KnisterEnvironmentWrapper()
agent = KnisterDQN_Agent(lr=5e-5)

scores_history = deque(maxlen=500)
best_avg = -float('inf')

epsilon, eps_min, eps_decay = 1.0, 0.02, 0.9999947

beta_start = 0.4
beta_frames = 850000
EPISODES = 1000000
TRAIN_FREQ = 18 
global_step = 0
if __name__ == "__main__":
    print(f"Inizio addestramento da zero ({EPISODES} episodi) con Double DQN + PER...")

    for ep in range(EPISODES):
        state, mask = env.reset()
        done = False

        current_beta = min(1.0, beta_start + ep * (1.0 - beta_start) / beta_frames)
        
        while not done:
            action = agent.get_masked_action(state, mask, epsilon)
            n_state, reward, done, n_mask = env.step(action)

            augmented_transitions = get_symmetries(state, action, reward, n_state, done, n_mask)
            for t in augmented_transitions:
                agent.memory.add(t)

            global_step += 1

            if global_step % TRAIN_FREQ == 0:
                agent.train_step(beta=current_beta)
                agent.soft_update_target_network()
            
            state, mask = n_state, n_mask

        score = env.game.get_total_reward()
        scores_history.append(score)

        if epsilon > eps_min: 
            epsilon *= eps_decay

        if ep > 0 and ep % 100 == 0:
            avg = np.mean(scores_history)
            print(f"EP {ep} | Media ultimi 500: {avg:.2f} | Eps: {epsilon:.3f} | Buffer: {len(agent.memory.buffer)}")
            
            if avg > best_avg and ep > 500:
                best_avg = avg
                print("NUOVO RECORD DI MEDIA!")
                torch.save(agent.q_net.state_dict(), "checkpoint_knister.pth")
                print_grid_expert(state[0])