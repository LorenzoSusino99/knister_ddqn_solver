import torch
import numpy as np
from knister_ai import KnisterEnvironmentWrapper, KnisterDQN_Agent # Assicurati che l'import del tuo file main sia corretto
from collections import deque

def evaluate_checkpoint(checkpoint_path="checkpoint_knister.pth", num_episodes=500):
    print(f"--- INIZIO VERIFICA MODELLO: {checkpoint_path} ---")
    print(f"Partite previste: {num_episodes}")
    
    # 1. Inizializza ambiente e agente
    env = KnisterEnvironmentWrapper()
    agent = KnisterDQN_Agent(lr=0.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.q_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.q_net.eval()
    
    scores = []
    
    for ep in range(1, num_episodes + 1):
        state, mask = env.reset()
        done = False
        
        while not done:
            
            action = agent.get_masked_action(state, mask, epsilon=0.0)
            next_state, reward, done, next_mask = env.step(action)
            state, mask = next_state, next_mask
            
        final_score = env.game.get_total_reward()
        scores.append(final_score)
        
        if ep % 50 == 0:
            print(f"Partite giocate: {ep}/{num_episodes} | Media Provvisoria: {np.mean(scores):.2f}")

    final_avg = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    print("\n" + "="*40)
    print(f"RISULTATI FINALI SU {num_episodes} PARTITE:")
    print(f"Media Totale:   {final_avg:.2f} punti")
    print(f"Punteggio Max:  {max_score} punti")
    print(f"Punteggio Min:  {min_score} punti")
    print("="*40)

if __name__ == "__main__":
    evaluate_checkpoint("checkpoint_knister.pth", 500)