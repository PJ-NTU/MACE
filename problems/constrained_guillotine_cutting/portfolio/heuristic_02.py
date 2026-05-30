# MACE evolved heuristic 02/10 for problem: constrained_guillotine_cutting
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A non-greedy, stochastic metaheuristic focusing on 'Tabu-inspired' 
    hill climbing with large-scale neighborhood moves.
    
    The portfolio is dominated by:
    1. Deterministic/Greedy construction (Bottom-Left / BFD).
    2. Local swaps/replacements (1-for-1 swaps).
    
    This heuristic differs by:
    1. 'Destructive-Reconstructive' strategy: Instead of small swaps, it 
       periodically clears large portions of the stock (shattering) to allow 
       the greedy placer to re-fill with a different global bias.
    2. No reliance on a single 'best' greedy order; it uses a temperature-controlled
       probabilistic selection of piece types to explore the state space.
    """
    start_time = time.time()
    
    m = tools['n_piece_types']()
    indices = list(range(1, m + 1))
    
    def get_shattered_placements(current_placements, keep_ratio=0.5):
        """Randomly removes a subset of pieces to open up space."""
        if not current_placements:
            return []
        keep_count = int(len(current_placements) * keep_ratio)
        return random.sample(current_placements, keep_count)

    def get_stochastic_order():
        """Returns a permutation biased by value density but with noise."""
        # Calculate base density
        densities = []
        for i in indices:
            l, w = tools['piece_dims'](i)
            v = tools['piece_value'](i)
            densities.append((v / (l * w + 1e-9), i))
        
        # Softmax-like selection or jittered sort
        densities.sort(key=lambda x: x[0], reverse=True)
        # Apply jitter
        order = [d[1] for d in densities]
        for i in range(len(order)):
            if random.random() < 0.2:
                j = random.randint(0, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        return order

    best_sol = {"total_value": 0, "placements": []}
    
    # Run loop
    while time.time() - start_time < time_limit_s * 0.9:
        try:
            # 1. Start with a destruction move if we have an existing solution
            if best_sol["placements"]:
                work_placements = get_shattered_placements(best_sol["placements"])
            else:
                work_placements = []
            
            # 2. Fill the remainder using a stochastic greedy order
            # The tool try_place_piece is used here to fill gaps one by one
            order = get_stochastic_order()
            for p_idx in order:
                if tools['used_count'](work_placements, p_idx) < tools['piece_demand_max'](p_idx):
                    attempt = tools['try_place_piece'](work_placements, p_idx)
                    if attempt is not None:
                        work_placements = attempt
            
            val = tools['total_value_of'](work_placements)
            
            # 3. Acceptance (Simple Hill Climbing)
            if val > best_sol["total_value"]:
                best_sol = {"total_value": val, "placements": work_placements}
        except Exception:
            continue
            
    # Final check
    is_valid, _ = tools['is_feasible'](best_sol)
    if not is_valid:
        # Fallback to standard greedy if stochastic search diverged
        fallback = tools['bottom_left_pack_demand_aware']()
        return {"total_value": tools['total_value_of'](fallback), "placements": fallback}
        
    return best_sol