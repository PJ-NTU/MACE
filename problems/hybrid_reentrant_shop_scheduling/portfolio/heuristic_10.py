# MACE evolved heuristic 10/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random
import math
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Population-based Iterated Local Search (ILS) with a focus on 
    'Ruined and Recreated' (R&R) perturbations rather than simple 
    hill-climbing or pure SA.
    
    Differences from portfolio:
    - Uses a population of solutions (not just a single trajectory).
    - Employs 'Ruin and Recreate' (removing random segments of the 
      permutation and greedily re-inserting) which is more robust than 
      simple swaps or insertions for re-entrant scheduling.
    - Uses a tournament selection mechanism for crossover/mutation.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    init_time = instance['init_time']

    # Deterministic batch assignment
    machine_heap = [(0, i) for i in range(1, n_machines + 1)]
    heapq.heapify(machine_heap)
    batch_assignment = [0] * n_jobs
    for j in range(1, n_jobs + 1):
        avail, m_id = heapq.heappop(machine_heap)
        batch_assignment[j - 1] = m_id
        heapq.heappush(machine_heap, (avail + init_time, m_id))

    def get_makespan(perm):
        try:
            return tools['simulate_schedule'](perm)['makespan']
        except:
            return float('inf')

    # Initial population using diversified priority rules
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    population = []
    for rule in rules:
        try:
            p = tools['list_scheduling_priority'](rule=rule)
            population.append({'perm': p, 'score': get_makespan(p)})
        except:
            continue
            
    best_sol = min(population, key=lambda x: x['score'])

    # Main Loop: Ruin and Recreate
    while time.time() - start_time < time_limit_s * 0.95:
        # Select parent
        parent = random.choice(population)
        
        # Ruin: Remove a random subsegment (10-30% of sequence)
        size = max(1, int(n_jobs * random.uniform(0.1, 0.3)))
        start_idx = random.randint(0, n_jobs - size)
        
        ruined = parent['perm'][:start_idx] + parent['perm'][start_idx+size:]
        removed = parent['perm'][start_idx:start_idx+size]
        
        # Recreate: Greedy re-insertion
        recreated = list(ruined)
        for job in removed:
            best_pos_score = float('inf')
            best_pos = 0
            # Try all positions to re-insert the removed job
            for pos in range(len(recreated) + 1):
                temp = recreated[:pos] + [job] + recreated[pos:]
                score = get_makespan(temp)
                if score < best_pos_score:
                    best_pos_score = score
                    best_pos = pos
            recreated.insert(best_pos, job)
            
        new_score = get_makespan(recreated)
        
        # Update population: replace worst
        if len(population) < 10:
            population.append({'perm': recreated, 'score': new_score})
        else:
            worst_idx = population.index(max(population, key=lambda x: x['score']))
            if new_score < population[worst_idx]['score']:
                population[worst_idx] = {'perm': recreated, 'score': new_score}
        
        if new_score < best_sol['score']:
            best_sol = {'perm': recreated, 'score': new_score}
            
    return {
        "permutation": best_sol['perm'],
        "batch_assignment": batch_assignment
    }