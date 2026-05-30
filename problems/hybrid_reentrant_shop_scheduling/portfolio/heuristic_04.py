# MACE evolved heuristic 04/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Hybrid Reentrant Shop Scheduling problem using a GRASP-inspired 
    approach: generating multiple candidate permutations based on priority rules 
    and local search refinement.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    
    # The problem description notes that eval_func ignores batch_assignment 
    # and uses a deterministic round-robin assignment for primary machines.
    # We must provide a valid batch_assignment as per the feasibility check.
    # Job (j-1) is assigned to machine ((j-1) % n_machines) + 1.
    batch_assignment = [((i - 1) % n_machines) + 1 for i in range(1, n_jobs + 1)]
    
    best_makespan = float('inf')
    best_permutation = None
    
    # Rules to try for initial seed permutations
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    
    candidates = []
    for rule in rules:
        try:
            perm = tools['list_scheduling_priority'](rule=rule)
            candidates.append(perm)
        except Exception:
            continue
            
    # Add a random permutation to diversify
    random_perm = list(range(1, n_jobs + 1))
    random.shuffle(random_perm)
    candidates.append(random_perm)
    
    # Process candidates
    for perm in candidates:
        if time.time() - start_time > time_limit_s * 0.9:
            break
            
        # Refine using local search (first-improvement)
        # We use a smaller time slice per candidate to allow multiple restarts
        remaining = time_limit_s * 0.9 - (time.time() - start_time)
        if remaining <= 0:
            break
            
        refined_perm = tools['apply_local_swap'](
            permutation=list(perm), 
            time_limit_s=remaining / len(candidates), 
            neighbourhood='insert'
        )
        
        # Evaluate
        try:
            res = tools['simulate_schedule'](refined_perm)
            ms = res['makespan']
            if ms < best_makespan:
                best_makespan = ms
                best_permutation = refined_perm
        except Exception:
            continue
            
    # Fallback in case of failure
    if best_permutation is None:
        best_permutation = list(range(1, n_jobs + 1))
        
    return {
        "permutation": best_permutation,
        "batch_assignment": batch_assignment
    }