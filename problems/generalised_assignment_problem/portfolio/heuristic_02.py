# MACE evolved heuristic 02/10 for problem: generalised_assignment_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for GAP.
    
    Hypothesis:
    - Small/Dense/Constrained instances are better handled by ILP/Tabu (Parent B).
    - Large/Sparse/Loose instances are better handled by GRASP-style local search (Parent A).
    
    We estimate density by comparing total capacity to total mean resource usage.
    """
    start_time = time.time()
    n = tools['n_tasks']()
    m = tools['n_agents']()
    
    # Calculate density heuristic: ratio of total capacity to total resource demand
    # A low ratio implies a highly constrained problem (B-style).
    # A high ratio implies a loose problem (A-style).
    total_capacity = sum(instance['capacities'])
    total_avg_consumption = sum(sum(row) for row in instance['consumption_matrix']) / m
    density = total_capacity / (total_avg_consumption + 1e-9)
    
    # Decide strategy
    # B (Tabu/ILP) is robust for constrained; A (GRASP) is faster for large/loose.
    use_b_strategy = (density < 1.5) or (n < 50)

    if use_b_strategy:
        # Parent B: Tabu Search + ILP initialization
        current_assignment = None
        if time_limit_s > 1.0:
            current_assignment = tools['ilp_gap'](time_limit_s=min(1.5, time_limit_s * 0.2))
            
        if current_assignment is None:
            current_assignment = tools['greedy_min_cost']()
            feasible, _ = tools['is_feasible_assignment'](current_assignment)
            if not feasible:
                current_assignment = [1] * n
                for j in range(n):
                    for a in range(1, m + 1):
                        temp = tools['apply_reassign'](current_assignment, j, a)
                        if temp and tools['is_feasible_assignment'](temp)[0]:
                            current_assignment = temp
                            break

        best_assignment = list(current_assignment)
        best_score = tools['objective']({'assignments': best_assignment})
        tabu_list = {}
        tabu_tenure = max(5, int(n * 0.1))
        
        while time.time() - start_time < time_limit_s * 0.95:
            best_candidate = None
            best_candidate_score = float('inf')
            move_to_record = None
            
            # Neighborhood exploration
            for j in range(n):
                for a in range(1, m + 1):
                    if current_assignment[j] == a: continue
                    candidate = tools['apply_reassign'](current_assignment, j, a)
                    if candidate and tools['is_feasible_assignment'](candidate)[0]:
                        score = tools['objective']({'assignments': candidate})
                        if tabu_list.get((j, a), 0) < time.time() or score < best_score:
                            if score < best_candidate_score:
                                best_candidate = candidate
                                best_candidate_score = score
                                move_to_record = (j, a)
            if best_candidate is None: break
            current_assignment = best_candidate
            if move_to_record: tabu_list[move_to_record] = time.time() + 0.1
            if best_candidate_score < best_score:
                best_score = best_candidate_score
                best_assignment = list(best_candidate)
                
    else:
        # Parent A: GRASP-style Metaheuristic
        best_assignment = tools['greedy_min_cost']()
        best_score = tools['objective']({'assignments': best_assignment})
        
        while time.time() - start_time < time_limit_s * 0.9:
            improved = False
            for _ in range(200):
                if time.time() - start_time > time_limit_s * 0.95: break
                candidate = None
                if random.random() < 0.8:
                    candidate = tools['apply_reassign'](best_assignment, random.randint(0, n-1), random.randint(1, m))
                else:
                    candidate = tools['apply_swap_assignments'](best_assignment, random.randint(0, n-1), random.randint(0, n-1))
                
                if candidate:
                    score = tools['objective']({'assignments': candidate})
                    if score < best_score:
                        best_score = score
                        best_assignment = candidate
                        improved = True
            if not improved: break
            
    return {'assignments': best_assignment}