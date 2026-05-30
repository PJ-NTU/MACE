# MACE evolved heuristic 04/10 for problem: flow_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Flow Shop Scheduling.
    
    Hypothesis: 
    - Small instances (n <= 20) benefit from exhaustive search/ILP or 
      aggressive iterative refinement.
    - Large instances (n > 20) are dominated by the quality of the NEH 
      construction, where standard insertion search provides diminishing 
      returns.
    - For high machine counts (m > n), the problem becomes more sensitive 
      to bottleneck identification, favoring iterative refinement (ILS).
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    
    # Feature extraction:
    # A-style (ILS) is better for harder, larger instances where local minima 
    # are frequent. B-style (Basic NEH + single Insertion) is better for 
    # smaller instances or tight time budgets where overhead matters.
    is_large = n > 20 or (n * m > 500)
    
    # 1. Construction
    try:
        # Special case: Johnson's rule for 2 machines is optimal
        if m == 2:
            johnson_perm = tools['johnson_2machine_construct']()
            if johnson_perm:
                return tools['make_solution'](johnson_perm)
        
        current_perm = tools['neh_construct']()
    except Exception:
        current_perm = list(range(1, n + 1))
        random.shuffle(current_perm)
    
    # 2. Refinement strategy
    if is_large:
        # ILS Strategy (Parent A-like)
        # Large instances need perturbation to escape local optima.
        current_perm = tools['apply_insertion_search'](
            current_perm, 
            time_limit_s=time_limit_s * 0.5, 
            first_improvement=True
        )
        
        while (time.time() - start_time) < time_limit_s * 0.9:
            # Perturb
            perturbed = list(current_perm)
            # Swap 2 elements
            idx1, idx2 = random.sample(range(n), 2)
            perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
            
            # Local Search
            candidate = tools['apply_insertion_search'](
                perturbed, 
                time_limit_s=0.1, 
                first_improvement=True
            )
            
            if tools['simulate_makespan'](candidate) < tools['simulate_makespan'](current_perm):
                current_perm = candidate
    else:
        # Standard Insertion Search (Parent B-like)
        # For smaller instances, NEH + one strong pass is usually sufficient.
        current_perm = tools['apply_insertion_search'](
            current_perm, 
            time_limit_s=time_limit_s * 0.8, 
            first_improvement=True
        )
        
    return tools['make_solution'](current_perm)