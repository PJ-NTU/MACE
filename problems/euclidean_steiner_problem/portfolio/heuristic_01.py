# MACE evolved heuristic 01/10 for problem: euclidean_steiner_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined heuristic for the Euclidean Steiner Problem:
    1. Seed with high-quality Fermat points.
    2. Iteratively refine with coordinate descent.
    3. Apply a multi-stage randomized local perturbation (shake) that
       increases the intensity of exploration as the time limit approaches.
    """
    start_time = time.time()
    
    # 1. Initial Seeding
    steiner_points = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-8)
    
    def get_remaining():
        return time_limit_s - (time.time() - start_time)

    # 2. Refinement Loop
    if steiner_points and get_remaining() > 0.1:
        steiner_points = tools['local_relocate_steiner'](
            steiner_points,
            time_limit_s=min(get_remaining() * 0.5, 1.0),
            step=0.05
        )

    # 3. Enhanced Randomized Perturbation (Shake)
    # Modification: Perform multiple small shakes instead of one, 
    # scaling the perturbation magnitude based on remaining time.
    shake_count = 0
    while get_remaining() > 0.3 and shake_count < 3:
        perturbed = list(steiner_points)
        # Scale range by 0.2 down to 0.05 to refine local search
        magnitude = 0.2 - (shake_count * 0.05)
        idx = random.randrange(len(perturbed))
        x, y = perturbed[idx]
        perturbed[idx] = (x + random.uniform(-magnitude, magnitude), y + random.uniform(-magnitude, magnitude))
        
        refined_perturbed = tools['local_relocate_steiner'](
            perturbed,
            time_limit_s=min(get_remaining() * 0.4, 0.4),
            step=0.01
        )
        
        if tools['mst_length'](refined_perturbed) < tools['mst_length'](steiner_points):
            steiner_points = refined_perturbed
        shake_count += 1

    # 4. Finalization
    return tools['make_solution'](steiner_points=steiner_points)