# MACE evolved heuristic 09/10 for problem: crew_scheduling
import time
import math
import random
import heapq
from collections import defaultdict, deque


def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Crew Scheduling via Ant Colony Optimization + DP Chain Building + Tabu Search.
    
    Core differences from portfolio:
    1. ACO pheromone-guided construction (vs greedy/random)
    2. DP shortest-path for optimal chain building within crews
    3. Population-based search with diversity maintenance
    4. Tabu search acceptance (vs simulated annealing)
    5. Chain merging via greedy matching (vs insertion repair)
    6. Time-bucket decomposition for large instances
    """
    start_time = time.time()

    N = instance['N']
    K = instance['K']
    time_limit = instance['time_limit']
    tasks = instance['tasks']
    arcs = instance['arcs']

    def elapsed():
        return time.time() - start_time

    def remaining():
        return time_limit_s - elapsed()

    # ------------------------------------------------------------------ #
    # Precompute structures
    # ------------------------------------------------------------------ #
    arc_set = set(arcs.keys())
    
    # Sorted successors/predecessors
    succ = defaultdict(list)  # t -> [(start_j, j, cost)]
    pred = defaultdict(list)  # t -> [(finish_i, i, cost)]
    for (i, j), cost in arcs.items():
        if tasks[i][1] <= tasks[j][0]:
            succ[i].append((tasks[j][0], j, cost))
            pred[j].append((tasks[i][1], i, cost))
    for t in range(1, N + 1):
        succ[t].sort()
        pred[t].sort()

    tasks_by_start = sorted(range(1, N + 1), key=lambda t: (tasks[t][0], tasks[t][1]))
    task_start = {t: tasks[t][0] for t in range(1, N + 1)}
    task_finish = {t: tasks[t][1] for t in range(1, N + 1)}

    # ------------------------------------------------------------------ #
    # Fast feasibility helpers
    # ------------------------------------------------------------------ #
    def check_crew(crew):
        if not crew:
            return False
        if task_finish[crew[-1]] - task_start[crew[0]] > time_limit:
            return False
        for idx in range(len(crew) - 1):
            t1, t2 = crew[idx], crew[idx + 1]
            if task_finish[t1] > task_start[t2]:
                return False
            if (t1, t2) not in arc_set:
                return False
        return True

    def crew_cost(crew):
        cost = 0.0
        for idx in range(len(crew) - 1):
            c = arcs.get((crew[idx], crew[idx + 1]))
            if c is None:
                return float('inf')
            cost += c
        return cost

    def sol_cost(crews):
        return sum(crew_cost(c) for c in crews)

    # ------------------------------------------------------------------ #
    # Best solution tracking
    # ------------------------------------------------------------------ #
    best_sol = None
    best_cost = float('inf')

    def update_best(sol):
        nonlocal best_sol, best_cost
        if sol is None:
            return False
        try:
            feasible, _ = tools['is_feasible'](sol)
            if not feasible:
                return False
            cost = tools['objective'](sol)
            if cost < best_cost:
                best_cost = cost
                best_sol = {'crews': [list(c) for c in sol['crews']]}
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------ #
    # Phase 0: Quick initial solution from tools
    # ------------------------------------------------------------------ #
    init_budget = min(remaining() * 0.25, time_limit_s * 0.25)
    if init_budget > 0.5:
        try:
            sol = tools['solve_default'](time_limit_s=init_budget)
            update_best(sol)
        except Exception:
            pass

    if best_sol is None and remaining() > 1:
        try:
            chains = tools['greedy_chain_pack']()
            if chains and len(chains) <= K:
                sol = tools['make_solution'](chains)
                update_best(sol)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # DP-based optimal chain builder
    # Given a set of tasks, build minimum-cost chains using DP on the DAG
    # ------------------------------------------------------------------ #
    def dp_build_chains(task_set, max_chains):
        """
        Build chains covering task_set using DP shortest paths.
        Returns list of chains or None if infeasible.
        
        Strategy: for each task in topological order, compute the minimum
        cost to reach it from any chain start, tracking predecessors.
        """
        if not task_set:
            return []
        
        task_list = sorted(task_set, key=lambda t: task_start[t])
        task_idx = {t: i for i, t in enumerate(task_list)}
        n = len(task_list)
        
        INF = float('inf')
        # dp[i] = min cost to end a chain at task_list[i]
        # prev[i] = predecessor task in optimal chain (None = chain start)
        dp = [INF] * n
        prev_task = [None] * n
        
        # Initialize: each task can start a new chain (cost 0 to reach)
        for i in range(n):
            dp[i] = 0.0  # cost of chain starting at task_list[i]
        
        # Forward DP
        for i in range(n):
            t_i = task_list[i]
            # Try to extend chain ending at t_i to subsequent tasks
            for _, t_j, arc_cost_val in succ[t_i]:
                if t_j not in task_idx:
                    continue
                j = task_idx[t_j]
                # Check duty time: need to know chain head
                # Find head of chain ending at i
                head = t_i
                cur = i
                visited = set()
                while prev_task[cur] is not None and cur not in visited:
                    visited.add(cur)
                    cur = task_idx[prev_task[cur]]
                    head = task_list[cur]
                
                duty = task_finish[t_j] - task_start[head]
                if duty > time_limit:
                    continue
                
                new_cost = dp[i] + arc_cost_val
                if new_cost < dp[j]:
                    dp[j] = new_cost
                    prev_task[j] = t_i
        
        # Reconstruct chains
        # Find chain ends: tasks not pointed to by any prev_task
        is_interior = set()
        for i in range(n):
            if prev_task[i] is not None:
                is_interior.add(prev_task[i])
        
        chains = []
        for i in range(n):
            t = task_list[i]
            if t not in is_interior:
                # This is a chain end - trace back
                chain = []
                cur_t = t
                visited = set()
                while cur_t is not None and cur_t not in visited:
                    visited.add(cur_t)
                    chain.append(cur_t)
                    cur_t = prev_task[task_idx[cur_t]]
                chain.reverse()
                if check_crew(chain):
                    chains.append(chain)
                else:
                    # Break into individual tasks
                    for task in chain:
                        chains.append([task])
        
        if len(chains) > max_chains:
            return None
        return chains

    # ------------------------------------------------------------------ #
    # ACO Pheromone-guided construction
    # ------------------------------------------------------------------ #
    # Initialize pheromone matrix: tau[(i,j)] = initial pheromone
    # Only for valid arcs
    tau_init = 1.0
    tau = {}
    for (i, j) in arc_set:
        tau[(i, j)] = tau_init
    
    tau_min = 0.01
    tau_max = 10.0
    alpha = 1.0   # pheromone importance
    beta = 2.0    # heuristic importance (1/cost)
    rho = 0.1     # evaporation rate
    
    def heuristic(i, j):
        """Heuristic value: inverse of arc cost (prefer cheap arcs)."""
        c = arcs.get((i, j), float('inf'))
        if c <= 0:
            return 10.0
        return 1.0 / c
    
    def aco_construct_solution(rng, alpha_param=1.0, beta_param=2.0):
        """
        Build a solution using ACO:
        - Process tasks in start-time order
        - For each task, choose which crew to append it to using pheromone + heuristic
        """
        # Each crew is represented as a list; we maintain crew state
        crews = []
        crew_last = []   # last task in each crew
        crew_start_t = []  # start time of first task
        crew_finish_t = []  # finish time of last task
        
        for task in tasks_by_start:
            t_s = task_start[task]
            t_f = task_finish[task]
            
            # Compute probabilities for each valid crew
            candidates = []
            for i, crew in enumerate(crews):
                last = crew_last[i]
                if crew_finish_t[i] > t_s:
                    continue
                if (last, task) not in arc_set:
                    continue
                if t_f - crew_start_t[i] > time_limit:
                    continue
                
                # Pheromone * heuristic
                tau_val = tau.get((last, task), tau_init)
                eta_val = heuristic(last, task)
                score = (tau_val ** alpha_param) * (eta_val ** beta_param)
                candidates.append((score, i))
            
            if candidates:
                # Roulette wheel selection
                total_score = sum(s for s, _ in candidates)
                if total_score <= 0:
                    chosen_idx = rng.choice([i for _, i in candidates])
                else:
                    r = rng.random() * total_score
                    cumulative = 0.0
                    chosen_idx = candidates[-1][1]
                    for score, idx in candidates:
                        cumulative += score
                        if r <= cumulative:
                            chosen_idx = idx
                            break
                
                crews[chosen_idx].append(task)
                crew_last[chosen_idx] = task
                crew_finish_t[chosen_idx] = t_f
            elif len(crews) < K:
                # Start new crew
                crews.append([task])
                crew_last.append(task)
                crew_start_t.append(t_s)
                crew_finish_t.append(t_f)
            else:
                # Force: find any valid crew
                forced = False
                for i in range(len(crews)):
                    last = crew_last[i]
                    if crew_finish_t[i] <= t_s and (last, task) in arc_set:
                        if t_f - crew_start_t[i] <= time_limit:
                            crews[i].append(task)
                            crew_last[i] = task
                            crew_finish_t[i] = t_f
                            forced = True
                            break
                if not forced:
                    return None
        
        if not crews:
            return None
        
        sol = {'crews': crews}
        try:
            feasible, _ = tools['is_feasible'](sol)
            return sol if feasible else None
        except Exception:
            return None
    
    def update_pheromones(solutions_with_costs, best_cost_global):
        """
        Update pheromone trails:
        - Evaporate all trails
        - Deposit on arcs used by good solutions
        """
        # Evaporation
        for key in tau:
            tau[key] = max(tau_min, tau[key] * (1 - rho))
        
        # Deposit
        for sol, cost in solutions_with_costs:
            if cost >= float('inf'):
                continue
            deposit = 1.0 / max(cost, 1e-6)
            # Bonus for best solution
            if abs(cost - best_cost_global) < 1e-6:
                deposit *= 2.0
            
            for crew in sol['crews']:
                for idx in range(len(crew) - 1):
                    key = (crew[idx], crew[idx + 1])
                    if key in tau:
                        tau[key] = min(tau_max, tau[key] + deposit)

    # ------------------------------------------------------------------ #
    # Tabu Search
    # ------------------------------------------------------------------ #
    def tabu_search(initial_crews, time_budget, tabu_tenure=7):
        """
        Tabu search on crew assignments.
        Moves: relocate a task from one crew to another.
        Tabu: (task, src_crew_id) pairs recently moved.
        """
        ts_start = time.time()
        
        def ts_remaining():
            return time_budget - (time.time() - ts_start)
        
        crews = [list(c) for c in initial_crews]
        current_cost = sol_cost(crews)
        best_ts_crews = [list(c) for c in crews]
        best_ts_cost = current_cost
        
        tabu_list = {}  # (task, src_crew_hash) -> iteration_added
        iteration = 0
        no_improve = 0
        
        while ts_remaining() > 0.1 and no_improve < 30:
            iteration += 1
            
            # Clean expired tabu entries
            expired = [k for k, v in tabu_list.items() if iteration - v > tabu_tenure]
            for k in expired:
                del tabu_list[k]
            
            # Find best non-tabu move
            best_move_gain = float('-inf')
            best_move = None
            
            for ci in range(len(crews)):
                crew_i = crews[ci]
                if len(crew_i) <= 1:
                    continue
                
                for ti in range(len(crew_i)):
                    task = crew_i[ti]
                    
                    # Check if move is tabu
                    tabu_key = (task, ci)
                    if tabu_key in tabu_list:
                        # Only allow if aspiration criterion met
                        pass
                    
                    # Cost of removing task from crew_i
                    old_cost_i = crew_cost(crew_i)
                    new_crew_i = crew_i[:ti] + crew_i[ti + 1:]
                    
                    if not new_crew_i:
                        continue
                    
                    # Check bridge validity
                    if ti > 0 and ti < len(crew_i) - 1:
                        prev_t = crew_i[ti - 1]
                        next_t = crew_i[ti + 1]
                        if (prev_t, next_t) not in arc_set:
                            continue
                        if task_finish[prev_t] > task_start[next_t]:
                            continue
                    
                    if task_finish[new_crew_i[-1]] - task_start[new_crew_i[0]] > time_limit:
                        continue
                    
                    new_cost_i = crew_cost(new_crew_i)
                    saving_i = old_cost_i - new_cost_i
                    
                    ts_task = task_start[task]
                    tf_task = task_finish[task]
                    
                    # Try inserting into other crews
                    for cj in range(len(crews)):
                        if cj == ci:
                            continue
                        crew_j = crews[cj]
                        
                        for pos in range(len(crew_j) + 1):
                            prev_j = crew_j[pos - 1] if pos > 0 else None
                            next_j = crew_j[pos] if pos < len(crew_j) else None
                            
                            if prev_j is not None:
                                if task_finish[prev_j] > ts_task:
                                    continue
                                if (prev_j, task) not in arc_set:
                                    continue
                            if next_j is not None:
                                if tf_task > task_start[next_j]:
                                    continue
                                if (task, next_j) not in arc_set:
                                    continue
                            
                            # Duty time check
                            new_first = task_start[crew_j[0]] if pos > 0 else ts_task
                            new_last = task_finish[crew_j[-1]] if pos < len(crew_j) else tf_task
                            if new_last - new_first > time_limit:
                                continue
                            
                            ins_cost = 0.0
                            if prev_j is not None:
                                ins_cost += arcs[(prev_j, task)]
                            if next_j is not None:
                                ins_cost += arcs[(task, next_j)]
                            bridge = 0.0
                            if prev_j is not None and next_j is not None:
                                bridge = arcs.get((prev_j, next_j), 0.0)
                            
                            gain = saving_i - (ins_cost - bridge)
                            
                            is_tabu = tabu_key in tabu_list
                            aspiration = (current_cost - gain) < best_ts_cost
                            
                            if (not is_tabu or aspiration) and gain > best_move_gain:
                                best_move_gain = gain
                                best_move = (ci, ti, cj, pos)
            
            if best_move is None:
                no_improve += 1
                continue
            
            ci, ti, cj, pos = best_move
            task = crews[ci][ti]
            
            new_crew_i = crews[ci][:ti] + crews[ci][ti + 1:]
            new_crew_j = crews[cj][:pos] + [task] + crews[cj][pos:]
            
            old_cost_ij = crew_cost(crews[ci]) + crew_cost(crews[cj])
            new_cost_ij = crew_cost(new_crew_i) + crew_cost(new_crew_j)
            
            crews[ci] = new_crew_i
            crews[cj] = new_crew_j
            current_cost = current_cost - old_cost_ij + new_cost_ij
            
            # Add to tabu list
            tabu_list[(task, ci)] = iteration
            
            if current_cost < best_ts_cost - 1e-9:
                best_ts_cost = current_cost
                best_ts_crews = [list(c) for c in crews]
                no_improve = 0
            else:
                no_improve += 1
        
        return best_ts_crews, best_ts_cost

    # ------------------------------------------------------------------ #
    # Population-based search with diversity maintenance
    # ------------------------------------------------------------------ #
    def solution_similarity(sol1_crews, sol2_crews):
        """
        Measure similarity between two solutions as fraction of shared consecutive pairs.
        """
        pairs1 = set()
        for crew in sol1_crews:
            for idx in range(len(crew) - 1):
                pairs1.add((crew[idx], crew[idx + 1]))
        
        pairs2 = set()
        for crew in sol2_crews:
            for idx in range(len(crew) - 1):
                pairs2.add((crew[idx], crew[idx + 1]))
        
        if not pairs1 and not pairs2:
            return 1.0
        if not pairs1 or not pairs2:
            return 0.0
        
        intersection = len(pairs1 & pairs2)
        union = len(pairs1 | pairs2)
        return intersection / union if union > 0 else 0.0

    def population_search(initial_sol, time_budget, pop_size=8):
        """
        Maintain a population of solutions, combine diversity and quality.
        """
        ps_start = time.time()
        
        def ps_remaining():
            return time_budget - (time.time() - ps_start)
        
        rng = random.Random(777)
        
        # Initialize population
        population = []  # list of (cost, crews)
        
        if initial_sol is not None:
            init_crews = [list(c) for c in initial_sol['crews']]
            init_cost = sol_cost(init_crews)
            population.append((init_cost, init_crews))
        
        # Generate diverse initial solutions via ACO with different parameters
        aco_params = [
            (1.0, 1.0), (1.0, 3.0), (2.0, 1.0), (0.5, 2.0),
            (1.5, 1.5), (0.5, 3.0), (2.0, 2.0), (1.0, 0.5)
        ]
        
        for alpha_p, beta_p in aco_params:
            if ps_remaining() < 0.3:
                break
            sol = aco_construct_solution(rng, alpha_param=alpha_p, beta_param=beta_p)
            if sol is not None:
                cost = sol_cost(sol['crews'])
                population.append((cost, [list(c) for c in sol['crews']]))
        
        if not population:
            return None
        
        # Sort by cost
        population.sort(key=lambda x: x[0])
        population = population[:pop_size]
        
        best_pop_cost = population[0][0]
        best_pop_crews = [list(c) for c in population[0][1]]
        
        # Main population loop
        iteration = 0
        while ps_remaining() > 0.5:
            iteration += 1
            
            # Select solution for improvement (tournament)
            # Bias toward diverse solutions
            if len(population) >= 2 and rng.random() < 0.3:
                # Select most diverse from top half
                top_half = population[:max(2, len(population) // 2)]
                if len(top_half) >= 2:
                    idx1, idx2 = rng.sample(range(len(top_half)), 2)
                    sim = solution_similarity(top_half[idx1][1], top_half[idx2][1])
                    # Pick the more diverse one
                    selected = top_half[idx2][1] if rng.random() < 0.5 else top_half[idx1][1]
                else:
                    selected = top_half[0][1]
            else:
                # Tournament selection favoring quality
                candidates_idx = rng.sample(range(len(population)), min(3, len(population)))
                best_idx = min(candidates_idx, key=lambda i: population[i][0])
                selected = population[best_idx][1]
            
            # Apply tabu search to selected solution
            ts_budget = min(ps_remaining() * 0.3, 5.0)
            if ts_budget < 0.2:
                break
            
            improved_crews, improved_cost = tabu_search(selected, ts_budget)
            
            # Check if improved solution is valid
            all_tasks = set(t for c in improved_crews for t in c)
            if len(all_tasks) != N or len(improved_crews) > K:
                continue
            
            if not all(check_crew(c) for c in improved_crews):
                continue
            
            # Add to population if good enough
            if improved_cost < best_pop_cost - 1e-9:
                best_pop_cost = improved_cost
                best_pop_crews = [list(c) for c in improved_crews]
                
                # Verify with tools
                candidate = {'crews': [list(c) for c in improved_crews]}
                try:
                    feasible, _ = tools['is_feasible'](candidate)
                    if feasible:
                        obj = tools['objective'](candidate)
                        if obj < best_pop_cost:
                            best_pop_cost = obj
                except Exception:
                    pass
            
            # Insert into population if diverse enough
            min_sim = min(solution_similarity(improved_crews, p[1]) for p in population)
            if min_sim < 0.8 or improved_cost < population[-1][0]:
                population.append((improved_cost, [list(c) for c in improved_crews]))
                population.sort(key=lambda x: x[0])
                
                # Maintain population size with diversity
                if len(population) > pop_size:
                    # Remove most similar pair with worse quality
                    to_remove = -1
                    max_sim_score = -1
                    for i in range(len(population) - 1, 0, -1):
                        for j in range(i):
                            sim = solution_similarity(population[i][1], population[j][1])
                            # Score = similarity * (cost ratio vs best)
                            score = sim * (population[i][0] / max(population[0][0], 1e-6))
                            if score > max_sim_score:
                                max_sim_score = score
                                to_remove = i
                    if to_remove >= 0:
                        population.pop(to_remove)
            
            # Periodically inject new ACO solutions
            if iteration % 5 == 0 and ps_remaining() > 1.0:
                alpha_p = rng.uniform(0.5, 2.5)
                beta_p = rng.uniform(0.5, 3.0)
                new_sol = aco_construct_solution(rng, alpha_param=alpha_p, beta_param=beta_p)
                if new_sol is not None:
                    new_cost = sol_cost(new_sol['crews'])
                    population.append((new_cost, [list(c) for c in new_sol['crews']]))
                    population.sort(key=lambda x: x[0])
                    if len(population) > pop_size:
                        population.pop()
                
                # Update pheromones based on current population
                top_sols = [(({'crews': p[1]}, p[0])) for p in population[:3]]
                update_pheromones([(s, c) for s, c in top_sols], best_pop_cost)
        
        if best_pop_crews:
            return {'crews': [list(c) for c in best_pop_crews]}
        return None

    # ------------------------------------------------------------------ #
    # Chain merging: try to merge short chains
    # ------------------------------------------------------------------ #
    def merge_chains(crews):
        """
        Greedily merge chains that can be connected at low cost.
        Returns merged crews list.
        """
        merged = [list(c) for c in crews]
        changed = True
        
        while changed and len(merged) > 1:
            changed = False
            best_merge_gain = 1e-9
            best_i = -1
            best_j = -1
            
            for i in range(len(merged)):
                for j in range(len(merged)):
                    if i == j:
                        continue
                    # Try appending merged[j] after merged[i]
                    tail = merged[i][-1]
                    head = merged[j][0]
                    
                    if task_finish[tail] > task_start[head]:
                        continue
                    if (tail, head) not in arc_set:
                        continue
                    
                    merged_crew = merged[i] + merged[j]
                    if task_finish[merged_crew[-1]] - task_start[merged_crew[0]] > time_limit:
                        continue
                    
                    # Cost of merge arc
                    merge_cost = arcs[(tail, head)]
                    gain = -merge_cost  # We save nothing but add a connection; only merge if 0-cost
                    # Actually merge if it reduces crew count (allowing more flexibility)
                    # For now: merge if arc cost is 0 or very low
                    if merge_cost < 1e-6:
                        best_merge_gain = 1.0
                        best_i = i
                        best_j = j
                        break
                if best_i >= 0:
                    break
            
            if best_i >= 0 and best_j >= 0:
                new_crew = merged[best_i] + merged[best_j]
                new_merged = []
                for idx, c in enumerate(merged):
                    if idx == best_i:
                        new_merged.append(new_crew)
                    elif idx == best_j:
                        pass
                    else:
                        new_merged.append(c)
                merged = new_merged
                changed = True
        
        return merged

    # ------------------------------------------------------------------ #
    # Time-bucket decomposition for large instances
    # ------------------------------------------------------------------ #
    def time_bucket_solve():
        """
        Decompose the problem into time buckets and solve each bucket.
        Useful for large instances with many tasks.
        """
        if N <= 50:
            return None
        
        # Create time buckets based on task start times
        all_starts = sorted(set(task_start[t] for t in range(1, N + 1)))
        
        if len(all_starts) <= 3:
            return None
        
        # Split into ~4 buckets
        n_buckets = min(4, len(all_starts) // 2)
        bucket_size = len(all_starts) // n_buckets
        
        bucket_boundaries = []
        for b in range(n_buckets):
            start_idx = b * bucket_size
            bucket_boundaries.append(all_starts[start_idx])
        bucket_boundaries.append(float('inf'))
        
        # Assign tasks to buckets
        buckets = [[] for _ in range(n_buckets)]
        for t in range(1, N + 1):
            for b in range(n_buckets):
                if bucket_boundaries[b] <= task_start[t] < bucket_boundaries[b + 1]:
                    buckets[b].append(t)
                    break
        
        # Build chains per bucket using DP
        all_chains = []
        for bucket in buckets:
            if not bucket:
                continue
            bucket_set = set(bucket)
            chains = dp_build_chains(bucket_set, K)
            if chains is not None:
                all_chains.extend(chains)
            else:
                # Fall back: each task in own chain
                for t in bucket:
                    all_chains.append([t])
        
        if len(all_chains) > K:
            # Try to merge chains
            all_chains = merge_chains(all_chains)
        
        if len(all_chains) > K:
            return None
        
        # Verify
        all_tasks_in = set(t for c in all_chains for t in c)
        if len(all_tasks_in) != N:
            return None
        
        if not all(check_crew(c) for c in all_chains):
            return None
        
        return {'crews': all_chains}

    # ------------------------------------------------------------------ #
    # Main algorithm execution
    # ------------------------------------------------------------------ #
    rng = random.Random(42)

    # Try time-bucket decomposition for large instances
    if N > 50 and remaining() > 2:
        try:
            sol = time_bucket_solve()
            update_best(sol)
        except Exception:
            pass

    # Run ACO construction phase
    aco_time = min(remaining() * 0.20, time_limit_s * 0.20)
    aco_end = time.time() + aco_time
    aco_solutions = []
    
    while time.time() < aco_end and remaining() > 0.5:
        alpha_p = rng.uniform(0.5, 2.5)
        beta_p = rng.uniform(0.5, 3.5)
        sol = aco_construct_solution(rng, alpha_param=alpha_p, beta_param=beta_p)
        if sol is not None:
            cost = sol_cost(sol['crews'])
            aco_solutions.append((sol, cost))
            update_best(sol)
    
    # Update pheromones based on ACO solutions
    if aco_solutions:
        aco_solutions.sort(key=lambda x: x[1])
        top_aco = aco_solutions[:3]
        update_pheromones([(s, c) for s, c in top_aco], best_cost)

    # Run tabu search on best solution so far
    if best_sol is not None and remaining() > 1.5:
        ts_budget = min(remaining() * 0.25, time_limit_s * 0.25)
        try:
            ts_crews, ts_cost = tabu_search(best_sol['crews'], ts_budget, tabu_tenure=10)
            if ts_crews and len(ts_crews) <= K:
                all_t = set(t for c in ts_crews for t in c)
                if len(all_t) == N and all(check_crew(c) for c in ts_crews):
                    sol = {'crews': ts_crews}
                    update_best(sol)
        except Exception:
            pass

    # Run population-based search
    if remaining() > 2.0:
        pop_budget = remaining() - 0.5
        try:
            pop_sol = population_search(best_sol, pop_budget, pop_size=6)
            update_best(pop_sol)
        except Exception:
            pass

    # Final tabu polish
    if best_sol is not None and remaining() > 0.8:
        final_budget = remaining() - 0.3
        try:
            final_crews, final_cost = tabu_search(best_sol['crews'], final_budget, tabu_tenure=5)
            if final_crews and len(final_crews) <= K:
                all_t = set(t for c in final_crews for t in c)
                if len(all_t) == N and all(check_crew(c) for c in final_crews):
                    sol = {'crews': final_crews}
                    update_best(sol)
        except Exception:
            pass

    # Emergency fallback
    if best_sol is None:
        if K >= N:
            best_sol = {'crews': [[i] for i in range(1, N + 1)]}
        else:
            task_order = sorted(range(1, N + 1), key=lambda t: task_start[t])
            crews_e = [[] for _ in range(min(K, N))]
            for idx, t in enumerate(task_order):
                crews_e[idx % len(crews_e)].append(t)
            best_sol = {'crews': [c for c in crews_e if c]}

    # Verify final solution
    try:
        feasible, _ = tools['is_feasible'](best_sol)
        if feasible:
            return best_sol
    except Exception:
        pass

    # Last resort: return any feasible solution from tools
    try:
        sol = tools['solve_default'](time_limit_s=min(remaining() - 0.1, 5.0))
        if sol is not None:
            feasible, _ = tools['is_feasible'](sol)
            if feasible:
                return sol
    except Exception:
        pass

    return best_sol