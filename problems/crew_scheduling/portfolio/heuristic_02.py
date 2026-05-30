# MACE evolved heuristic 02/10 for problem: crew_scheduling
import time
import math
import random
import heapq
from collections import defaultdict


def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Crew Scheduling via Large Neighborhood Search + Simulated Annealing.
    
    Core differences from portfolio:
    1. LNS destroy/repair instead of move-based local search
    2. Simulated annealing acceptance (not pure greedy descent)
    3. DP-based chain repair (shortest path on task DAG)
    4. Regret-based insertion for repair phase
    5. Destroy targets highest-cost crews preferentially
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
    # Sorted successors for each task
    succ = defaultdict(list)  # task -> [(start_j, j, cost)]
    pred = defaultdict(list)  # task -> [(finish_i, i, cost)]

    for (i, j), cost in arcs.items():
        if tasks[i][1] <= tasks[j][0]:
            succ[i].append((tasks[j][0], j, cost))
            pred[j].append((tasks[i][1], i, cost))

    for t in range(1, N + 1):
        succ[t].sort()
        pred[t].sort()

    # Tasks sorted by start time
    tasks_by_start = sorted(range(1, N + 1), key=lambda t: tasks[t][0])
    task_start_rank = {t: i for i, t in enumerate(tasks_by_start)}

    # ------------------------------------------------------------------ #
    # Fast feasibility helpers
    # ------------------------------------------------------------------ #
    def check_crew(crew):
        if not crew:
            return False
        if tasks[crew[-1]][1] - tasks[crew[0]][0] > time_limit:
            return False
        for idx in range(len(crew) - 1):
            t1, t2 = crew[idx], crew[idx + 1]
            if tasks[t1][1] > tasks[t2][0]:
                return False
            if (t1, t2) not in arcs:
                return False
        return True

    def crew_cost_fn(crew):
        cost = 0.0
        for idx in range(len(crew) - 1):
            key = (crew[idx], crew[idx + 1])
            if key not in arcs:
                return float('inf')
            cost += arcs[key]
        return cost

    def total_cost_fn(crews_list):
        return sum(crew_cost_fn(c) for c in crews_list)

    # ------------------------------------------------------------------ #
    # Phase 1: Get initial solution
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

    # Use solve_default for initial solution (50% of budget)
    init_budget = min(remaining() * 0.50, time_limit_s * 0.50)
    if init_budget > 0.5:
        try:
            sol = tools['solve_default'](time_limit_s=init_budget)
            update_best(sol)
        except Exception:
            pass

    # Greedy fallback
    if best_sol is None and remaining() > 1:
        try:
            chains = tools['greedy_chain_pack']()
            if chains and len(chains) <= K:
                sol = tools['make_solution'](chains)
                update_best(sol)
        except Exception:
            pass

    if best_sol is None and remaining() > 1:
        sol = _greedy_construct(N, K, tasks, arcs, time_limit)
        update_best(sol)

    if best_sol is None:
        if K >= N:
            best_sol = {'crews': [[i] for i in range(1, N + 1)]}
        else:
            task_order = sorted(range(1, N + 1), key=lambda t: tasks[t][0])
            crews_e = [[] for _ in range(K)]
            for idx, t in enumerate(task_order):
                crews_e[idx % K].append(t)
            best_sol = {'crews': [c for c in crews_e if c]}

    # ------------------------------------------------------------------ #
    # Phase 2: LNS + Simulated Annealing
    # ------------------------------------------------------------------ #
    if remaining() > 1.0:
        best_sol = _lns_sa(
            best_sol, N, K, tasks, arcs, time_limit,
            succ, pred, tasks_by_start, tools,
            start_time, time_limit_s
        )

    return best_sol


def _greedy_construct(N, K, tasks, arcs, time_limit):
    """Greedy construction by start time."""
    task_order = sorted(range(1, N + 1), key=lambda t: tasks[t][0])
    crews = []
    crew_last = []
    crew_start = []
    crew_finish = []

    for task in task_order:
        t_start, t_finish = tasks[task]
        best_crew = -1
        best_cost_val = float('inf')

        for i in range(len(crews)):
            last = crew_last[i]
            if crew_finish[i] > t_start:
                continue
            if (last, task) not in arcs:
                continue
            if t_finish - crew_start[i] > time_limit:
                continue
            c = arcs[(last, task)]
            if c < best_cost_val:
                best_cost_val = c
                best_crew = i

        if best_crew >= 0:
            crews[best_crew].append(task)
            crew_last[best_crew] = task
            crew_finish[best_crew] = t_finish
        elif len(crews) < K:
            crews.append([task])
            crew_last.append(task)
            crew_start.append(t_start)
            crew_finish.append(t_finish)
        else:
            forced = False
            for i in range(len(crews)):
                last = crew_last[i]
                if crew_finish[i] <= t_start and (last, task) in arcs:
                    if t_finish - crew_start[i] <= time_limit:
                        crews[i].append(task)
                        crew_last[i] = task
                        crew_finish[i] = t_finish
                        forced = True
                        break
            if not forced:
                return None

    return {'crews': crews}


def _lns_sa(initial_sol, N, K, tasks, arcs, time_limit,
            succ, pred, tasks_by_start, tools,
            start_time, time_limit_s):
    """
    Large Neighborhood Search with Simulated Annealing acceptance.
    
    Destroy: remove tasks from selected crews (segment or random removal)
    Repair: DP-based shortest path chain construction for unassigned tasks
    Accept: Simulated annealing (accept worse solutions with decreasing probability)
    """
    def remaining():
        return time_limit_s - (time.time() - start_time)

    def check_crew(crew):
        if not crew:
            return False
        if tasks[crew[-1]][1] - tasks[crew[0]][0] > time_limit:
            return False
        for idx in range(len(crew) - 1):
            t1, t2 = crew[idx], crew[idx + 1]
            if tasks[t1][1] > tasks[t2][0]:
                return False
            if (t1, t2) not in arcs:
                return False
        return True

    def crew_cost_fn(crew):
        cost = 0.0
        for idx in range(len(crew) - 1):
            key = (crew[idx], crew[idx + 1])
            if key not in arcs:
                return float('inf')
            cost += arcs[key]
        return cost

    def total_cost_fn(crews_list):
        return sum(crew_cost_fn(c) for c in crews_list)

    # ------------------------------------------------------------------ #
    # DP-based chain builder: given a set of tasks to cover, build
    # minimum-cost chains using shortest path on the task DAG
    # ------------------------------------------------------------------ #
    def dp_build_chains(task_set, max_chains):
        """
        Build minimum-cost chains covering all tasks in task_set.
        Uses a greedy DP approach: for each task in start-time order,
        extend the cheapest valid chain or start a new one.
        
        Returns list of chains or None if infeasible.
        """
        if not task_set:
            return []

        ordered = sorted(task_set, key=lambda t: tasks[t][0])

        # dp[task] = (min_cost_to_reach, prev_task_or_None)
        # We use a priority queue to build optimal chains greedily
        # Strategy: build chains using Dijkstra-like approach on DAG

        # For each task, find best predecessor within task_set
        # dp_cost[t] = min cost of chain ending at t
        # dp_prev[t] = predecessor of t in optimal chain
        INF = float('inf')
        dp_cost = {t: 0.0 for t in task_set}  # cost to start new chain at t
        dp_prev = {t: None for t in task_set}

        # Process in topological order (by start time, then finish time)
        for t in ordered:
            ts_t, tf_t = tasks[t]
            # Try all predecessors in task_set
            for (_, finish_i, i, cost_i) in sorted(
                [(tasks[i][0], tasks[i][1], i, arcs.get((i, t), INF))
                 for i in task_set if i != t
                 and tasks[i][1] <= ts_t
                 and (i, t) in arcs],
                key=lambda x: x[0]
            ):
                new_cost = dp_cost[i] + cost_i
                # Check duty time
                # Find chain head of i
                head = i
                h = dp_prev[i]
                visited = set()
                while h is not None and h not in visited:
                    visited.add(head)
                    head = h
                    h = dp_prev[h]
                duty = tasks[t][1] - tasks[head][0]
                if duty > time_limit:
                    continue
                if new_cost < dp_cost[t]:
                    dp_cost[t] = new_cost
                    dp_prev[t] = i

        # Reconstruct chains
        in_chain = set()
        chain_ends = []  # tasks that are chain ends (no successor in dp)

        # Find which tasks are "pointed to" by others
        has_successor = set()
        for t in task_set:
            if dp_prev[t] is not None:
                has_successor.add(dp_prev[t])

        # Chain ends: tasks with no successor in dp
        for t in task_set:
            if t not in has_successor:
                chain_ends.append(t)

        chains = []
        for end in chain_ends:
            chain = []
            cur = end
            visited = set()
            while cur is not None and cur not in visited:
                visited.add(cur)
                chain.append(cur)
                cur = dp_prev[cur]
            chain.reverse()
            if check_crew(chain):
                chains.append(chain)
            else:
                # Fall back: each task in its own chain
                for t in chain:
                    chains.append([t])

        if len(chains) > max_chains:
            return None  # Can't fit in max_chains

        return chains

    # ------------------------------------------------------------------ #
    # Regret-based insertion repair
    # ------------------------------------------------------------------ #
    def regret_repair(unassigned, current_crews, max_total_crews):
        """
        Insert unassigned tasks using regret-2 heuristic.
        Regret = (2nd best insertion cost) - (best insertion cost)
        Tasks with high regret are inserted first.
        
        Returns updated crews or None if infeasible.
        """
        if not unassigned:
            return current_crews

        crews = [list(c) for c in current_crews]
        remaining_tasks = list(unassigned)

        while remaining_tasks:
            # Compute best and 2nd best insertion for each task
            regrets = []

            for task in remaining_tasks:
                ts_t, tf_t = tasks[task]
                insertions = []  # (cost_delta, crew_idx, pos)

                for ci, crew in enumerate(crews):
                    for pos in range(len(crew) + 1):
                        # Temporal check
                        if pos > 0 and tasks[crew[pos - 1]][1] > ts_t:
                            continue
                        if pos < len(crew) and tf_t > tasks[crew[pos]][0]:
                            continue

                        # Arc check
                        cost_add = 0.0
                        ok = True
                        if pos > 0:
                            c = arcs.get((crew[pos - 1], task))
                            if c is None:
                                ok = False
                            else:
                                cost_add += c
                        if ok and pos < len(crew):
                            c = arcs.get((task, crew[pos]))
                            if c is None:
                                ok = False
                            else:
                                cost_add += c
                        if not ok:
                            continue

                        # Bridge removal
                        bridge = 0.0
                        if pos > 0 and pos < len(crew):
                            bridge = arcs.get((crew[pos - 1], crew[pos]), 0.0)

                        # Duty time check
                        new_crew = crew[:pos] + [task] + crew[pos:]
                        dt = tasks[new_crew[-1]][1] - tasks[new_crew[0]][0]
                        if dt > time_limit:
                            continue

                        delta = cost_add - bridge
                        insertions.append((delta, ci, pos))

                # Also consider opening a new crew
                if len(crews) < max_total_crews:
                    insertions.append((0.0, -1, 0))  # new crew, cost = 0

                if not insertions:
                    return None  # Can't insert this task

                insertions.sort(key=lambda x: x[0])
                best_cost_ins = insertions[0][0]
                second_cost_ins = insertions[1][0] if len(insertions) > 1 else best_cost_ins + 1e9

                regret_val = second_cost_ins - best_cost_ins
                regrets.append((regret_val, task, insertions[0]))

            # Insert task with highest regret
            regrets.sort(key=lambda x: -x[0])
            _, task_to_insert, (delta, ci, pos) = regrets[0]

            if ci == -1:
                # Open new crew
                crews.append([task_to_insert])
            else:
                crews[ci] = crews[ci][:pos] + [task_to_insert] + crews[ci][pos:]

            remaining_tasks.remove(task_to_insert)

        return crews

    # ------------------------------------------------------------------ #
    # Greedy repair: insert by cheapest position
    # ------------------------------------------------------------------ #
    def greedy_repair(unassigned, current_crews, max_total_crews):
        """Simple greedy repair: insert each task at cheapest valid position."""
        if not unassigned:
            return current_crews

        crews = [list(c) for c in current_crews]
        # Sort unassigned by start time
        task_order = sorted(unassigned, key=lambda t: tasks[t][0])

        for task in task_order:
            ts_t, tf_t = tasks[task]
            best_delta = float('inf')
            best_ci = -1
            best_pos = -1

            for ci, crew in enumerate(crews):
                for pos in range(len(crew) + 1):
                    if pos > 0 and tasks[crew[pos - 1]][1] > ts_t:
                        continue
                    if pos < len(crew) and tf_t > tasks[crew[pos]][0]:
                        continue

                    cost_add = 0.0
                    ok = True
                    if pos > 0:
                        c = arcs.get((crew[pos - 1], task))
                        if c is None:
                            ok = False
                        else:
                            cost_add += c
                    if ok and pos < len(crew):
                        c = arcs.get((task, crew[pos]))
                        if c is None:
                            ok = False
                        else:
                            cost_add += c
                    if not ok:
                        continue

                    bridge = 0.0
                    if pos > 0 and pos < len(crew):
                        bridge = arcs.get((crew[pos - 1], crew[pos]), 0.0)

                    new_crew = crew[:pos] + [task] + crew[pos:]
                    dt = tasks[new_crew[-1]][1] - tasks[new_crew[0]][0]
                    if dt > time_limit:
                        continue

                    delta = cost_add - bridge
                    if delta < best_delta:
                        best_delta = delta
                        best_ci = ci
                        best_pos = pos

            if best_ci >= 0:
                crews[best_ci] = crews[best_ci][:best_pos] + [task] + crews[best_ci][best_pos:]
            elif len(crews) < max_total_crews:
                crews.append([task])
            else:
                return None  # Can't insert

        return crews

    # ------------------------------------------------------------------ #
    # Destroy operators
    # ------------------------------------------------------------------ #
    def destroy_random_tasks(crews, n_remove, rng):
        """Remove n_remove random tasks from crews."""
        all_tasks = [(ci, pos, t)
                     for ci, crew in enumerate(crews)
                     for pos, t in enumerate(crew)]
        if not all_tasks:
            return crews, []

        n_remove = min(n_remove, len(all_tasks))
        removed_entries = rng.sample(all_tasks, n_remove)
        removed_tasks = [t for _, _, t in removed_entries]
        removed_set = set(removed_tasks)

        new_crews = []
        for crew in crews:
            new_crew = [t for t in crew if t not in removed_set]
            if new_crew:
                new_crews.append(new_crew)

        return new_crews, removed_tasks

    def destroy_worst_crew(crews, n_crews_to_destroy, rng):
        """Remove tasks from the most expensive crews."""
        if not crews:
            return crews, []

        crew_costs = [(crew_cost_fn(crew), ci) for ci, crew in enumerate(crews)]
        crew_costs.sort(reverse=True)

        n_destroy = min(n_crews_to_destroy, len(crews))
        destroyed_indices = set(ci for _, ci in crew_costs[:n_destroy])

        removed_tasks = []
        new_crews = []
        for ci, crew in enumerate(crews):
            if ci in destroyed_indices:
                removed_tasks.extend(crew)
            else:
                new_crews.append(crew)

        return new_crews, removed_tasks

    def destroy_segment(crews, rng):
        """Remove a contiguous segment from a randomly chosen crew."""
        if not crews:
            return crews, []

        # Pick crew weighted by cost
        crew_costs = [max(crew_cost_fn(crew), 0.01) for crew in crews]
        total_w = sum(crew_costs)
        probs = [c / total_w for c in crew_costs]

        r = rng.random()
        cumulative = 0.0
        ci = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                ci = i
                break

        crew = crews[ci]
        if len(crew) <= 1:
            return crews, []

        # Remove a random contiguous segment
        seg_len = rng.randint(1, max(1, len(crew) // 2))
        start_pos = rng.randint(0, len(crew) - seg_len)
        removed = crew[start_pos:start_pos + seg_len]
        new_crew = crew[:start_pos] + crew[start_pos + seg_len:]

        new_crews = []
        for i, c in enumerate(crews):
            if i == ci:
                if new_crew:
                    new_crews.append(new_crew)
            else:
                new_crews.append(c)

        return new_crews, removed

    def destroy_related_tasks(crews, n_remove, rng):
        """
        Remove tasks that are 'related' (share arcs or close in time).
        Starts from a random task, then removes its neighbors.
        """
        all_tasks_flat = [t for crew in crews for t in crew]
        if not all_tasks_flat:
            return crews, []

        seed_task = rng.choice(all_tasks_flat)
        removed_set = {seed_task}

        # Expand by relatedness: tasks connected by arcs to removed tasks
        candidates = set()
        for t in removed_set:
            for _, j, _ in succ[t]:
                if j in all_tasks_flat:
                    candidates.add(j)
            for _, i, _ in pred[t]:
                if i in all_tasks_flat:
                    candidates.add(i)

        # Also add temporally close tasks
        seed_start = tasks[seed_task][0]
        for t in all_tasks_flat:
            if abs(tasks[t][0] - seed_start) < (time_limit * 0.1):
                candidates.add(t)

        candidates -= removed_set
        candidates = list(candidates)
        rng.shuffle(candidates)

        n_remove = min(n_remove, len(all_tasks_flat))
        for t in candidates:
            if len(removed_set) >= n_remove:
                break
            removed_set.add(t)

        removed_tasks = list(removed_set)
        new_crews = []
        for crew in crews:
            new_crew = [t for t in crew if t not in removed_set]
            if new_crew:
                new_crews.append(new_crew)

        return new_crews, removed_tasks

    # ------------------------------------------------------------------ #
    # Main LNS + SA loop
    # ------------------------------------------------------------------ #
    current_crews = [list(c) for c in initial_sol['crews']]
    current_cost = total_cost_fn(current_crews)
    best_crews = [list(c) for c in current_crews]
    best_cost = current_cost

    rng = random.Random(42)

    # SA parameters
    # Initial temperature: accept 5% worse solutions with ~50% probability
    # T = -delta / ln(0.5) => T = delta / ln(2)
    # Estimate initial delta as ~5% of current cost
    if current_cost > 0 and current_cost < float('inf'):
        T_init = max(0.05 * current_cost / math.log(2), 1.0)
    else:
        T_init = 100.0

    T_min = 1e-4
    T = T_init

    # Cooling: geometric cooling
    # We want T to reach T_min after ~80% of iterations
    # Estimate iterations based on remaining time
    ls_time = remaining() - 0.5
    if ls_time <= 0:
        return initial_sol

    # Estimate ~100 iterations per second for LNS
    estimated_iters = max(50, int(ls_time * 30))
    cooling_rate = (T_min / T_init) ** (1.0 / max(1, estimated_iters))

    # Adaptive destroy size
    n_destroy_base = max(1, N // 10)

    iteration = 0
    no_improve_streak = 0

    # Destroy operator selection (adaptive weights)
    destroy_ops = ['random', 'worst_crew', 'segment', 'related']
    op_weights = [1.0, 1.0, 1.0, 1.0]
    op_success = [0, 0, 0, 0]
    op_tries = [1, 1, 1, 1]

    def select_destroy_op():
        # UCB-style selection
        total_tries = sum(op_tries)
        scores = []
        for i in range(len(destroy_ops)):
            exploit = op_success[i] / op_tries[i]
            explore = math.sqrt(2 * math.log(total_tries) / op_tries[i])
            scores.append(exploit + 0.3 * explore)
        return scores.index(max(scores))

    while remaining() > 0.5:
        iteration += 1
        T = max(T * cooling_rate, T_min)

        # Adaptive destroy size: increase when stuck
        n_destroy = n_destroy_base + min(no_improve_streak // 5, N // 5)
        n_destroy = min(n_destroy, max(1, N // 2))

        # Select destroy operator
        op_idx = select_destroy_op()
        op_tries[op_idx] += 1
        destroy_op = destroy_ops[op_idx]

        # Destroy
        if destroy_op == 'random':
            partial_crews, removed = destroy_random_tasks(current_crews, n_destroy, rng)
        elif destroy_op == 'worst_crew':
            n_crews_destroy = max(1, min(3, len(current_crews) // 3))
            partial_crews, removed = destroy_worst_crew(current_crews, n_crews_destroy, rng)
        elif destroy_op == 'segment':
            partial_crews, removed = destroy_segment(current_crews, rng)
        else:  # related
            partial_crews, removed = destroy_related_tasks(current_crews, n_destroy, rng)

        if not removed:
            continue

        # Repair: alternate between regret and greedy repair
        max_crews = K
        if iteration % 3 == 0:
            # Use regret repair (slower but better)
            if len(removed) <= 20:  # Only for small sets (regret is O(n^2))
                new_crews = regret_repair(removed, partial_crews, max_crews)
            else:
                new_crews = greedy_repair(removed, partial_crews, max_crews)
        else:
            new_crews = greedy_repair(removed, partial_crews, max_crews)

        if new_crews is None:
            continue

        # Verify all tasks present
        all_covered = set(t for crew in new_crews for t in crew)
        if len(all_covered) != N:
            # Try to add missing tasks
            missing = set(range(1, N + 1)) - all_covered
            new_crews = greedy_repair(list(missing), new_crews, max_crews)
            if new_crews is None:
                continue
            all_covered2 = set(t for crew in new_crews for t in crew)
            if len(all_covered2) != N:
                continue

        if len(new_crews) > K:
            continue

        # Quick validity check
        all_valid = True
        for crew in new_crews:
            if not check_crew(crew):
                all_valid = False
                break
        if not all_valid:
            continue

        new_cost = total_cost_fn(new_crews)

        # SA acceptance
        delta = new_cost - current_cost
        accept = False
        if delta < 0:
            accept = True
        elif T > T_min * 10:
            prob = math.exp(-delta / T)
            accept = rng.random() < prob

        if accept:
            current_crews = new_crews
            current_cost = new_cost

            if new_cost < best_cost - 1e-9:
                # Verify with tools
                candidate = {'crews': [list(c) for c in new_crews]}
                try:
                    feasible, _ = tools['is_feasible'](candidate)
                    if feasible:
                        obj = tools['objective'](candidate)
                        if obj < best_cost:
                            best_cost = obj
                            best_crews = [list(c) for c in new_crews]
                            op_success[op_idx] += 1
                            no_improve_streak = 0
                except Exception:
                    pass
            else:
                no_improve_streak += 1
        else:
            no_improve_streak += 1

        # Periodically reset to best (diversification restart)
        if no_improve_streak > 50 and remaining() > 2.0:
            current_crews = [list(c) for c in best_crews]
            current_cost = best_cost
            no_improve_streak = 0
            # Reheat slightly
            T = max(T, T_init * 0.1)

    # Final solution
    result = {'crews': [list(c) for c in best_crews if c]}
    try:
        feasible, _ = tools['is_feasible'](result)
        if feasible:
            return result
    except Exception:
        pass

    return initial_sol