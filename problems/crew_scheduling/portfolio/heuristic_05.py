# MACE evolved heuristic 05/10 for problem: crew_scheduling
import time
import math
import random
import heapq
from collections import defaultdict


def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
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
    # Instance feature analysis for dispatch
    # ------------------------------------------------------------------ #
    n_arcs = len(arcs)
    arc_density = n_arcs / max(1, N * (N - 1))
    k_ratio = K / max(1, N)
    
    # Average successors per task
    succ_counts = defaultdict(int)
    for (i, j) in arcs.keys():
        if tasks[i][1] <= tasks[j][0]:
            succ_counts[i] += 1
    avg_successors = sum(succ_counts.values()) / max(1, N)
    
    # Time span analysis
    all_starts = [tasks[t][0] for t in range(1, N + 1)]
    all_finishes = [tasks[t][1] for t in range(1, N + 1)]
    time_span = max(all_finishes) - min(all_starts)
    avg_task_duration = sum(tasks[t][1] - tasks[t][0] for t in range(1, N + 1)) / max(1, N)
    
    # Regime decision:
    # Use A-style (LNS+SA) when:
    #   - N is large (> 150): LNS scales better
    #   - K/N ratio is small (tight packing): needs global repair
    #   - arc density is high: many repair options available
    # Use B-style (move-based LS) when:
    #   - N is small (<= 150): exhaustive moves feasible
    #   - K/N ratio is generous: fine-tuning works well
    #   - arc density is low: moves must be careful
    
    use_lns_style = (
        N > 150 or
        k_ratio < 0.3 or
        (avg_successors > 3.0 and N > 80)
    )
    
    # ------------------------------------------------------------------ #
    # Shared data structures
    # ------------------------------------------------------------------ #
    arc_set = set(arcs.keys())
    
    succ_list = defaultdict(list)
    pred_list = defaultdict(list)
    for (i, j), cost in arcs.items():
        if tasks[i][1] <= tasks[j][0]:
            succ_list[i].append(j)
            pred_list[j].append(i)
    for t in range(1, N + 1):
        succ_list[t].sort(key=lambda x: tasks[x][0])
        pred_list[t].sort(key=lambda x: tasks[x][1])

    task_start = {t: tasks[t][0] for t in range(1, N + 1)}
    task_finish = {t: tasks[t][1] for t in range(1, N + 1)}

    def crew_cost_fast(crew):
        cost = 0.0
        for idx in range(len(crew) - 1):
            c = arcs.get((crew[idx], crew[idx + 1]))
            if c is None:
                return float('inf')
            cost += c
        return cost

    def check_crew_fast(crew):
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

    def solution_cost_fast(crews):
        return sum(crew_cost_fast(c) for c in crews)

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
    # Phase 1: Initial solution construction
    # ------------------------------------------------------------------ #
    if use_lns_style:
        # A-style: use solve_default with generous budget
        init_budget = min(remaining() * 0.50, time_limit_s * 0.50)
        if init_budget > 0.5:
            try:
                sol = tools['solve_default'](time_limit_s=init_budget)
                update_best(sol)
            except Exception:
                pass
    else:
        # B-style: try MCF first, then ILP for small instances
        if remaining() > 3:
            mcf_budget = min(remaining() * 0.25, 15.0)
            try:
                sol = tools['solve_min_cost_flow'](time_limit_s=mcf_budget)
                update_best(sol)
            except Exception:
                pass

        if remaining() > 5 and N <= 200:
            ilp_budget = min(remaining() * 0.35, 40.0)
            try:
                sol = tools['ilp_crew_scheduling'](time_limit_s=ilp_budget)
                update_best(sol)
            except Exception:
                pass

        if best_sol is None and remaining() > 3:
            sd_budget = min(remaining() * 0.25, 15.0)
            try:
                sol = tools['solve_default'](time_limit_s=sd_budget)
                update_best(sol)
            except Exception:
                pass

    # Greedy chain pack fallback
    if remaining() > 1:
        try:
            chains = tools['greedy_chain_pack']()
            if chains and len(chains) <= K:
                sol = tools['make_solution'](chains)
                update_best(sol)
        except Exception:
            pass

    # Custom greedy construction
    def greedy_construct(shuffle_seed=None, alpha=0.0):
        task_order = sorted(range(1, N + 1), key=lambda t: task_start[t])
        rng_g = random.Random(shuffle_seed) if shuffle_seed is not None else None

        crews_g = []
        crew_last = []
        crew_start_t = []
        crew_finish_t = []

        for task in task_order:
            t_start = task_start[task]
            t_finish = task_finish[task]

            candidates = []
            for i in range(len(crews_g)):
                last = crew_last[i]
                if crew_finish_t[i] > t_start:
                    continue
                if (last, task) not in arc_set:
                    continue
                if t_finish - crew_start_t[i] > time_limit:
                    continue
                c = arcs[(last, task)]
                candidates.append((c, i))

            if candidates:
                if rng_g is not None and alpha > 0 and len(candidates) > 1:
                    candidates.sort()
                    min_c = candidates[0][0]
                    max_c = candidates[-1][0]
                    threshold = min_c + alpha * (max_c - min_c)
                    rcl = [i for c, i in candidates if c <= threshold]
                    chosen = rng_g.choice(rcl)
                else:
                    chosen = min(candidates, key=lambda x: x[0])[1]
                crews_g[chosen].append(task)
                crew_last[chosen] = task
                crew_finish_t[chosen] = t_finish
            elif len(crews_g) < K:
                crews_g.append([task])
                crew_last.append(task)
                crew_start_t.append(t_start)
                crew_finish_t.append(t_finish)
            else:
                forced = False
                for i in range(len(crews_g)):
                    last = crew_last[i]
                    if crew_finish_t[i] <= t_start and (last, task) in arc_set:
                        if t_finish - crew_start_t[i] <= time_limit:
                            crews_g[i].append(task)
                            crew_last[i] = task
                            crew_finish_t[i] = t_finish
                            forced = True
                            break
                if not forced:
                    return None

        if not crews_g:
            return None
        sol = {'crews': crews_g}
        try:
            feasible, _ = tools['is_feasible'](sol)
            return sol if feasible else None
        except Exception:
            return None

    if remaining() > 0.5:
        sol = greedy_construct()
        update_best(sol)

    for seed in range(3):
        if remaining() < 0.3:
            break
        for alpha in [0.1, 0.3]:
            if remaining() < 0.2:
                break
            sol = greedy_construct(shuffle_seed=seed, alpha=alpha)
            update_best(sol)

    # Emergency fallback
    if best_sol is None:
        if K >= N:
            sol = {'crews': [[i] for i in range(1, N + 1)]}
            update_best(sol)

    if best_sol is None:
        task_order = sorted(range(1, N + 1), key=lambda t: task_start[t])
        crews_em = [[] for _ in range(min(K, N))]
        for idx, task in enumerate(task_order):
            crews_em[idx % len(crews_em)].append(task)
        sol = {'crews': [c for c in crews_em if c]}
        update_best(sol)

    if best_sol is None:
        return {'crews': [[i] for i in range(1, min(K + 1, N + 1))]}

    # ================================================================== #
    # Phase 2: Optimization - dispatch to A-style or B-style
    # ================================================================== #

    if use_lns_style:
        # ---------------------------------------------------------------- #
        # A-style: LNS + Simulated Annealing
        # ---------------------------------------------------------------- #
        result = _run_lns_sa(
            best_sol, N, K, tasks, arcs, time_limit,
            succ_list, pred_list, arc_set,
            task_start, task_finish,
            tools, start_time, time_limit_s
        )
        if result is not None:
            update_best(result)
    else:
        # ---------------------------------------------------------------- #
        # B-style: Move-based Local Search + SA
        # ---------------------------------------------------------------- #
        result = _run_move_ls_sa(
            best_sol, N, K, tasks, arcs, time_limit,
            arc_set, task_start, task_finish,
            tools, start_time, time_limit_s
        )
        if result is not None:
            update_best(result)

    return best_sol if best_sol is not None else {'crews': [[i] for i in range(1, min(K + 1, N + 1))]}


# ====================================================================== #
# A-style: LNS + SA helpers
# ====================================================================== #

def _run_lns_sa(initial_sol, N, K, tasks, arcs, time_limit,
                succ_list, pred_list, arc_set,
                task_start, task_finish,
                tools, start_time, time_limit_s):

    def remaining():
        return time_limit_s - (time.time() - start_time)

    def crew_cost_fn(crew):
        cost = 0.0
        for idx in range(len(crew) - 1):
            c = arcs.get((crew[idx], crew[idx + 1]))
            if c is None:
                return float('inf')
            cost += c
        return cost

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

    def total_cost_fn(crews_list):
        return sum(crew_cost_fn(c) for c in crews_list)

    def greedy_repair(unassigned, current_crews, max_total_crews):
        if not unassigned:
            return current_crews
        crews = [list(c) for c in current_crews]
        task_order = sorted(unassigned, key=lambda t: task_start[t])

        for task in task_order:
            ts_t = task_start[task]
            tf_t = task_finish[task]
            best_delta = float('inf')
            best_ci = -1
            best_pos = -1

            for ci, crew in enumerate(crews):
                for pos in range(len(crew) + 1):
                    if pos > 0 and task_finish[crew[pos - 1]] > ts_t:
                        continue
                    if pos < len(crew) and tf_t > task_start[crew[pos]]:
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
                    dt = task_finish[new_crew[-1]] - task_start[new_crew[0]]
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
                return None

        return crews

    def regret_repair(unassigned, current_crews, max_total_crews):
        if not unassigned:
            return current_crews
        if len(unassigned) > 25:
            return greedy_repair(unassigned, current_crews, max_total_crews)

        crews = [list(c) for c in current_crews]
        remaining_tasks = list(unassigned)

        while remaining_tasks:
            regrets = []
            for task in remaining_tasks:
                ts_t = task_start[task]
                tf_t = task_finish[task]
                insertions = []

                for ci, crew in enumerate(crews):
                    for pos in range(len(crew) + 1):
                        if pos > 0 and task_finish[crew[pos - 1]] > ts_t:
                            continue
                        if pos < len(crew) and tf_t > task_start[crew[pos]]:
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
                        dt = task_finish[new_crew[-1]] - task_start[new_crew[0]]
                        if dt > time_limit:
                            continue

                        delta = cost_add - bridge
                        insertions.append((delta, ci, pos))

                if len(crews) < max_total_crews:
                    insertions.append((0.0, -1, 0))

                if not insertions:
                    return None

                insertions.sort(key=lambda x: x[0])
                best_c = insertions[0][0]
                second_c = insertions[1][0] if len(insertions) > 1 else best_c + 1e9
                regret_val = second_c - best_c
                regrets.append((regret_val, task, insertions[0]))

            regrets.sort(key=lambda x: -x[0])
            _, task_to_insert, (delta, ci, pos) = regrets[0]

            if ci == -1:
                crews.append([task_to_insert])
            else:
                crews[ci] = crews[ci][:pos] + [task_to_insert] + crews[ci][pos:]

            remaining_tasks.remove(task_to_insert)

        return crews

    def destroy_random_tasks(crews, n_remove, rng):
        all_tasks = [(ci, pos, t) for ci, crew in enumerate(crews) for pos, t in enumerate(crew)]
        if not all_tasks:
            return crews, []
        n_remove = min(n_remove, len(all_tasks))
        removed_entries = rng.sample(all_tasks, n_remove)
        removed_tasks = [t for _, _, t in removed_entries]
        removed_set = set(removed_tasks)
        new_crews = [[t for t in crew if t not in removed_set] for crew in crews]
        new_crews = [c for c in new_crews if c]
        return new_crews, removed_tasks

    def destroy_worst_crew(crews, n_crews_to_destroy, rng):
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
        if not crews:
            return crews, []
        crew_costs_w = [max(crew_cost_fn(crew), 0.01) for crew in crews]
        total_w = sum(crew_costs_w)
        probs = [c / total_w for c in crew_costs_w]
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
        all_tasks_flat = [t for crew in crews for t in crew]
        if not all_tasks_flat:
            return crews, []
        seed_task = rng.choice(all_tasks_flat)
        removed_set = {seed_task}
        candidates = set()
        for j in succ_list[seed_task]:
            candidates.add(j)
        for i in pred_list[seed_task]:
            candidates.add(i)
        seed_s = task_start[seed_task]
        for t in all_tasks_flat:
            if abs(task_start[t] - seed_s) < (time_limit * 0.1):
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
        new_crews = [[t for t in crew if t not in removed_set] for crew in crews]
        new_crews = [c for c in new_crews if c]
        return new_crews, removed_tasks

    # Main LNS+SA loop
    current_crews = [list(c) for c in initial_sol['crews']]
    current_cost = total_cost_fn(current_crews)
    best_crews = [list(c) for c in current_crews]
    best_cost = current_cost

    rng = random.Random(42)

    if current_cost > 0 and current_cost < float('inf'):
        T_init = max(0.05 * current_cost / math.log(2), 1.0)
    else:
        T_init = 100.0

    T_min = 1e-4
    T = T_init

    ls_time = remaining() - 0.5
    if ls_time <= 0:
        return initial_sol

    estimated_iters = max(50, int(ls_time * 30))
    cooling_rate = (T_min / T_init) ** (1.0 / max(1, estimated_iters))

    n_destroy_base = max(1, N // 10)

    destroy_ops = ['random', 'worst_crew', 'segment', 'related']
    op_success = [0, 0, 0, 0]
    op_tries = [1, 1, 1, 1]

    def select_destroy_op():
        total_tries = sum(op_tries)
        scores = []
        for i in range(len(destroy_ops)):
            exploit = op_success[i] / op_tries[i]
            explore = math.sqrt(2 * math.log(total_tries) / op_tries[i])
            scores.append(exploit + 0.3 * explore)
        return scores.index(max(scores))

    iteration = 0
    no_improve_streak = 0

    while remaining() > 0.5:
        iteration += 1
        T = max(T * cooling_rate, T_min)

        n_destroy = n_destroy_base + min(no_improve_streak // 5, N // 5)
        n_destroy = min(n_destroy, max(1, N // 2))

        op_idx = select_destroy_op()
        op_tries[op_idx] += 1
        destroy_op = destroy_ops[op_idx]

        if destroy_op == 'random':
            partial_crews, removed = destroy_random_tasks(current_crews, n_destroy, rng)
        elif destroy_op == 'worst_crew':
            n_crews_destroy = max(1, min(3, len(current_crews) // 3))
            partial_crews, removed = destroy_worst_crew(current_crews, n_crews_destroy, rng)
        elif destroy_op == 'segment':
            partial_crews, removed = destroy_segment(current_crews, rng)
        else:
            partial_crews, removed = destroy_related_tasks(current_crews, n_destroy, rng)

        if not removed:
            continue

        if iteration % 3 == 0 and len(removed) <= 20:
            new_crews = regret_repair(removed, partial_crews, K)
        else:
            new_crews = greedy_repair(removed, partial_crews, K)

        if new_crews is None:
            continue

        all_covered = set(t for crew in new_crews for t in crew)
        if len(all_covered) != N:
            missing = set(range(1, N + 1)) - all_covered
            new_crews = greedy_repair(list(missing), new_crews, K)
            if new_crews is None:
                continue
            if len(set(t for crew in new_crews for t in crew)) != N:
                continue

        if len(new_crews) > K:
            continue

        all_valid = all(check_crew(crew) for crew in new_crews)
        if not all_valid:
            continue

        new_cost = total_cost_fn(new_crews)
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

        if no_improve_streak > 50 and remaining() > 2.0:
            current_crews = [list(c) for c in best_crews]
            current_cost = best_cost
            no_improve_streak = 0
            T = max(T, T_init * 0.1)

    result = {'crews': [list(c) for c in best_crews if c]}
    try:
        feasible, _ = tools['is_feasible'](result)
        if feasible:
            return result
    except Exception:
        pass
    return initial_sol


# ====================================================================== #
# B-style: Move-based LS + SA helpers
# ====================================================================== #

def _run_move_ls_sa(initial_sol, N, K, tasks, arcs, time_limit,
                    arc_set, task_start, task_finish,
                    tools, start_time, time_limit_s):

    def remaining():
        return time_limit_s - (time.time() - start_time)

    def crew_cost_fast(crew):
        cost = 0.0
        for idx in range(len(crew) - 1):
            c = arcs.get((crew[idx], crew[idx + 1]))
            if c is None:
                return float('inf')
            cost += c
        return cost

    def check_crew_fast(crew):
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

    def compute_crew_costs(crews):
        return [crew_cost_fast(c) for c in crews]

    def relocation_best_improvement(crews, crew_costs):
        best_gain = 1e-9
        best_move = None

        for ci in range(len(crews)):
            crew_i = crews[ci]
            if len(crew_i) <= 1:
                continue
            old_cost_i = crew_costs[ci]

            for ti in range(len(crew_i)):
                task = crew_i[ti]
                t_s = task_start[task]
                t_f = task_finish[task]

                new_crew_i = crew_i[:ti] + crew_i[ti + 1:]
                if not new_crew_i:
                    continue

                valid_removal = True
                if ti > 0 and ti < len(crew_i) - 1:
                    prev_t = crew_i[ti - 1]
                    next_t = crew_i[ti + 1]
                    if task_finish[prev_t] > task_start[next_t]:
                        valid_removal = False
                    elif (prev_t, next_t) not in arc_set:
                        valid_removal = False
                if valid_removal:
                    dt = task_finish[new_crew_i[-1]] - task_start[new_crew_i[0]]
                    if dt > time_limit:
                        valid_removal = False
                if not valid_removal:
                    continue

                new_cost_i = crew_cost_fast(new_crew_i)
                saving_i = old_cost_i - new_cost_i

                for cj in range(len(crews)):
                    if cj == ci:
                        continue
                    crew_j = crews[cj]
                    old_cost_j = crew_costs[cj]

                    for pos in range(len(crew_j) + 1):
                        prev_j = crew_j[pos - 1] if pos > 0 else None
                        next_j = crew_j[pos] if pos < len(crew_j) else None

                        if prev_j is not None:
                            if task_finish[prev_j] > t_s:
                                continue
                            if (prev_j, task) not in arc_set:
                                continue
                        if next_j is not None:
                            if t_f > task_start[next_j]:
                                continue
                            if (task, next_j) not in arc_set:
                                continue

                        new_first = task_start[crew_j[0]] if pos > 0 else t_s
                        new_last = task_finish[crew_j[-1]] if pos < len(crew_j) else t_f
                        if new_last - new_first > time_limit:
                            continue

                        insert_cost = 0.0
                        if prev_j is not None:
                            insert_cost += arcs[(prev_j, task)]
                        if next_j is not None:
                            insert_cost += arcs[(task, next_j)]
                        bridge_cost = 0.0
                        if prev_j is not None and next_j is not None:
                            bridge_cost = arcs.get((prev_j, next_j), 0.0)

                        gain = saving_i - (insert_cost - bridge_cost)
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (ci, ti, cj, pos)

        if best_move is None:
            return False, crews, crew_costs

        ci, ti, cj, pos = best_move
        task = crews[ci][ti]
        new_crew_i = crews[ci][:ti] + crews[ci][ti + 1:]
        new_crew_j = crews[cj][:pos] + [task] + crews[cj][pos:]
        crews[ci] = new_crew_i
        crews[cj] = new_crew_j
        crew_costs[ci] = crew_cost_fast(new_crew_i)
        crew_costs[cj] = crew_cost_fast(new_crew_j)
        return True, crews, crew_costs

    def swap_tasks(crews, crew_costs):
        best_gain = 1e-9
        best_move = None

        for ci in range(len(crews)):
            for cj in range(ci + 1, len(crews)):
                crew_i = crews[ci]
                crew_j = crews[cj]
                old_cost_ij = crew_costs[ci] + crew_costs[cj]

                for pi in range(len(crew_i)):
                    for pj in range(len(crew_j)):
                        task_i = crew_i[pi]
                        task_j = crew_j[pj]

                        ti_s = task_start[task_i]
                        ti_f = task_finish[task_i]
                        tj_s = task_start[task_j]
                        tj_f = task_finish[task_j]

                        ok_i = True
                        if pi > 0:
                            prev = crew_i[pi - 1]
                            if task_finish[prev] > tj_s or (prev, task_j) not in arc_set:
                                ok_i = False
                        if ok_i and pi < len(crew_i) - 1:
                            nxt = crew_i[pi + 1]
                            if tj_f > task_start[nxt] or (task_j, nxt) not in arc_set:
                                ok_i = False
                        if ok_i:
                            nf_i = crew_i[0] if pi > 0 else task_j
                            nl_i = crew_i[-1] if pi < len(crew_i) - 1 else task_j
                            if task_finish[nl_i] - task_start[nf_i] > time_limit:
                                ok_i = False
                        if not ok_i:
                            continue

                        ok_j = True
                        if pj > 0:
                            prev = crew_j[pj - 1]
                            if task_finish[prev] > ti_s or (prev, task_i) not in arc_set:
                                ok_j = False
                        if ok_j and pj < len(crew_j) - 1:
                            nxt = crew_j[pj + 1]
                            if ti_f > task_start[nxt] or (task_i, nxt) not in arc_set:
                                ok_j = False
                        if ok_j:
                            nf_j = crew_j[0] if pj > 0 else task_i
                            nl_j = crew_j[-1] if pj < len(crew_j) - 1 else task_i
                            if task_finish[nl_j] - task_start[nf_j] > time_limit:
                                ok_j = False
                        if not ok_j:
                            continue

                        new_ci = list(crew_i)
                        new_cj = list(crew_j)
                        new_ci[pi] = task_j
                        new_cj[pj] = task_i
                        new_cost = crew_cost_fast(new_ci) + crew_cost_fast(new_cj)
                        gain = old_cost_ij - new_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (ci, pi, cj, pj)

        if best_move is None:
            return False, crews, crew_costs

        ci, pi, cj, pj = best_move
        task_i = crews[ci][pi]
        task_j = crews[cj][pj]
        new_ci = list(crews[ci])
        new_cj = list(crews[cj])
        new_ci[pi] = task_j
        new_cj[pj] = task_i
        crews[ci] = new_ci
        crews[cj] = new_cj
        crew_costs[ci] = crew_cost_fast(new_ci)
        crew_costs[cj] = crew_cost_fast(new_cj)
        return True, crews, crew_costs

    def oropt_move(crews, crew_costs, seg_len=2):
        best_gain = 1e-9
        best_move = None

        for ci in range(len(crews)):
            crew_i = crews[ci]
            if len(crew_i) <= seg_len:
                continue
            old_cost_i = crew_costs[ci]

            for ti in range(len(crew_i) - seg_len + 1):
                segment = crew_i[ti:ti + seg_len]
                seg_valid = all((segment[k], segment[k + 1]) in arc_set for k in range(seg_len - 1))
                if not seg_valid:
                    continue

                new_crew_i = crew_i[:ti] + crew_i[ti + seg_len:]
                if not new_crew_i:
                    continue

                if ti > 0 and ti + seg_len <= len(crew_i) - 1:
                    prev_t = crew_i[ti - 1]
                    next_t = crew_i[ti + seg_len]
                    if task_finish[prev_t] > task_start[next_t]:
                        continue
                    if (prev_t, next_t) not in arc_set:
                        continue

                dt_i = task_finish[new_crew_i[-1]] - task_start[new_crew_i[0]]
                if dt_i > time_limit:
                    continue

                new_cost_i = crew_cost_fast(new_crew_i)
                saving_i = old_cost_i - new_cost_i
                seg_s = task_start[segment[0]]
                seg_f = task_finish[segment[-1]]
                seg_internal = crew_cost_fast(segment)

                for cj in range(len(crews)):
                    if cj == ci:
                        continue
                    crew_j = crews[cj]
                    old_cost_j = crew_costs[cj]

                    for pos in range(len(crew_j) + 1):
                        prev_j = crew_j[pos - 1] if pos > 0 else None
                        next_j = crew_j[pos] if pos < len(crew_j) else None

                        if prev_j is not None and task_finish[prev_j] > seg_s:
                            continue
                        if next_j is not None and seg_f > task_start[next_j]:
                            continue

                        ok = True
                        extra_cost = seg_internal
                        if prev_j is not None:
                            c = arcs.get((prev_j, segment[0]))
                            if c is None:
                                ok = False
                            else:
                                extra_cost += c
                        if ok and next_j is not None:
                            c = arcs.get((segment[-1], next_j))
                            if c is None:
                                ok = False
                            else:
                                extra_cost += c
                        if not ok:
                            continue

                        bridge_cost = 0.0
                        if prev_j is not None and next_j is not None:
                            bridge_cost = arcs.get((prev_j, next_j), 0.0)

                        new_first_t = task_start[crew_j[0]] if pos > 0 else seg_s
                        new_last_t = task_finish[crew_j[-1]] if pos < len(crew_j) else seg_f
                        if new_last_t - new_first_t > time_limit:
                            continue

                        gain = saving_i - (extra_cost - bridge_cost)
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (ci, ti, cj, pos, seg_len)

        if best_move is None:
            return False, crews, crew_costs

        ci, ti, cj, pos, sl = best_move
        segment = crews[ci][ti:ti + sl]
        new_crew_i = crews[ci][:ti] + crews[ci][ti + sl:]
        new_crew_j = crews[cj][:pos] + segment + crews[cj][pos:]
        crews[ci] = new_crew_i
        crews[cj] = new_crew_j
        crew_costs[ci] = crew_cost_fast(new_crew_i)
        crew_costs[cj] = crew_cost_fast(new_crew_j)
        return True, crews, crew_costs

    def intra_relocate(crews, crew_costs):
        improved = False
        for ci in range(len(crews)):
            crew = crews[ci]
            n = len(crew)
            if n < 3:
                continue
            old_cost = crew_costs[ci]
            best_gain = 1e-9
            best_move = None

            for ti in range(n):
                task = crew[ti]
                new_crew_base = crew[:ti] + crew[ti + 1:]

                if ti > 0 and ti < n - 1:
                    prev_t = crew[ti - 1]
                    next_t = crew[ti + 1]
                    if task_finish[prev_t] > task_start[next_t] or (prev_t, next_t) not in arc_set:
                        continue

                for pos in range(len(new_crew_base) + 1):
                    if pos == ti:
                        continue
                    prev_j = new_crew_base[pos - 1] if pos > 0 else None
                    next_j = new_crew_base[pos] if pos < len(new_crew_base) else None

                    if prev_j is not None:
                        if task_finish[prev_j] > task_start[task] or (prev_j, task) not in arc_set:
                            continue
                    if next_j is not None:
                        if task_finish[task] > task_start[next_j] or (task, next_j) not in arc_set:
                            continue

                    new_crew = new_crew_base[:pos] + [task] + new_crew_base[pos:]
                    dt = task_finish[new_crew[-1]] - task_start[new_crew[0]]
                    if dt > time_limit:
                        continue

                    new_cost = crew_cost_fast(new_crew)
                    gain = old_cost - new_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (ti, pos, new_crew, new_cost)

            if best_move is not None:
                _, _, new_crew, new_cost = best_move
                crews[ci] = new_crew
                crew_costs[ci] = new_cost
                improved = True

        return improved, crews, crew_costs

    def run_local_search(crews_in, time_budget):
        crews = [list(c) for c in crews_in]
        crew_costs = compute_crew_costs(crews)
        ls_start = time.time()

        def ls_rem():
            return time_budget - (time.time() - ls_start)

        improved_global = True
        while improved_global and ls_rem() > 0.05:
            improved_global = False

            if ls_rem() > 0.02:
                imp, crews, crew_costs = intra_relocate(crews, crew_costs)
                if imp:
                    improved_global = True

            while ls_rem() > 0.05:
                imp, crews, crew_costs = relocation_best_improvement(crews, crew_costs)
                if not imp:
                    break
                improved_global = True

            if ls_rem() < 0.05:
                break

            while ls_rem() > 0.05:
                imp, crews, crew_costs = swap_tasks(crews, crew_costs)
                if not imp:
                    break
                improved_global = True

            if ls_rem() < 0.05:
                break

            while ls_rem() > 0.05:
                imp, crews, crew_costs = oropt_move(crews, crew_costs, seg_len=2)
                if not imp:
                    break
                improved_global = True

            if ls_rem() < 0.05:
                break

            if N <= 300 and ls_rem() > 0.1:
                while ls_rem() > 0.05:
                    imp, crews, crew_costs = oropt_move(crews, crew_costs, seg_len=3)
                    if not imp:
                        break
                    improved_global = True

        return crews, sum(crew_costs)

    def simulated_annealing(crews_in, time_budget, initial_temp=None, cooling=0.997):
        crews = [list(c) for c in crews_in]
        crew_costs = compute_crew_costs(crews)
        current_cost = sum(crew_costs)
        sa_best_cost = current_cost
        sa_best_crews = [list(c) for c in crews]

        sa_start = time.time()
        rng = random.Random(123)

        def sa_rem():
            return time_budget - (time.time() - sa_start)

        if initial_temp is None:
            initial_temp = max(current_cost * 0.05, 1.0)

        temp = initial_temp
        n_iter = 0

        while sa_rem() > 0.05:
            n_iter += 1
            if n_iter % 1000 == 0:
                temp *= cooling

            move_type = rng.randint(0, 2)

            if move_type == 0:
                eligible = [i for i in range(len(crews)) if len(crews[i]) > 1]
                if not eligible:
                    continue
                ci = rng.choice(eligible)
                ti = rng.randint(0, len(crews[ci]) - 1)
                task = crews[ci][ti]
                t_s = task_start[task]
                t_f = task_finish[task]

                new_crew_i = crews[ci][:ti] + crews[ci][ti + 1:]
                if not new_crew_i:
                    continue

                bridge_ok = True
                if ti > 0 and ti < len(crews[ci]) - 1:
                    prev_t = crews[ci][ti - 1]
                    next_t = crews[ci][ti + 1]
                    if task_finish[prev_t] > task_start[next_t] or (prev_t, next_t) not in arc_set:
                        bridge_ok = False
                if not bridge_ok:
                    continue

                dt_i = task_finish[new_crew_i[-1]] - task_start[new_crew_i[0]]
                if dt_i > time_limit:
                    continue

                other_crews = [j for j in range(len(crews)) if j != ci]
                if not other_crews:
                    continue
                cj = rng.choice(other_crews)
                crew_j = crews[cj]
                pos = rng.randint(0, len(crew_j))

                prev_j = crew_j[pos - 1] if pos > 0 else None
                next_j = crew_j[pos] if pos < len(crew_j) else None

                if prev_j is not None:
                    if task_finish[prev_j] > t_s or (prev_j, task) not in arc_set:
                        continue
                if next_j is not None:
                    if t_f > task_start[next_j] or (task, next_j) not in arc_set:
                        continue

                new_first = task_start[crew_j[0]] if pos > 0 else t_s
                new_last = task_finish[crew_j[-1]] if pos < len(crew_j) else t_f
                if new_last - new_first > time_limit:
                    continue

                new_crew_j = crew_j[:pos] + [task] + crew_j[pos:]
                new_cost_i = crew_cost_fast(new_crew_i)
                new_cost_j = crew_cost_fast(new_crew_j)
                delta = (new_cost_i + new_cost_j) - (crew_costs[ci] + crew_costs[cj])

                if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                    crews[ci] = new_crew_i
                    crews[cj] = new_crew_j
                    crew_costs[ci] = new_cost_i
                    crew_costs[cj] = new_cost_j
                    current_cost += delta
                    if current_cost < sa_best_cost:
                        sa_best_cost = current_cost
                        sa_best_crews = [list(c) for c in crews]

            elif move_type == 1:
                if len(crews) < 2:
                    continue
                ci = rng.randint(0, len(crews) - 1)
                cj = rng.randint(0, len(crews) - 1)
                if ci == cj or not crews[ci] or not crews[cj]:
                    continue
                pi = rng.randint(0, len(crews[ci]) - 1)
                pj = rng.randint(0, len(crews[cj]) - 1)

                new_ci = list(crews[ci])
                new_cj = list(crews[cj])
                new_ci[pi] = crews[cj][pj]
                new_cj[pj] = crews[ci][pi]

                if not check_crew_fast(new_ci) or not check_crew_fast(new_cj):
                    continue

                new_cost_ci = crew_cost_fast(new_ci)
                new_cost_cj = crew_cost_fast(new_cj)
                delta = (new_cost_ci + new_cost_cj) - (crew_costs[ci] + crew_costs[cj])

                if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                    crews[ci] = new_ci
                    crews[cj] = new_cj
                    crew_costs[ci] = new_cost_ci
                    crew_costs[cj] = new_cost_cj
                    current_cost += delta
                    if current_cost < sa_best_cost:
                        sa_best_cost = current_cost
                        sa_best_crews = [list(c) for c in crews]

            else:
                eligible = [i for i in range(len(crews)) if len(crews[i]) >= 3]
                if not eligible:
                    continue
                ci = rng.choice(eligible)
                crew = crews[ci]
                n = len(crew)
                ti = rng.randint(0, n - 1)
                task = crew[ti]
                new_crew_base = crew[:ti] + crew[ti + 1:]

                if ti > 0 and ti < n - 1:
                    prev_t = crew[ti - 1]
                    next_t = crew[ti + 1]
                    if task_finish[prev_t] > task_start[next_t] or (prev_t, next_t) not in arc_set:
                        continue

                pos = rng.randint(0, len(new_crew_base))
                if pos == ti:
                    continue

                prev_j = new_crew_base[pos - 1] if pos > 0 else None
                next_j = new_crew_base[pos] if pos < len(new_crew_base) else None

                if prev_j is not None:
                    if task_finish[prev_j] > task_start[task] or (prev_j, task) not in arc_set:
                        continue
                if next_j is not None:
                    if task_finish[task] > task_start[next_j] or (task, next_j) not in arc_set:
                        continue

                new_crew = new_crew_base[:pos] + [task] + new_crew_base[pos:]
                dt = task_finish[new_crew[-1]] - task_start[new_crew[0]]
                if dt > time_limit:
                    continue

                new_cost = crew_cost_fast(new_crew)
                delta = new_cost - crew_costs[ci]

                if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                    crews[ci] = new_crew
                    crew_costs[ci] = new_cost
                    current_cost += delta
                    if current_cost < sa_best_cost:
                        sa_best_cost = current_cost
                        sa_best_crews = [list(c) for c in crews]

        return sa_best_crews, sa_best_cost

    # Main B-style optimization
    best_local_sol = initial_sol
    best_local_cost = float('inf')

    def update_local(sol):
        nonlocal best_local_sol, best_local_cost
        if sol is None:
            return False
        try:
            feasible, _ = tools['is_feasible'](sol)
            if not feasible:
                return False
            cost = tools['objective'](sol)
            if cost < best_local_cost:
                best_local_cost = cost
                best_local_sol = {'crews': [list(c) for c in sol['crews']]}
                return True
        except Exception:
            pass
        return False

    update_local(initial_sol)

    # Initial LS pass
    ls_budget = min(remaining() * 0.40, 30.0)
    if remaining() > 1.0:
        ls_crews, ls_cost = run_local_search(best_local_sol['crews'], ls_budget)
        update_local({'crews': ls_crews})

    # SA phase
    if remaining() > 2.0:
        sa_budget = min(remaining() * 0.50, 25.0)
        init_temp = max(best_local_cost * 0.03, 0.5)
        sa_crews, sa_cost = simulated_annealing(
            best_local_sol['crews'], sa_budget,
            initial_temp=init_temp, cooling=0.998
        )
        update_local({'crews': sa_crews})

        # Post-SA LS
        if remaining() > 1.5:
            ls_budget2 = min(remaining() * 0.55, 18.0)
            ls_crews2, _ = run_local_search(best_local_sol['crews'], ls_budget2)
            update_local({'crews': ls_crews2})

    # Final polish
    if remaining() > 1.0:
        polish_budget = remaining() * 0.85
        polished, _ = run_local_search(best_local_sol['crews'], polish_budget)
        update_local({'crews': polished})

    return best_local_sol