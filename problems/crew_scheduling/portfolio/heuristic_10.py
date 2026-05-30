# MACE evolved heuristic 10/10 for problem: crew_scheduling
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
    # Precompute data structures
    # ------------------------------------------------------------------ #
    arc_set = set(arcs.keys())
    task_start = {t: tasks[t][0] for t in range(1, N + 1)}
    task_finish = {t: tasks[t][1] for t in range(1, N + 1)}

    succ_list = defaultdict(list)
    pred_list = defaultdict(list)
    for (i, j), cost in arcs.items():
        if task_finish[i] <= task_start[j]:
            succ_list[i].append(j)
            pred_list[j].append(i)
    for t in range(1, N + 1):
        succ_list[t].sort(key=lambda x: task_start[x])
        pred_list[t].sort(key=lambda x: task_finish[x])

    # ------------------------------------------------------------------ #
    # Core utility functions
    # ------------------------------------------------------------------ #
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
    # Instance analysis for strategy dispatch
    # ------------------------------------------------------------------ #
    k_ratio = K / max(1, N)
    n_arcs = len(arcs)
    arc_density = n_arcs / max(1, N * (N - 1))
    avg_succ = sum(len(succ_list[t]) for t in range(1, N + 1)) / max(1, N)

    # Use ILP for small instances, MCF/LNS for larger
    use_ilp = (N <= 100)
    use_lns = (N > 80 or k_ratio < 0.35)

    # ------------------------------------------------------------------ #
    # Phase 1: Initial solution construction
    # ------------------------------------------------------------------ #

    # Strategy: try multiple solvers in order of expected quality
    if use_ilp and remaining() > 2.0:
        ilp_budget = min(remaining() * 0.45, 45.0)
        try:
            sol = tools['ilp_crew_scheduling'](time_limit_s=ilp_budget)
            update_best(sol)
        except Exception:
            pass

    if remaining() > 2.0:
        mcf_budget = min(remaining() * 0.30, 20.0)
        try:
            sol = tools['solve_min_cost_flow'](time_limit_s=mcf_budget)
            update_best(sol)
        except Exception:
            pass

    if best_sol is None and remaining() > 2.0:
        sd_budget = min(remaining() * 0.40, 25.0)
        try:
            sol = tools['solve_default'](time_limit_s=sd_budget)
            update_best(sol)
        except Exception:
            pass

    # Greedy chain pack
    if remaining() > 0.5:
        try:
            chains = tools['greedy_chain_pack']()
            if chains and len(chains) <= K:
                sol = tools['make_solution'](chains)
                update_best(sol)
        except Exception:
            pass

    # Custom greedy construction (deterministic)
    def greedy_construct(rng=None, alpha=0.0):
        task_order = sorted(range(1, N + 1), key=lambda t: task_start[t])
        crews_g = []
        crew_last = []
        crew_start_t = []
        crew_finish_t = []

        for task in task_order:
            ts = task_start[task]
            tf = task_finish[task]
            candidates = []

            for i in range(len(crews_g)):
                last = crew_last[i]
                if crew_finish_t[i] > ts:
                    continue
                if (last, task) not in arc_set:
                    continue
                if tf - crew_start_t[i] > time_limit:
                    continue
                c = arcs[(last, task)]
                candidates.append((c, i))

            if candidates:
                if rng is not None and alpha > 0 and len(candidates) > 1:
                    candidates.sort()
                    min_c = candidates[0][0]
                    max_c = candidates[-1][0]
                    threshold = min_c + alpha * (max_c - min_c)
                    rcl = [i for c, i in candidates if c <= threshold + 1e-9]
                    chosen = rng.choice(rcl)
                else:
                    chosen = min(candidates, key=lambda x: x[0])[1]
                crews_g[chosen].append(task)
                crew_last[chosen] = task
                crew_finish_t[chosen] = tf
            elif len(crews_g) < K:
                crews_g.append([task])
                crew_last.append(task)
                crew_start_t.append(ts)
                crew_finish_t.append(tf)
            else:
                forced = False
                for i in range(len(crews_g)):
                    last = crew_last[i]
                    if crew_finish_t[i] <= ts and (last, task) in arc_set:
                        if tf - crew_start_t[i] <= time_limit:
                            crews_g[i].append(task)
                            crew_last[i] = task
                            crew_finish_t[i] = tf
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

    if remaining() > 0.3:
        sol = greedy_construct()
        update_best(sol)

    # Multiple randomized constructions
    rng_init = random.Random(42)
    for _ in range(5):
        if remaining() < 0.2:
            break
        alpha = rng_init.uniform(0.1, 0.5)
        sol = greedy_construct(rng=rng_init, alpha=alpha)
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

    # ------------------------------------------------------------------ #
    # Phase 2: Local search + LNS optimization
    # ------------------------------------------------------------------ #

    def greedy_repair(unassigned, current_crews, max_crews):
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
                n = len(crew)
                for pos in range(n + 1):
                    if pos > 0 and task_finish[crew[pos - 1]] > ts_t:
                        continue
                    if pos < n and tf_t > task_start[crew[pos]]:
                        continue

                    cost_add = 0.0
                    ok = True
                    if pos > 0:
                        c = arcs.get((crew[pos - 1], task))
                        if c is None:
                            ok = False
                        else:
                            cost_add += c
                    if ok and pos < n:
                        c = arcs.get((task, crew[pos]))
                        if c is None:
                            ok = False
                        else:
                            cost_add += c
                    if not ok:
                        continue

                    bridge = 0.0
                    if 0 < pos < n:
                        bridge = arcs.get((crew[pos - 1], crew[pos]), 0.0)

                    head_start = task_start[crew[0]] if pos > 0 else ts_t
                    tail_finish = task_finish[crew[-1]] if pos < n else tf_t
                    if tail_finish - head_start > time_limit:
                        continue

                    delta = cost_add - bridge
                    if delta < best_delta:
                        best_delta = delta
                        best_ci = ci
                        best_pos = pos

            if best_ci >= 0:
                crews[best_ci] = crews[best_ci][:best_pos] + [task] + crews[best_ci][best_pos:]
            elif len(crews) < max_crews:
                crews.append([task])
            else:
                return None

        return crews

    def regret_repair(unassigned, current_crews, max_crews):
        if not unassigned:
            return current_crews
        if len(unassigned) > 30:
            return greedy_repair(unassigned, current_crews, max_crews)

        crews = [list(c) for c in current_crews]
        remaining_tasks = list(unassigned)

        while remaining_tasks:
            regrets = []
            for task in remaining_tasks:
                ts_t = task_start[task]
                tf_t = task_finish[task]
                insertions = []

                for ci, crew in enumerate(crews):
                    n = len(crew)
                    for pos in range(n + 1):
                        if pos > 0 and task_finish[crew[pos - 1]] > ts_t:
                            continue
                        if pos < n and tf_t > task_start[crew[pos]]:
                            continue

                        cost_add = 0.0
                        ok = True
                        if pos > 0:
                            c = arcs.get((crew[pos - 1], task))
                            if c is None:
                                ok = False
                            else:
                                cost_add += c
                        if ok and pos < n:
                            c = arcs.get((task, crew[pos]))
                            if c is None:
                                ok = False
                            else:
                                cost_add += c
                        if not ok:
                            continue

                        bridge = 0.0
                        if 0 < pos < n:
                            bridge = arcs.get((crew[pos - 1], crew[pos]), 0.0)

                        head_start = task_start[crew[0]] if pos > 0 else ts_t
                        tail_finish = task_finish[crew[-1]] if pos < n else tf_t
                        if tail_finish - head_start > time_limit:
                            continue

                        delta = cost_add - bridge
                        insertions.append((delta, ci, pos))

                if len(crews) < max_crews:
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

    # ------------------------------------------------------------------ #
    # Local search moves
    # ------------------------------------------------------------------ #

    def intra_relocate_pass(crews, crew_costs):
        """Move tasks within a single crew to better positions."""
        improved = False
        for ci in range(len(crews)):
            crew = crews[ci]
            n = len(crew)
            if n < 3:
                continue
            old_cost = crew_costs[ci]
            best_gain = 1e-9
            best_new_crew = None
            best_new_cost = old_cost

            for ti in range(n):
                task = crew[ti]
                # Check if removal is valid (bridge arc exists)
                if ti > 0 and ti < n - 1:
                    prev_t = crew[ti - 1]
                    next_t = crew[ti + 1]
                    if task_finish[prev_t] > task_start[next_t]:
                        continue
                    if (prev_t, next_t) not in arc_set:
                        continue

                new_crew_base = crew[:ti] + crew[ti + 1:]

                for pos in range(len(new_crew_base) + 1):
                    if pos == ti:
                        continue
                    prev_j = new_crew_base[pos - 1] if pos > 0 else None
                    next_j = new_crew_base[pos] if pos < len(new_crew_base) else None

                    if prev_j is not None:
                        if task_finish[prev_j] > task_start[task]:
                            continue
                        if (prev_j, task) not in arc_set:
                            continue
                    if next_j is not None:
                        if task_finish[task] > task_start[next_j]:
                            continue
                        if (task, next_j) not in arc_set:
                            continue

                    new_crew = new_crew_base[:pos] + [task] + new_crew_base[pos:]
                    dt = task_finish[new_crew[-1]] - task_start[new_crew[0]]
                    if dt > time_limit:
                        continue

                    new_cost = crew_cost_fast(new_crew)
                    gain = old_cost - new_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_new_crew = new_crew
                        best_new_cost = new_cost

            if best_new_crew is not None:
                crews[ci] = best_new_crew
                crew_costs[ci] = best_new_cost
                improved = True

        return improved, crews, crew_costs

    def inter_relocate_best(crews, crew_costs):
        """Move a single task from one crew to another, best improvement."""
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

                # Check removal validity
                if ti > 0 and ti < len(crew_i) - 1:
                    prev_t = crew_i[ti - 1]
                    next_t = crew_i[ti + 1]
                    if task_finish[prev_t] > task_start[next_t]:
                        continue
                    if (prev_t, next_t) not in arc_set:
                        continue

                new_crew_i = crew_i[:ti] + crew_i[ti + 1:]
                dt_i = task_finish[new_crew_i[-1]] - task_start[new_crew_i[0]]
                if dt_i > time_limit:
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

                        head_start = task_start[crew_j[0]] if pos > 0 else t_s
                        tail_finish = task_finish[crew_j[-1]] if pos < len(crew_j) else t_f
                        if tail_finish - head_start > time_limit:
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

    def swap_best(crews, crew_costs):
        """Swap single tasks between two crews, best improvement."""
        best_gain = 1e-9
        best_move = None

        for ci in range(len(crews)):
            for cj in range(ci + 1, len(crews)):
                crew_i = crews[ci]
                crew_j = crews[cj]
                old_cost = crew_costs[ci] + crew_costs[cj]

                for pi in range(len(crew_i)):
                    task_i = crew_i[pi]
                    ti_s = task_start[task_i]
                    ti_f = task_finish[task_i]

                    for pj in range(len(crew_j)):
                        task_j = crew_j[pj]
                        tj_s = task_start[task_j]
                        tj_f = task_finish[task_j]

                        # Check task_j in crew_i at position pi
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
                            first_i = crew_i[0] if pi > 0 else task_j
                            last_i = crew_i[-1] if pi < len(crew_i) - 1 else task_j
                            if task_finish[last_i] - task_start[first_i] > time_limit:
                                ok_i = False
                        if not ok_i:
                            continue

                        # Check task_i in crew_j at position pj
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
                            first_j = crew_j[0] if pj > 0 else task_i
                            last_j = crew_j[-1] if pj < len(crew_j) - 1 else task_i
                            if task_finish[last_j] - task_start[first_j] > time_limit:
                                ok_j = False
                        if not ok_j:
                            continue

                        new_ci = list(crew_i)
                        new_cj = list(crew_j)
                        new_ci[pi] = task_j
                        new_cj[pj] = task_i
                        new_cost = crew_cost_fast(new_ci) + crew_cost_fast(new_cj)
                        gain = old_cost - new_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (ci, pi, cj, pj)

        if best_move is None:
            return False, crews, crew_costs

        ci, pi, cj, pj = best_move
        new_ci = list(crews[ci])
        new_cj = list(crews[cj])
        new_ci[pi] = crews[cj][pj]
        new_cj[pj] = crews[ci][pi]
        crews[ci] = new_ci
        crews[cj] = new_cj
        crew_costs[ci] = crew_cost_fast(new_ci)
        crew_costs[cj] = crew_cost_fast(new_cj)
        return True, crews, crew_costs

    def oropt_best(crews, crew_costs, seg_len):
        """Move a segment of seg_len tasks from one crew to another."""
        best_gain = 1e-9
        best_move = None

        for ci in range(len(crews)):
            crew_i = crews[ci]
            if len(crew_i) <= seg_len:
                continue
            old_cost_i = crew_costs[ci]

            for ti in range(len(crew_i) - seg_len + 1):
                segment = crew_i[ti:ti + seg_len]

                # Segment internal arcs must be valid
                seg_ok = all((segment[k], segment[k + 1]) in arc_set
                             for k in range(seg_len - 1))
                if not seg_ok:
                    continue

                # Check bridge after removal
                if ti > 0 and ti + seg_len < len(crew_i):
                    prev_t = crew_i[ti - 1]
                    next_t = crew_i[ti + seg_len]
                    if task_finish[prev_t] > task_start[next_t]:
                        continue
                    if (prev_t, next_t) not in arc_set:
                        continue

                new_crew_i = crew_i[:ti] + crew_i[ti + seg_len:]
                if new_crew_i:
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

                        extra_cost = seg_internal
                        ok = True
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

                        head_start = task_start[crew_j[0]] if pos > 0 else seg_s
                        tail_finish = task_finish[crew_j[-1]] if pos < len(crew_j) else seg_f
                        if tail_finish - head_start > time_limit:
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

    def run_local_search(crews_in, time_budget):
        crews = [list(c) for c in crews_in]
        crew_costs = [crew_cost_fast(c) for c in crews]
        ls_start = time.time()

        def ls_rem():
            return time_budget - (time.time() - ls_start)

        improved_global = True
        while improved_global and ls_rem() > 0.05:
            improved_global = False

            if ls_rem() > 0.02:
                imp, crews, crew_costs = intra_relocate_pass(crews, crew_costs)
                if imp:
                    improved_global = True

            while ls_rem() > 0.05:
                imp, crews, crew_costs = inter_relocate_best(crews, crew_costs)
                if not imp:
                    break
                improved_global = True

            if ls_rem() < 0.05:
                break

            while ls_rem() > 0.05:
                imp, crews, crew_costs = swap_best(crews, crew_costs)
                if not imp:
                    break
                improved_global = True

            if ls_rem() < 0.05:
                break

            for sl in [2, 3]:
                if ls_rem() < 0.05:
                    break
                while ls_rem() > 0.05:
                    imp, crews, crew_costs = oropt_best(crews, crew_costs, sl)
                    if not imp:
                        break
                    improved_global = True

        return crews, sum(crew_costs)

    # ------------------------------------------------------------------ #
    # LNS destroy operators
    # ------------------------------------------------------------------ #

    def destroy_random_tasks(crews, n_remove, rng):
        all_tasks = [(ci, pos, t) for ci, crew in enumerate(crews)
                     for pos, t in enumerate(crew)]
        if not all_tasks:
            return crews, []
        n_remove = min(n_remove, len(all_tasks))
        removed_entries = rng.sample(all_tasks, n_remove)
        removed_set = set(t for _, _, t in removed_entries)
        new_crews = [[t for t in crew if t not in removed_set] for crew in crews]
        new_crews = [c for c in new_crews if c]
        return new_crews, list(removed_set)

    def destroy_worst_crew(crews, n_destroy, rng):
        if not crews:
            return crews, []
        crew_costs_sorted = sorted(range(len(crews)),
                                   key=lambda i: crew_cost_fast(crews[i]),
                                   reverse=True)
        n_destroy = min(n_destroy, len(crews))
        destroyed = set(crew_costs_sorted[:n_destroy])
        removed = []
        new_crews = []
        for ci, crew in enumerate(crews):
            if ci in destroyed:
                removed.extend(crew)
            else:
                new_crews.append(crew)
        return new_crews, removed

    def destroy_segment(crews, rng):
        if not crews:
            return crews, []
        crew_weights = [max(crew_cost_fast(crew), 0.01) for crew in crews]
        total_w = sum(crew_weights)
        r = rng.random() * total_w
        cumulative = 0.0
        ci = 0
        for i, w in enumerate(crew_weights):
            cumulative += w
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
        time_window = time_limit * 0.15
        for t in all_tasks_flat:
            if abs(task_start[t] - seed_s) < time_window:
                candidates.add(t)
        candidates -= removed_set
        candidates = sorted(candidates, key=lambda t: abs(task_start[t] - seed_s))
        n_remove = min(n_remove, len(all_tasks_flat))
        for t in candidates:
            if len(removed_set) >= n_remove:
                break
            removed_set.add(t)
        removed_tasks = list(removed_set)
        new_crews = [[t for t in crew if t not in removed_set] for crew in crews]
        new_crews = [c for c in new_crews if c]
        return new_crews, removed_tasks

    def destroy_expensive_arcs(crews, n_remove, rng):
        """Remove tasks around most expensive arcs."""
        arc_costs_in_sol = []
        for ci, crew in enumerate(crews):
            for idx in range(len(crew) - 1):
                c = arcs.get((crew[idx], crew[idx + 1]), 0)
                arc_costs_in_sol.append((c, ci, idx, idx + 1))
        if not arc_costs_in_sol:
            return crews, []
        arc_costs_in_sol.sort(reverse=True)
        removed_set = set()
        for c, ci, idx1, idx2 in arc_costs_in_sol[:max(1, n_remove // 2)]:
            removed_set.add(crews[ci][idx1])
            removed_set.add(crews[ci][idx2])
            if len(removed_set) >= n_remove:
                break
        new_crews = [[t for t in crew if t not in removed_set] for crew in crews]
        new_crews = [c for c in new_crews if c]
        return new_crews, list(removed_set)

    # ------------------------------------------------------------------ #
    # Main optimization loop: adaptive LNS + SA with periodic LS
    # ------------------------------------------------------------------ #

    current_crews = [list(c) for c in best_sol['crews']]
    current_cost = solution_cost_fast(current_crews)
    lns_best_crews = [list(c) for c in current_crews]
    lns_best_cost = current_cost

    rng = random.Random(42)

    # Initial local search pass
    if remaining() > 1.5:
        ls_budget = min(remaining() * 0.35, 30.0)
        ls_crews, ls_cost = run_local_search(current_crews, ls_budget)
        ls_sol = {'crews': ls_crews}
        if update_best(ls_sol):
            current_crews = [list(c) for c in ls_crews]
            current_cost = ls_cost
            lns_best_crews = [list(c) for c in ls_crews]
            lns_best_cost = ls_cost
        else:
            try:
                feasible, _ = tools['is_feasible'](ls_sol)
                if feasible:
                    cost = tools['objective'](ls_sol)
                    if cost < lns_best_cost:
                        lns_best_cost = cost
                        lns_best_crews = [list(c) for c in ls_crews]
                    current_crews = ls_crews
                    current_cost = cost
            except Exception:
                pass

    if remaining() < 0.5:
        return best_sol

    # SA temperature setup
    if current_cost > 0 and current_cost < float('inf'):
        T_init = max(0.05 * current_cost / math.log(2), 1.0)
    else:
        T_init = 100.0
    T_min = 1e-4
    T = T_init

    lns_time = remaining() - 0.3
    if lns_time <= 0:
        return best_sol

    estimated_iters = max(100, int(lns_time * 40))
    cooling_rate = (T_min / T_init) ** (1.0 / max(1, estimated_iters))

    n_destroy_base = max(1, N // 10)

    destroy_ops = ['random', 'worst_crew', 'segment', 'related', 'expensive']
    op_success = [0] * len(destroy_ops)
    op_tries = [1] * len(destroy_ops)

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
    last_ls_iter = 0
    restart_count = 0

    while remaining() > 0.4:
        iteration += 1
        T = max(T * cooling_rate, T_min)

        # Adaptive destroy size
        n_destroy = n_destroy_base + min(no_improve_streak // 6, N // 5)
        n_destroy = max(1, min(n_destroy, N // 2))

        op_idx = select_destroy_op()
        op_tries[op_idx] += 1
        destroy_op = destroy_ops[op_idx]

        if destroy_op == 'random':
            partial_crews, removed = destroy_random_tasks(current_crews, n_destroy, rng)
        elif destroy_op == 'worst_crew':
            n_cd = max(1, min(3, len(current_crews) // 3))
            partial_crews, removed = destroy_worst_crew(current_crews, n_cd, rng)
        elif destroy_op == 'segment':
            partial_crews, removed = destroy_segment(current_crews, rng)
        elif destroy_op == 'related':
            partial_crews, removed = destroy_related_tasks(current_crews, n_destroy, rng)
        else:
            partial_crews, removed = destroy_expensive_arcs(current_crews, n_destroy, rng)

        if not removed:
            continue

        # Alternate between greedy and regret repair
        use_regret = (iteration % 4 == 0) and (len(removed) <= 25)
        if use_regret:
            new_crews = regret_repair(removed, partial_crews, K)
        else:
            new_crews = greedy_repair(removed, partial_crews, K)

        if new_crews is None:
            continue

        # Verify coverage
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

        if not all(check_crew_fast(crew) for crew in new_crews):
            continue

        new_cost = solution_cost_fast(new_crews)
        delta = new_cost - current_cost

        accept = False
        if delta < 0:
            accept = True
        elif T > T_min * 10 and delta < float('inf'):
            prob = math.exp(-delta / T)
            accept = rng.random() < prob

        if accept:
            current_crews = new_crews
            current_cost = new_cost

            if new_cost < lns_best_cost - 1e-9:
                candidate = {'crews': [list(c) for c in new_crews]}
                try:
                    feasible, _ = tools['is_feasible'](candidate)
                    if feasible:
                        obj = tools['objective'](candidate)
                        if obj < lns_best_cost:
                            lns_best_cost = obj
                            lns_best_crews = [list(c) for c in new_crews]
                            op_success[op_idx] += 1
                            no_improve_streak = 0
                            update_best(candidate)
                except Exception:
                    pass
            else:
                no_improve_streak += 1
        else:
            no_improve_streak += 1

        # Periodic local search refinement
        ls_interval = max(15, 50 - N // 10)
        if (iteration - last_ls_iter >= ls_interval and
                remaining() > 1.5 and no_improve_streak > 10):
            ls_budget = min(remaining() * 0.15, 8.0)
            ls_crews, ls_cost = run_local_search(
                [list(c) for c in current_crews], ls_budget)
            ls_sol = {'crews': ls_crews}
            try:
                feasible, _ = tools['is_feasible'](ls_sol)
                if feasible:
                    obj = tools['objective'](ls_sol)
                    if obj < lns_best_cost:
                        lns_best_cost = obj
                        lns_best_crews = [list(c) for c in ls_crews]
                        update_best(ls_sol)
                        no_improve_streak = 0
                    if obj < current_cost:
                        current_crews = ls_crews
                        current_cost = obj
            except Exception:
                pass
            last_ls_iter = iteration

        # Restart strategy: mix exploitation and diversification
        if no_improve_streak > 60 and remaining() > 1.5:
            restart_count += 1
            no_improve_streak = 0

            if restart_count % 4 == 0 and remaining() > 3.0:
                # Diversification: randomized greedy construction
                alpha = rng.uniform(0.1, 0.5)
                rand_sol = greedy_construct(rng=rng, alpha=alpha)
                if rand_sol is not None:
                    rand_crews = rand_sol['crews']
                    if (len(rand_crews) <= K and
                            all(check_crew_fast(c) for c in rand_crews) and
                            len(set(t for c in rand_crews for t in c)) == N):
                        rand_cost = solution_cost_fast(rand_crews)
                        if rand_cost < lns_best_cost * 1.6:
                            current_crews = rand_crews
                            current_cost = rand_cost
                            if rand_cost < lns_best_cost - 1e-9:
                                candidate = {'crews': [list(c) for c in rand_crews]}
                                update_best(candidate)
                                lns_best_cost = rand_cost
                                lns_best_crews = rand_crews
                        else:
                            current_crews = [list(c) for c in lns_best_crews]
                            current_cost = lns_best_cost
                    else:
                        current_crews = [list(c) for c in lns_best_crews]
                        current_cost = lns_best_cost
                else:
                    current_crews = [list(c) for c in lns_best_crews]
                    current_cost = lns_best_cost
                T = max(T, T_init * 0.25)
            else:
                # Exploitation: return to best known
                current_crews = [list(c) for c in lns_best_crews]
                current_cost = lns_best_cost
                T = max(T, T_init * 0.15)

    # Final local search polish
    if remaining() > 0.5:
        polish_budget = remaining() * 0.9
        polished, _ = run_local_search(
            [list(c) for c in lns_best_crews], polish_budget)
        update_best({'crews': polished})

    return best_sol if best_sol is not None else {
        'crews': [[i] for i in range(1, min(K + 1, N + 1))]
    }