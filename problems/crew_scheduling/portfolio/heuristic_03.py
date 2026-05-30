# MACE evolved heuristic 03/10 for problem: crew_scheduling
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
    # Fast data structures
    # ------------------------------------------------------------------ #
    succ_list = defaultdict(list)
    pred_list = defaultdict(list)
    for (i, j), cost in arcs.items():
        if tasks[i][1] <= tasks[j][0]:
            succ_list[i].append((tasks[j][0], j, cost))
            pred_list[j].append((tasks[i][1], i, cost))
    for t in range(1, N + 1):
        succ_list[t].sort()
        pred_list[t].sort()

    def check_crew_fast(crew):
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

    def crew_cost_fast(crew):
        cost = 0.0
        for idx in range(len(crew) - 1):
            key = (crew[idx], crew[idx + 1])
            if key not in arcs:
                return float('inf')
            cost += arcs[key]
        return cost

    def solution_cost_fast(crews):
        return sum(crew_cost_fast(c) for c in crews)

    # ------------------------------------------------------------------ #
    # Instance feature analysis for dispatch
    # ------------------------------------------------------------------ #
    # Arc density: number of valid arcs relative to N^2
    arc_density = len(arcs) / max(N * N, 1)

    # Average successors per task
    avg_successors = sum(len(succ_list[t]) for t in range(1, N + 1)) / max(N, 1)

    # N/K ratio: tasks per crew
    tasks_per_crew = N / max(K, 1)

    # Time span analysis
    all_starts = [tasks[t][0] for t in range(1, N + 1)]
    all_finishes = [tasks[t][1] for t in range(1, N + 1)]
    total_span = max(all_finishes) - min(all_starts) if N > 0 else 1.0
    time_limit_ratio = time_limit / max(total_span, 1.0)

    # Classify instance regime
    # ILP regime: small N, low arc density (hard to find feasible), tight time limits
    # MCF/ILS regime: larger N, higher density, more flexible
    is_tiny = N <= 40
    is_small = 40 < N <= 100
    is_medium = 100 < N <= 250
    is_large = N > 250

    # Dense arc graph -> greedy/MCF works well; sparse -> ILP more needed
    is_sparse = avg_successors < 2.0
    is_dense = avg_successors >= 5.0

    # Tight duty time -> harder to chain tasks -> ILP more useful for small instances
    is_tight_time = time_limit_ratio < 0.5

    # Dispatch decision:
    # "ILP-first" regime: tiny instances OR (small + sparse + tight time)
    # "MCF/ILS-first" regime: everything else
    use_ilp_first = is_tiny or (is_small and is_sparse and is_tight_time)
    use_mcf_first = is_large or (is_medium and not is_sparse) or (is_dense and not is_tiny)

    # ------------------------------------------------------------------ #
    # Solution tracking
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
    # Custom greedy construction
    # ------------------------------------------------------------------ #
    def greedy_construct(shuffle_seed=None, prefer_low_cost=True):
        task_order = sorted(range(1, N + 1), key=lambda t: tasks[t][0])
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            i = 0
            while i < len(task_order):
                j = i
                while j < len(task_order) and tasks[task_order[j]][0] == tasks[task_order[i]][0]:
                    j += 1
                seg = task_order[i:j]
                rng.shuffle(seg)
                task_order[i:j] = seg
                i = j

        crews_g = []
        crew_last = []
        crew_start = []
        crew_finish = []

        for task in task_order:
            t_start, t_finish = tasks[task]
            best_crew_idx = -1
            best_c = float('inf')

            for i in range(len(crews_g)):
                last = crew_last[i]
                if crew_finish[i] > t_start:
                    continue
                if (last, task) not in arcs:
                    continue
                if t_finish - crew_start[i] > time_limit:
                    continue
                c = arcs[(last, task)]
                if c < best_c:
                    best_c = c
                    best_crew_idx = i

            if best_crew_idx >= 0:
                crews_g[best_crew_idx].append(task)
                crew_last[best_crew_idx] = task
                crew_finish[best_crew_idx] = t_finish
            elif len(crews_g) < K:
                crews_g.append([task])
                crew_last.append(task)
                crew_start.append(t_start)
                crew_finish.append(t_finish)
            else:
                forced = False
                for i in range(len(crews_g)):
                    last = crew_last[i]
                    if crew_finish[i] <= t_start and (last, task) in arcs:
                        if t_finish - crew_start[i] <= time_limit:
                            crews_g[i].append(task)
                            crew_last[i] = task
                            crew_finish[i] = t_finish
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

    # ------------------------------------------------------------------ #
    # Phase 1: Construction based on dispatch
    # ------------------------------------------------------------------ #

    if use_ilp_first:
        # ILP-first: exact solver with generous budget for tiny/hard instances
        ilp_budget = min(remaining() * 0.45, 40.0)
        if remaining() > 3:
            try:
                sol = tools['ilp_crew_scheduling'](time_limit_s=ilp_budget)
                update_best(sol)
            except Exception:
                pass

        # MCF as complement
        if remaining() > 2:
            mcf_budget = min(remaining() * 0.35, 15.0)
            try:
                sol = tools['solve_min_cost_flow'](time_limit_s=mcf_budget)
                update_best(sol)
            except Exception:
                pass

    elif use_mcf_first:
        # MCF-first: polynomial solver for large/dense instances
        mcf_budget = min(remaining() * 0.30, 20.0)
        if remaining() > 2:
            try:
                sol = tools['solve_min_cost_flow'](time_limit_s=mcf_budget)
                update_best(sol)
            except Exception:
                pass

        # solve_default as backup for very large
        if is_large and remaining() > 3 and best_sol is None:
            sd_budget = min(remaining() * 0.25, 20.0)
            try:
                sol = tools['solve_default'](time_limit_s=sd_budget)
                update_best(sol)
            except Exception:
                pass

    else:
        # Balanced: medium instances - MCF then short ILP
        mcf_budget = min(remaining() * 0.28, 18.0)
        if remaining() > 2:
            try:
                sol = tools['solve_min_cost_flow'](time_limit_s=mcf_budget)
                update_best(sol)
            except Exception:
                pass

        if remaining() > 5:
            ilp_budget = min(remaining() * 0.25, 20.0)
            try:
                sol = tools['ilp_crew_scheduling'](time_limit_s=ilp_budget)
                update_best(sol)
            except Exception:
                pass

    # Greedy chain pack as fallback
    if remaining() > 1:
        try:
            chains = tools['greedy_chain_pack']()
            if chains and len(chains) <= K:
                sol = tools['make_solution'](chains)
                update_best(sol)
        except Exception:
            pass

    # Custom greedy constructions
    if remaining() > 0.5:
        sol = greedy_construct()
        update_best(sol)

    for seed in range(10):
        if remaining() < 0.3:
            break
        sol = greedy_construct(shuffle_seed=seed)
        update_best(sol)

    # Emergency fallback
    if best_sol is None:
        if K >= N:
            sol = {'crews': [[i] for i in range(1, N + 1)]}
            update_best(sol)
        if best_sol is None:
            task_order = sorted(range(1, N + 1), key=lambda t: tasks[t][0])
            crews_em = [[] for _ in range(min(K, N))]
            for idx, task in enumerate(task_order):
                crews_em[idx % len(crews_em)].append(task)
            sol = {'crews': [c for c in crews_em if c]}
            update_best(sol)

    if best_sol is None:
        return {'crews': [[i] for i in range(1, min(K + 1, N + 1))]}

    # ------------------------------------------------------------------ #
    # Local Search Neighborhoods
    # ------------------------------------------------------------------ #

    def relocation_pass(crews, cost):
        n_crews = len(crews)
        for ci in range(n_crews):
            crew_i = crews[ci]
            if len(crew_i) <= 1:
                continue
            for ti in range(len(crew_i)):
                task = crew_i[ti]
                t_start_task, t_finish_task = tasks[task]

                new_crew_i = crew_i[:ti] + crew_i[ti + 1:]
                if not check_crew_fast(new_crew_i):
                    continue

                old_cost_i = crew_cost_fast(crew_i)
                new_cost_i = crew_cost_fast(new_crew_i)
                saving_i = old_cost_i - new_cost_i

                best_gain = 1e-9
                best_cj = -1
                best_pos = -1

                for cj in range(n_crews):
                    if cj == ci:
                        continue
                    crew_j = crews[cj]
                    old_cost_j = crew_cost_fast(crew_j)

                    for pos in range(len(crew_j) + 1):
                        prev_j = crew_j[pos - 1] if pos > 0 else None
                        next_j = crew_j[pos] if pos < len(crew_j) else None

                        if prev_j is not None:
                            if tasks[prev_j][1] > t_start_task:
                                continue
                            if (prev_j, task) not in arcs:
                                continue
                        if next_j is not None:
                            if t_finish_task > tasks[next_j][0]:
                                continue
                            if (task, next_j) not in arcs:
                                continue

                        insert_cost = 0.0
                        if prev_j is not None:
                            insert_cost += arcs[(prev_j, task)]
                        if next_j is not None:
                            insert_cost += arcs[(task, next_j)]

                        bridge_cost = 0.0
                        if prev_j is not None and next_j is not None:
                            bridge_cost = arcs.get((prev_j, next_j), 0.0)

                        new_crew_j = crew_j[:pos] + [task] + crew_j[pos:]
                        duty = tasks[new_crew_j[-1]][1] - tasks[new_crew_j[0]][0]
                        if duty > time_limit:
                            continue

                        gain = saving_i - (insert_cost - bridge_cost)
                        if gain > best_gain:
                            best_gain = gain
                            best_cj = cj
                            best_pos = pos

                if best_cj >= 0:
                    old_cost_j = crew_cost_fast(crews[best_cj])
                    new_crew_j = crews[best_cj][:best_pos] + [task] + crews[best_cj][best_pos:]
                    new_cost_j = crew_cost_fast(new_crew_j)
                    cost = cost - old_cost_i - old_cost_j + new_cost_i + new_cost_j
                    crews[ci] = new_crew_i
                    crews[best_cj] = new_crew_j
                    return crews, cost, True

        return crews, cost, False

    def swap_pass(crews, cost):
        n_crews = len(crews)
        for ci in range(n_crews):
            for cj in range(ci + 1, n_crews):
                crew_i = crews[ci]
                crew_j = crews[cj]
                old_cost_ij = crew_cost_fast(crew_i) + crew_cost_fast(crew_j)
                best_gain = 1e-9
                best_pi = -1
                best_pj = -1

                for pi in range(len(crew_i)):
                    for pj in range(len(crew_j)):
                        task_i = crew_i[pi]
                        task_j = crew_j[pj]

                        ti_start, ti_finish = tasks[task_i]
                        tj_start, tj_finish = tasks[task_j]

                        if pi > 0 and tasks[crew_i[pi - 1]][1] > tj_start:
                            continue
                        if pi < len(crew_i) - 1 and tj_finish > tasks[crew_i[pi + 1]][0]:
                            continue
                        if pj > 0 and tasks[crew_j[pj - 1]][1] > ti_start:
                            continue
                        if pj < len(crew_j) - 1 and ti_finish > tasks[crew_j[pj + 1]][0]:
                            continue

                        new_ci = list(crew_i)
                        new_cj = list(crew_j)
                        new_ci[pi] = task_j
                        new_cj[pj] = task_i

                        if not check_crew_fast(new_ci):
                            continue
                        if not check_crew_fast(new_cj):
                            continue

                        new_cost_ij = crew_cost_fast(new_ci) + crew_cost_fast(new_cj)
                        gain = old_cost_ij - new_cost_ij
                        if gain > best_gain:
                            best_gain = gain
                            best_pi = pi
                            best_pj = pj

                if best_pi >= 0:
                    task_i = crew_i[best_pi]
                    task_j = crew_j[best_pj]
                    new_ci = list(crew_i)
                    new_cj = list(crew_j)
                    new_ci[best_pi] = task_j
                    new_cj[best_pj] = task_i
                    new_cost_ij = crew_cost_fast(new_ci) + crew_cost_fast(new_cj)
                    cost = cost - old_cost_ij + new_cost_ij
                    crews[ci] = new_ci
                    crews[cj] = new_cj
                    return crews, cost, True

        return crews, cost, False

    def oropt_pass(crews, cost, seg_len=2):
        n_crews = len(crews)
        for ci in range(n_crews):
            crew_i = crews[ci]
            if len(crew_i) <= seg_len:
                continue
            for ti in range(len(crew_i) - seg_len + 1):
                segment = crew_i[ti:ti + seg_len]
                new_crew_i = crew_i[:ti] + crew_i[ti + seg_len:]
                if not new_crew_i:
                    continue
                if not check_crew_fast(new_crew_i):
                    continue

                seg_valid = True
                for k in range(seg_len - 1):
                    if (segment[k], segment[k + 1]) not in arcs:
                        seg_valid = False
                        break
                if not seg_valid:
                    continue

                old_cost_i = crew_cost_fast(crew_i)
                new_cost_i = crew_cost_fast(new_crew_i)
                saving_i = old_cost_i - new_cost_i

                seg_start = tasks[segment[0]][0]
                seg_finish = tasks[segment[-1]][1]

                best_gain = 1e-9
                best_cj = -1
                best_pos = -1

                for cj in range(n_crews):
                    if cj == ci:
                        continue
                    crew_j = crews[cj]
                    old_cost_j = crew_cost_fast(crew_j)

                    for pos in range(len(crew_j) + 1):
                        prev_j = crew_j[pos - 1] if pos > 0 else None
                        next_j = crew_j[pos] if pos < len(crew_j) else None

                        if prev_j is not None and tasks[prev_j][1] > seg_start:
                            continue
                        if next_j is not None and seg_finish > tasks[next_j][0]:
                            continue

                        ok = True
                        insert_cost = crew_cost_fast(segment)
                        if prev_j is not None:
                            c = arcs.get((prev_j, segment[0]))
                            if c is None:
                                ok = False
                            else:
                                insert_cost += c
                        if ok and next_j is not None:
                            c = arcs.get((segment[-1], next_j))
                            if c is None:
                                ok = False
                            else:
                                insert_cost += c
                        if not ok:
                            continue

                        bridge_cost = 0.0
                        if prev_j is not None and next_j is not None:
                            bridge_cost = arcs.get((prev_j, next_j), 0.0)

                        new_crew_j = crew_j[:pos] + segment + crew_j[pos:]
                        duty = tasks[new_crew_j[-1]][1] - tasks[new_crew_j[0]][0]
                        if duty > time_limit:
                            continue

                        new_cost_j = crew_cost_fast(new_crew_j)
                        gain = saving_i - (new_cost_j - old_cost_j)
                        if gain > best_gain:
                            best_gain = gain
                            best_cj = cj
                            best_pos = pos

                if best_cj >= 0:
                    old_cost_j = crew_cost_fast(crews[best_cj])
                    new_crew_j = crews[best_cj][:best_pos] + segment + crews[best_cj][best_pos:]
                    new_cost_j = crew_cost_fast(new_crew_j)
                    cost = cost - old_cost_i - old_cost_j + new_cost_i + new_cost_j
                    crews[ci] = new_crew_i
                    crews[best_cj] = new_crew_j
                    return crews, cost, True

        return crews, cost, False

    def intra_2opt_pass(crews, cost):
        improved = False
        for ci in range(len(crews)):
            crew = crews[ci]
            n = len(crew)
            if n < 3:
                continue
            old_cost_c = crew_cost_fast(crew)
            best_gain = 1e-9
            best_ij = (-1, -1)
            for i in range(n - 1):
                for j in range(i + 2, n):
                    new_crew = crew[:i + 1] + crew[i + 1:j + 1][::-1] + crew[j + 1:]
                    if not check_crew_fast(new_crew):
                        continue
                    new_cost_c = crew_cost_fast(new_crew)
                    gain = old_cost_c - new_cost_c
                    if gain > best_gain:
                        best_gain = gain
                        best_ij = (i, j)
            if best_ij[0] >= 0:
                i, j = best_ij
                new_crew = crew[:i + 1] + crew[i + 1:j + 1][::-1] + crew[j + 1:]
                new_cost_c = crew_cost_fast(new_crew)
                cost = cost - old_cost_c + new_cost_c
                crews[ci] = new_crew
                improved = True
        return crews, cost, improved

    def merge_pass(crews, cost):
        n_crews = len(crews)
        for ci in range(n_crews):
            for cj in range(n_crews):
                if ci == cj:
                    continue
                tail_i = crews[ci][-1]
                head_j = crews[cj][0]
                if tasks[tail_i][1] > tasks[head_j][0]:
                    continue
                if (tail_i, head_j) not in arcs:
                    continue
                merged = crews[ci] + crews[cj]
                if not check_crew_fast(merged):
                    continue
                old_cost_ij = crew_cost_fast(crews[ci]) + crew_cost_fast(crews[cj])
                new_cost_m = crew_cost_fast(merged)
                if new_cost_m < old_cost_ij - 1e-9:
                    new_crews = []
                    for k in range(n_crews):
                        if k == ci:
                            new_crews.append(merged)
                        elif k == cj:
                            pass
                        else:
                            new_crews.append(list(crews[k]))
                    cost = cost - old_cost_ij + new_cost_m
                    return new_crews, cost, True
        return crews, cost, False

    def cross_crew_2opt(crews, cost):
        """Exchange tails between two crews: crew_i[:p+1] + crew_j[q+1:] and crew_j[:q+1] + crew_i[p+1:]"""
        n_crews = len(crews)
        for ci in range(n_crews):
            for cj in range(ci + 1, n_crews):
                crew_i = crews[ci]
                crew_j = crews[cj]
                old_cost_ij = crew_cost_fast(crew_i) + crew_cost_fast(crew_j)
                best_gain = 1e-9
                best_p = -1
                best_q = -1

                for p in range(len(crew_i)):
                    for q in range(len(crew_j)):
                        # new_ci = crew_i[:p+1] + crew_j[q+1:]
                        # new_cj = crew_j[:q+1] + crew_i[p+1:]
                        new_ci = crew_i[:p + 1] + crew_j[q + 1:]
                        new_cj = crew_j[:q + 1] + crew_i[p + 1:]

                        if not new_ci or not new_cj:
                            continue
                        if not check_crew_fast(new_ci):
                            continue
                        if not check_crew_fast(new_cj):
                            continue

                        new_cost_ij = crew_cost_fast(new_ci) + crew_cost_fast(new_cj)
                        gain = old_cost_ij - new_cost_ij
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                            best_q = q

                if best_p >= 0:
                    new_ci = crew_i[:best_p + 1] + crew_j[best_q + 1:]
                    new_cj = crew_j[:best_q + 1] + crew_i[best_p + 1:]
                    new_cost_ij = crew_cost_fast(new_ci) + crew_cost_fast(new_cj)
                    cost = cost - old_cost_ij + new_cost_ij
                    crews[ci] = new_ci
                    crews[cj] = new_cj
                    return crews, cost, True

        return crews, cost, False

    def run_local_search(crews_in, time_budget):
        crews = [list(c) for c in crews_in]
        cost = solution_cost_fast(crews)
        ls_start = time.time()

        def ls_remaining():
            return time_budget - (time.time() - ls_start)

        avg_crew_sz = N / max(len(crews), 1)
        use_2opt = avg_crew_sz <= 30
        use_cross = avg_crew_sz <= 20

        improved_global = True
        while improved_global and ls_remaining() > 0.05:
            improved_global = False

            # Relocation (most powerful)
            while ls_remaining() > 0.05:
                crews, cost, imp = relocation_pass(crews, cost)
                if not imp:
                    break
                improved_global = True

            if ls_remaining() < 0.05:
                break

            # Swap
            while ls_remaining() > 0.05:
                crews, cost, imp = swap_pass(crews, cost)
                if not imp:
                    break
                improved_global = True

            if ls_remaining() < 0.05:
                break

            # Or-opt seg=2
            while ls_remaining() > 0.05:
                crews, cost, imp = oropt_pass(crews, cost, seg_len=2)
                if not imp:
                    break
                improved_global = True

            if ls_remaining() < 0.05:
                break

            # Or-opt seg=3
            while ls_remaining() > 0.05:
                crews, cost, imp = oropt_pass(crews, cost, seg_len=3)
                if not imp:
                    break
                improved_global = True

            if ls_remaining() < 0.05:
                break

            # Cross-crew 2-opt (tail exchange) - good for small instances
            if use_cross:
                while ls_remaining() > 0.05:
                    crews, cost, imp = cross_crew_2opt(crews, cost)
                    if not imp:
                        break
                    improved_global = True

            if ls_remaining() < 0.05:
                break

            # 2-opt within crews
            if use_2opt:
                while ls_remaining() > 0.05:
                    crews, cost, imp = intra_2opt_pass(crews, cost)
                    if not imp:
                        break
                    improved_global = True

            if ls_remaining() < 0.05:
                break

            # Merge
            while ls_remaining() > 0.05:
                crews, cost, imp = merge_pass(crews, cost)
                if not imp:
                    break
                improved_global = True

        return crews, cost

    def perturbation(crews_in, rng, n_moves=3):
        crews = [list(c) for c in crews_in]
        for _ in range(n_moves):
            if not crews:
                break
            eligible = [i for i in range(len(crews)) if len(crews[i]) > 1]
            if not eligible:
                break
            ci = rng.choice(eligible)
            crew_i = crews[ci]
            ti = rng.randint(0, len(crew_i) - 1)
            task = crew_i[ti]
            new_crew_i = crew_i[:ti] + crew_i[ti + 1:]
            if not check_crew_fast(new_crew_i):
                continue
            other_crews = [j for j in range(len(crews)) if j != ci]
            if not other_crews:
                continue
            rng.shuffle(other_crews)
            inserted = False
            for cj in other_crews:
                crew_j = crews[cj]
                positions = list(range(len(crew_j) + 1))
                rng.shuffle(positions)
                for pos in positions:
                    new_crew_j = crew_j[:pos] + [task] + crew_j[pos:]
                    if check_crew_fast(new_crew_j):
                        crews[ci] = new_crew_i
                        crews[cj] = new_crew_j
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted and len(crews) < K:
                crews[ci] = new_crew_i
                crews.append([task])
        return [c for c in crews if c]

    def double_bridge_perturbation(crews_in, rng):
        """Double-bridge style perturbation: split two crews and recombine."""
        crews = [list(c) for c in crews_in]
        eligible = [i for i in range(len(crews)) if len(crews[i]) >= 2]
        if len(eligible) < 2:
            return crews
        ci, cj = rng.sample(eligible, 2)
        crew_i = crews[ci]
        crew_j = crews[cj]
        # Split points
        pi = rng.randint(1, len(crew_i) - 1) if len(crew_i) > 1 else 1
        pj = rng.randint(1, len(crew_j) - 1) if len(crew_j) > 1 else 1
        # Try recombining: crew_i[:pi] + crew_j[pj:] and crew_j[:pj] + crew_i[pi:]
        new_ci = crew_i[:pi] + crew_j[pj:]
        new_cj = crew_j[:pj] + crew_i[pi:]
        if new_ci and new_cj and check_crew_fast(new_ci) and check_crew_fast(new_cj):
            crews[ci] = new_ci
            crews[cj] = new_cj
        return crews

    # ------------------------------------------------------------------ #
    # ILS Main Loop
    # ------------------------------------------------------------------ #
    if best_sol is not None and remaining() > 1.0:
        rng = random.Random(42)

        # Tune initial LS budget based on regime
        if use_ilp_first:
            # For ILP regime, ILP may already be near-optimal; moderate LS
            initial_ls_frac = 0.45
            max_initial_ls = 25.0
        elif is_large:
            initial_ls_frac = 0.38
            max_initial_ls = 25.0
        else:
            initial_ls_frac = 0.42
            max_initial_ls = 30.0

        initial_ls_budget = min(remaining() * initial_ls_frac, max_initial_ls)
        current_crews, current_cost = run_local_search(best_sol['crews'], initial_ls_budget)

        sol = {'crews': current_crews}
        update_best(sol)

        # ILS loop
        ils_crews = [list(c) for c in best_sol['crews']]
        no_improve_count = 0
        iteration = 0

        # For ILP-first regime, also try a second ILP call with remaining time
        # if the ILS is stuck early
        ilp_retry_done = False

        while remaining() > 1.5:
            iteration += 1

            # Adaptive perturbation strength
            perturb_strength = 3 + no_improve_count // 3

            # Alternate between relocation perturbation and double-bridge
            if iteration % 4 == 0 and not is_large:
                perturbed = double_bridge_perturbation(ils_crews, rng)
                # Also do some relocations
                perturbed = perturbation(perturbed, rng, n_moves=2)
            else:
                perturbed = perturbation(ils_crews, rng, n_moves=perturb_strength)

            if not perturbed:
                break

            # Verify all tasks present after perturbation
            all_t = set()
            for c in perturbed:
                all_t.update(c)
            if len(all_t) != N:
                ils_crews = [list(c) for c in best_sol['crews']]
                no_improve_count = 0
                continue

            # LS budget per iteration - scale with instance size
            if is_tiny:
                ls_time = min(remaining() * 0.50, 15.0)
            elif is_small:
                ls_time = min(remaining() * 0.45, 12.0)
            elif is_medium:
                ls_time = min(remaining() * 0.38, 9.0)
            else:
                ls_time = min(remaining() * 0.30, 6.0)

            new_crews, new_cost = run_local_search(perturbed, ls_time)
            sol = {'crews': new_crews}

            if update_best(sol):
                ils_crews = [list(c) for c in new_crews]
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Accept slightly worse for diversification
                if new_cost < best_cost * 1.02:
                    ils_crews = [list(c) for c in new_crews]
                else:
                    ils_crews = [list(c) for c in best_sol['crews']]

            # For ILP-first regime: retry ILP if stuck and time allows
            if use_ilp_first and not ilp_retry_done and no_improve_count >= 5 and remaining() > 8.0:
                ilp_retry_budget = min(remaining() * 0.35, 15.0)
                try:
                    sol2 = tools['ilp_crew_scheduling'](time_limit_s=ilp_retry_budget)
                    if update_best(sol2):
                        ils_crews = [list(c) for c in best_sol['crews']]
                        no_improve_count = 0
                except Exception:
                    pass
                ilp_retry_done = True

            # Restart from best if stuck
            if no_improve_count >= 8:
                ils_crews = [list(c) for c in best_sol['crews']]
                no_improve_count = 0

    return best_sol if best_sol is not None else {'crews': [[i] for i in range(1, min(K + 1, N + 1))]}