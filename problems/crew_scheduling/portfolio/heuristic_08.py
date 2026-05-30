# MACE evolved heuristic 08/10 for problem: crew_scheduling
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

    # Precompute adjacency
    succ = defaultdict(list)
    pred = defaultdict(list)
    for (i, j), cost in arcs.items():
        if tasks[i][1] <= tasks[j][0]:
            succ[i].append(j)
            pred[j].append(i)

    arc_set = set(arcs.keys())

    # ------------------------------------------------------------------ #
    # Helper functions
    # ------------------------------------------------------------------ #
    def crew_valid(crew):
        if not crew:
            return False
        if tasks[crew[-1]][1] - tasks[crew[0]][0] > time_limit:
            return False
        for idx in range(len(crew) - 1):
            t1, t2 = crew[idx], crew[idx + 1]
            if tasks[t1][1] > tasks[t2][0]:
                return False
            if (t1, t2) not in arc_set:
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

    def make_sol_dict(crews):
        return {'crews': [list(c) for c in crews if c]}

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
    # Dispatch: ILP for small N, flow/greedy+LNS for larger
    # ------------------------------------------------------------------ #
    arc_count = len(arcs)
    max_arcs = N * (N - 1)
    arc_density = arc_count / max_arcs if max_arcs > 0 else 0.0
    crew_slack = K / N if N > 0 else 1.0

    use_ilp = (N <= 70) or (arc_density < 0.04) or (crew_slack >= 0.8)

    # ------------------------------------------------------------------ #
    # Phase 1: Initial solution
    # ------------------------------------------------------------------ #
    if use_ilp:
        ilp_budget = min(remaining() * 0.50, time_limit_s * 0.50)
        if ilp_budget > 0.5:
            try:
                sol = tools['ilp_crew_scheduling'](time_limit_s=ilp_budget)
                update_best(sol)
            except Exception:
                pass

    # Try min cost flow
    if (best_sol is None or not use_ilp) and remaining() > 1.0:
        mcf_budget = min(remaining() * 0.35, 20.0)
        try:
            sol = tools['solve_min_cost_flow'](time_limit_s=mcf_budget)
            update_best(sol)
        except Exception:
            pass

    # Try solve_default
    if best_sol is None and remaining() > 1.0:
        init_budget = min(remaining() * 0.40, 30.0)
        try:
            sol = tools['solve_default'](time_limit_s=init_budget)
            update_best(sol)
        except Exception:
            pass

    # Greedy chain pack
    if best_sol is None and remaining() > 0.5:
        try:
            chains = tools['greedy_chain_pack']()
            if chains and len(chains) <= K:
                sol = tools['make_solution'](chains)
                update_best(sol)
        except Exception:
            pass

    # Custom greedy construction
    if best_sol is None and remaining() > 0.5:
        sol = _greedy_construct(N, K, tasks, arcs, arc_set, time_limit)
        update_best(sol)

    # Emergency fallback
    if best_sol is None:
        if K >= N:
            best_sol = {'crews': [[i] for i in range(1, N + 1)]}
            best_cost = 0.0
        else:
            task_order = sorted(range(1, N + 1), key=lambda t: tasks[t][0])
            crews_e = [[] for _ in range(K)]
            for idx, t in enumerate(task_order):
                crews_e[idx % K].append(t)
            best_sol = {'crews': [c for c in crews_e if c]}
            best_cost = float('inf')

    # ------------------------------------------------------------------ #
    # Phase 2: Local search improvement
    # ------------------------------------------------------------------ #
    if remaining() > 1.0:
        best_sol = _improve(
            best_sol, N, K, tasks, arcs, arc_set, time_limit,
            succ, pred, tools, start_time, time_limit_s,
            crew_valid, crew_cost_fn, total_cost_fn, make_sol_dict
        )

    return best_sol


def _greedy_construct(N, K, tasks, arcs, arc_set, time_limit):
    """Greedy construction: process tasks by start time, assign to best crew."""
    task_order = sorted(range(1, N + 1), key=lambda t: (tasks[t][0], tasks[t][1]))
    crews = []
    crew_last = []
    crew_start = []
    crew_finish = []

    for task in task_order:
        t_start, t_finish = tasks[task]
        best_crew_idx = -1
        best_cost_val = float('inf')

        for i in range(len(crews)):
            last = crew_last[i]
            if crew_finish[i] > t_start:
                continue
            if (last, task) not in arc_set:
                continue
            if t_finish - crew_start[i] > time_limit:
                continue
            c = arcs[(last, task)]
            if c < best_cost_val:
                best_cost_val = c
                best_crew_idx = i

        if best_crew_idx >= 0:
            crews[best_crew_idx].append(task)
            crew_last[best_crew_idx] = task
            crew_finish[best_crew_idx] = t_finish
        elif len(crews) < K:
            crews.append([task])
            crew_last.append(task)
            crew_start.append(t_start)
            crew_finish.append(t_finish)
        else:
            # Force assign: find any valid crew
            forced = False
            for i in range(len(crews)):
                last = crew_last[i]
                if crew_finish[i] <= t_start and (last, task) in arc_set:
                    if t_finish - crew_start[i] <= time_limit:
                        crews[i].append(task)
                        crew_last[i] = task
                        crew_finish[i] = t_finish
                        forced = True
                        break
            if not forced:
                return None

    return {'crews': crews}


def _improve(initial_sol, N, K, tasks, arcs, arc_set, time_limit,
             succ, pred, tools, start_time, time_limit_s,
             crew_valid, crew_cost_fn, total_cost_fn, make_sol_dict):
    """Combined LNS + SA + local search improvement."""

    def remaining():
        return time_limit_s - (time.time() - start_time)

    # ------------------------------------------------------------------ #
    # Repair operators
    # ------------------------------------------------------------------ #
    def greedy_repair(unassigned, current_crews, max_crews):
        if not unassigned:
            return current_crews
        crews = [list(c) for c in current_crews]
        task_order = sorted(unassigned, key=lambda t: tasks[t][0])

        for task in task_order:
            ts_t, tf_t = tasks[task]
            best_delta = float('inf')
            best_ci = -1
            best_pos = -1

            for ci, crew in enumerate(crews):
                n = len(crew)
                # Try appending at end first (most common case)
                positions = list(range(n + 1))
                for pos in positions:
                    if pos > 0 and tasks[crew[pos - 1]][1] > ts_t:
                        continue
                    if pos < n and tf_t > tasks[crew[pos]][0]:
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

                    # Duty time check
                    head_start = tasks[crew[0]][0] if pos > 0 else ts_t
                    tail_finish = tasks[crew[-1]][1] if pos < n else tf_t
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

    def regret2_repair(unassigned, current_crews, max_crews):
        """Regret-2 repair: insert task with highest regret first."""
        if not unassigned:
            return current_crews
        crews = [list(c) for c in current_crews]
        remaining_tasks = list(unassigned)

        while remaining_tasks:
            regrets = []
            for task in remaining_tasks:
                ts_t, tf_t = tasks[task]
                insertions = []

                for ci, crew in enumerate(crews):
                    n = len(crew)
                    for pos in range(n + 1):
                        if pos > 0 and tasks[crew[pos - 1]][1] > ts_t:
                            continue
                        if pos < n and tf_t > tasks[crew[pos]][0]:
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

                        head_start = tasks[crew[0]][0] if pos > 0 else ts_t
                        tail_finish = tasks[crew[-1]][1] if pos < n else tf_t
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
    # Destroy operators
    # ------------------------------------------------------------------ #
    def destroy_random(crews, n_remove, rng):
        all_tasks = [(ci, t) for ci, crew in enumerate(crews) for t in crew]
        if not all_tasks:
            return crews, []
        n_remove = min(n_remove, len(all_tasks))
        removed_entries = rng.sample(all_tasks, n_remove)
        removed_set = set(t for _, t in removed_entries)
        new_crews = [[t for t in crew if t not in removed_set] for crew in crews]
        new_crews = [c for c in new_crews if c]
        return new_crews, list(removed_set)

    def destroy_worst_crew(crews, n_destroy, rng):
        if not crews:
            return crews, []
        sorted_crews = sorted(range(len(crews)), key=lambda i: crew_cost_fn(crews[i]), reverse=True)
        n_destroy = min(n_destroy, len(crews))
        destroyed = set(sorted_crews[:n_destroy])
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
        # Weighted by cost
        crew_costs = [max(crew_cost_fn(crew), 0.01) for crew in crews]
        total_w = sum(crew_costs)
        r = rng.random() * total_w
        cumulative = 0.0
        ci = 0
        for i, c in enumerate(crew_costs):
            cumulative += c
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

    def destroy_related(crews, n_remove, rng):
        all_tasks_flat = [t for crew in crews for t in crew]
        if not all_tasks_flat:
            return crews, []
        seed_task = rng.choice(all_tasks_flat)
        removed_set = {seed_task}
        candidates = set()
        for j in succ[seed_task]:
            candidates.add(j)
        for i in pred[seed_task]:
            candidates.add(i)
        seed_start = tasks[seed_task][0]
        time_window = time_limit * 0.15
        for t in all_tasks_flat:
            if abs(tasks[t][0] - seed_start) < time_window:
                candidates.add(t)
        candidates -= removed_set
        candidates = sorted(candidates, key=lambda t: abs(tasks[t][0] - seed_start))
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
        """Remove tasks adjacent to expensive arcs."""
        arc_costs_list = []
        for ci, crew in enumerate(crews):
            for pos in range(len(crew) - 1):
                c = arcs.get((crew[pos], crew[pos + 1]), 0)
                arc_costs_list.append((c, ci, pos))
        if not arc_costs_list:
            return crews, []
        arc_costs_list.sort(reverse=True)
        removed_set = set()
        for c, ci, pos in arc_costs_list[:max(1, n_remove // 2)]:
            removed_set.add(crews[ci][pos])
            removed_set.add(crews[ci][pos + 1])
            if len(removed_set) >= n_remove:
                break
        new_crews = [[t for t in crew if t not in removed_set] for crew in crews]
        new_crews = [c for c in new_crews if c]
        return new_crews, list(removed_set)

    # ------------------------------------------------------------------ #
    # Local search: single-task relocation (best improvement)
    # ------------------------------------------------------------------ #
    def relocate_pass(crews, rng):
        """One pass of single-task relocation."""
        improved = False

        task_to_crew = {}
        task_to_pos = {}
        for ci, crew in enumerate(crews):
            for pos, t in enumerate(crew):
                task_to_crew[t] = ci
                task_to_pos[t] = pos

        all_task_ids = list(range(1, N + 1))
        rng.shuffle(all_task_ids)

        for task in all_task_ids:
            if remaining() < 0.5:
                break
            src_ci = task_to_crew[task]
            src_pos = task_to_pos[task]
            src_crew = crews[src_ci]

            # Check if removing task is valid (bridge check)
            if src_pos > 0 and src_pos < len(src_crew) - 1:
                prev_t = src_crew[src_pos - 1]
                next_t = src_crew[src_pos + 1]
                if (prev_t, next_t) not in arc_set:
                    continue
                if tasks[prev_t][1] > tasks[next_t][0]:
                    continue

            new_src = src_crew[:src_pos] + src_crew[src_pos + 1:]
            if new_src and tasks[new_src[-1]][1] - tasks[new_src[0]][0] > time_limit:
                continue

            # Cost change from removal
            removal_gain = 0.0
            if src_pos > 0:
                removal_gain += arcs.get((src_crew[src_pos - 1], task), 0.0)
            if src_pos < len(src_crew) - 1:
                removal_gain += arcs.get((task, src_crew[src_pos + 1]), 0.0)
            if src_pos > 0 and src_pos < len(src_crew) - 1:
                removal_gain -= arcs.get((src_crew[src_pos - 1], src_crew[src_pos + 1]), 0.0)

            ts_t, tf_t = tasks[task]
            best_ins_delta = float('inf')
            best_dst_ci = -1
            best_ins_pos = -1

            for dst_ci, dst_crew in enumerate(crews):
                if dst_ci == src_ci:
                    continue
                n = len(dst_crew)
                for pos in range(n + 1):
                    if pos > 0 and tasks[dst_crew[pos - 1]][1] > ts_t:
                        continue
                    if pos < n and tf_t > tasks[dst_crew[pos]][0]:
                        continue

                    cost_add = 0.0
                    ok = True
                    if pos > 0:
                        c = arcs.get((dst_crew[pos - 1], task))
                        if c is None:
                            ok = False
                        else:
                            cost_add += c
                    if ok and pos < n:
                        c = arcs.get((task, dst_crew[pos]))
                        if c is None:
                            ok = False
                        else:
                            cost_add += c
                    if not ok:
                        continue

                    bridge = 0.0
                    if 0 < pos < n:
                        bridge = arcs.get((dst_crew[pos - 1], dst_crew[pos]), 0.0)

                    head_start = tasks[dst_crew[0]][0] if pos > 0 else ts_t
                    tail_finish = tasks[dst_crew[-1]][1] if pos < n else tf_t
                    if tail_finish - head_start > time_limit:
                        continue

                    delta = cost_add - bridge
                    if delta < best_ins_delta:
                        best_ins_delta = delta
                        best_dst_ci = dst_ci
                        best_ins_pos = pos

            if best_dst_ci < 0:
                continue

            total_delta = best_ins_delta - removal_gain
            if total_delta < -1e-9:
                dst_crew = crews[best_dst_ci]
                new_dst = dst_crew[:best_ins_pos] + [task] + dst_crew[best_ins_pos:]

                crews[src_ci] = new_src
                crews[best_dst_ci] = new_dst

                # Update mappings
                if new_src:
                    for p, t in enumerate(new_src):
                        task_to_crew[t] = src_ci
                        task_to_pos[t] = p
                task_to_crew[task] = best_dst_ci
                task_to_pos[task] = best_ins_pos
                for p, t in enumerate(new_dst):
                    task_to_crew[t] = best_dst_ci
                    task_to_pos[t] = p

                improved = True

        return crews, improved

    def swap_pass(crews, rng):
        """One pass of pairwise task swaps between crews."""
        improved = False
        crew_indices = list(range(len(crews)))
        rng.shuffle(crew_indices)

        for ci in crew_indices:
            if remaining() < 0.5:
                break
            for cj in crew_indices:
                if ci >= cj:
                    continue
                if remaining() < 0.5:
                    break

                crew_i = crews[ci]
                crew_j = crews[cj]

                for pi in range(len(crew_i)):
                    if remaining() < 0.5:
                        break
                    for pj in range(len(crew_j)):
                        ti = crew_i[pi]
                        tj = crew_j[pj]

                        # Build candidate swapped crews
                        new_ci = crew_i[:pi] + [tj] + crew_i[pi + 1:]
                        new_cj = crew_j[:pj] + [ti] + crew_j[pj + 1:]

                        if not crew_valid(new_ci) or not crew_valid(new_cj):
                            continue

                        old_cost = crew_cost_fn(crew_i) + crew_cost_fn(crew_j)
                        new_cost = crew_cost_fn(new_ci) + crew_cost_fn(new_cj)

                        if new_cost < old_cost - 1e-9:
                            crews[ci] = new_ci
                            crews[cj] = new_cj
                            improved = True

        return crews, improved

    def or_opt_pass(crews, seg_len, rng):
        """Or-opt: move a segment of seg_len tasks between crews."""
        improved = False
        crew_indices = list(range(len(crews)))
        rng.shuffle(crew_indices)

        for src_ci in crew_indices:
            if remaining() < 0.5:
                break
            src_crew = crews[src_ci]
            if len(src_crew) <= seg_len:
                continue

            for src_pos in range(len(src_crew) - seg_len + 1):
                segment = src_crew[src_pos:src_pos + seg_len]
                seg_start_t = tasks[segment[0]][0]
                seg_finish_t = tasks[segment[-1]][1]

                # Cost of removing segment
                removal_gain = 0.0
                if src_pos > 0:
                    removal_gain += arcs.get((src_crew[src_pos - 1], segment[0]), 0.0)
                if src_pos + seg_len < len(src_crew):
                    removal_gain += arcs.get((segment[-1], src_crew[src_pos + seg_len]), 0.0)

                # Bridge cost
                bridge_cost = 0.0
                if src_pos > 0 and src_pos + seg_len < len(src_crew):
                    prev_t = src_crew[src_pos - 1]
                    next_t = src_crew[src_pos + seg_len]
                    if (prev_t, next_t) not in arc_set:
                        continue
                    bridge_cost = arcs[(prev_t, next_t)]

                new_src = src_crew[:src_pos] + src_crew[src_pos + seg_len:]
                if new_src and tasks[new_src[-1]][1] - tasks[new_src[0]][0] > time_limit:
                    continue

                best_delta = -1e-9
                best_dst_ci = -1
                best_ins_pos = -1

                for dst_ci in crew_indices:
                    if dst_ci == src_ci:
                        continue
                    dst_crew = crews[dst_ci]

                    for ins_pos in range(len(dst_crew) + 1):
                        if ins_pos > 0 and tasks[dst_crew[ins_pos - 1]][1] > seg_start_t:
                            continue
                        if ins_pos < len(dst_crew) and seg_finish_t > tasks[dst_crew[ins_pos]][0]:
                            continue

                        ins_cost = 0.0
                        ok = True
                        if ins_pos > 0:
                            c = arcs.get((dst_crew[ins_pos - 1], segment[0]))
                            if c is None:
                                ok = False
                            else:
                                ins_cost += c
                        if ok and ins_pos < len(dst_crew):
                            c = arcs.get((segment[-1], dst_crew[ins_pos]))
                            if c is None:
                                ok = False
                            else:
                                ins_cost += c
                        if not ok:
                            continue

                        dst_bridge = 0.0
                        if 0 < ins_pos < len(dst_crew):
                            dst_bridge = arcs.get((dst_crew[ins_pos - 1], dst_crew[ins_pos]), 0.0)

                        new_dst = dst_crew[:ins_pos] + segment + dst_crew[ins_pos:]
                        if tasks[new_dst[-1]][1] - tasks[new_dst[0]][0] > time_limit:
                            continue

                        delta = (bridge_cost + ins_cost - dst_bridge) - removal_gain
                        if delta < best_delta:
                            best_delta = delta
                            best_dst_ci = dst_ci
                            best_ins_pos = ins_pos

                if best_dst_ci >= 0:
                    dst_crew = crews[best_dst_ci]
                    new_dst = dst_crew[:best_ins_pos] + segment + dst_crew[best_ins_pos:]
                    crews[src_ci] = new_src
                    crews[best_dst_ci] = new_dst
                    improved = True
                    # Restart this crew
                    break

        return crews, improved

    # ------------------------------------------------------------------ #
    # Main improvement loop
    # ------------------------------------------------------------------ #
    current_crews = [list(c) for c in initial_sol['crews']]
    current_cost = total_cost_fn(current_crews)
    best_crews = [list(c) for c in current_crews]
    best_cost = current_cost
    best_sol_out = {'crews': [list(c) for c in current_crews]}

    rng = random.Random(42)

    # SA temperature
    if current_cost > 0 and current_cost < float('inf'):
        T_init = max(0.05 * current_cost / math.log(2), 1.0)
    else:
        T_init = 100.0
    T_min = 1e-4
    T = T_init

    ls_time = remaining() - 0.5
    if ls_time <= 0:
        return initial_sol

    estimated_iters = max(100, int(ls_time * 30))
    cooling_rate = (T_min / T_init) ** (1.0 / max(1, estimated_iters))

    n_destroy_base = max(1, N // 8)

    # Adaptive operator selection (UCB)
    destroy_ops = ['random', 'worst_crew', 'segment', 'related', 'expensive']
    op_success = [0] * len(destroy_ops)
    op_tries = [1] * len(destroy_ops)

    def select_op_ucb():
        total_tries = sum(op_tries)
        scores = []
        for i in range(len(destroy_ops)):
            exploit = op_success[i] / op_tries[i]
            explore = math.sqrt(2 * math.log(total_tries) / op_tries[i])
            scores.append(exploit + 0.3 * explore)
        return scores.index(max(scores))

    no_improve_streak = 0
    iteration = 0
    last_ls_iter = 0
    last_or_opt_iter = 0

    while remaining() > 0.5:
        iteration += 1
        T = max(T * cooling_rate, T_min)

        # Adaptive destroy size
        n_destroy = n_destroy_base + min(no_improve_streak // 6, N // 5)
        n_destroy = max(1, min(n_destroy, N // 2))

        op_idx = select_op_ucb()
        op_tries[op_idx] += 1
        destroy_op = destroy_ops[op_idx]

        # Destroy
        if destroy_op == 'random':
            partial_crews, removed = destroy_random(current_crews, n_destroy, rng)
        elif destroy_op == 'worst_crew':
            n_cd = max(1, min(3, len(current_crews) // 3))
            partial_crews, removed = destroy_worst_crew(current_crews, n_cd, rng)
        elif destroy_op == 'segment':
            partial_crews, removed = destroy_segment(current_crews, rng)
        elif destroy_op == 'related':
            partial_crews, removed = destroy_related(current_crews, n_destroy, rng)
        else:
            partial_crews, removed = destroy_expensive_arcs(current_crews, n_destroy, rng)

        if not removed:
            continue

        # Repair: alternate between greedy and regret-2
        use_regret = (iteration % 3 == 0) and (len(removed) <= 30)
        if use_regret:
            new_crews = regret2_repair(removed, partial_crews, K)
        else:
            new_crews = greedy_repair(removed, partial_crews, K)

        if new_crews is None:
            continue

        # Check coverage
        all_covered = set(t for crew in new_crews for t in crew)
        if len(all_covered) != N:
            missing = list(set(range(1, N + 1)) - all_covered)
            new_crews = greedy_repair(missing, new_crews, K)
            if new_crews is None:
                continue
            if len(set(t for crew in new_crews for t in crew)) != N:
                continue

        if len(new_crews) > K:
            continue

        if not all(crew_valid(crew) for crew in new_crews):
            continue

        new_cost = total_cost_fn(new_crews)

        # SA acceptance
        delta = new_cost - current_cost
        accept = False
        if delta < 0:
            accept = True
        elif T > T_min * 10 and delta < float('inf'):
            accept = rng.random() < math.exp(-delta / T)

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
                            best_sol_out = candidate
                            op_success[op_idx] += 1
                            no_improve_streak = 0
                except Exception:
                    pass
            else:
                no_improve_streak += 1
        else:
            no_improve_streak += 1

        # Periodic or-opt (every 15 iterations)
        if iteration - last_or_opt_iter >= 15 and remaining() > 1.0:
            last_or_opt_iter = iteration
            seg_len = rng.choice([1, 2, 3])
            opt_crews, opt_improved = or_opt_pass([list(c) for c in current_crews], seg_len, rng)
            if opt_improved and len(opt_crews) <= K:
                if all(crew_valid(c) for c in opt_crews):
                    opt_cost = total_cost_fn(opt_crews)
                    if opt_cost < current_cost:
                        current_crews = opt_crews
                        current_cost = opt_cost
                        if opt_cost < best_cost - 1e-9:
                            candidate = {'crews': [list(c) for c in opt_crews]}
                            try:
                                feasible, _ = tools['is_feasible'](candidate)
                                if feasible:
                                    obj = tools['objective'](candidate)
                                    if obj < best_cost:
                                        best_cost = obj
                                        best_crews = [list(c) for c in opt_crews]
                                        best_sol_out = candidate
                                        no_improve_streak = 0
                            except Exception:
                                pass

        # Periodic relocate pass (every 25 iterations when stagnating)
        if iteration - last_ls_iter >= 25 and remaining() > 1.5 and no_improve_streak > 10:
            last_ls_iter = iteration
            ls_crews, ls_improved = relocate_pass([list(c) for c in current_crews], rng)
            if ls_improved:
                current_crews = ls_crews
                current_cost = total_cost_fn(current_crews)
                if current_cost < best_cost - 1e-9:
                    candidate = {'crews': [list(c) for c in current_crews if c]}
                    try:
                        feasible, _ = tools['is_feasible'](candidate)
                        if feasible:
                            obj = tools['objective'](candidate)
                            if obj < best_cost:
                                best_cost = obj
                                best_crews = [list(c) for c in current_crews if c]
                                best_sol_out = candidate
                                no_improve_streak = 0
                    except Exception:
                        pass

        # Periodic swap pass
        if iteration % 40 == 0 and remaining() > 2.0 and N <= 100:
            sw_crews, sw_improved = swap_pass([list(c) for c in current_crews], rng)
            if sw_improved:
                current_crews = sw_crews
                current_cost = total_cost_fn(current_crews)
                if current_cost < best_cost - 1e-9:
                    candidate = {'crews': [list(c) for c in current_crews if c]}
                    try:
                        feasible, _ = tools['is_feasible'](candidate)
                        if feasible:
                            obj = tools['objective'](candidate)
                            if obj < best_cost:
                                best_cost = obj
                                best_crews = [list(c) for c in current_crews if c]
                                best_sol_out = candidate
                                no_improve_streak = 0
                    except Exception:
                        pass

        # Restart from best when stagnating
        if no_improve_streak > 60 and remaining() > 2.0:
            current_crews = [list(c) for c in best_crews]
            current_cost = best_cost
            no_improve_streak = 0
            T = max(T, T_init * 0.1)

    # Final validation
    result = {'crews': [list(c) for c in best_crews if c]}
    try:
        feasible, _ = tools['is_feasible'](result)
        if feasible:
            return result
    except Exception:
        pass

    return best_sol_out