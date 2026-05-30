# MACE evolved heuristic 07/10 for problem: crew_scheduling
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

    succ = defaultdict(list)
    pred = defaultdict(list)

    for (i, j), cost in arcs.items():
        if tasks[i][1] <= tasks[j][0]:
            succ[i].append((tasks[j][0], j, cost))
            pred[j].append((tasks[i][1], i, cost))

    for t in range(1, N + 1):
        succ[t].sort()
        pred[t].sort()

    arc_set = set(arcs.keys())
    tasks_by_start = sorted(range(1, N + 1), key=lambda t: (tasks[t][0], tasks[t][1]))

    def check_crew(crew):
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

    if N <= 80:
        ilp_budget = min(remaining() * 0.45, time_limit_s * 0.45)
        if ilp_budget > 0.5:
            try:
                sol = tools['ilp_crew_scheduling'](time_limit_s=ilp_budget)
                update_best(sol)
            except Exception:
                pass

    if best_sol is None and remaining() > 1.0:
        mcf_budget = min(remaining() * 0.40, 20.0)
        try:
            sol = tools['solve_min_cost_flow'](time_limit_s=mcf_budget)
            update_best(sol)
        except Exception:
            pass

    if best_sol is None and remaining() > 1.0:
        init_budget = min(remaining() * 0.45, 30.0)
        try:
            sol = tools['solve_default'](time_limit_s=init_budget)
            update_best(sol)
        except Exception:
            pass

    if best_sol is None and remaining() > 0.5:
        try:
            chains = tools['greedy_chain_pack']()
            if chains and len(chains) <= K:
                sol = tools['make_solution'](chains)
                update_best(sol)
        except Exception:
            pass

    if best_sol is None and remaining() > 0.5:
        sol = _greedy_construct(N, K, tasks, arcs, time_limit, arc_set)
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
        best_cost = total_cost_fn(best_sol['crews'])

    if remaining() > 1.0:
        improved_sol = _lns_sa_improve(
            best_sol, N, K, tasks, arcs, time_limit,
            succ, pred, arc_set, tasks_by_start, tools,
            start_time, time_limit_s,
            check_crew, crew_cost_fn, total_cost_fn,
            best_cost
        )
        if improved_sol is not None:
            try:
                feasible, _ = tools['is_feasible'](improved_sol)
                if feasible:
                    cost = tools['objective'](improved_sol)
                    if cost < best_cost:
                        best_cost = cost
                        best_sol = improved_sol
            except Exception:
                pass

    return best_sol


def _greedy_construct(N, K, tasks, arcs, time_limit, arc_set):
    task_order = sorted(range(1, N + 1), key=lambda t: (tasks[t][0], tasks[t][1]))
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
            if (last, task) not in arc_set:
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


def _greedy_construct_randomized(N, K, tasks, arcs, time_limit, arc_set, rng, alpha=0.3):
    """Randomized greedy construction using RCL (restricted candidate list)."""
    task_order = sorted(range(1, N + 1), key=lambda t: (tasks[t][0], tasks[t][1]))
    crews = []
    crew_last = []
    crew_start = []
    crew_finish = []

    for task in task_order:
        t_start, t_finish = tasks[task]
        candidates = []

        for i in range(len(crews)):
            last = crew_last[i]
            if crew_finish[i] > t_start:
                continue
            if (last, task) not in arc_set:
                continue
            if t_finish - crew_start[i] > time_limit:
                continue
            c = arcs[(last, task)]
            candidates.append((c, i))

        if candidates:
            candidates.sort()
            c_min = candidates[0][0]
            c_max = candidates[-1][0]
            threshold = c_min + alpha * (c_max - c_min)
            rcl = [i for c, i in candidates if c <= threshold + 1e-9]
            best_crew = rng.choice(rcl)
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


def _lns_sa_improve(initial_sol, N, K, tasks, arcs, time_limit,
                    succ, pred, arc_set, tasks_by_start, tools,
                    start_time, time_limit_s,
                    check_crew, crew_cost_fn, total_cost_fn,
                    initial_best_cost):

    def remaining():
        return time_limit_s - (time.time() - start_time)

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

    def destroy_random_tasks(crews, n_remove, rng):
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

    def destroy_related_tasks(crews, n_remove, rng):
        all_tasks_flat = [t for crew in crews for t in crew]
        if not all_tasks_flat:
            return crews, []
        seed_task = rng.choice(all_tasks_flat)
        removed_set = {seed_task}
        candidates = set()
        for _, j, _ in succ[seed_task]:
            candidates.add(j)
        for _, i, _ in pred[seed_task]:
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

    def destroy_chain_break(crews, rng):
        if not crews:
            return crews, []
        best_cost_arc = -1
        best_ci = -1
        best_pos = -1
        for ci, crew in enumerate(crews):
            for pos in range(len(crew) - 1):
                c = arcs.get((crew[pos], crew[pos + 1]), 0)
                if c > best_cost_arc:
                    best_cost_arc = c
                    best_ci = ci
                    best_pos = pos
        if best_ci < 0:
            return crews, []
        crew = crews[best_ci]
        window = rng.randint(1, max(1, len(crew) // 4))
        lo = max(0, best_pos - window + 1)
        hi = min(len(crew), best_pos + window + 1)
        removed = crew[lo:hi]
        new_crew = crew[:lo] + crew[hi:]
        new_crews = []
        for i, c in enumerate(crews):
            if i == best_ci:
                if new_crew:
                    new_crews.append(new_crew)
            else:
                new_crews.append(c)
        return new_crews, removed

    def or_opt_move(crews, seg_len, rng):
        best_delta = -1e-9
        best_move = None

        crew_indices = list(range(len(crews)))
        rng.shuffle(crew_indices)

        for src_ci in crew_indices[:min(len(crew_indices), 8)]:
            src_crew = crews[src_ci]
            if len(src_crew) <= seg_len:
                continue

            for src_pos in range(len(src_crew) - seg_len + 1):
                segment = src_crew[src_pos:src_pos + seg_len]
                seg_start = tasks[segment[0]][0]
                seg_finish = tasks[segment[-1]][1]

                removal_gain = 0.0
                if src_pos > 0:
                    removal_gain += arcs.get((src_crew[src_pos - 1], segment[0]), 0.0)
                if src_pos + seg_len < len(src_crew):
                    removal_gain += arcs.get((segment[-1], src_crew[src_pos + seg_len]), 0.0)
                bridge_cost = 0.0
                if src_pos > 0 and src_pos + seg_len < len(src_crew):
                    prev_t = src_crew[src_pos - 1]
                    next_t = src_crew[src_pos + seg_len]
                    if (prev_t, next_t) in arc_set:
                        bridge_cost = arcs[(prev_t, next_t)]
                    else:
                        continue

                new_src = src_crew[:src_pos] + src_crew[src_pos + seg_len:]
                if new_src and tasks[new_src[-1]][1] - tasks[new_src[0]][0] > time_limit:
                    continue

                for dst_ci in crew_indices:
                    if dst_ci == src_ci:
                        continue
                    dst_crew = crews[dst_ci]

                    for ins_pos in range(len(dst_crew) + 1):
                        if ins_pos > 0 and tasks[dst_crew[ins_pos - 1]][1] > seg_start:
                            continue
                        if ins_pos < len(dst_crew) and seg_finish > tasks[dst_crew[ins_pos]][0]:
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
                            best_move = (src_ci, src_pos, seg_len, dst_ci, ins_pos,
                                         new_src, new_dst)

        if best_move is None:
            return crews, False

        src_ci, src_pos, seg_len_m, dst_ci, ins_pos, new_src, new_dst = best_move
        new_crews = []
        for i, c in enumerate(crews):
            if i == src_ci:
                if new_src:
                    new_crews.append(new_src)
            elif i == dst_ci:
                new_crews.append(new_dst)
            else:
                new_crews.append(c)
        return new_crews, True

    def relocate_pass(crews):
        """One pass of single-task relocation, best improvement."""
        improved = False

        task_to_crew = {}
        task_to_pos = {}
        for ci, crew in enumerate(crews):
            for pos, t in enumerate(crew):
                task_to_crew[t] = ci
                task_to_pos[t] = pos

        all_task_ids = list(range(1, N + 1))
        random.shuffle(all_task_ids)

        for task in all_task_ids:
            if remaining() < 0.3:
                break
            src_ci = task_to_crew[task]
            src_pos = task_to_pos[task]
            src_crew = crews[src_ci]

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

    # Main LNS + SA loop
    current_crews = [list(c) for c in initial_sol['crews']]
    current_cost = total_cost_fn(current_crews)
    best_crews = [list(c) for c in current_crews]
    best_cost_lns = current_cost
    best_sol_lns = {'crews': [list(c) for c in current_crews]}

    rng = random.Random(42)

    if current_cost > 0 and current_cost < float('inf'):
        T_init = max(0.05 * current_cost / math.log(2), 1.0)
    else:
        T_init = 100.0
    T_min = 1e-4
    T = T_init

    ls_time = remaining() - 0.3
    if ls_time <= 0:
        return initial_sol

    estimated_iters = max(50, int(ls_time * 25))
    cooling_rate = (T_min / T_init) ** (1.0 / max(1, estimated_iters))

    n_destroy_base = max(1, N // 8)

    destroy_ops = ['random', 'worst_crew', 'segment', 'related', 'chain_break']
    op_success = [0] * len(destroy_ops)
    op_tries = [1] * len(destroy_ops)

    def select_op():
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
    restart_count = 0

    while remaining() > 0.3:
        iteration += 1
        T = max(T * cooling_rate, T_min)

        n_destroy = n_destroy_base + min(no_improve_streak // 8, N // 4)
        n_destroy = max(1, min(n_destroy, N // 2))

        op_idx = select_op()
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
            partial_crews, removed = destroy_chain_break(current_crews, rng)

        if not removed:
            continue

        use_regret = (iteration % 4 == 0) and (len(removed) <= 25)
        if use_regret:
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

        if not all(check_crew(crew) for crew in new_crews):
            continue

        new_cost = total_cost_fn(new_crews)

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

            if new_cost < best_cost_lns - 1e-9:
                candidate = {'crews': [list(c) for c in new_crews]}
                try:
                    feasible, _ = tools['is_feasible'](candidate)
                    if feasible:
                        obj = tools['objective'](candidate)
                        if obj < best_cost_lns:
                            best_cost_lns = obj
                            best_crews = [list(c) for c in new_crews]
                            best_sol_lns = candidate
                            op_success[op_idx] += 1
                            no_improve_streak = 0
                except Exception:
                    pass
            else:
                no_improve_streak += 1
        else:
            no_improve_streak += 1

        if iteration % 20 == 0 and remaining() > 1.0:
            seg_len = rng.choice([1, 2, 3])
            new_crews_opt, improved_opt = or_opt_move(current_crews, seg_len, rng)
            if improved_opt and len(new_crews_opt) <= K:
                if all(check_crew(c) for c in new_crews_opt):
                    new_cost_opt = total_cost_fn(new_crews_opt)
                    if new_cost_opt < current_cost:
                        current_crews = new_crews_opt
                        current_cost = new_cost_opt
                        if new_cost_opt < best_cost_lns - 1e-9:
                            candidate = {'crews': [list(c) for c in new_crews_opt]}
                            try:
                                feasible, _ = tools['is_feasible'](candidate)
                                if feasible:
                                    obj = tools['objective'](candidate)
                                    if obj < best_cost_lns:
                                        best_cost_lns = obj
                                        best_crews = [list(c) for c in new_crews_opt]
                                        best_sol_lns = candidate
                                        no_improve_streak = 0
                            except Exception:
                                pass

        if iteration - last_ls_iter >= 30 and remaining() > 1.5 and no_improve_streak > 15:
            ls_crews, ls_improved = relocate_pass([list(c) for c in current_crews])
            if ls_improved:
                current_crews = ls_crews
                current_cost = total_cost_fn(current_crews)
                if current_cost < best_cost_lns - 1e-9:
                    candidate = {'crews': [list(c) for c in current_crews if c]}
                    try:
                        feasible, _ = tools['is_feasible'](candidate)
                        if feasible:
                            obj = tools['objective'](candidate)
                            if obj < best_cost_lns:
                                best_cost_lns = obj
                                best_crews = [list(c) for c in current_crews if c]
                                best_sol_lns = candidate
                                no_improve_streak = 0
                    except Exception:
                        pass
            last_ls_iter = iteration

        # MODIFIED RESTART STRATEGY: diversify with randomized construction
        # instead of always returning to best known solution
        if no_improve_streak > 60 and remaining() > 2.0:
            restart_count += 1
            # Every 3rd restart, try a randomized greedy construction for diversification
            # Other restarts return to best known (exploitation)
            if restart_count % 3 == 0 and remaining() > 3.0:
                # Try randomized greedy construction with varying alpha for diversification
                alpha = rng.uniform(0.1, 0.5)
                try:
                    rand_sol = _greedy_construct_randomized(
                        N, K, tasks, arcs, time_limit, arc_set, rng, alpha=alpha
                    )
                    if rand_sol is not None:
                        rand_crews = rand_sol['crews']
                        if (len(rand_crews) <= K and
                                all(check_crew(c) for c in rand_crews) and
                                len(set(t for c in rand_crews for t in c)) == N):
                            rand_cost = total_cost_fn(rand_crews)
                            # Accept the random solution as new current if it's not
                            # catastrophically worse than best (within 50% overhead)
                            if rand_cost < best_cost_lns * 1.5:
                                current_crews = rand_crews
                                current_cost = rand_cost
                                # Also update best if better
                                if rand_cost < best_cost_lns - 1e-9:
                                    candidate = {'crews': [list(c) for c in rand_crews]}
                                    try:
                                        feasible, _ = tools['is_feasible'](candidate)
                                        if feasible:
                                            obj = tools['objective'](candidate)
                                            if obj < best_cost_lns:
                                                best_cost_lns = obj
                                                best_crews = [list(c) for c in rand_crews]
                                                best_sol_lns = candidate
                                    except Exception:
                                        pass
                            else:
                                # Fall back to best known
                                current_crews = [list(c) for c in best_crews]
                                current_cost = best_cost_lns
                        else:
                            current_crews = [list(c) for c in best_crews]
                            current_cost = best_cost_lns
                    else:
                        current_crews = [list(c) for c in best_crews]
                        current_cost = best_cost_lns
                except Exception:
                    current_crews = [list(c) for c in best_crews]
                    current_cost = best_cost_lns
            else:
                # Standard restart: return to best known solution
                current_crews = [list(c) for c in best_crews]
                current_cost = best_cost_lns

            no_improve_streak = 0
            # Reheat temperature more aggressively on diversification restarts
            if restart_count % 3 == 0:
                T = max(T, T_init * 0.25)
            else:
                T = max(T, T_init * 0.15)

    result = {'crews': [list(c) for c in best_crews if c]}
    try:
        feasible, _ = tools['is_feasible'](result)
        if feasible:
            return result
    except Exception:
        pass

    return best_sol_lns