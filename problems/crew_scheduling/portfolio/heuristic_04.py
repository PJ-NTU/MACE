# MACE evolved heuristic 04/10 for problem: crew_scheduling
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

    tasks_by_start = sorted(range(1, N + 1), key=lambda t: tasks[t][0])

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

    # Use 35% of budget for initial solution (down from 50%)
    init_budget = min(remaining() * 0.35, time_limit_s * 0.35)
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

    if remaining() > 1.0:
        best_sol = _lns_sa(
            best_sol, N, K, tasks, arcs, time_limit,
            succ, pred, tasks_by_start, tools,
            start_time, time_limit_s
        )

    return best_sol


def _greedy_construct(N, K, tasks, arcs, time_limit):
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

    def regret_repair(unassigned, current_crews, max_total_crews):
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
                        insertions.append((delta, ci, pos))

                if len(crews) < max_total_crews:
                    insertions.append((0.0, -1, 0))

                if not insertions:
                    return None

                insertions.sort(key=lambda x: x[0])
                best_cost_ins = insertions[0][0]
                second_cost_ins = insertions[1][0] if len(insertions) > 1 else best_cost_ins + 1e9

                regret_val = second_cost_ins - best_cost_ins
                regrets.append((regret_val, task, insertions[0]))

            regrets.sort(key=lambda x: -x[0])
            _, task_to_insert, (delta, ci, pos) = regrets[0]

            if ci == -1:
                crews.append([task_to_insert])
            else:
                crews[ci] = crews[ci][:pos] + [task_to_insert] + crews[ci][pos:]

            remaining_tasks.remove(task_to_insert)

        return crews

    def greedy_repair(unassigned, current_crews, max_total_crews):
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
                return None

        return crews

    def destroy_random_tasks(crews, n_remove, rng):
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
        for t in removed_set:
            for _, j, _ in succ[t]:
                if j in all_tasks_flat:
                    candidates.add(j)
            for _, i, _ in pred[t]:
                if i in all_tasks_flat:
                    candidates.add(i)

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

    # Main LNS + SA loop
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

    # Use a less aggressive cooling rate to allow more exploration
    estimated_iters = max(50, int(ls_time * 40))
    cooling_rate = (T_min / T_init) ** (1.0 / max(1, estimated_iters))

    n_destroy_base = max(1, N // 10)

    iteration = 0
    no_improve_streak = 0

    destroy_ops = ['random', 'worst_crew', 'segment', 'related']
    op_weights = [1.0, 1.0, 1.0, 1.0]
    op_success = [0, 0, 0, 0]
    op_tries = [1, 1, 1, 1]

    # --- MODIFICATION: Track acceptance rate for adaptive reheating ---
    # Use a sliding window to monitor recent acceptance rate
    accept_window_size = 20
    accept_history = []  # 1 = accepted, 0 = rejected
    reheat_count = 0
    max_reheats = 5  # limit total reheats to avoid infinite warming

    def select_destroy_op():
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

        # --- MODIFICATION: Adaptive reheating based on acceptance rate ---
        # If acceptance rate in recent window is too low, reheat temperature
        if (len(accept_history) >= accept_window_size and
                reheat_count < max_reheats and
                remaining() > 1.5):
            recent_accept_rate = sum(accept_history[-accept_window_size:]) / accept_window_size
            if recent_accept_rate < 0.05:
                # Acceptance rate too low: reheat to 20% of initial temperature
                T = max(T, T_init * 0.20)
                reheat_count += 1
                # Reset accept history to avoid immediate re-trigger
                accept_history = []
                # Reset to best solution on reheat for fresh exploration
                current_crews = [list(c) for c in best_crews]
                current_cost = best_cost
                no_improve_streak = 0

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
            accept_history.append(0)
            continue

        max_crews = K
        if iteration % 3 == 0:
            if len(removed) <= 20:
                new_crews = regret_repair(removed, partial_crews, max_crews)
            else:
                new_crews = greedy_repair(removed, partial_crews, max_crews)
        else:
            new_crews = greedy_repair(removed, partial_crews, max_crews)

        if new_crews is None:
            accept_history.append(0)
            continue

        all_covered = set(t for crew in new_crews for t in crew)
        if len(all_covered) != N:
            missing = set(range(1, N + 1)) - all_covered
            new_crews = greedy_repair(list(missing), new_crews, max_crews)
            if new_crews is None:
                accept_history.append(0)
                continue
            all_covered2 = set(t for crew in new_crews for t in crew)
            if len(all_covered2) != N:
                accept_history.append(0)
                continue

        if len(new_crews) > K:
            accept_history.append(0)
            continue

        all_valid = True
        for crew in new_crews:
            if not check_crew(crew):
                all_valid = False
                break
        if not all_valid:
            accept_history.append(0)
            continue

        new_cost = total_cost_fn(new_crews)

        delta = new_cost - current_cost
        accept = False
        if delta < 0:
            accept = True
        elif T > T_min * 10:
            prob = math.exp(-delta / T)
            accept = rng.random() < prob

        accept_history.append(1 if accept else 0)

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
            # Only reheat here if we haven't hit the adaptive reheat recently
            T = max(T, T_init * 0.1)

    result = {'crews': [list(c) for c in best_crews if c]}
    try:
        feasible, _ = tools['is_feasible'](result)
        if feasible:
            return result
    except Exception:
        pass

    return initial_sol