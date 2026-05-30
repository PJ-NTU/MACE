# MACE evolved heuristic 01/10 for problem: crew_scheduling
import time
import math
import random
import heapq
from collections import defaultdict


def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """Solve crew scheduling via greedy construction + local search."""
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
    
    # Try the default solver first with a portion of the budget
    try:
        default_budget = min(remaining() * 0.7, time_limit_s * 0.7)
        sol = tools['solve_default'](time_limit_s=default_budget)
        if sol is not None:
            feasible, _ = tools['is_feasible'](sol)
            if feasible:
                best_sol = sol
                best_cost = tools['objective'](best_sol)
                # Try local search with remaining time
                if remaining() > 2.0:
                    best_sol = local_search(
                        best_sol, N, K, tasks, arcs, time_limit, tools, 
                        start_time, time_limit_s
                    )
                return best_sol
    except Exception:
        pass
    
    # Fallback: greedy construction
    try:
        chains = tools['greedy_chain_pack']()
        if len(chains) <= K:
            sol = tools['make_solution'](chains)
            feasible, _ = tools['is_feasible'](sol)
            if feasible:
                if remaining() > 2.0:
                    sol = local_search(
                        sol, N, K, tasks, arcs, time_limit, tools,
                        start_time, time_limit_s
                    )
                return sol
    except Exception:
        pass
    
    # Manual greedy construction
    sol = greedy_construct(N, K, tasks, arcs, time_limit, tools)
    if sol is not None:
        feasible, _ = tools['is_feasible'](sol)
        if feasible:
            if remaining() > 2.0:
                sol = local_search(
                    sol, N, K, tasks, arcs, time_limit, tools,
                    start_time, time_limit_s
                )
            return sol
    
    # Last resort: trivial assignment (one task per crew if K >= N)
    return trivial_solution(N, K, tasks, arcs, time_limit, tools)


def greedy_construct(N, K, tasks, arcs, time_limit, tools):
    """Greedy construction: assign tasks to crews minimizing transition costs."""
    # Sort tasks by start time
    task_order = sorted(range(1, N + 1), key=lambda t: tasks[t][0])
    
    crews = []  # list of lists of task ids
    crew_last = []  # last task in each crew
    crew_start = []  # start time of first task in each crew
    crew_finish = []  # finish time of last task in each crew
    
    for task in task_order:
        t_start, t_finish = tasks[task]
        
        best_crew = -1
        best_cost = float('inf')
        
        for i, crew in enumerate(crews):
            last = crew_last[i]
            last_finish = crew_finish[i]
            first_start = crew_start[i]
            
            # Check no overlap
            if last_finish > t_start:
                continue
            
            # Check arc exists
            if (last, task) not in arcs:
                continue
            
            # Check duty time
            duty = t_finish - first_start
            if duty > time_limit:
                continue
            
            cost = arcs[(last, task)]
            if cost < best_cost:
                best_cost = cost
                best_crew = i
        
        if best_crew >= 0:
            crews[best_crew].append(task)
            crew_last[best_crew] = task
            crew_finish[best_crew] = t_finish
        else:
            # Open new crew
            if len(crews) < K:
                crews.append([task])
                crew_last.append(task)
                crew_start.append(t_start)
                crew_finish.append(t_finish)
            else:
                # Try to force assign to some crew (may fail constraints)
                # Find best crew ignoring duty time, then check
                forced = False
                for i in range(len(crews)):
                    last = crew_last[i]
                    if crew_finish[i] <= t_start and (last, task) in arcs:
                        duty = t_finish - crew_start[i]
                        if duty <= time_limit:
                            crews[i].append(task)
                            crew_last[i] = task
                            crew_finish[i] = t_finish
                            forced = True
                            break
                if not forced:
                    return None
    
    sol = {'crews': crews}
    feasible, _ = tools['is_feasible'](sol)
    if feasible:
        return sol
    return None


def local_search(sol, N, K, tasks, arcs, time_limit, tools, start_time, time_limit_s):
    """Local search to improve solution quality."""
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return time_limit_s - elapsed()
    
    best_sol = sol
    best_cost = tools['objective'](sol)
    
    crews = [list(c) for c in sol['crews']]
    
    improved = True
    iteration = 0
    
    while improved and remaining() > 0.5:
        improved = False
        iteration += 1
        
        # Try relocate: move a task from one crew to another position
        for ci in range(len(crews)):
            if remaining() < 0.3:
                break
            for ti in range(len(crews[ci])):
                if remaining() < 0.3:
                    break
                task = crews[ci][ti]
                
                # Try inserting task into other crews
                for cj in range(len(crews)):
                    if ci == cj:
                        continue
                    if remaining() < 0.3:
                        break
                    
                    # Try each position in cj
                    for pos in range(len(crews[cj]) + 1):
                        new_ci = crews[ci][:ti] + crews[ci][ti+1:]
                        new_cj = crews[cj][:pos] + [task] + crews[cj][pos:]
                        
                        # Validate both crews
                        if len(new_ci) == 0 and len(crews) > 1:
                            # Would create empty crew - skip if K allows
                            # Actually empty crews are not allowed, but we can just remove them
                            pass
                        
                        valid_ci = len(new_ci) == 0 or check_crew(new_ci, tasks, arcs, time_limit)
                        valid_cj = check_crew(new_cj, tasks, arcs, time_limit)
                        
                        if valid_ci and valid_cj:
                            new_crews = []
                            for k, c in enumerate(crews):
                                if k == ci:
                                    if len(new_ci) > 0:
                                        new_crews.append(new_ci)
                                elif k == cj:
                                    new_crews.append(new_cj)
                                else:
                                    new_crews.append(c)
                            
                            if len(new_crews) > K:
                                continue
                            if len(new_crews) == 0:
                                continue
                            
                            new_cost = sum(crew_cost_local(c, arcs) for c in new_crews)
                            if new_cost < best_cost - 1e-9:
                                best_cost = new_cost
                                crews = new_crews
                                best_sol = {'crews': [list(c) for c in crews]}
                                improved = True
                                break
                    
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-opt within a crew: reverse a subsequence
        for ci in range(len(crews)):
            if remaining() < 0.3:
                break
            crew = crews[ci]
            n = len(crew)
            for i in range(n - 1):
                if remaining() < 0.3:
                    break
                for j in range(i + 2, n):
                    # Reverse segment [i+1..j]
                    new_crew = crew[:i+1] + crew[i+1:j+1][::-1] + crew[j+1:]
                    if check_crew(new_crew, tasks, arcs, time_limit):
                        new_cost_crew = crew_cost_local(new_crew, arcs)
                        old_cost_crew = crew_cost_local(crew, arcs)
                        if new_cost_crew < old_cost_crew - 1e-9:
                            delta = new_cost_crew - old_cost_crew
                            best_cost += delta
                            crews[ci] = new_crew
                            best_sol = {'crews': [list(c) for c in crews]}
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try swap: exchange tasks between two crews
        for ci in range(len(crews)):
            if remaining() < 0.3:
                break
            for ti in range(len(crews[ci])):
                if remaining() < 0.3:
                    break
                for cj in range(ci + 1, len(crews)):
                    if remaining() < 0.3:
                        break
                    for tj in range(len(crews[cj])):
                        task_i = crews[ci][ti]
                        task_j = crews[cj][tj]
                        
                        new_ci = crews[ci][:ti] + [task_j] + crews[ci][ti+1:]
                        new_cj = crews[cj][:tj] + [task_i] + crews[cj][tj+1:]
                        
                        if check_crew(new_ci, tasks, arcs, time_limit) and \
                           check_crew(new_cj, tasks, arcs, time_limit):
                            old_cost = crew_cost_local(crews[ci], arcs) + crew_cost_local(crews[cj], arcs)
                            new_cost = crew_cost_local(new_ci, arcs) + crew_cost_local(new_cj, arcs)
                            if new_cost < old_cost - 1e-9:
                                best_cost += new_cost - old_cost
                                crews[ci] = new_ci
                                crews[cj] = new_cj
                                best_sol = {'crews': [list(c) for c in crews]}
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Verify final solution
    feasible, _ = tools['is_feasible'](best_sol)
    if feasible:
        return best_sol
    return sol


def check_crew(crew, tasks, arcs, time_limit):
    """Check if a crew sequence is valid."""
    if len(crew) == 0:
        return True
    
    first_start = tasks[crew[0]][0]
    last_finish = tasks[crew[-1]][1]
    
    if last_finish - first_start > time_limit:
        return False
    
    for i in range(len(crew) - 1):
        t1, t2 = crew[i], crew[i+1]
        if tasks[t1][1] > tasks[t2][0]:
            return False
        if (t1, t2) not in arcs:
            return False
    
    return True


def crew_cost_local(crew, arcs):
    """Compute cost of a crew."""
    cost = 0.0
    for i in range(len(crew) - 1):
        key = (crew[i], crew[i+1])
        if key in arcs:
            cost += arcs[key]
        else:
            return float('inf')
    return cost


def trivial_solution(N, K, tasks, arcs, time_limit, tools):
    """Last resort: try to build any feasible solution."""
    if K >= N:
        # Each task in its own crew
        crews = [[i] for i in range(1, N + 1)]
        sol = {'crews': crews}
        feasible, _ = tools['is_feasible'](sol)
        if feasible:
            return sol
    
    # Try min cost flow
    try:
        sol = tools['solve_min_cost_flow'](time_limit_s=10.0)
        if sol is not None:
            feasible, _ = tools['is_feasible'](sol)
            if feasible:
                return sol
    except Exception:
        pass
    
    # Return whatever we have
    crews = [[i] for i in range(1, min(N, K) + 1)]
    if N > K:
        # Assign remaining tasks greedily
        for task in range(K + 1, N + 1):
            crews[0].append(task)
    return {'crews': crews}