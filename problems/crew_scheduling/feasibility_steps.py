"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    crews = solution.get("crews")
    if crews is None:
        return False, "solution missing 'crews' key"

    if not isinstance(crews, (list, tuple)):
        return False, f"'crews' must be list, got {type(crews).__name__}"

    if len(crews) > K:
        return False, f"number of crews {len(crews)} exceeds K={K}"

    all_tasks_in_output = [task for crew in crews for task in crew]

    if len(all_tasks_in_output) != N:
        return False, f"total tasks assigned {len(all_tasks_in_output)} != N={N}"

    if set(all_tasks_in_output) != set(range(1, N + 1)):
        return False, "tasks in crews do not match expected task IDs 1..N"

    for i, crew in enumerate(crews):
        if not isinstance(crew, (list, tuple)):
            return False, f"crew {i} must be a list, got {type(crew).__name__}"

        if len(crew) == 0:
            return False, f"crew {i} has an empty schedule"

        first_task = crew[0]
        last_task = crew[-1]
        duty_time = tasks[last_task][1] - tasks[first_task][0]
        if duty_time > time_limit:
            return False, f"crew {i} duty time {duty_time} exceeds time_limit={time_limit}"

        for idx in range(len(crew) - 1):
            current_task = crew[idx]
            next_task = crew[idx + 1]

            if tasks[current_task][1] > tasks[next_task][0]:
                return False, f"tasks {current_task} and {next_task} in crew {i} overlap"

            if (current_task, next_task) not in arcs:
                return False, f"no valid transition arc between tasks {current_task} and {next_task} in crew {i}"

    return True, None
'''
