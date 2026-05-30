"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    schedule = solution.get("schedule") if isinstance(solution, dict) else None

    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    schedule = solution.get("schedule")
    if schedule is None:
        return False, "solution missing 'schedule' key"
    if not isinstance(schedule, dict):
        return False, f"'schedule' must be dict, got {type(schedule).__name__}"
    if len(schedule) != num_planes:
        return False, f"schedule has {len(schedule)} entries, expected {num_planes}"

    for plane_id in range(1, num_planes + 1):
        if plane_id not in schedule:
            return False, f"plane {plane_id} is missing from schedule"
        entry = schedule[plane_id]
        if not isinstance(entry, dict):
            return False, f"schedule entry for plane {plane_id} must be a dict"
        if "landing_time" not in entry:
            return False, f"schedule entry for plane {plane_id} missing 'landing_time'"
        if "runway" not in entry:
            return False, f"schedule entry for plane {plane_id} missing 'runway'"
        runway = entry["runway"]
        if not isinstance(runway, int):
            return False, f"plane {plane_id} runway must be int, got {type(runway).__name__}"
        if runway < 1 or runway > num_runways:
            return False, f"plane {plane_id} runway {runway} out of range [1, {num_runways}]"

    for i in range(1, num_planes + 1):
        landing_time = schedule[i]["landing_time"]
        earliest = planes[i - 1]["earliest"]
        latest = planes[i - 1]["latest"]
        if landing_time < earliest or landing_time > latest:
            return False, (f"plane {i}: landing time {landing_time} outside window "
                           f"[{earliest}, {latest}]")

    for i in range(1, num_planes + 1):
        for j in range(1, num_planes + 1):
            if i == j:
                continue
            entry_i = schedule[i]
            entry_j = schedule[j]
            if entry_i["runway"] == entry_j["runway"]:
                L_i = entry_i["landing_time"]
                L_j = entry_j["landing_time"]
                if L_i <= L_j:
                    required_gap = separation[i - 1][j - 1]
                    if (L_j - L_i) < required_gap:
                        return False, (f"separation violation on runway {entry_i['runway']}: "
                                       f"plane {i} lands at {L_i}, plane {j} at {L_j}, "
                                       f"required gap {required_gap}")

    return True, None
'''
