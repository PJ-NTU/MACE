"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "selected_schedules" not in solution:
        return False, "solution missing 'selected_schedules' key"
    if "tours" not in solution:
        return False, "solution missing 'tours' key"

    # (3) correct types for required keys
    selected_schedules = solution["selected_schedules"]
    if not isinstance(selected_schedules, dict):
        return False, f"'selected_schedules' must be dict, got {type(selected_schedules).__name__}"
    tours = solution["tours"]
    if not isinstance(tours, dict):
        return False, f"'tours' must be dict, got {type(tours).__name__}"

    # Build customer lookup
    customer_lookup = {cust["id"]: cust for cust in customers}

    # (4) per-element constraints on selected_schedules
    for cust in customers:
        if cust["id"] == 0:
            continue
        if cust["id"] not in selected_schedules:
            return False, f"missing selected schedule for customer {cust['id']}"

    for cid, sel_sched in selected_schedules.items():
        cust = customer_lookup.get(cid)
        if cust is None:
            return False, f"customer id {cid} in selected_schedules not found in customer list"
        if len(sel_sched) != period_length:
            return False, f"selected schedule for customer {cid} has length {len(sel_sched)}, expected {period_length}"
        if sel_sched not in cust["schedules"]:
            return False, f"selected schedule {sel_sched} for customer {cid} is not a valid candidate schedule"

    # (5) cross-element / global constraints on tours
    for day in range(1, period_length + 1):
        tours_day = tours.get(day, [])
        vehicles_available = vehicles_per_day[day - 1]

        if len(tours_day) > vehicles_available:
            return False, (f"on day {day}: number of tours ({len(tours_day)}) "
                           f"exceeds available vehicles ({vehicles_available})")

        expected_customers = set()
        for cust in customers:
            if cust["id"] == 0:
                continue
            sched = selected_schedules.get(cust["id"])
            if sched is not None and sched[day - 1] == 1:
                expected_customers.add(cust["id"])

        visited_today = []
        for tour in tours_day:
            if len(tour) < 3:
                return False, f"tour {tour} on day {day} is too short (must have depot + >=1 customer + depot)"
            if tour[0] != 0 or tour[-1] != 0:
                return False, f"tour {tour} on day {day} must start and end at depot (id 0)"
            if 0 in tour[1:-1]:
                return False, f"tour {tour} on day {day} contains depot visit in the middle"

            seen_in_tour = set()
            for vid in tour[1:-1]:
                if vid not in customer_lookup:
                    return False, f"customer id {vid} in tour on day {day} not found in customer list"
                if vid in seen_in_tour:
                    return False, f"tour on day {day} visits customer {vid} more than once"
                seen_in_tour.add(vid)
                visited_today.append(vid)

            capacity_used = sum(customer_lookup[vid]["demand"] for vid in tour[1:-1])
            if capacity_used > vehicle_capacity:
                return False, (f"tour on day {day} exceeds capacity: "
                               f"used {capacity_used}, capacity is {vehicle_capacity}")

        if set(visited_today) != expected_customers:
            missing = expected_customers - set(visited_today)
            extra = set(visited_today) - expected_customers
            msg = f"on day {day}: "
            if missing:
                msg += f"missing visits for customers {list(missing)[:10]}. "
            if extra:
                msg += f"extra visits for customers {list(extra)}."
            return False, msg

    return True, None
'''
