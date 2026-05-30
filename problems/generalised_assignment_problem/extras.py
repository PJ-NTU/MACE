"""Per-problem extras for Generalised Assignment Problem (GAP).

Provides primitive building blocks so the LLM can compose construction /
repair / LNS heuristics for GAP without re-deriving the ILP, capacity
bookkeeping, or basic neighborhood moves.

Tool groups:
  (1) Queries:        cost, resource, agent_capacity, n_agents, n_tasks
  (2) Feasibility:    agent_load, agent_remaining, unassigned_tasks,
                      is_feasible_assignment
  (3) Construction /
      improvement:    greedy_min_cost, greedy_min_resource_ratio,
                      apply_reassign, apply_swap_assignments
  (4) Exact / heavy:  ilp_gap

The solution dict CO-Bench expects is {'assignments': [a_1, ..., a_n]} with
1-INDEXED agent ids. All tools here accept and return that same 1-indexed
convention so values flow directly into tools['is_feasible'] / ['objective'].

NOTE: GAP comes in two flavors (gap1..gap12 are 'max', gapa..gapd are 'min').
Greedy 'min_cost' picks the BEST cost in the problem's own sense (smallest
for 'min', largest for 'max'), and the ILP optimizes in the same direction.
"""
from __future__ import annotations
from typing import Optional, Iterable, List

from mip import Model, BINARY, MINIMIZE, MAXIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns GAP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench GAP load_data, one case):
      - m:                  int, number of agents
      - n:                  int, number of jobs/tasks
      - cost_matrix:        list[list[float]] of shape m x n
      - consumption_matrix: list[list[float]] of shape m x n
      - capacities:         list[float] of length m
      - problem_type:       'max' or 'min'
    """
    m: int = int(instance["m"])
    n: int = int(instance["n"])
    C = instance["cost_matrix"]          # C[i][j], 0-indexed
    R = instance["consumption_matrix"]   # R[i][j], 0-indexed
    cap = instance["capacities"]         # cap[i], 0-indexed
    problem_type: str = instance.get("problem_type", "max")
    is_min = (problem_type == "min")

    # Helper: validate / convert a 1-indexed agent id to a 0-indexed one.
    def _ai(agent_1: int) -> int:
        a = int(agent_1) - 1
        if not (0 <= a < m):
            raise ValueError(f"agent id {agent_1} out of range [1, {m}]")
        return a

    def _tj(task: int) -> int:
        j = int(task)
        if not (0 <= j < n):
            raise ValueError(f"task index {task} out of range [0, {n})")
        return j

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def cost(i: int, j: int) -> float:
        """Cost of assigning task j (0-indexed) to agent i (1-indexed)."""
        return float(C[_ai(i)][_tj(j)])

    def resource(i: int, j: int) -> float:
        """Resource consumed when task j (0-indexed) is assigned to agent i (1-indexed)."""
        return float(R[_ai(i)][_tj(j)])

    def agent_capacity(i: int) -> float:
        """Resource capacity of agent i (1-indexed)."""
        return float(cap[_ai(i)])

    def n_agents() -> int:
        return m

    def n_tasks() -> int:
        return n

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _validate_assignment_shape(assignment: Iterable[int]) -> List[int]:
        a = list(assignment)
        if len(a) != n:
            raise ValueError(f"assignment has length {len(a)}, expected {n}")
        return a

    def agent_load(i: int, assignment: Iterable[int]) -> float:
        """Sum of consumption_matrix[i-1][j] over tasks j assigned to agent i.
        Tasks assigned to other agents (including the sentinel 0 / None used
        for 'unassigned') contribute nothing."""
        a0 = _ai(i)
        a = _validate_assignment_shape(assignment)
        total = 0.0
        for j, aj in enumerate(a):
            if aj is None:
                continue
            try:
                if int(aj) - 1 == a0:
                    total += float(R[a0][j])
            except (TypeError, ValueError):
                continue
        return total

    def agent_remaining(i: int, assignment: Iterable[int]) -> float:
        """capacity[i-1] - agent_load(i, assignment). Can be negative if the
        assignment overuses agent i."""
        return float(cap[_ai(i)]) - agent_load(i, assignment)

    def unassigned_tasks(assignment: Iterable[int]) -> List[int]:
        """List of task indices (0-indexed) whose entry is 0, None, or otherwise
        outside [1, m]. The CO-Bench solution requires every task to be
        assigned to a valid agent; this tool helps you find ones still to
        place during incremental construction."""
        a = _validate_assignment_shape(assignment)
        out = []
        for j, aj in enumerate(a):
            if aj is None:
                out.append(j)
                continue
            try:
                v = int(aj)
            except (TypeError, ValueError):
                out.append(j)
                continue
            if v < 1 or v > m:
                out.append(j)
        return out

    def is_feasible_assignment(assignment: Iterable[int]) -> tuple:
        """Lightweight feasibility precheck WITHOUT calling eval_func: checks
        that every task is assigned to a valid agent (1..m) and that no
        agent's resource usage exceeds capacity. Returns (True, None) on
        success, else (False, message). Cheap to call inside a tight local
        search loop."""
        a = _validate_assignment_shape(assignment)
        load = [0.0] * m
        for j, aj in enumerate(a):
            try:
                v = int(aj)
            except (TypeError, ValueError):
                return False, f"task {j}: agent id {aj!r} not an integer"
            if v < 1 or v > m:
                return False, f"task {j}: agent id {v} out of range [1, {m}]"
            load[v - 1] += float(R[v - 1][j])
        for i in range(m):
            if load[i] > float(cap[i]):
                return False, (f"agent {i + 1} overcapacity: load {load[i]} > "
                               f"capacity {cap[i]}")
        return True, None

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def _best_cost_better(a: float, b: float) -> bool:
        # For 'min' problems we want SMALLER cost; for 'max' we want LARGER.
        return (a < b) if is_min else (a > b)

    def greedy_min_cost() -> List[int]:
        """Greedy construction: for each task (processed in an order that
        favors tight tasks first), pick the agent whose cost is BEST in the
        problem's own direction ('min' -> smallest, 'max' -> largest) and
        still has enough remaining capacity.

        Returns a list of length n with 1-indexed agent ids. May fall back
        to the agent with the most remaining capacity (ignoring capacity
        violation) if no feasible agent exists -- the caller should run
        is_feasible_assignment on the result. Often a decent warm start."""
        # Order tasks by hardness: the task whose best (cheapest-resource)
        # agent uses the most resource (harder to place) goes first.
        task_order = sorted(
            range(n),
            key=lambda j: -min(R[i][j] for i in range(m)),
        )
        load = [0.0] * m
        assignment = [0] * n
        for j in task_order:
            best_i = None
            best_c = float("inf") if is_min else float("-inf")
            for i in range(m):
                if load[i] + R[i][j] <= cap[i] + 1e-9:
                    c = C[i][j]
                    if best_i is None or _best_cost_better(c, best_c):
                        best_c = c
                        best_i = i
            if best_i is None:
                # No feasible agent -- pick the one with most slack.
                best_i = max(range(m), key=lambda i: cap[i] - load[i] - R[i][j])
            assignment[j] = best_i + 1
            load[best_i] += R[best_i][j]
        return assignment

    def greedy_min_resource_ratio() -> List[int]:
        """Greedy construction variant: for each task pick the agent that
        minimizes (resource / |cost|) -- i.e., the agent for whom the task is
        most resource-efficient given its cost. Capacity-respecting; falls
        back to the agent with most slack if none feasible. Often gives a
        different (sometimes better) warm start than greedy_min_cost."""
        load = [0.0] * m
        assignment = [0] * n
        # Process tasks in order of decreasing 'tightness' (max resource cost).
        task_order = sorted(
            range(n),
            key=lambda j: -max(R[i][j] for i in range(m)),
        )
        for j in task_order:
            best_i = None
            best_ratio = float("inf")
            for i in range(m):
                if load[i] + R[i][j] > cap[i] + 1e-9:
                    continue
                cij = C[i][j]
                # For 'max' problems higher cost is better, so use
                # resource / cost directly (smaller is better when cost large).
                # For 'min' problems lower cost is better, so use cost*resource
                # (smaller is better when cost small AND resource small).
                if is_min:
                    score = (abs(cij) + 1.0) * (R[i][j] + 1e-9)
                else:
                    score = (R[i][j] + 1e-9) / (abs(cij) + 1.0)
                if score < best_ratio:
                    best_ratio = score
                    best_i = i
            if best_i is None:
                best_i = max(range(m), key=lambda i: cap[i] - load[i] - R[i][j])
            assignment[j] = best_i + 1
            load[best_i] += R[best_i][j]
        return assignment

    def apply_reassign(assignment: Iterable[int], task: int,
                       new_agent: int) -> Optional[List[int]]:
        """Return a NEW assignment with task `task` (0-indexed) moved to
        `new_agent` (1-indexed). Returns None if the move would overflow the
        new agent's capacity (the original assignment is never mutated).

        Use as a neighborhood move in local search; combine with objective()
        to evaluate the move's effect on total cost."""
        a = _validate_assignment_shape(assignment)
        j = _tj(task)
        new_i0 = _ai(new_agent)
        # Recompute load for the destination agent under the new assignment.
        new_load = 0.0
        for jj, aj in enumerate(a):
            try:
                if int(aj) - 1 == new_i0 and jj != j:
                    new_load += float(R[new_i0][jj])
            except (TypeError, ValueError):
                continue
        new_load += float(R[new_i0][j])
        if new_load > float(cap[new_i0]) + 1e-9:
            return None
        out = list(a)
        out[j] = new_i0 + 1
        return out

    def apply_swap_assignments(assignment: Iterable[int], t1: int,
                               t2: int) -> Optional[List[int]]:
        """Return a NEW assignment with tasks t1 and t2 (0-indexed) swapping
        their agents. Returns None if either of the two affected agents
        would exceed capacity after the swap. The input is never mutated."""
        a = _validate_assignment_shape(assignment)
        j1, j2 = _tj(t1), _tj(t2)
        if j1 == j2:
            return list(a)
        try:
            i1 = int(a[j1]) - 1
            i2 = int(a[j2]) - 1
        except (TypeError, ValueError):
            return None
        if not (0 <= i1 < m and 0 <= i2 < m):
            return None
        if i1 == i2:
            return list(a)
        # Recompute loads for the two affected agents only.
        load_i1 = 0.0
        load_i2 = 0.0
        for jj, aj in enumerate(a):
            try:
                v = int(aj) - 1
            except (TypeError, ValueError):
                continue
            if jj == j1 or jj == j2:
                continue
            if v == i1:
                load_i1 += float(R[i1][jj])
            elif v == i2:
                load_i2 += float(R[i2][jj])
        # After the swap, task j1 goes to i2 and task j2 goes to i1.
        load_i1 += float(R[i1][j2])
        load_i2 += float(R[i2][j1])
        if load_i1 > float(cap[i1]) + 1e-9:
            return None
        if load_i2 > float(cap[i2]) + 1e-9:
            return None
        out = list(a)
        out[j1] = i2 + 1
        out[j2] = i1 + 1
        return out

    # ==================================================================
    # (4) Exact / heavy: ILP
    # ==================================================================
    def ilp_gap(time_limit_s: float = 10.0) -> Optional[List[int]]:
        """Solve the GAP exactly via CBC (open-source MILP, via python-mip).

        Variables: x[i,j] in {0,1}  ('task j assigned to agent i')
        Objective: sum_{i,j} C[i][j] * x[i,j]   (minimize if 'min', else maximize)
        Constraints:
          - sum_i x[i,j] == 1  for each task j   (each task to exactly one agent)
          - sum_j R[i,j] * x[i,j] <= cap[i] for each agent i

        Returns a 1-indexed assignment list of length n, or None if the
        solver does not find a feasible solution within time_limit_s.
        Use as the primary solver when the instance is small enough, or
        within an LNS loop by fixing part of the assignment and re-solving."""
        sense = MINIMIZE if is_min else MAXIMIZE
        model = Model(sense=sense)
        model.verbose = 0
        model.max_seconds = float(time_limit_s)
        x = [[model.add_var(var_type=BINARY, name=f"x_{i}_{j}")
              for j in range(n)] for i in range(m)]
        model.objective = xsum(
            float(C[i][j]) * x[i][j] for i in range(m) for j in range(n)
        )
        for j in range(n):
            model += xsum(x[i][j] for i in range(m)) == 1, f"task_{j}"
        for i in range(m):
            model += xsum(float(R[i][j]) * x[i][j]
                          for j in range(n)) <= float(cap[i]), f"cap_{i}"
        status = model.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if model.num_solutions < 1:
            return None
        assignment = [0] * n
        for j in range(n):
            chosen = None
            for i in range(m):
                val = x[i][j].x
                if val is not None and val > 0.5:
                    chosen = i
                    break
            if chosen is None:
                return None
            assignment[j] = chosen + 1
        return assignment

    return {
        # (1) queries
        "cost": cost,
        "resource": resource,
        "agent_capacity": agent_capacity,
        "n_agents": n_agents,
        "n_tasks": n_tasks,
        # (2) feasibility
        "agent_load": agent_load,
        "agent_remaining": agent_remaining,
        "unassigned_tasks": unassigned_tasks,
        "is_feasible_assignment": is_feasible_assignment,
        # (3) construction / improvement
        "greedy_min_cost": greedy_min_cost,
        "greedy_min_resource_ratio": greedy_min_resource_ratio,
        "apply_reassign": apply_reassign,
        "apply_swap_assignments": apply_swap_assignments,
        # (4) exact
        "ilp_gap": ilp_gap,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "cost",
        "input": "i: int (1-indexed agent), j: int (0-indexed task)",
        "output": "float",
        "purpose": "cost_matrix[i-1][j]: cost of assigning task j to agent i.",
    },
    {
        "name": "resource",
        "input": "i: int (1-indexed agent), j: int (0-indexed task)",
        "output": "float",
        "purpose": "consumption_matrix[i-1][j]: resource consumed when task j is assigned to agent i.",
    },
    {
        "name": "agent_capacity",
        "input": "i: int (1-indexed agent)",
        "output": "float",
        "purpose": "capacities[i-1]: resource capacity of agent i.",
    },
    {
        "name": "n_agents",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of agents m.",
    },
    {
        "name": "n_tasks",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of tasks (jobs) n.",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "agent_load",
        "input": "i: int (1-indexed agent), assignment: list[int]",
        "output": "float",
        "purpose": (
            "Total resource consumption_matrix[i-1][j] over tasks j currently "
            "assigned to agent i. Entries in `assignment` outside [1, m] (e.g. "
            "0 or None used as 'unassigned') are ignored."
        ),
    },
    {
        "name": "agent_remaining",
        "input": "i: int (1-indexed agent), assignment: list[int]",
        "output": "float",
        "purpose": (
            "capacities[i-1] - agent_load(i, assignment). Negative if agent i "
            "is over capacity under `assignment`."
        ),
    },
    {
        "name": "unassigned_tasks",
        "input": "assignment: list[int]",
        "output": "list[int]",
        "purpose": (
            "0-indexed task indices whose assignment is missing/invalid (None, "
            "0, or outside [1, m]). Use for incremental construction -- the "
            "final CO-Bench solution requires every task to have an agent in "
            "[1, m]."
        ),
    },
    {
        "name": "is_feasible_assignment",
        "input": "assignment: list[int]",
        "output": "(bool, str | None)",
        "purpose": (
            "Cheap feasibility precheck WITHOUT calling eval_func: verifies "
            "length == n, every entry in [1, m], and no agent exceeds its "
            "capacity. Use inside tight local-search loops to filter "
            "infeasible neighbors before objective evaluation."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "greedy_min_cost",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Greedy construction: tasks are placed in order of hardness "
            "(highest min-resource first); each task is assigned to the "
            "feasible agent with the BEST cost in the problem's own direction "
            "('min' -> smallest, 'max' -> largest). May produce an infeasible "
            "assignment for tight instances -- verify with "
            "is_feasible_assignment or tools['is_feasible']. Good warm start "
            "for local search (apply_reassign / apply_swap_assignments)."
        ),
    },
    {
        "name": "greedy_min_resource_ratio",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Greedy variant that scores each (agent, task) by a "
            "resource-per-cost ratio: for 'min' problems uses |cost| * "
            "resource; for 'max' problems uses resource / |cost|. Often "
            "complements greedy_min_cost as an alternative warm start in a "
            "multi-start scheme."
        ),
    },
    {
        "name": "apply_reassign",
        "input": "assignment: list[int], task: int (0-indexed), new_agent: int (1-indexed)",
        "output": "list[int] | None",
        "purpose": (
            "Returns a NEW assignment with `task` moved to `new_agent`. "
            "Returns None if the move would exceed `new_agent`'s capacity. "
            "Pure function -- the input is not mutated. The basic 1-task move "
            "for local search."
        ),
    },
    {
        "name": "apply_swap_assignments",
        "input": "assignment: list[int], t1: int (0-indexed), t2: int (0-indexed)",
        "output": "list[int] | None",
        "purpose": (
            "Returns a NEW assignment with tasks t1 and t2 swapping their "
            "agents. Returns None if either of the two affected agents would "
            "exceed capacity. Pure function. Complementary to apply_reassign "
            "-- can escape local minima where every single-task move is "
            "infeasible."
        ),
    },
    # ----- (4) Exact / heavy -----
    {
        "name": "ilp_gap",
        "input": "time_limit_s: float = 10.0",
        "output": "list[int] | None",
        "purpose": (
            "Solve the GAP exactly via CBC (python-mip) with a wall-clock "
            "budget. Variables x[i,j] in {0,1}, objective sum C[i][j]*x[i,j] "
            "(minimised if problem_type=='min', maximised otherwise), "
            "constraints `each task to exactly one agent` and `agent load <= "
            "capacity`. Returns a 1-indexed assignment list (length n), or "
            "None if no feasible solution was found within the budget. "
            "Primary tool when the instance fits; can also be used in LNS by "
            "warm-starting from a heuristic and re-running with a tight "
            "budget."
        ),
    },
]
