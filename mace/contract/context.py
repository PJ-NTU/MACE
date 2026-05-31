from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ContractContext:
    """Accumulating state threaded across the I -> O -> T -> helpers designers.

    No eval_func: a new CO problem has none. The tool library T is generated
    directly as separate functions: is_feasible + objective (+ a few helpers).
    """
    # inputs
    nl_description: str
    sample_instance_text: str
    # locked after I
    description: str | None = None        # cleaned DESCRIPTION for config.py
    load_data_code: str | None = None     # source of load_data(path)
    input_schema: str | None = None       # human-readable typed field list
    # locked after O
    output_schema: str | None = None      # solution dict shape (Returns: text)
    placeholder_solution_code: str | None = None  # make_solution(instance) -> dict
    # locked after T core
    is_feasible_code: str | None = None   # def is_feasible(instance, solution) -> (bool, msg)
    objective_code: str | None = None     # def objective(instance, solution) -> float
    # locked after helpers
    helpers_code: str | None = None       # source defining a few helper functions
    helper_names: list | None = None      # names of the helper functions, for wiring
    tools_description: list | None = None # [{name,input,output,purpose}] for the H-gen prompt
    direction: str = "min"                # 'min' or 'max'
