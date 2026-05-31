from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ContractContext:
    """Accumulating state threaded across the I -> O -> T designers."""
    # inputs
    nl_description: str
    sample_instance_text: str
    # locked after I
    description: str | None = None
    load_data_code: str | None = None
    input_schema: str | None = None
    # locked after O
    output_schema: str | None = None
    placeholder_solution_code: str | None = None
    # locked after T
    eval_func_code: str | None = None
    feasibility_steps: str | None = None
    extras_code: str | None = None
    direction: str = "min"
