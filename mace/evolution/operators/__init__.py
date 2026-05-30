"""Stage Two operators (O1-O7) + bootstrap helper.

Generation operators (O1-O5) export `generate(spec, portfolio, F, R, llm_client)
                                                    -> (code, meta_dict)`.

Ordering convention: by number of input algorithms (1 -> 1 -> 2 -> 2 -> N).

  O1 Weighted Mutation        (1 parent, weighted by 1/r_bar)
  O2 Reflective Redesign      (1 parent, weighted by r_bar)
  O3 Complementary Crossover  (2 parents, by rank-L1 distance)
  O4 Comparative Synthesis    (2 parents: 1 strong + 1 weak, by contrast)
  O5 Diversity Injection      (N parents = whole portfolio)

Repair operators (O6, O7) export `repair(spec, broken_code, info, llm_client)
                                              -> code`.

`_bootstrap_random` is NOT part of GENERATION_OPERATORS; it is a from-scratch
generator used only by `build_initial_portfolio` when no parent exists yet.
"""
from . import (
    o1_weighted_mutation,
    o2_reflective_redesign,
    o3_complementary_crossover,
    o4_comparative_synthesis,
    o5_diversity_injection,
    o6_error_repair,
    o7_efficiency_repair,
    _bootstrap_random,
)

GENERATION_OPERATORS = [
    o1_weighted_mutation,
    o2_reflective_redesign,
    o3_complementary_crossover,
    o4_comparative_synthesis,
    o5_diversity_injection,
]
