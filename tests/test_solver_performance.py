import time
import numpy as np

from moneyrepair.simulate import make_synthetic_fragments
from moneyrepair.compat import compute_compatibility_fast
from moneyrepair.solver import solve_covering_sets


def test_solver_vectorized_performance_and_correctness():
    # Generate a medium synthetic dataset
    pieces = 18
    template, fragments = make_synthetic_fragments(pieces=pieces, width=160, height=80, seed=7)
    
    matrix = compute_compatibility_fast(fragments)
    
    # Time execution of the vectorized solver
    start = time.perf_counter()
    solutions = solve_covering_sets(
        fragments,
        matrix,
        target_coverage=0.95,
        max_solutions=10,
        time_limit_seconds=10,
        order_strategy="area_degree"
    )
    elapsed = time.perf_counter() - start
    
    print(f"Vectorized search solved {len(solutions)} solutions in {elapsed*1000:.2f} ms")
    
    # Assert correctness
    assert len(solutions) > 0
    # Top solution should have near 100% coverage
    assert solutions[0].coverage >= 0.95
    # The solver should run very fast (typically < 100ms for 18 pieces)
    assert elapsed < 1.0
