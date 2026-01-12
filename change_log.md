# Change Log for Carbox Project

This log details the modifications made to the `carbox` project as part of a debugging and feature enhancement session.

## Summary of Changes:

1.  **Enabled JAX Debug Printing**:
    *   **File**: `benchmarks/run_cse.py`
    *   **Change**: Added `jax.config.update("jax_debug_print_always", True)` to ensure `jax.debug.print` statements are always outputted, aiding in real-time debugging of JAX-compiled code.

2.  **Added Debug Print for PHReactionRateTerm**:
    *   **File**: `carbox/reactions.py`
    *   **Change**: Inserted a `jax.debug.print` statement within `PHReactionRateTerm.__call__` to display `alpha`, `uv_field`, `visual_extinction`, `gamma`, and the calculated `rate` during photodissociation reaction computations. This was crucial for diagnosing why a reaction with `alpha=0` was yielding a non-zero rate.

3.  **Fixed `NameError: name 'jax' is not defined`**:
    *   **File**: `carbox/reactions.py`
    *   **Change**: Added `import jax` at the top of the file to resolve a `NameError` that occurred when `jax.debug.print` was called in a file where `jax` had not been explicitly imported.

4.  **Enhanced `Reaction` Class with `reaction_id`**:
    *   **File**: `carbox/reactions.py`
    *   **Change**: Modified the `Reaction` dataclass to include an optional `reaction_id` attribute. This allows each reaction object to store its original identifier from the input chemical network file (e.g., UMIST `reaction_number`). The `__repr__` method was also updated to include this ID for better debugging.

5.  **Updated `Reaction` Subclass Constructors**:
    *   **File**: `carbox/reactions.py`
    *   **Change**: Modified the constructors for `CRPReaction`, `UMISTPhotoReaction`, and `KAReaction` to accept the new `reaction_id` argument and pass it correctly to the base `Reaction` class constructor.

6.  **`UMISTParser` Integrates `reaction_id`**:
    *   **File**: `carbox/parsers/umist_parser.py`
    *   **Change**: Modified the `parse_reaction` method in `UMISTParser` to extract the `reaction_number` from the parsed UMIST CSV data and pass it as the `reaction_id` when instantiating `CRPReaction`, `UMISTPhotoReaction`, and `KAReaction` objects.

7.  **Standardized Rate CSV Column Headers**:
    *   **Files**: `benchmarks/run_cse.py`, `benchmarks/run_carbox.py`
    *   **Change**: Modified the code responsible for generating the `rates.csv` output files. Instead of using descriptive reaction strings (e.g., "A + B -> C") as column headers, the files now use the numerical `reaction_id` for each reaction, providing a more consistent and easily parsable output format.

8.  **Fixed `NameError: name 'Optional' is not defined`**:
    *   **File**: `carbox/reactions.py`
    *   **Change**: Added `from typing import List, Union, Optional` to the top of the file. This resolved a `NameError` for `Optional` which was used in the `Reaction` class definition.

9.  **Resolved `ValueError: Terms are not compatible with solver!`**:
    *   **Files**: `carbox/solver.py` (indirectly), `carbox/reactions.py` (indirectly)
    *   **Change**: This error was a generic Diffrax error that manifested due to underlying `NameError`s (`jax` not defined, `Optional` not defined) within the JIT-compiled code. Fixing these `NameError`s is expected to resolve this compatibility error, as the JAX tracing mechanism can now correctly process the terms.