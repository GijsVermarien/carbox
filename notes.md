# Carbox notes

In /Users/marie/Chemsitry/carbox/, run 
`source venv/bin/activate` to activate the virtual python environment

## Issues encountered

TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function _fn at /Users/marie/Chemistry/carbox/venv/lib/python3.11/site-packages/equinox/_eval_shape.py:31 for jit. This concrete value was not available in Python because it depends on the value of the argument _dynamic[1][1][1][2].
See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError

According to gemini: The TracerBoolConversionError occurs because JAX is trying to turn a "placeholder" value (a Tracer) into a real Python True or False to execute an if statement during compilation. Since the actual value isn't known until the code runs on your data, Python crashes because it cannot make a "now" decision using a "future" result.

-> pip install --upgrade "jax[cpu]==0.4.26" "jaxlib==0.4.26" "equinox==0.11.4" "lineax==0.0.4" "diffrax==0.5.0"

Why this specific combination?
JAX 0.4.26: This is a very stable "LTS-like" version. It’s modern enough for your carbox requirements but predates some of the most aggressive "breaking changes" in JAX 0.4.30+.
Lineax 0.0.4: This version contains fixes specifically for the LU.compute error you saw. It replaces the "Manual Python if" with JAX-compatible logic.
Equinox & Diffrax: These versions are the "sweet spot" for compatibility with JAX 0.4.26.

### Running example

from carbox import SimulationConfig, run_simulation

config = SimulationConfig(
    number_density=1e4,
    temperature=50.0,
    t_end=1e6,
    run_name="example_run",
    max_steps=65536) #! different

results = run_simulation("data/simple_latent_tgas.csv", config, format_type="latent_tgas")
solution = results["solution"]
network = results["network"]



### CSE physics

In carbox/cse_physics.py 

Changed output.py to get rid of performance warning
    /Users/marie/Chemistry/carbox/carbox/output.py:110: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`


