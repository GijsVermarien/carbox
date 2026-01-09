# Original benchmarks for later
```NETWORK_CONFIGS = {
    "small_chemistry": {
        "description": "Small gas-phase chemistry (~20 species)",
        "input_file": "../data/uclchem_small_chemistry.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/small_chemistry_initial.yaml",
    },
    "gas_phase_only": {
        "description": "Gas-phase only chemistry (~183 species)",
        "input_file": "../data/uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/gas_phase_only_initial.yaml",
    },
    "gas_phase_only_cse": {
        "description": "Gas-phase only chemistry (~183 species)",
        "input_file": "../data/uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/orich_cse_umist.yaml",
    },
    "orich_cse": {
        "description": "UMIST Rate22 network with O-rich parent species",
        "input_file": "../data/umist22.csv",
        "input_format": "umist",
        "initial_conditions": "initial_conditions/orich_cse_umist.yaml",
    }
}
```

```PHYSICAL_PARAMS = {
    "number_density": 1.0e4,  # cm^-3
    "temperature": 250.0,  # K
    "cr_rate": 1.0,  # s^-1
    "fuv_field": 1.0,  # Habing units
    "visual_extinction": 2.9643750143703076,  # mag (used if not self-consistent)
    # Self-consistent Av calculation (optional)
    "use_self_consistent_av": True,  # Enable self-consistent Av
    "base_av": 2.0,  # Base Av before column density contribution
    "cloud_radius_pc": 1.0,
    "t_start": 0.0,  # years
    "t_end": 5.0e6,  # years
    "n_snapshots": 100,  # output timesteps (increased for detail)
    "rtol": 1.0e-9,
    "atol": 1.0e-30,
    "solver": "kvaerno5",  # lowercase required
    "max_steps": 65536,  # max steps, always use power of 16 (e.g., 4096, 65536)
}
```

results/orig_gasphasonly = original benchmark 
results/orig_smallchemstry
results/orig_gasphasonly_cse == orig_gasphasonly but with CSE initial conditions

## Fixed Pandas DataFrame error when writing output
    orig_gasphasonly_cse_output
    Output is the same as orig_gasphasonly_cse, works fine


# Start UMIST adventure

## cse_0
Shorten calculation time
``` PHYSICAL_PARAMS = {
    "number_density": 1.0e4,  # cm^-3
    "temperature": 250.0,  # K
    "cr_rate": 1.0,  # s^-1
    "fuv_field": 1.0,  # Habing units
    "visual_extinction": 2.9643750143703076,  # mag (used if not self-consistent)
    # Self-consistent Av calculation (optional)
    "use_self_consistent_av": True,  # Enable self-consistent Av
    "base_av": 2.0,  # Base Av before column density contribution
    "cloud_radius_pc": 1.0,
    "t_start": 0.0,  # years
    "t_end": 1e2,  # years
    "n_snapshots": 100,  # output timesteps (increased for detail)
    "rtol": 1.0e-5,
    "atol": 1.0e-25,
    "solver": "kvaerno5",  # lowercase required
    "max_steps": 65536,  # max steps, always use power of 16 (e.g., 4096, 65536)
}
```

And changed this in solver/solve_network
```
    # Solve
    # solution = dx.diffeqsolve(
    #     ode_term,
    #     solver,
    #     t0=t_start_sec,
    #     t1=t_end_sec,
    #     dt0=1e-6,  # Initial timestep [s]
    #     y0=y0,
    #     stepsize_controller=dx.PIDController(
    #         atol=config.atol,
    #         rtol=config.rtol,
    #     ),
    #     saveat=dx.SaveAt(ts=t_snapshots_sec),
    #     args=params,
    #     max_steps=config.max_steps,
    # )

    # Solve (JIT compiled for performance)
    @eqx.filter_jit
    def _solve(t0, t1, y0, args, saveat_ts):
        return dx.diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t1,
            dt0=1e-6,  # Initial timestep [s]
            y0=y0,
            stepsize_controller=dx.PIDController(
                atol=config.atol,
                rtol=config.rtol,
            ),
            saveat=dx.SaveAt(ts=saveat_ts),
            args=args,
            max_steps=config.max_steps,
        )

    solution = _solve(t_start_sec, t_end_sec, y0, params, t_snapshots_sec)
```

Explanation 
```1. The Python Bottleneck: Without JIT, every single step of your ODE solver (and there can be tens of thousands) requires the Python interpreter to dispatch operations. It has to say "multiply these arrays," "add this vector," "check this tolerance" for every single timestep. This "interpreter overhead" is often slower than the math itself for chemical networks.
2. The JIT Solution: When you wrap the solver in @eqx.filter_jit, JAX traces the entire integration loop once. It sees the logic: "Run this Newton-Raphson step 65,000 times." It then compiles this entire loop into a single, fused binary executable.
3. The Result: When you run the simulation, Python calls this binary once. The binary runs the entire chemistry evolution on your CPU/GPU at C++ speeds, and returns the final result.
```

 python run_cse.py --network orich_cse --output results/cse_0
"Simulation complete! Total time: 631.76 seconds"
A bit faster than before, but not much!

## cse_1
Made carbox/cse_physics.py, which calculates density, temperature and av 
Start and end of the model are now radii
cr_rate and fuv_field constant inputs (scaling factors)

## cse_gpo
Takes a very long time, let's test with the smaller network  "gas_phase_only_cse", 1e14 to 1e18 cm
python run_cse.py --network gas_phase_only_cse --output results/cse_gpo

!! Geometric dilution term needed in ODEs !!! 
Without this term, all abundances increase

## cse_1 part 2
Back to UMIST
TracerBoolConversionError 
```
ERROR: Carbox failed after 50.34s
  TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function _fn at /Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_eval_shape.py:31 for jit. This concrete value was not available in Python because it depends on the value of the argument _dynamic[1][1][1][2].
See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError
Traceback (most recent call last):
  File "/Users/marie/Chemistry/carbox/benchmarks/run_cse.py", line 264, in run_carbox
    results = run_simulation(
              ^^^^^^^^^^^^^^^
  File "/Users/marie/Chemistry/carbox/carbox/main.py", line 153, in run_simulation
    solution = solve_network(jnetwork, y0, config)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Chemistry/carbox/carbox/solver.py", line 169, in solve_network
    solution = _solve(t_start_sec, t_end_sec, y0, physics, t_snapshots_sec)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_jit.py", line 209, in __call__
    return _call(self, False, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_jit.py", line 263, in _call
    marker, _, _ = out = jit_wrapper._cached(
                         ^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Chemistry/carbox/carbox/solver.py", line 153, in _solve
    return dx.diffeqsolve(
           ^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/diffrax/_integrate.py", line 1416, in diffeqsolve
    final_state, aux_stats = adjoint.loop(
                             ^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_module/_prebuilt.py", line 33, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/diffrax/_adjoint.py", line 299, in loop
    final_state = self._loop(
                  ^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/diffrax/_integrate.py", line 619, in loop
    _, traced_jump, traced_result = eqx.filter_eval_shape(body_fun_aux, init_state)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/diffrax/_integrate.py", line 349, in body_fun_aux
    (y, y_error, dense_info, solver_state, solver_result) = solver.step(
                                                            ^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_module/_prebuilt.py", line 33, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/diffrax/_solver/runge_kutta.py", line 1149, in step
    final_val = eqxi.while_loop(
                ^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/internal/_loop/loop.py", line 107, in while_loop
    return checkpointed_while_loop(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/internal/_loop/checkpointed.py", line 247, in checkpointed_while_loop
    body_fun_ = filter_closure_convert(body_fun_, init_val_)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/internal/_loop/common.py", line 511, in new_body_fun
    buffer_val2 = body_fun(buffer_val)
                  ^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/diffrax/_solver/runge_kutta.py", line 984, in rk_stage
    nonlinear_sol = optx.root_find(
                    ^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/optimistix/_root_find.py", line 218, in root_find
    return iterative_solve(
           ^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/optimistix/_iterate.py", line 344, in iterative_solve
    ) = adjoint.apply(_iterate, rewrite_fn, inputs, tags)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_module/_prebuilt.py", line 33, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/optimistix/_adjoint.py", line 133, in apply
    return implicit_jvp(primal_fn, rewrite_fn, inputs, tags, self.linear_solver)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/optimistix/_ad.py", line 60, in implicit_jvp
    root, residual = _implicit_impl(fn_primal, fn_rewrite, inputs, tags, linear_solver)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/optimistix/_ad.py", line 67, in _implicit_impl
    return jtu.tree_map(jnp.asarray, fn_primal(inputs))
                                     ^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/optimistix/_iterate.py", line 240, in _iterate
    final_carry = while_loop(cond_fun, body_fun, init_carry, max_steps=max_steps)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/internal/_loop/loop.py", line 103, in while_loop
    _, _, _, final_val = lax.while_loop(cond_fun_, body_fun_, init_val_)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/internal/_loop/common.py", line 511, in new_body_fun
    buffer_val2 = body_fun(buffer_val)
                  ^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/optimistix/_iterate.py", line 230, in body_fun
    new_y, new_state, aux = solver.step(fn, y, args, options, state, tags)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_module/_prebuilt.py", line 33, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/diffrax/_root_finder/_verychord.py", line 127, in step
    sol = lx.linear_solve(
          ^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/lineax/_solve.py", line 820, in linear_solve
    solution, result, stats = eqxi.filter_primitive_bind(
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/internal/_primitive.py", line 271, in filter_primitive_bind
    flat_out = prim.bind(*dynamic, treedef=treedef, static=static, flatten=flatten)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/internal/_primitive.py", line 156, in _wrapper
    out = rule(*args)
          ^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/lineax/_solve.py", line 126, in _linear_solve_abstract_eval
    out = eqx.filter_eval_shape(
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/lineax/_solve.py", line 87, in _linear_solve_impl
    out = solver.compute(state, vector, options)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_module/_prebuilt.py", line 33, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/lineax/_solve.py", line 648, in compute
    solution, result, _ = solver.compute(state, vector, options)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_module/_prebuilt.py", line 33, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/marie/Library/Python/3.11/lib/python/site-packages/lineax/_solver/lu.py", line 61, in compute
    trans = 1 if transpose else 0
            ^^^^^^^^^^^^^^^^^^^^^
jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function _fn at /Users/marie/Library/Python/3.11/lib/python/site-packages/equinox/_eval_shape.py:31 for jit. This concrete value was not available in Python because it depends on the value of the argument _dynamic[1][1][1][2].
See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
```

Claude to the rescue. Switched to explicit tsit5 solver, issue lied with the first step, which wasn't 0.

Works for small network, not for umist. This is because JNetwork isn't JAX safe. 
Moved get_reactant_multipliers form JNetwork to Network

In original code:

    @jax.jit
    def get_rates(self, temperature, cr_rate, fuv_rate, visual_extinction, abundances):
        """
        Get the reaction rates for the given temperature, cosmic ray ionisation rate,
        FUV radiation field, and abundance vector.
        """
        # TODO: optimization: The most Jax way to do optimize would be to create one class with all the reactions of one type and all their constants.
        # rates = jnp.empty(len(self.reactions))
        # for i, reaction in enumerate(self.reactions):
        #     rates = rates.at[i].set(reaction(temperature, cr_rate, fuv_rate))
        # return rates
        return jnp.hstack(
            [
                reaction(temperature, cr_rate, fuv_rate, visual_extinction, abundances)
                for reaction in self.reactions
            ]
        )

"get_rates still uses a Python list comprehension:
Inside JIT, this is unrolled into a sequence of constant additions to the graph. For a "large network," this causes:
Graph Bloat: JAX has to track the derivative of every single stacked rate.
Tracer Leakage: If any of these reactions (even the standard ones) have a hidden if or a type-cast that depends on the abundances tracer, the error will persist."

"Recommendation: Refactor JNetwork for Dense Logic
To truly fix the "heavy" Jacobian for large networks, you should try to group all reactions of the same type into single vectorized calls (e.g., one call for all KAReaction types) rather than iterating through a list."

"The error specifically happens because Lineax (the linear algebra library used by Diffrax) is trying to optimize the LU decomposition and hitting a tracer. You can bypass this by explicitly defining a non-branching solver in your solver.py:"

Reactions were fine! 

- moved get_rates out of JNetwork
- being super pedantic about the solver
```
    root_finder = optx.Newton(rtol=config.rtol, atol=config.atol, linear_solver=lx.LU())
    
    @eqx.filter_jit
    def _solve(t0, t1, y0, args, saveat_ts):
        my_kvaerno = dx.Kvaerno5(root_finder=root_finder)
        return dx.diffeqsolve(
            ode_term,
            solver=my_kvaerno,
            t0=t0,
            t1=t1,
            dt0=1e-6,  # Initial timestep [s]
            y0=y0,
            stepsize_controller=dx.PIDController(
                atol=config.atol,
                rtol=config.rtol,
            ),
            saveat=dx.SaveAt(ts=saveat_ts),
            args=args,
            max_steps=config.max_steps,
        )
```
"When your chemical network is small, the solver uses very simple math. But as the network grows, the Jacobian matrix (the matrix describing how every species affects every other species) becomes large and complex.

The library Lineax (the math engine behind the solver) tries to be "smart" to save time. It looks at your large matrix and tries to run code like this internally:
    if matrix_is_very_sparse:
        use_algorithm_A()
    else:
        use_algorithm_B()
Because your network is "large," the "sparsity" of the matrix is being traced. When Lineax hits that if statement, it crashes because it's trying to make a Python decision based on a JAX Tracer.

By passing linear_solver=lx.LU() inside the root_finder, you effectively told the solver: "Do not look at the matrix. Do not try to choose an algorithm. Just use the standard LU decomposition."




