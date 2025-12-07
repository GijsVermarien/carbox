"""Testing the umist network."""

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.parsers import NetworkNames

config = SimulationConfig(
    number_density=1e4,
    temperature=50.0,
    t_end=1e5,
    solver="kvaerno5",
    max_steps=500000,
    n_snapshots=5,
    atol=1e-14,
    rtol=1e-8,
)

results = run_simulation(
    network_file="data/rate22_final.rates",
    config=config,
    format_type=NetworkNames.umist,
)

print("Simulation finished.")
print(
    f"Final abundances stored in: {config.output_dir}/{config.run_name}_abundances.csv"
)
