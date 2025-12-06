"""Example of using the UMIST network with Carbox."""

from carbox.initial_conditions import InitialConditions
from carbox.output import Output
from carbox.parsers.umist_parser import UMISTParser
from carbox.solver import Solver


def run_umist_network(umist_filepath: str):
    """Runs the UMIST network simulation."""
    # 1. Parse the UMIST reaction file
    parser = UMISTParser()
    network = parser.parse_network(umist_filepath)

    # 2. Define initial conditions (example values)
    initial_conditions = InitialConditions(
        species_names=[s.name for s in network.species],
        initial_densities={"H2": 1e3, "CO": 1e2},  # Example initial densities
        temperature=10.0,
        cosmic_ray_rate=1.0e-17,
        uv_field=1.0,
        visual_extinction=1.0,
    )

    # 3. Create a solver
    solver = Solver(
        network=network,
        initial_conditions=initial_conditions,
        t_end=1e6,  # End time for simulation (e.g., 1 million years)
        dt_initial=1e-2,  # Initial timestep
    )

    # 4. Run the simulation
    results = solver.run()

    # 5. Output results (example: print final abundances)
    output = Output(results)
    final_abundances = output.get_final_abundances()
    print("Final abundances:")
    for species, abundance in final_abundances.items():
        print(f"{species}: {abundance:.2e}")


if __name__ == "__main__":
    # Path to your UMIST reaction file
    # You will need to provide a UMIST formatted reaction file.
    # For example, you can download one from the KIDA database or use a local one.
    # Example: umist_data_file = "path/to/your/umist_reactions.dat"
    # For demonstration, we'll assume a file named 'umist_reactions.dat' exists in the current directory.
    # Replace with the actual path to your UMIST reaction file.
    umist_data_file = "umist_reactions.dat"
    run_umist_network(umist_data_file)
