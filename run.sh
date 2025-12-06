cd examples/latent_tgas/jax && python main_data.py
cd examples/latent_tgas/scipy && python main_data.py
python carbox/main.py
python examples/latent_tgas/plot_flux_no_heating.py
python examples/latent_tgas/plot_abunds_no_heating.py
python benchmarks/plot_publication_comparison.py
python benchmarks/compare_results.py
python notebooks/parser_testing.py
python notebooks/parse_network.py
bash benchmarks/run_benchmarks.sh
bash sensitivity_analysis/run_cr_sensitivity.sh
