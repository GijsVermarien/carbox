cd jax/
python main_data.py > jax_heatcool.txt
cd ../scipy
python main_data.py > scipy_heatcool.txt
cd ..
python compare.py
