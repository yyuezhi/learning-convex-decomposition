#!/bin/bash

conda install -y nvidia/label/cuda-12.4.0::cuda 
pip install psutil
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru pytz scikit-learn
pip install Werkzeug==2.2.2  pymeshlab  libigl==2.6.1 embreex einops  rtree pyfqmr  matplotlib boto3 numba pymeshlab==2023.12.post3
pip install mesh2sdf tetgen==0.7.0 vtk
conda install -y -c conda-forge cmake=3.27 ninja pybind11 compilers
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
