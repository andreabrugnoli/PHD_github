#!/bin/bash

#SBATCH -N1 -n1 -t00:20:00
#SBATCH -o conv_wave_deg1.sh.log-%j 

# Loading the required module
source /home/brugnolia/sources/firedrake/bin/activate
cd codes/ph_firedrake/cluster/ 

# Run the script
python3 conv_wave_deg1_cluster.py 
