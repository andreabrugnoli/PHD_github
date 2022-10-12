
ENV["PYTHON"] = "~/anaconda3/envs/fenicslegacy-env/bin/python3.10"
using Pkg
Pkg.add(PackageSpec(name="PyCall", rev="master"))
Pkg.build("PyCall")
using PyCall

fenics = pyimport_conda("fenics")
#= py"""
from fenics import *
"""
 =#


Pkg.add("FEniCS")  # Some incompatibility issue
