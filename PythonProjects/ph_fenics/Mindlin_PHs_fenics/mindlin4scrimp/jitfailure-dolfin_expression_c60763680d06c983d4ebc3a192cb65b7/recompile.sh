#!/bin/bash
# Execute this file to recompile locally
/home/a.brugnoli/anaconda3/envs/fenicsproject/bin/x86_64-conda_cos6-linux-gnu-c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/home/a.brugnoli/anaconda3/envs/fenicsproject/include -I/home/a.brugnoli/anaconda3/envs/fenicsproject/include/eigen3 -I/home/a.brugnoli/anaconda3/envs/fenicsproject/.cache/dijitso/include dolfin_expression_c60763680d06c983d4ebc3a192cb65b7.cpp -L/home/a.brugnoli/anaconda3/envs/fenicsproject/lib -L/home/a.brugnoli/anaconda3/envs/fenicsproject/home/a.brugnoli/anaconda3/envs/fenicsproject/lib -L/home/a.brugnoli/anaconda3/envs/fenicsproject/.cache/dijitso/lib -Wl,-rpath,/home/a.brugnoli/anaconda3/envs/fenicsproject/.cache/dijitso/lib -lmpi -lmpicxx -lpetsc -lslepc -lz -lhdf5 -lboost_timer -ldolfin -olibdijitso-dolfin_expression_c60763680d06c983d4ebc3a192cb65b7.so