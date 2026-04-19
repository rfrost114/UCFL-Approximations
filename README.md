# UCFL-Approximations
Python implementation of three approximation algorithms for UCFL

This was created as part of a term project for CSE 894 at Michigan State university

Main implementation is found in metric_UCFL.py which provides three approximations algorithms for the metric Uncapacitcated Facility Location problem (UCFL). 
- A 4-approximation based on deterministic rounding
- A randomized 3-approximation
- A 3-approximation using the primal-dual method
The implementations are based largely on the descriptions provided in the textbook by Williamson and Shmoys.

stats.ipynb contains the necessary code to produce the figures and statistics in the write up. 
The primary dependancies are cvxpy and numpy 

As is, the implementation uses the default cvxpy solvers, however it can easily be modified to use a solver of ones choice.

The test instances used in the write up can be downloaded from https://resources.mpi-inf.mpg.de/departments/d1/projects/benchmarks/UflLib/Euklid.html
