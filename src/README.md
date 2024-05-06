# Code
This folder contains the code for the thesis.

Under the folder `schedule_generator` you can find the code for Ant Colony Optimization with a exact local search. To be able to run it (with exact search) you need a MIP solver, such as CPLEX (recomended), coinor-cbc, or glpk. At the moment you have to change to the correct solver manually in the file `schedule_generator/poc_aoc_local_search.py`.