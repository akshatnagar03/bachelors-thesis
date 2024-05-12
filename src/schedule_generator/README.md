# Schedule Generator

This folder contains the implementation of the Ant Colony Optimization (ACO) algorithm for the job shop scheduling problem. The ACO algorithm is a probabilistic technique used in problem-solving designed to mimic the behavior of ants in finding paths from the colony to food.

> [!IMPORTANT]
> This is a **Proof of Concept** of the two-stage ACO, and not final in any way.

## Files

- [`poc_aco_main.py`](https://github.com/AlbinLind/bachelors-thesis/blob/master/src/schedule_generator/poc_aco_main.py): This is the main entry point for the ACO algorithm. It includes the `FullACO` class which is responsible for running the ACO algorithm and generating the schedule.

- [`poc_aco_v2.py`](https://github.com/AlbinLind/bachelors-thesis/blob/master/src/schedule_generator/poc_aco_v2.py): This file contains the `JobShopProblem` and `Job` classes which represent the job shop problem and a job/task/operation in the problem respectively. It also includes the `ACO` class which is the base class for the ACO algorithm.

- [`poc_aco_machine_assignment.py`](https://github.com/AlbinLind/bachelors-thesis/blob/master/src/schedule_generator/poc_aco_machine_assignment.py): This file contains the `FullJobShopProblem` class which extends the `JobShopProblem` class with additional functionality for assigning machines to jobs.

- [`poc_aco_local_search.py`](https://github.com/AlbinLind/bachelors-thesis/blob/master/src/schedule_generator/poc_aco_local_search.py): This file contains functions for performing local search in the ACO algorithm.

## Usage

To run the ACO algorithm, you need to execute the `poc_aco_main.py` file. This will parse the data, assign machines to jobs, run the ACO algorithm, and generate the schedule.

```sh
python poc_aco_main.py
```

## Dependencies

> [!NOTE]
> A solver is not needed for running `poc_aco_main.py`

To be able to run the ACO algorithm with exact local search, you need a MIP solver, such as CPLEX (recommended), coinor-cbc, or glpk. You have to change to the correct solver manually in the file `poc_aco_local_search.py`.

It is also highly advised to follow the installation instructions in the project's [`README.md`](https://github.com/AlbinLind/bachelors-thesis/blob/master/README.md)