# Stable Schedule Generator

This folder contains the implementation of the Ant Colony Optimization (ACO) algorithm for the job shop scheduling problem. The ACO algorithm is a probabilistic technique used in problem-solving designed to mimic the behavior of ants in finding paths from the colony to food.

> [!NOTE]
> This is the "stable" version. It contains a somewhat good implementations of the two-stage ACO, with batches.

## Usage

To run the ACO algorithm, you need to execute the `ant_colony_optimisation.py` file. This will parse the data, assign machines to jobs, run the ACO algorithm, and generate the schedule.

```sh
python ant_colony_optimisation.py
```

## Dependencies

It is also highly advised to follow the installation instructions in the project's [`README.md`](https://github.com/AlbinLind/bachelors-thesis/blob/master/README.md)