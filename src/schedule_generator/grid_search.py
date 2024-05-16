import time
from src.stable_schedule_generator.ant_colony_optimisation import TwoStageACO
from src.stable_schedule_generator.main import JobShopProblem, ObjectiveFunction
from src.production_orders import parse_data
import pandas as pd
import numpy as np


def grid_search(jssp: JobShopProblem):
    parameters = {
        "rho": [0.05, 0.25],
        "alpha": [0.05, 0.25],
        "q_zero": [0.75, 0.9],
        "n_ants": [200],
        "tau_zero": [
            1.0 / (200 * 15000.0), # 3.13e-7
            1.0 / (200 * 18000.0), # 2.77e-7
            1.0 / (200 * 20000.0), # 2.5e-7
        ],
    }
    results = dict()
    i = 0
    total_iters = np.prod([len(x) for x in parameters.values()])
    for rho in parameters["rho"]:
        for alpha in parameters["alpha"]:
            for q_zero in parameters["q_zero"]:
                for n_ants in parameters["n_ants"]:
                    for tau_zero in parameters["tau_zero"]:
                        aco = TwoStageACO(
                            jssp,
                            ObjectiveFunction.TARDINESS,
                            n_ants=n_ants,
                            n_iter=50,
                            rho=rho,
                            alpha=alpha,
                            beta=0,
                            q_zero=q_zero,
                            tau_zero=tau_zero,
                        )
                        start_time = time.time()
                        aco.run()
                        total_time = time.time() - start_time
                        results[i] = {
                            "rho": rho,
                            "alpha": alpha,
                            "beta": 0,
                            "q_zero": q_zero,
                            "n_ants": n_ants,
                            "tau_zero": tau_zero,
                            "tardiness": aco.best_solution[0],
                            "time": total_time,
                        }
                        i += 1
                        print(f"Completed {i}/{total_iters} iterations")

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.to_csv("grid_search_results.csv", index=False)
    print(results_df.sort_values(by="tardiness").head(10))


if __name__ == "__main__":
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    grid_search(jssp)
