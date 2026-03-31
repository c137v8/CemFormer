from optimization.nsga2_optimization import (
    run_nsga2,
    plot_pareto,
    plot_convergence
)
if __name__ == "__main__":

    result = run_nsga2()
    print("Pareto Decision Variables:")
    print(result.X)
    
    print("Pareto Objectives:")
    print(result.F)
    # Generate plots
    plot_pareto(result)
    plot_convergence(result)

    print("Plots saved successfully.")