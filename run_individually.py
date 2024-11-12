import numpy as np
import pandas as pd
import time
import os
import re

# Ensure output directory exists
if not os.path.exists("Lab_simple_solved"):
    os.makedirs("Lab_simple_solved")

def print_table(costs, allocation, supply, demand):
    table = pd.DataFrame(costs, columns=[f"D{i+1}" for i in range(len(demand))], dtype=str)
    for i in range(len(supply)):
        for j in range(len(demand)):
            if allocation[i][j] > 0:
                table.iloc[i, j] = f"{costs[i][j]} ({allocation[i][j]})"
            else:
                table.iloc[i, j] = f"{costs[i][j]}"
    table["Supply"] = [str(s) for s in supply]
    demand_row = [str(d) for d in demand] + [str(sum(supply))]
    table.loc["Demand"] = demand_row
    print(table)

def save_output_file(costs, allocation, supply, demand, method_name, instance_name):
    total_cost = int(np.sum(np.multiply(costs, allocation)))
    Uj = [1 if sum(allocation[i]) > 0 else 0 for i in range(len(supply))]
    Dk_str = " ".join(f"{val}" for val in demand)

    output = (
        f"Xjk=\t{allocation}\n"
        f"Uj=\t\t [{Uj}]\n"
        f"Dk=\t\t [{Dk_str}]\n"
        f"Optim        = {total_cost}\n"
    )

    file_name = f"{instance_name}_{method_name}.txt"
    with open(os.path.join("Lab_simple_solved", file_name), "w") as f:
        f.write(output)
    
    return total_cost

def solve_instance(costs, supply, demand, method_name, instance_name, print_steps=False):
    allocation = np.zeros_like(costs)
    supply_left, demand_left = supply.copy(), demand.copy()
    iterations = 0
    start_time = time.perf_counter()
    
    if method_name == "nv":  # North-West Corner
        i, j = 0, 0
        print("NorthWest Method")
        while i < len(supply) and j < len(demand):
            alloc = min(supply_left[i], demand_left[j])
            allocation[i][j] = alloc
            supply_left[i] -= alloc
            demand_left[j] -= alloc
            iterations += 1
            if print_steps:
                print_table(costs, allocation, supply_left, demand_left)
            if supply_left[i] == 0: i += 1
            elif demand_left[j] == 0: j += 1

    elif method_name == "rm":  # Row Minimum Method
        print("Row Minimum Method")
        for i in range(len(supply)):
            while supply_left[i] > 0 and np.sum(demand_left) > 0:
                min_cost = float('inf')
                min_j = -1
                for j in range(len(demand)):
                    if demand_left[j] > 0 and costs[i][j] < min_cost:
                        min_cost = costs[i][j]
                        min_j = j
                if min_j == -1:
                    break
                alloc = min(supply_left[i], demand_left[min_j])
                allocation[i][min_j] = alloc
                supply_left[i] -= alloc
                demand_left[min_j] -= alloc
                iterations += 1
                if print_steps:
                    print_table(costs, allocation, supply_left, demand_left)

    elif method_name == "mm":  # Minimum on Matrix Method
        print("Global Minimum Method")

        while np.sum(supply_left) > 0 and np.sum(demand_left) > 0:
            min_cost = float('inf')
            min_i, min_j = -1, -1
            for i in range(len(supply)):
                for j in range(len(demand)):
                    if supply_left[i] > 0 and demand_left[j] > 0 and costs[i][j] < min_cost:
                        min_cost = costs[i][j]
                        min_i, min_j = i, j
            alloc = min(supply_left[min_i], demand_left[min_j])
            allocation[min_i][min_j] = alloc
            supply_left[min_i] -= alloc
            demand_left[min_j] -= alloc
            iterations += 1
            if print_steps:
                print_table(costs, allocation, supply_left, demand_left)
                
    elif method_name == "vam":  # Vogel's Approximation Method
        while np.sum(supply_left) > 0 and np.sum(demand_left) > 0:
            penalties = []

            # Calculate penalties for rows with remaining supply
            for i in range(len(supply)):
                if supply_left[i] > 0:  # Only consider rows with remaining supply
                    # Collect costs for columns with remaining demand only
                    row_costs = [(costs[i][j], j) for j in range(len(demand)) if demand_left[j] > 0]
                    if len(row_costs) > 1:
                        # Sort by cost and calculate penalty as the difference between the two lowest costs
                        sorted_row_costs = sorted(row_costs)
                        penalty = sorted_row_costs[1][0] - sorted_row_costs[0][0]
                        penalties.append((penalty, i, 'row'))
                        print(f"Row {i} penalty: {penalty}, based on costs {row_costs}")
                    else:
                        print(f"Skipping penalty calculation for Row {i} as it has only one remaining cost: {row_costs}")

            # Calculate penalties for columns with remaining demand
            for j in range(len(demand)):
                if demand_left[j] > 0:  # Only consider columns with remaining demand
                    # Collect costs for rows with remaining supply only
                    col_costs = [(costs[i][j], i) for i in range(len(supply)) if supply_left[i] > 0]
                    if len(col_costs) > 1:
                        # Sort by cost and calculate penalty as the difference between the two lowest costs
                        sorted_col_costs = sorted(col_costs)
                        penalty = sorted_col_costs[1][0] - sorted_col_costs[0][0]
                        penalties.append((penalty, j, 'col'))
                        print(f"Column {j} penalty: {penalty}, based on costs {col_costs}")
                    else:
                        print(f"Skipping penalty calculation for Column {j} as it has only one remaining cost: {col_costs}")

            # If no penalties are available, break
            if not penalties:
                print("No more penalties available, stopping allocation.")
                break

            # Select the maximum penalty; if tied, choose the minimum cost cell
            penalties.sort(reverse=True, key=lambda x: x[0])
            max_penalty, idx, axis = penalties[0]
            print(f"Selected highest penalty: {max_penalty} for {'row' if axis == 'row' else 'column'} {idx}")

            # Allocate based on row or column with lowest cost in the chosen row or column
            if axis == 'row':
                # Choose the cell with minimum cost in the selected row
                min_cost, min_j = min((costs[idx][j], j) for j in range(len(demand)) if demand_left[j] > 0)
                alloc = min(supply_left[idx], demand_left[min_j])
                allocation[idx][min_j] = alloc
                supply_left[idx] -= alloc
                demand_left[min_j] -= alloc
                print(f"Allocating {alloc} units to cell ({idx}, {min_j}) with cost {min_cost}")
                print(f"Updated supply for row {idx}: {supply_left[idx]}, updated demand for column {min_j}: {demand_left[min_j]}")

            elif axis == 'col':
                # Choose the cell with minimum cost in the selected column
                min_cost, min_i = min((costs[i][idx], i) for i in range(len(supply)) if supply_left[i] > 0)
                alloc = min(supply_left[min_i], demand_left[idx])
                allocation[min_i][idx] = alloc
                supply_left[min_i] -= alloc
                demand_left[idx] -= alloc
                print(f"Allocating {alloc} units to cell ({min_i}, {idx}) with cost {min_cost}")
                print(f"Updated supply for row {min_i}: {supply_left[min_i]}, updated demand for column {idx}: {demand_left[idx]}")

            iterations += 1
            if print_steps:
                print("Allocation Table after this step:")
                print_table(costs, allocation, supply_left, demand_left)
            print("\n" + "="*40 + "\n")

        # Final allocation check for any remaining supply or demand
        for i in range(len(supply)):
            for j in range(len(demand)):
                if supply_left[i] > 0 and demand_left[j] > 0:
                    alloc = min(supply_left[i], demand_left[j])
                    allocation[i][j] += alloc
                    supply_left[i] -= alloc
                    demand_left[j] -= alloc
                    print(f"Final allocation of {alloc} units to cell ({i}, {j}) with cost {costs[i][j]}")
                    if print_steps:
                        print("Final Allocation Table:")
                        print_table(costs, allocation, supply_left, demand_left)


    runtime = time.perf_counter() - start_time
    total_cost = save_output_file(costs, allocation, supply, demand, method_name, instance_name)
    solved_status = "Solved" if all(sum(allocation[:, j]) == demand[j] for j in range(len(demand))) else "Not Solved"
    
    return instance_name, total_cost, iterations, runtime, solved_status


def process_specific_instance(file_name):
    methods = ["nv", "rm", "mm", "vam"]
    results = {method: [] for method in methods}

    file_path = os.path.join("Lab_simple_instances", file_name)
    with open(file_path, "r") as f:
        data = f.read()
        
        instance_name = re.search(r'instance_name\s*=\s*"([^"]+)";', data).group(1)
        d = int(re.search(r'd\s*=\s*(\d+);', data).group(1))
        r = int(re.search(r'r\s*=\s*(\d+);', data).group(1))
        SCj = list(map(int, re.search(r'SCj\s*=\s*\[([^\]]+)\];', data).group(1).split()))
        Dk = list(map(int, re.search(r'Dk\s*=\s*\[([^\]]+)\];', data).group(1).split()))
        Cjk_data_section = data.split("Cjk = ")[1].split(";")[0].replace("[", "").replace("]", "").strip()
        Cjk_numbers = list(map(int, Cjk_data_section.split()))
        supply, demand = SCj, Dk
        Cjk = np.array(Cjk_numbers).reshape((len(supply), len(demand)))

    for method in methods:
        result = solve_instance(
            Cjk, SCj, Dk, method, instance_name,
            print_steps=True  # Set to True if you want detailed steps for this instance
        )
        results[method].append(result)

    for method, result_data in results.items():
        df = pd.DataFrame(result_data, columns=["Instance", "Optimal", "Iterations", "Runtime (seconds)", "Solved Status"])
        df.to_excel(f"Lab_simple_solved_{method.upper()}_{instance_name}.xlsx", index=False)

# Call this function with the specific file you want to process, e.g.:
process_specific_instance("Lab01_simple_small_01.dat")
