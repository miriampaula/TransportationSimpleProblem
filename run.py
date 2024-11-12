import numpy as np
import pandas as pd
import time
import os
import re

if not os.path.exists("Lab_simple_solved"):
    os.makedirs("Lab_simple_solved")

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

def solve_instance(costs, supply, demand, method_name, instance_name):
    allocation = np.zeros_like(costs)
    supply_left, demand_left = supply.copy(), demand.copy()
    iterations = 0
    start_time = time.perf_counter()
    
    if method_name == "nv":  # North-West Corner
        i, j = 0, 0
        while i < len(supply) and j < len(demand):
            alloc = min(supply_left[i], demand_left[j])
            allocation[i][j] = alloc
            supply_left[i] -= alloc
            demand_left[j] -= alloc
            iterations += 1
            if supply_left[i] == 0: i += 1
            elif demand_left[j] == 0: j += 1

    elif method_name == "rm":  # Row Minimum Method
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

    elif method_name == "mm":  # Minimum on Matrix Method
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

    elif method_name == "vam":  # Vogel's Approximation Method
            while np.sum(supply_left) > 0 and np.sum(demand_left) > 0:
                penalties = []

                for i in range(len(supply)):
                    if supply_left[i] > 0: 
                        row_costs = [(costs[i][j], j) for j in range(len(demand)) if demand_left[j] > 0]
                        if len(row_costs) > 1:
                            sorted_row_costs = sorted(row_costs)
                            penalty = sorted_row_costs[1][0] - sorted_row_costs[0][0]
                            penalties.append((penalty, i, 'row'))

                for j in range(len(demand)):
                    if demand_left[j] > 0:  
                        col_costs = [(costs[i][j], i) for i in range(len(supply)) if supply_left[i] > 0]
                        if len(col_costs) > 1:
                            sorted_col_costs = sorted(col_costs)
                            penalty = sorted_col_costs[1][0] - sorted_col_costs[0][0]
                            penalties.append((penalty, j, 'col'))

                if not penalties:
                    break

                penalties.sort(reverse=True, key=lambda x: x[0])
                max_penalty, idx, axis = penalties[0]

                if axis == 'row':
                    min_cost, min_j = min((costs[idx][j], j) for j in range(len(demand)) if demand_left[j] > 0)
                    alloc = min(supply_left[idx], demand_left[min_j])
                    allocation[idx][min_j] = alloc
                    supply_left[idx] -= alloc
                    demand_left[min_j] -= alloc

                elif axis == 'col':
                    min_cost, min_i = min((costs[i][idx], i) for i in range(len(supply)) if supply_left[i] > 0)
                    alloc = min(supply_left[min_i], demand_left[idx])
                    allocation[min_i][idx] = alloc
                    supply_left[min_i] -= alloc
                    demand_left[idx] -= alloc

                iterations += 1

            for i in range(len(supply)):
                for j in range(len(demand)):
                    if supply_left[i] > 0 and demand_left[j] > 0:
                        alloc = min(supply_left[i], demand_left[j])
                        allocation[i][j] += alloc
                        supply_left[i] -= alloc
                        demand_left[j] -= alloc

    runtime = time.perf_counter() - start_time
    total_cost = save_output_file(costs, allocation, supply, demand, method_name, instance_name)
    solved_status = "Solved" if all(sum(allocation[:, j]) == demand[j] for j in range(len(demand))) else "Not Solved"
    
    return instance_name, total_cost, iterations, runtime, solved_status


def process_all_instances():
    methods = ["nv", "rm", "mm", "vam"]
    results = {method: [] for method in methods}

    for file_name in os.listdir("Lab_simple_instances"):
        if file_name.endswith(".dat"):
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
                result = solve_instance(Cjk, SCj, Dk, method, instance_name)
                results[method].append(result)

    for method, result_data in results.items():
        df = pd.DataFrame(result_data, columns=["Instance", "Optimal", "Iterations", "Runtime (seconds)", "Solved Status"])
        df.to_excel(f"Lab_simple_solved_{method.upper()}.xlsx", index=False)

process_all_instances()
