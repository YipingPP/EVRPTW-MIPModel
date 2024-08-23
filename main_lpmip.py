import pandas as pd
import pulp
import re
import random
import numpy as np
import os
import time
from solver_functions import read_data, process_folder_t, duplicate_charging_stations, extract_parameters, create_mip_model, create_robust_mip_model, create_robust_lp_model

def process_files_in_folder(folder_path, output_folder, average_times):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            lines, data1 = read_data(file_path)
            data = duplicate_charging_stations(data1)
            Q1, C1, r, g1, v = extract_parameters(lines)
            C = 10
            Q = 100
            g = 1 / g1
            N1 = list(range(1, 11))  # Assume 10 vehicles

            # Step 1: Solve the LP relaxation problem
            problem_lp, R, T, V, A, x, e, z, y = create_robust_lp_model(data, average_times, N1, Q, C, r, g, 0.3)

            # Solve LP model using HiGHS or Gurobi
            solver_lp = pulp.GUROBI(msg=False, timeLimit=600)
            start_time_lp = time.time()
            problem_lp.solve(solver_lp)
            end_time_lp = time.time()
            lp_time = end_time_lp - start_time_lp

            # Extract LP solution
            lp_solution = {v.name: v.varValue for v in problem_lp.variables()}

            # Save LP relaxation results
            lp_output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result_lp.txt")
            with open(lp_output_file_path, 'w') as lp_output_file:
                lp_output_file.write(f"LP Relaxation solved in {lp_time} seconds\n")
                lp_output_file.write("Non-zero x variables in LP relaxation:\n")
                for v in problem_lp.variables():
                    if v.name.startswith("x_") and v.varValue > 0:
                        lp_output_file.write(f"{v.name} = {v.varValue}\n")

            # Step 2: Modify and solve MIP model iteratively with threshold adjustment
            threshold = 0.1 # Initial threshold for fixing variables
            min_threshold = 0.01  # Minimum threshold to stop reducing
            mip_solved = False  # Flag to check if MIP has been solved
            max_iterations = 5  # Maximum iterations to try adjusting threshold
            iteration = 0

            while not mip_solved and iteration < max_iterations:
                # Create MIP model and apply LP-based fixing with current threshold
                problem_mip, R, T, V, A, x, e, z, y = create_robust_mip_model(data, average_times, N1, Q, C, r, g, 0.3)

                for v in problem_mip.variables():
                    if v.name in lp_solution and v.cat == pulp.LpBinary:
                        if lp_solution[v.name] >= (1 - threshold):
                            v.setInitialValue(1)
                            v.fixValue()
                        elif lp_solution[v.name] <= threshold:
                            v.setInitialValue(0)
                            v.fixValue()

                # Solve the modified MIP model
                start_time_mip = time.time()
                solver_mip = pulp.GUROBI_CMD(timeLimit=3600, msg=True)
                problem_mip.solve(solver_mip)
                end_time_mip = time.time()
                mip_time = end_time_mip - start_time_mip

                # Check if MIP was successfully solved
                status = pulp.LpStatus[problem_mip.status]
                if status == 'Infeasible':
                    # If infeasible, reduce the threshold and retry
                    threshold = max(threshold * 0.7, min_threshold)  # Reduce threshold, but not below min_threshold
                    iteration += 1
                    print(f"MIP infeasible, reducing threshold to {threshold} and retrying...")
                else:
                    mip_solved = True  # Mark MIP as solved if not infeasible

            if mip_solved:
                # Extract MIP solution if solved
                total_cost = pulp.value(problem_mip.objective)
                used_vehicles = [n for n in N1 if any(pulp.value(x[n, i, j]) > 0.5 for i, j in A)]
                charging_schedule = {(n, i): pulp.value(e[n, i]) for n in N1 for i in R}
                start_times = {(n, i): pulp.value(z[n, i]) for n in N1 for i in V}
                remaining_energy = {(n, i): pulp.value(y[n, i]) for n in N1 for i in V}

                # Create a mapping from node ID to its StringID
                node_meaning = {row['ID']: row['StringID'] for _, row in data.iterrows()}

                # Save MIP results
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result_lpmip2.txt")
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"MIP Model Status: {status}\n")
                    output_file.write(f"MIP Model Objective value: {total_cost}\n")
                    output_file.write(f"MIP Model Run Time: {mip_time} seconds\n")
                    output_file.write(f"Number of EVs used: {len(used_vehicles)}\n")
                    output_file.write(f"Threshold: {threshold}\n")
                    
                    for n in N1:
                        route = []
                        route2 = []
                        node = 'O1'
                        while True:
                            for (i, j) in A:
                                if i == node and pulp.value(x[n, i, j]) > 0.5:
                                    if j in R:
                                        charged_energy = pulp.value(charging_schedule[(n, j)])
                                        if charged_energy > 0:
                                            route.append(j)
                                            route2.append(node_meaning[j])
                                    else:
                                        route.append(j)
                                        route2.append(node_meaning[j])
                                    node = j
                                    break
                            else:
                                break

                        if route:
                            output_file.write(f"Vehicle {n} route: {'O1'} -> {' -> '.join(route2)}\n")
                            for node in route:
                                arrival_time = start_times[(n, node)]
                                remaining_battery = remaining_energy[(n, node)]
                                if node in R:
                                    charged_energy = pulp.value(charging_schedule[(n, node)])
                                    if charged_energy > 0:
                                        output_file.write(f"  {node_meaning[node]}: start time = {arrival_time}, remaining battery = {remaining_battery}, charged energy = {charged_energy}\n")
                                else:
                                    output_file.write(f"  {node_meaning[node]}: start time = {arrival_time}, remaining battery = {remaining_battery}\n")
            else:
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result_lpmip.txt")
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"MIP Model Status: {status}\n")
                print(f"Failed to solve MIP after {max_iterations} iterations with threshold adjustments.")


folder_path = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C15\\C15_instances1'
output_folder = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C15\\lpmip_R_results'
# folder_path = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C330_10\\30_10_instances_less'
# output_folder = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C30_10\\lpmip_R_results2'

folder_path_t = "c:\\Users\\ypwan\\Downloads\\dissertation_MSc\\allinstances"
arc_statistics = process_folder_t(folder_path_t)
average_times = {arc: stats[0] for arc, stats in arc_statistics.items()}

process_files_in_folder(folder_path, output_folder, average_times)


# python C:\Users\ypwan\Downloads\EVRPTW_C15\lpmip_R.py >> C:\Users\ypwan\Downloads\EVRPTW_C15\lpmip_R_results\terminal_informationc.txt
# python C:\Users\ypwan\Downloads\EVRPTW_C15\lpmip_R.py >> C:\Users\ypwan\Downloads\EVRPTW_C30_10\lpmip_R_results2\terminal_information.txt
