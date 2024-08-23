import pandas as pd
import pulp
import re
import random
import numpy as np
import os
import time
from solver_functions import read_data, process_folder_t, duplicate_charging_stations, extract_parameters, create_mip_model, create_robust_mip_model, create_robust_lp_model

def process_files_in_folder(folder_path, output_folder, average_times):
    thresholds_to_test = [0.05, 0.1, 0.2, 0.3, 0.4]  # Define the thresholds to test
    threshold_results = {}  # Dictionary to store average time for each threshold
    infeasible_thresholds = {}  # Dictionary to track how many files were infeasible for each threshold

    for threshold in thresholds_to_test:
        total_time_for_threshold = 0  # To accumulate the total time for the current threshold
        infeasible_count = 0  # To count the number of infeasible files for this threshold
        file_count = 0  # To keep track of how many files were processed
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_count += 1
                file_path = os.path.join(folder_path, filename)
                lines, data1 = read_data(file_path)
                data = duplicate_charging_stations(data1)
                Q1, C1, r, g1, v = extract_parameters(lines)
                C = 10
                Q = 100
                g = 1 / g1
                N1 = list(range(1, 5))  # Assume 4 vehicles

                # Step 1: Solve the LP relaxation problem
                problem_lp, R, T, V, A, x, e, z, y = create_robust_lp_model(data, average_times, N1, Q, C, r, g, 0.3)

                # Solve LP model using HiGHS or Gurobi
                solver_lp = pulp.GUROBI(msg=False, timeLimit=5400)
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

                # Step 2: Modify and solve MIP model with the current threshold
                problem_mip, R, T, V, A, x, e, z, y = create_robust_mip_model(data, average_times, N1, Q, C, r, g, 0.3)

                for v in problem_mip.variables():
                    if v.name in lp_solution and v.cat == pulp.LpBinary:
                        if lp_solution[v.name] >= (1 - threshold):
                            v.setInitialValue(1)
                            v.fixValue()
                        elif lp_solution[v.name] <= 1e-6:
                            v.setInitialValue(0)
                            v.fixValue()

                # Solve the modified MIP model
                start_time_mip = time.time()
                solver_mip = pulp.GUROBI_CMD(timeLimit=3600, msg=True, gapRel=0.05)
                problem_mip.solve(solver_mip)
                end_time_mip = time.time()
                mip_time = end_time_mip - start_time_mip

                # Check if the MIP solution was feasible
                status = pulp.LpStatus[problem_mip.status]
                if status == 'Infeasible':
                    infeasible_count += 1  # Count the infeasible file
                    print(f"File {filename} was infeasible for threshold {threshold}")
                else:
                    total_time_for_threshold += mip_time  # Accumulate the time for this file

                # Save MIP results even if infeasible (for debugging or tracking purposes)
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result_lpmip_threshold_{threshold}.txt")
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"MIP Model Status: {status}\n")
                    if status != 'Infeasible':
                        total_cost = pulp.value(problem_mip.objective)
                        used_vehicles = [n for n in N1 if any(pulp.value(x[n, i, j]) > 0.5 for i, j in A)]
                        charging_schedule = {(n, i): pulp.value(e[n, i]) for n in N1 for i in R}
                        start_times = {(n, i): pulp.value(z[n, i]) for n in N1 for i in V}
                        remaining_energy = {(n, i): pulp.value(y[n, i]) for n in N1 for i in V}

                        # Create a mapping from node ID to its StringID
                        node_meaning = {row['ID']: row['StringID'] for _, row in data.iterrows()}

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

        # Calculate the average time for the current threshold, excluding infeasible files
        if file_count - infeasible_count > 0:
            average_time_for_threshold = total_time_for_threshold / (file_count - infeasible_count)
            threshold_results[threshold] = average_time_for_threshold
            infeasible_thresholds[threshold] = infeasible_count
            print(f"Average time for threshold {threshold}: {average_time_for_threshold} seconds (infeasible files: {infeasible_count})")
        else:
            print(f"All files were infeasible for threshold {threshold}")
            threshold_results[threshold] = None
            infeasible_thresholds[threshold] = infeasible_count

    # Save the overall results for all thresholds
    threshold_results_file = os.path.join(output_folder, "threshold_results.txt")
    with open(threshold_results_file, 'w') as f:
        f.write("Threshold\tAverage Time (s)\tInfeasible Files\n")
        for threshold, avg_time in threshold_results.items():
            if avg_time is not None:
                f.write(f"{threshold}\t{avg_time}\t{infeasible_thresholds[threshold]}\n")
            else:
                f.write(f"{threshold}\tInfeasible\t{infeasible_thresholds[threshold]}\n")

    print("Finished processing all thresholds.")


#folder_path = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C15\\C15_instances2'
#output_folder = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C15\\lpmip_R_results\\r15'
folder_path = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C5\\instances'
output_folder = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C5\\lpmip_test'

folder_path_t = "c:\\Users\\ypwan\\Downloads\\dissertation_MSc\\allinstances"
arc_statistics = process_folder_t(folder_path_t)
average_times = {arc: stats[0] for arc, stats in arc_statistics.items()}

process_files_in_folder(folder_path, output_folder, average_times)