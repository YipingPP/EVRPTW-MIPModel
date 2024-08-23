import pandas as pd
import pulp
import re
import random
import numpy as np
import os
import time
from solver_functions import read_data, process_folder_t, duplicate_charging_stations, extract_parameters, create_mip_model, create_robust_mip_model, create_robust_lp_model

def process_files_in_folder_tw(folder_path, output_folder, average_times, factor):

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            lines, data1 = read_data(file_path)
            data = duplicate_charging_stations(data1)
            Q1, C1, r, g1, v = extract_parameters(lines)
            C = 10
            Q = 100
            g = 1/g1
            N2 = list(range(1, 6))  # Assume 5 vehicles

            max_runtime_mip = 3600  
            gap_threshold_mip = 0.1  
            solver_mip = pulp.GUROBI(msg=True, timeLimit=max_runtime_mip, mipGap=gap_threshold_mip)

            # Create a mapping from node ID to its StringID
            node_meaning = {row['ID']: row['StringID'] for _, row in data.iterrows()}

            time_window_output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_time_window_results_{factor}.txt")

            # Update ReadyTime and DueTime
            adjusted_data = data.copy()
            middle = (adjusted_data.loc[adjusted_data['Type'] == 'c', 'ReadyTime'] + adjusted_data.loc[adjusted_data['Type'] == 'c', 'DueDate']) / 2
            original_duration = adjusted_data.loc[adjusted_data['Type'] == 'c', 'DueDate'] - adjusted_data.loc[adjusted_data['Type'] == 'c', 'ReadyTime']
            new_duration = original_duration * factor
            adjusted_data.loc[adjusted_data['Type'] == 'c', 'ReadyTime'] = middle - new_duration / 2
            adjusted_data.loc[adjusted_data['Type'] == 'c', 'DueDate'] = middle + new_duration / 2

            problemrt, Rrt, Trt, Vrt, Art, xrt, ert, zrt, yrt = create_robust_mip_model(adjusted_data, average_times, N2, Q, C, r, g, 0.3)
    
            start_time_mip = time.time()
            problemrt.solve(solver_mip)
            end_time_mip = time.time()
            mip_time = end_time_mip - start_time_mip

            # Get results
            status_mip = pulp.LpStatus[problemrt.status]
            total_cost_mip = pulp.value(problemrt.objective)
            charging_schedule_rt = {(n, i): pulp.value(ert[n, i]) for n in N2 for i in Rrt}
            start_times_rt = {(n, i): pulp.value(zrt[n, i]) for n in N2 for i in Vrt}
            remaining_energy_rt = {(n, i): pulp.value(yrt[n, i]) for n in N2 for i in Vrt}
            used_vehicles = [n for n in N2 if any(pulp.value(xrt[n, i, j]) > 0.5 for i, j in Art)]

            # Write MIP results
            with open(time_window_output_file_path, 'a') as output_file:
                output_file.write(f"\nMIP Model with Time Window Factor = {factor}\n")
                output_file.write(f"MIP Model Status: {status_mip}\n")
                output_file.write(f"MIP Model Objective value: {total_cost_mip}\n")
                output_file.write(f"MIP Model Run Time: {mip_time} seconds\n")
                output_file.write(f"Number of EVs used: {len(used_vehicles)}\n")

                for n in N2:
                    route_rt = []
                    route2_rt = []
                    node_rt = 'O1'
                    while True:
                        for (i, j) in Art:
                            if i == node_rt and pulp.value(xrt[n, i, j]) > 0.5:
                                if j in Rrt:
                                    charged_energy_rt = pulp.value(charging_schedule_rt[(n, j)])
                                    if charged_energy_rt > 0:
                                        route_rt.append(j)
                                        route2_rt.append(node_meaning[j])
                                else:
                                    route_rt.append(j)
                                    route2_rt.append(node_meaning[j])
                                node_rt = j
                                break
                        else:
                            break

                    if route_rt:
                        output_file.write(f"Vehicle {n} robust route: {'O1'} -> {' -> '.join(route2_rt)}\n")
                        for node in route_rt:
                            arrival_time_rt = start_times_rt[(n, node)]
                            remaining_battery_rt = remaining_energy_rt[(n, node)]
                            if node in Rrt:
                                charged_energy_rt = pulp.value(charging_schedule_rt[(n, node)])
                                if charged_energy_rt > 0:
                                    output_file.write(f"  {node_meaning[node]}: start time = {arrival_time_rt}, remaining battery = {remaining_battery_rt}, charged energy = {charged_energy_rt}\n")
                            else:
                                output_file.write(f"  {node_meaning[node]}: start time = {arrival_time_rt}, remaining battery = {remaining_battery_rt}\n")

folder_path = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C5\\instances'
output_folder = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C5\\TW'

folder_path_t = "c:\\Users\\ypwan\\Downloads\\dissertation_MSc\\allinstances"
arc_statistics = process_folder_t(folder_path_t)
average_times = {arc: stats[0] for arc, stats in arc_statistics.items()}

# TW factor
time_window_factors = [2.0, 1, 0.5, 0] 

for factor in time_window_factors:
    process_files_in_folder_tw(folder_path, output_folder, average_times, factor)

