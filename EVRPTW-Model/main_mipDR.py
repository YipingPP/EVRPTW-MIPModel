import pandas as pd
import pulp
import re
import random
import numpy as np
import os
import time
from solver_functions import read_data, process_folder_t, duplicate_charging_stations, extract_parameters, create_mip_model, create_robust_mip_model


folder_path_t = "c:\\Users\\ypwan\\Downloads\\dissertation_MSc\\allinstances"
arc_statistics = process_folder_t(folder_path_t)
average_times = {arc: stats[0] for arc, stats in arc_statistics.items()}

# 定义需要测试的theta值
theta_values = [0.3]

folder_path = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C10\\C10_instances'
output_folder = 'c:\\Users\\ypwan\\Downloads\\EVRPTW_C10\\mip_D_R_results'

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        lines, data1 = read_data(file_path)
        data = duplicate_charging_stations(data1)
        Q1, C1, r, g1, v = extract_parameters(lines)
        C = 10
        Q = 100
        g = 1/g1
        #distances, times_df = calculate_distances(data, v)
        N1 = list(range(1, 11))  # Assume 10 vehicles
        N2 = list(range(1, 11))  # Assume 10 vehicles

        # Solve the deterministic problem once
        problem, R, T, V, A, x, e, z, y = create_mip_model(data, average_times, N1, Q, C, r, g)

        # Solve the problem
        max_runtime = 3600
        gap_threshold = 0.1  # 设置相对gap阈值为10%
        #solver1 = pulp.HiGHS(time_limit=max_runtime, gapRel=gap_threshold, msg=True)
        solver = pulp.GUROBI_CMD(timeLimit=max_runtime, msg=True)
        #solver2 = pulp.GUROBI(timeLimit=7200, mipGap=gap_threshold, msg=True)
        solver2 = pulp.GUROBI_CMD(timeLimit=3600, msg=True)
        
        # 记录开始时间
        start_time = time.time()
        problem.solve(solver)
        
        end_time = time.time()
        deterministic_time = end_time - start_time
        print(f"11111111111111111111111111111111111111111 deterministic {filename} has been processed.111111111111111111111111111111111111111111\n")

##################################################################################################################################
##################################################################################################################################
        # Extracting deterministic results
        status = pulp.LpStatus[problem.status]
        total_cost = pulp.value(problem.objective)
        charging_schedule = {(n, i): pulp.value(e[n, i]) for n in N1 for i in R}
        start_times = {(n, i): pulp.value(z[n, i]) for n in N1 for i in V}
        remaining_energy = {(n, i): pulp.value(y[n, i]) for n in N1 for i in V}
        used_vehicles = [n for n in N1 if any(pulp.value(x[n, i, j]) > 0.5 for i, j in A)]

        # Create a mapping from node ID to its StringID
        node_meaning = {row['ID']: row['StringID'] for _, row in data.iterrows()}

        # # Save the results to a file
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result_2.txt")
        with open(output_file_path, 'w') as output_file:
            # Write deterministic results
            output_file.write(f"Deterministic Model Status: {status}\n")
            output_file.write(f"Deterministic Model Objective value: {total_cost}\n")
            output_file.write(f"Deterministic Model Run Time: {deterministic_time} seconds\n")
            output_file.write(f"Number of EVs used: {len(used_vehicles)}\n")  # 保存使用的 EV 数量
            #output_file.write(f"Deterministic Model MIP Gap: {mip_gap}\n")
            
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

###################################################################################################################################################
###################################################################################################################################################
            # Solve the robust problem for each theta
            for theta in theta_values:
                problemr, Rr, Tr, Vr, Ar, xr, er, zr, yr = create_robust_mip_model(data, average_times, N2, Q, C, r, g, theta)

                # 记录开始时间
                start_time_r = time.time()
                problemr.solve(solver2)
                end_time_r = time.time()
                robust_time = end_time_r - start_time_r
                print(f"2222222222222222222222222222222222222222222222222222 robust {filename} has been processed. theta = {theta}\n2222222222222222222222222222222222222222222")


                # Extracting robust results
                status_r = pulp.LpStatus[problemr.status]
                if status_r == 'Infeasible':
                    print(f"Robust model for {filename} is infeasible with theta = {theta}\n")
                    output_file.write(f"\nRobust Model with theta = {theta} is infeasible.\n")
                    continue  # 跳过不可行的模型，继续下一个 theta

                total_cost_r = pulp.value(problemr.objective)
                charging_schedule_r = {(n, i): pulp.value(er[n, i]) for n in N2 for i in Rr}
                start_times_r = {(n, i): pulp.value(zr[n, i]) for n in N2 for i in Vr}
                remaining_energy_r = {(n, i): pulp.value(yr[n, i]) for n in N2 for i in Vr}

                used_vehicles_r = [n for n in N2 if any(pulp.value(xr[n, i, j]) is not None and pulp.value(xr[n, i, j]) > 0.5 for i, j in Ar)]

                # Write robust results for this theta
                output_file.write(f"\nRobust Model with theta = {theta}\n")
                output_file.write(f"Robust Model Status: {status_r}\n")
                output_file.write(f"Robust Model Objective value: {total_cost_r}\n")
                output_file.write(f"Robust Model Run Time: {robust_time} seconds\n")
                #output_file.write(f"Gap: {gap_r}\n")
                output_file.write(f"Number of EVs used: {len(used_vehicles_r)}\n")  # 保存使用的 EV 数量

                for n in N2:
                    route_r = []
                    route2_r = []
                    node_r = 'O1'
                    while True:
                        for (i, j) in Ar:
                            #print(pulp.value(xr[n,i,j]))
                            if i == node_r and pulp.value(xr[n, i, j]) > 0.5:
                                if j in Rr:
                                    charged_energy_r = pulp.value(charging_schedule_r[(n, j)])
                                    if charged_energy_r > 0:
                                        route_r.append(j)
                                        route2_r.append(node_meaning[j])
                                else:
                                    route_r.append(j)
                                    route2_r.append(node_meaning[j])
                                node_r = j
                                break
                        else:
                            break

                    if route_r:
                        output_file.write(f"Vehicle {n} robust route: {'O1'} -> {' -> '.join(route2_r)}\n")
                        for node in route_r:
                            arrival_time_r = start_times_r[(n, node)]
                            remaining_battery_r = remaining_energy_r[(n, node)]
                            if node in Rr:
                                charged_energy_r = pulp.value(charging_schedule_r[(n, node)])
                                if charged_energy_r > 0:
                                    output_file.write(f"  {node_meaning[node]}: start time = {arrival_time_r}, remaining battery = {remaining_battery_r}, charged energy = {charged_energy_r}\n")
                            else:
                                output_file.write(f"  {node_meaning[node]}: start time = {arrival_time_r}, remaining battery = {remaining_battery_r}\n")


# python C:\Users\ypwan\Downloads\EVRPTW_C15\mip_D_R.py >> C:\Users\ypwan\Downloads\EVRPTW_C10\mip_D_R_results\terminal_informationrc.txt
# python C:\Users\ypwan\Downloads\EVRPTW_C15\mip_D_R.py >> C:\Users\ypwan\Downloads\EVRPTW_C30_10\MIP_D_R_results\terminal_information.txt