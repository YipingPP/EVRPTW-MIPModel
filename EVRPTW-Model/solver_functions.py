import pandas as pd
import pulp
import re
import random
import numpy as np
import os
import time

def read_data(file_path):
    # Read data from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Read data from the file
    data = pd.read_csv(file_path, delim_whitespace=True)
    
    # Find the index of the first row where the first column value is 'Q'
    q_index = data.index[data.iloc[:, 0] == 'Q'].tolist()
    
    # If 'Q' is found, remove all rows from 'Q' onwards
    if q_index:
        data = data[:q_index[0]]
    
    return lines, data

def extract_parameters(lines):
    for line in lines:
        if re.match(r'^Q', line):
            match = re.search(r'/(\d+.\d+)/', line)
            if match:
                Q = float(match.group(1))
        elif re.match(r'^C', line):
            match = re.search(r'/(\d+.\d+)/', line)
            if match:
                C = float(match.group(1))
        elif re.match(r'^r', line):
            match = re.search(r'/(\d+.\d+)/', line)
            if match:
                r = float(match.group(1))
        elif re.match(r'^g', line):
            match = re.search(r'/(\d+.\d+)/', line)
            if match:
                g = float(match.group(1))
        elif re.match(r'^v', line):
            match = re.search(r'/(\d+.\d+)/', line)
            if match:
                v = float(match.group(1))
    return Q, C, r, g, v

def duplicate_charging_stations(data):
    # Ensure ReadyTime and DueDate are numeric
    data['ReadyTime'] = pd.to_numeric(data['ReadyTime'])
    data['DueDate'] = pd.to_numeric(data['DueDate'])
    
    # Add new columns 'electricity_price', 'peak_load' with default values
    data['electricity_price'] = 0.0
    data['peak_load'] = 0
    
    charging_stations = data[data['Type'] == 'f']
    other_data = data[data['Type'] != 'f']
    
    new_charging_stations = []
    
    for _, row in charging_stations.iterrows():
        ready_time = row['ReadyTime']
        due_date = row['DueDate']
        num_intervals = 1
        time_intervals = pd.interval_range(start=ready_time, end=due_date, periods=num_intervals, closed='left')
        
        for interval in time_intervals:
            new_row = row.copy()
            new_row['ReadyTime'] = interval.left
            new_row['DueDate'] = interval.right
            new_row['electricity_price'] = round(random.uniform(2, 2), 2)
            new_row['peak_load'] = 100
            new_charging_stations.append(new_row)
    
    new_charging_stations_df = pd.DataFrame(new_charging_stations)
    updated_data = pd.concat([other_data, new_charging_stations_df], ignore_index=True)
        
    # Duplicate rows with type 'd' and change the type to 'o'
    depot_rows = data[data['Type'] == 'd'].copy()
    depot_rows['Type'] = 'o'
    updated_data = pd.concat([updated_data, depot_rows], ignore_index=True)

    # Add 'ID' column and assign IDs based on the type
    updated_data['ID'] = ''
    trip_counter = 1
    charging_counter = 1
    depot_counter = 1
    origin_counter = 1

    for idx, row in updated_data.iterrows():
        if row['Type'] == 'c':
            updated_data.at[idx, 'ID'] = f'T{trip_counter}'
            trip_counter += 1
        elif row['Type'] == 'f':
            updated_data.at[idx, 'ID'] = f'R{charging_counter}'
            charging_counter += 1
        elif row['Type'] == 'd':
            updated_data.at[idx, 'ID'] = f'D{depot_counter}'
            depot_counter += 1
        elif row['Type'] == 'o':
            updated_data.at[idx, 'ID'] = f'O{origin_counter}'
            origin_counter += 1
    
    return updated_data

def calculate_distances(data, v):
    # Ensure x and y are numeric
    data['x'] = pd.to_numeric(data['x'])
    data['y'] = pd.to_numeric(data['y'])
    
    # Create a distance matrix and time matrix
    points = data[['x', 'y']].values
    num_points = len(points)
    
    distances = np.zeros((num_points, num_points))
    times_list = []
    
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distances[i, j] = np.linalg.norm(points[i] - points[j])
                travel_time = distances[i, j] / v  # tij = dij / v
                times_list.append((data.iloc[i]['ID'], data.iloc[j]['ID'], travel_time))
    
    # Create a DataFrame for times with columns 'From', 'To', 'Time'
    times_df = pd.DataFrame(times_list, columns=['From', 'To', 'Time'])
    
    return distances, times_df

def read_data_t(file_path):
    # Read data from the file, ignoring lines with non-numeric headers
    datat = pd.read_csv(file_path, delim_whitespace=True, comment='Q')
    datat = datat[pd.to_numeric(datat['x'], errors='coerce').notnull()]  # Keep only rows where 'x' is numeric
    return datat

def calculate_distances_t(datat, velocity):
    # Ensure x and y are numeric
    datat['x'] = pd.to_numeric(datat['x'])
    datat['y'] = pd.to_numeric(datat['y'])
    
    # Create a distance matrix and time matrix
    points = datat[['x', 'y']].values
    num_points = len(points)
    
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distances[i, j] = np.linalg.norm(points[i] - points[j])
    
    times = distances / velocity
    return times

def process_folder_t(folder_path_t):
    arc_times = {}
    file_count_t = 0
    velocity = 1
    
    for filename_t in os.listdir(folder_path_t):
        if filename_t.endswith(".txt"):
            file_path_t = os.path.join(folder_path_t, filename_t)
            data_tt = read_data_t(file_path_t)
            
            times = calculate_distances_t(data_tt, velocity)
            file_count_t += 1
            
            ids = data_tt['StringID'].tolist()
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i != j:
                        arc = (ids[i], ids[j])
                        if arc not in arc_times:
                            arc_times[arc] = []
                        arc_times[arc].append(times[i, j])
    
    # Calculate the average time of each arc
    arc_statistics = {arc: (np.mean(times), np.max(times), np.min(times)) for arc, times in arc_times.items()}
    unique_ids = set([i for arc in arc_statistics.keys() for i in arc])
    for id in unique_ids:
        arc_statistics[(id, id)] = (0.0, 0.0, 0.0)
    
    return arc_statistics

def generate_nodes_arcs(data, times_df):
    V = data['ID'].tolist()
    A = [(i, j) for i in V for j in V if i != j]

    T = data[data['Type'] == 'c']['ID'].tolist()
    R = data[data['Type'] == 'f']['ID'].tolist()

    # Remove arcs where O1 is the destination or D1 is the origin
    A = [arc for arc in A if not (arc[1].startswith('O') or arc[0].startswith('D'))]

    # Further remove arcs (i, j) where i in T, j in T and readytime_i + si + tij > duedate_j
    T = data[data['Type'] == 'c']['ID'].tolist()
    A = [
        (i, j) for (i, j) in A
        if not (i in T and j in T and
                data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
                data.loc[data['ID'] == i, 'ServiceTime'].values[0] +
                times_df[(times_df['From'] == i) & (times_df['To'] == j)]['Time'].values[0] >
                data.loc[data['ID'] == j, 'DueDate'].values[0])
    ]
    
    # Remove arcs (i, j) where i in T, j in R and starttime_i + si + tij >= duedate_j
    A = [
        (i, j) for (i, j) in A
        if not (i in T and j in R and
                data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
                data.loc[data['ID'] == i, 'ServiceTime'].values[0] +
                times_df[(times_df['From'] == i) & (times_df['To'] == j)]['Time'].values[0] >=
                data.loc[data['ID'] == j, 'DueDate'].values[0])
    ]
    
    # Remove arcs (i, j) where i in R, j in R and starttime_i + tij >= endtime_j
    A = [
        (i, j) for (i, j) in A
        if not (
            i in R and j in R and
            data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
            times_df[(times_df['From'] == i) & (times_df['To'] == j)]['Time'].values[0] >=
            data.loc[data['ID'] == j, 'DueDate'].values[0]
        )
    ]

    # Remove arcs (i, j) where i in R, j in T and starttime_i + tij > endtime_j
    A = [
        (i, j) for (i, j) in A
        if not (
            i in R and j in T and
            data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
            times_df[(times_df['From'] == i) & (times_df['To'] == j)]['Time'].values[0] >
            data.loc[data['ID'] == j, 'DueDate'].values[0]
        )
    ]

    return V, A

def generate_nodes_arcs_r(data, average_times):
    V = data['ID'].tolist()
    A = [(i, j) for i in V for j in V if i != j]

    nm = {row['ID']: row['StringID'] for _, row in data.iterrows()}

    T = data[data['Type'] == 'c']['ID'].tolist()
    R = data[data['Type'] == 'f']['ID'].tolist()

    # Remove arcs where O1 is the destination or D1 is the origin
    A = [arc for arc in A if not (arc[1].startswith('O') or arc[0].startswith('D'))]

    # Further remove arcs (i, j) where i in T, j in T and readytime_i + si + tij > duedate_j
    T = data[data['Type'] == 'c']['ID'].tolist()
    A = [
        (i, j) for (i, j) in A
        if not (i in T and j in T and
                data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
                data.loc[data['ID'] == i, 'ServiceTime'].values[0] +
                average_times[(nm[i], nm[j])] - 0.2*average_times[(nm[i], nm[j])] >
                data.loc[data['ID'] == j, 'DueDate'].values[0])
    ]
    
    # Remove arcs (i, j) where i in T, j in R and starttime_i + si + tij >= duedate_j
    A = [
        (i, j) for (i, j) in A
        if not (i in T and j in R and
                data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
                data.loc[data['ID'] == i, 'ServiceTime'].values[0] +
                average_times[(nm[i], nm[j])] - 0.2*average_times[(nm[i], nm[j])] >=
                data.loc[data['ID'] == j, 'DueDate'].values[0])
    ]
    
    # Remove arcs (i, j) where i in R, j in R and starttime_i + tij >= endtime_j
    A = [
        (i, j) for (i, j) in A
        if not (
            i in R and j in R and
            data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
            average_times[(nm[i], nm[j])] - 0.2*average_times[(nm[i], nm[j])] >=
            data.loc[data['ID'] == j, 'DueDate'].values[0]
        )
    ]

    # Remove arcs (i, j) where i in R, j in T and starttime_i + tij > endtime_j
    A = [
        (i, j) for (i, j) in A
        if not (
            i in R and j in T and
            data.loc[data['ID'] == i, 'ReadyTime'].values[0] +
            average_times[(nm[i], nm[j])] - 0.2*average_times[(nm[i], nm[j])] >
            data.loc[data['ID'] == j, 'DueDate'].values[0]
        )
    ]

    return V, A


def create_mip_model(data, average_times, N, Q, C, r, g):

    V, A = generate_nodes_arcs_r(data, average_times)

    T = data[data['Type'] == 'c']['ID'].tolist()

    R = data[data['Type'] == 'f']['ID'].tolist()

    p = 100
    M = 1e6
    

    # Define decision variables
    x = pulp.LpVariable.dicts("x", ((n, i, j) for n in N for (i, j) in A), cat='Binary')
    e = pulp.LpVariable.dicts("e", ((n, i) for n in N for i in R), lowBound=0, upBound=Q, cat='Continuous')
    z = pulp.LpVariable.dicts("z", ((n, i) for n in N for i in V), lowBound=0, cat='Continuous')
    y = pulp.LpVariable.dicts("y", ((n, i) for n in N for i in V), lowBound=0, upBound=Q, cat='Continuous')

    # Define the problem
    problem = pulp.LpProblem("EV_Charging_Routing", pulp.LpMinimize)

    nm = {row['ID']: row['StringID'] for _, row in data.iterrows()}

    # Objective function
    problem += pulp.lpSum([C * average_times[(nm[i], nm[j])] * x[n, i, j] for n in N for (i, j) in A]) \
               + pulp.lpSum([p * x[n, 'O1', j] for n in N for j in V if ('O1', j) in A]) \
               + pulp.lpSum([data.loc[data['ID'] == i, 'electricity_price'].values[0] * e[n, i] for n in N for i in R])

    # Constraints

    for t in T:
        problem += pulp.lpSum([x[n, t, j] for n in N for j in V if (t, j) in A]) == 1

    for n in N:
        problem += pulp.lpSum([x[n, 'O1', j] for j in V if ('O1', j) in A]) <= 1

    for i in R:
        problem += pulp.lpSum([e[n, i] for n in N]) <= data.loc[data['ID'] == i, 'peak_load'].values[0]

    for n in N:
        problem += z[n, 'O1'] == 0

    for i in T + R:
        for n in N:
            problem += pulp.lpSum([x[n, i, j] for j in V if (i, j) in A]) - pulp.lpSum([x[n, j, i] for j in V if (j, i) in A]) == 0

    for t in T:
        for n in N:
            problem += pulp.lpSum([data.loc[data['ID'] == t, 'ReadyTime'].values[0]]) <= z[n, t]
            problem += z[n, t] <= pulp.lpSum([data.loc[data['ID'] == t, 'DueDate'].values[0]])

    for i in R:
        for n in N:
            problem += pulp.lpSum([data.loc[data['ID'] == i, 'ReadyTime'].values[0]]) <= z[n, i]
            problem += z[n, i] <= pulp.lpSum([data.loc[data['ID'] == i, 'DueDate'].values[0]]) - e[n, i]*1 / g

    for t in T:
        for j in V:
            if (t, j) in A:
                for n in N:
                    problem += z[n, t] + pulp.lpSum([data.loc[data['ID'] == t, 'ServiceTime'].values[0]]) + average_times[(nm[t], nm[j])] - (1 - x[n, t, j]) * M <= z[n, j]

    for i in R:
        for j in V:
            if (i, j) in A:
                for n in N:
                    problem += z[n, i] + e[n, i]*1 / g + average_times[(nm[i], nm[j])] - (1 - x[n, i, j]) * M <= z[n, j]

    for n in N:
        for (i, j) in A:
            if i == 'O1':
                problem += z[n, i] + average_times[(nm[i], nm[j])] - M * (1 - x[n, i, j]) <= z[n, j]
    
    for t in T + ['O1']:
        for j in V:
            if (t, j) in A:
                for n in N:
                    #problem += y[n, t] - pulp.lpSum([data.loc[data['ID'] == t, 'BatteryConsumption'].values[0]]) - r * times_df[(times_df['From'] == t) & (times_df['To'] == j)]['Time'].values[0] + (1 - x[n, t, j]) * 1e6 >= y[n, j]
                    problem += y[n, t] - r * average_times[(nm[t], nm[j])] + (1 - x[n, t, j]) * M >= y[n, j]

    for i in R:
        for j in V:
            if (i, j) in A:
                for n in N:
                    problem += y[n, i] + e[n, i] - r * average_times[(nm[i], nm[j])] + (1 - x[n, i, j]) * M >= y[n, j]

    for i in R:
        for n in N:
            problem += y[n, i] <= Q - e[n, i]

    for n in N:
        problem += y[n, 'O1'] == Q


    return problem, R, T, V, A, x, e, z, y

def create_robust_mip_model(data, average_times, N, Q, C, r, g, thet):

    V, A = generate_nodes_arcs_r(data, average_times)

    T = data[data['Type'] == 'c']['ID'].tolist()

    R = data[data['Type'] == 'f']['ID'].tolist()

    p = 100
    M = 1e6
    

    # Define decision variables
    x = pulp.LpVariable.dicts("x", ((n, i, j) for n in N for (i, j) in A), cat='Binary')
    e = pulp.LpVariable.dicts("e", ((n, i) for n in N for i in R), lowBound=0, upBound=Q, cat='Continuous')
    z = pulp.LpVariable.dicts("z", ((n, i) for n in N for i in V), lowBound=0, cat='Continuous')
    y = pulp.LpVariable.dicts("y", ((n, i) for n in N for i in V), lowBound=0, upBound=Q, cat='Continuous')
    
    u = pulp.LpVariable.dicts("u", ((n, i, j) for n in N for (i, j) in A), lowBound=0, cat='Continuous')
    lam = pulp.LpVariable.dicts("lam", ((n) for n in N), lowBound=0, cat='Continuous')
    sig = pulp.LpVariable.dicts("sig", ((n, i, j) for n in N for (i,j) in A), lowBound=0, cat='Continuous')

    # Define the problem
    problemr = pulp.LpProblem("EV_Charging_Routing", pulp.LpMinimize)

    nm = {row['ID']: row['StringID'] for _, row in data.iterrows()}

    # Objective function
    problemr += pulp.lpSum([C * average_times[(nm[i], nm[j])] * x[n, i, j] for n in N for (i, j) in A]) \
               + pulp.lpSum([p * x[n, 'O1', j] for n in N for j in V if ('O1', j) in A]) \
               + pulp.lpSum([data.loc[data['ID'] == i, 'electricity_price'].values[0] * e[n, i] for n in N for i in R])\
               + pulp.lpSum([C * thet * u[n, i, j] for n in N for (i, j) in A])\
               + pulp.lpSum([C * sig[n, i, j] for n in N for (i, j) in A])

    # Constraints
    for n in N:
        for (i,j) in A:
            problemr += u[n, i, j] >= lam[n] - M * (1 - x[n, i, j])
            problemr += lam[n] + sig[n, i, j] >= 0.2 * average_times[(nm[i], nm[j])] * x[n, i, j]

    for t in T:
        problemr += pulp.lpSum([x[n, t, j] for n in N for j in V if (t, j) in A]) == 1

    for n in N:
        problemr += pulp.lpSum([x[n, 'O1', j] for j in V if ('O1', j) in A]) <= 1

    for i in R:
        problemr += pulp.lpSum([e[n, i] for n in N]) <= data.loc[data['ID'] == i, 'peak_load'].values[0]

    for n in N:
        problemr += z[n, 'O1'] == 0

    for i in T + R:
        for n in N:
            problemr += pulp.lpSum([x[n, i, j] for j in V if (i, j) in A]) - pulp.lpSum([x[n, j, i] for j in V if (j, i) in A]) == 0

    for t in T:
        for n in N:
            problemr += pulp.lpSum([data.loc[data['ID'] == t, 'ReadyTime'].values[0]]) <= z[n, t]
            problemr += z[n, t] <= pulp.lpSum([data.loc[data['ID'] == t, 'DueDate'].values[0]])

    for i in R:
        for n in N:
            problemr += pulp.lpSum([data.loc[data['ID'] == i, 'ReadyTime'].values[0]]) <= z[n, i]
            problemr += z[n, i] <= pulp.lpSum([data.loc[data['ID'] == i, 'DueDate'].values[0]]) - e[n, i]*1 / g

    for t in T:
        for j in V:
            if (t, j) in A:
                for n in N:
                    problemr += z[n, t] + pulp.lpSum([data.loc[data['ID'] == t, 'ServiceTime'].values[0]]) + average_times[(nm[t], nm[j])] + 0.2 * average_times[(nm[t], nm[j])] - (1 - x[n, t, j]) * M <= z[n, j]

    for i in R:
        for j in V:
            if (i, j) in A:
                for n in N:
                    problemr += z[n, i] + e[n, i]*1 / g + average_times[(nm[i], nm[j])] + 0.2 * average_times[(nm[i], nm[j])] - (1 - x[n, i, j]) * M <= z[n, j]

    for n in N:
        for (i, j) in A:
            if i == 'O1':
                problemr += z[n, i] + average_times[(nm[i], nm[j])] + 0.2 * average_times[(nm[i], nm[j])] - M * (1 - x[n, i, j]) <= z[n, j]
    
    for t in T + ['O1']:
        for j in V:
            if (t, j) in A:
                for n in N:
                    #problem += y[n, t] - pulp.lpSum([data.loc[data['ID'] == t, 'BatteryConsumption'].values[0]]) - r * times_df[(times_df['From'] == t) & (times_df['To'] == j)]['Time'].values[0] + (1 - x[n, t, j]) * 1e6 >= y[n, j]
                    problemr += y[n, t] - r * (average_times[(nm[t], nm[j])] + 0.2 * average_times[(nm[t], nm[j])]) + (1 - x[n, t, j]) * M >= y[n, j]

    for i in R:
        for j in V:
            if (i, j) in A:
                for n in N:
                    problemr += y[n, i] + e[n, i] - r * (average_times[(nm[i], nm[j])] + 0.2 * average_times[(nm[i], nm[j])]) + (1 - x[n, i, j]) * M >= y[n, j]

    for i in R:
        for n in N:
            problemr += y[n, i] <= Q - e[n, i]

    for n in N:
        problemr += y[n, 'O1'] == Q


    return problemr, R, T, V, A, x, e, z, y

def create_robust_lp_model(data, average_times, N, Q, C, r, g, thet):
    V, A = generate_nodes_arcs_r(data, average_times)
    T = data[data['Type'] == 'c']['ID'].tolist()
    R = data[data['Type'] == 'f']['ID'].tolist()

    p = 100
    M = 1e6

    # Define decision variables (x relaxed to continuous variables)
    x = pulp.LpVariable.dicts("x", ((n, i, j) for n in N for (i, j) in A), lowBound=0, upBound=1, cat='Continuous')
    e = pulp.LpVariable.dicts("e", ((n, i) for n in N for i in R), lowBound=0, upBound=Q, cat='Continuous')
    z = pulp.LpVariable.dicts("z", ((n, i) for n in N for i in V), lowBound=0, cat='Continuous')
    y = pulp.LpVariable.dicts("y", ((n, i) for n in N for i in V), lowBound=0, upBound=Q, cat='Continuous')

    u = pulp.LpVariable.dicts("u", ((n, i, j) for n in N for (i, j) in A), lowBound=0, cat='Continuous')
    lam = pulp.LpVariable.dicts("lam", ((n) for n in N), lowBound=0, cat='Continuous')
    sig = pulp.LpVariable.dicts("sig", ((n, i, j) for n in N for (i,j) in A), lowBound=0, cat='Continuous')

    # Define the problem
    problem_lp = pulp.LpProblem("EV_Charging_Routing_LP", pulp.LpMinimize)

    nm = {row['ID']: row['StringID'] for _, row in data.iterrows()}

    # Objective function
    problem_lp += pulp.lpSum([C * average_times[(nm[i], nm[j])] * x[n, i, j] for n in N for (i, j) in A]) \
                + pulp.lpSum([p * x[n, 'O1', j] for n in N for j in V if ('O1', j) in A]) \
                + pulp.lpSum([data.loc[data['ID'] == i, 'electricity_price'].values[0] * e[n, i] for n in N for i in R])\
                + pulp.lpSum([C * thet * u[n, i, j] for n in N for (i, j) in A])\
                + pulp.lpSum([C * sig[n, i, j] for n in N for (i, j) in A])



    # Constraints (same as MIP model but x is continuous)

    for n in N:
        for (i,j) in A:
            problem_lp += u[n, i, j] >= lam[n] - M * (1 - x[n, i, j])
            problem_lp += lam[n] + sig[n, i, j] >= 0.2 * average_times[(nm[i], nm[j])] * x[n, i, j]

    for t in T:
        problem_lp += pulp.lpSum([x[n, t, j] for n in N for j in V if (t, j) in A]) == 1

    for n in N:
        problem_lp += pulp.lpSum([x[n, 'O1', j] for j in V if ('O1', j) in A]) <= 1

    for i in R:
        problem_lp += pulp.lpSum([e[n, i] for n in N]) <= data.loc[data['ID'] == i, 'peak_load'].values[0]

    for n in N:
        problem_lp += z[n, 'O1'] == 0

    for i in T + R:
        for n in N:
            problem_lp += pulp.lpSum([x[n, i, j] for j in V if (i, j) in A]) - pulp.lpSum([x[n, j, i] for j in V if (j, i) in A]) == 0

    for t in T:
        for n in N:
            problem_lp += pulp.lpSum([data.loc[data['ID'] == t, 'ReadyTime'].values[0]]) <= z[n, t]
            problem_lp += z[n, t] <= pulp.lpSum([data.loc[data['ID'] == t, 'DueDate'].values[0]])

    for i in R:
        for n in N:
            problem_lp += pulp.lpSum([data.loc[data['ID'] == i, 'ReadyTime'].values[0]]) <= z[n, i]
            problem_lp += z[n, i] <= pulp.lpSum([data.loc[data['ID'] == i, 'DueDate'].values[0]]) - e[n, i]*1 / g

    for t in T:
        for j in V:
            if (t, j) in A:
                for n in N:
                    problem_lp += z[n, t] + pulp.lpSum([data.loc[data['ID'] == t, 'ServiceTime'].values[0]]) + average_times[(nm[t], nm[j])] + 0.2 * average_times[(nm[t], nm[j])] - (1 - x[n, t, j]) * M <= z[n, j]

    for i in R:
        for j in V:
            if (i, j) in A:
                for n in N:
                    problem_lp += z[n, i] + e[n, i]*1 / g + average_times[(nm[i], nm[j])] + 0.2 * average_times[(nm[i], nm[j])] - (1 - x[n, i, j]) * M <= z[n, j]

    for n in N:
        for (i, j) in A:
            if i == 'O1':
                problem_lp += z[n, i] + average_times[(nm[i], nm[j])] + 0.2 * average_times[(nm[i], nm[j])] - M * (1 - x[n, i, j]) <= z[n, j]

    for t in T + ['O1']:
        for j in V:
            if (t, j) in A:
                for n in N:
                    problem_lp += y[n, t] - r * (average_times[(nm[t], nm[j])] + 0.2 * average_times[(nm[t], nm[j])]) + (1 - x[n, t, j]) * M >= y[n, j]

    for i in R:
        for j in V:
            if (i, j) in A:
                for n in N:
                    problem_lp += y[n, i] + e[n, i] - r * (average_times[(nm[i], nm[j])] + 0.2 * average_times[(nm[i], nm[j])]) + (1 - x[n, i, j]) * M >= y[n, j]

    for i in R:
        for n in N:
            problem_lp += y[n, i] <= Q - e[n, i]

    for n in N:
        problem_lp += y[n, 'O1'] == Q

    # Add the mutual exclusion constraint only for pairs of nodes in T
    for n in N:
        for i in T:
            for j in T:
                if (i != j) and ((i, j) in A) and ((j, i) in A):  # Ensure i and j are distinct and both arcs exist
                    problem_lp += x[n, i, j] + x[n, j, i] <= 1

    return problem_lp, R, T, V, A, x, e, z, y
