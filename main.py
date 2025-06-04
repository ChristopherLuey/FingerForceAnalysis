import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import copy  # Add import for deep copying
import os  # Add import for directory operations
extension_min = -68.37835
extension_max = extension_min + 20

spring_rate_N_per_mm = 1.79
spring_free_length = 76.2
spring_fully_compressed_length = 28.48

# Linkage lengths as symbolic variables
L = {
    "GE": sp.Symbol('L_GE'),
    "GD": sp.Symbol('L_GD'),
    "DF": sp.Symbol('L_DF'),
    "FE": sp.Symbol('L_FE'),
    "IC": sp.Symbol('L_IC'),
    "IH": sp.Symbol('L_IH'),
    "GH": sp.Symbol('L_GH'),
    "CG": sp.Symbol('L_CG'),
    "KB": sp.Symbol('L_KB'),
    "IB": sp.Symbol('L_IB'),
    "IJ": sp.Symbol('L_IJ'),
    "KJ": sp.Symbol('L_KJ'),
    "AB": sp.Symbol('L_AB'),
    "AK": sp.Symbol('L_AK')
}

# Angle variables (θ) for each linkage
global A
A = {
    key: sp.Symbol(f'θ_{key}') 
    for key in L
}

global A_dot
A_dot = {
    key: sp.Symbol(f'θ_dot_{key}') 
    for key in L
}

L_dot = {
    key: sp.Symbol(f'L_dot_{key}') 
    for key in L
}

# Substitution values for lengths and angles
values = {
    # Original length values from your data
    L["GE"]: 76.98564,
    L["GD"]: 10,
    L["DF"]: 68.37835,
    L["FE"]: 17.37500,
    L["IC"]: 15.0,
    L["IH"]: 30.0,
    L["GH"]: 9.0,
    L["CG"]: 38.63864,
    L["KB"]: 11,
    L["IB"]: 34.32550,
    L["IJ"]: 9.0,
    L["KJ"]: 30.5,
    L["AB"]: 25,
    L["AK"]: 23,

    A["DF"]: 0,
    A["FE"]: np.pi/2,
    "A_HGE": np.deg2rad(136.955998),
    "A_CGD": np.deg2rad(90.523104),
    "A_BIC": np.deg2rad(80.032587),
    "A_JIH": np.deg2rad(90),

    L_dot["FE"]: 0,
    L_dot["GE"]: 0,
    L_dot["GD"]: 0,
    A_dot["DF"]: 0,
    A_dot["FE"]: 0,
}

stored_solutions = []

# DF is the extension of the spring
# DF = 0 when the spring is at rest
# DF = 20 when the spring is fully extended

def calculate_points(values):    
    global A
    # Define the origin point (point F)
    F = np.array([0, 0])
    
    # Calculate positions of all points based on angles and lengths from the horizontal
    E = F + np.array([values[L["FE"]] * np.cos(values[A["FE"]]), 
                      values[L["FE"]] * np.sin(values[A["FE"]])])
    
    D = F + np.array([values[L["DF"]] * np.cos(values[A["DF"]] + np.pi), 
                      values[L["DF"]] * np.sin(values[A["DF"]] + np.pi)])
    
    G = D + np.array([values[L["GD"]] * np.cos(float(values[A["GD"]]) + np.pi), 
                      values[L["GD"]] * np.sin(float(values[A["GD"]]) + np.pi)])
    
    C = G + np.array([values[L["CG"]] * np.cos(float(values[A["CG"]])), 
                      values[L["CG"]] * np.sin(float(values[A["CG"]]))])
    
    H = G + np.array([values[L["GH"]] * np.cos(float(values[A["GH"]])), 
                      values[L["GH"]] * np.sin(float(values[A["GH"]]))])
    
    I = H + np.array([values[L["IH"]] * np.cos(float(values[A["IH"]]) + np.pi), 
                      values[L["IH"]] * np.sin(float(values[A["IH"]]) + np.pi)])
    
    B = I + np.array([values[L["IB"]] * np.cos(float(values[A["IB"]])), 
                      values[L["IB"]] * np.sin(float(values[A["IB"]]))])
    
    J = I + np.array([values[L["IJ"]] * np.cos(float(values[A["IJ"]])), 
                      values[L["IJ"]] * np.sin(float(values[A["IJ"]]))])
    
    K = J + np.array([values[L["KJ"]] * np.cos(float(values[A["KJ"]]) + np.pi), 
                      values[L["KJ"]] * np.sin(float(values[A["KJ"]]) + np.pi)])
    
    A_point = K + np.array([values[L["AK"]] * np.cos(float(values[A["AK"]]) + np.pi), 
                           values[L["AK"]] * np.sin(float(values[A["AK"]]) + np.pi)])
    
    values["A_CD"] = np.arctan2(D[1] - C[1], D[0] - C[0])
    values["A_BC"] = np.arctan2(C[1] - B[1], C[0] - B[0])
    
    # Return all points as a dictionary
    return {'A': A_point, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 
            'G': G, 'H': H, 'I': I, 'J': J, 'K': K}

def plot_mechanism(values, fig=None, ax=None, show=True, color=None):
    
    # Initialize the plot if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate all points
    points = calculate_points(values)
    
    # Unpack all points for convenience
    A, B, C, D, E, F, G, H, I, J, K = [points[key] for key in 'ABCDEFGHIJK']
    
    # Use specified color or default to regular colors
    fe_color = color or 'b'
    df_color = color or 'g'
    gd_color = color or 'r'
    ge_color = color or 'orange'
    
    # # Plot the links as lines
    ax.plot([F[0], E[0]], [F[1], E[1]], 'brown', linewidth=3)  # Link FE
    ax.plot([F[0], D[0]], [F[1], D[1]], 'brown', linewidth=3)  # Link DF
    ax.plot([D[0], G[0]], [D[1], G[1]], 'red', linewidth=3)  # Link GD
    ax.plot([G[0], E[0]], [G[1], E[1]], 'orange', linewidth=3)  # Link GE
    ax.plot([G[0], C[0]], [G[1], C[1]], 'k--', linewidth=1)  # Link CG
    ax.plot([G[0], H[0]], [G[1], H[1]], 'orange', linewidth=3)  # Link GH
    ax.plot([H[0], I[0]], [H[1], I[1]], 'yellow', linewidth=3)  # Link IH
    ax.plot([C[0], I[0]], [C[1], I[1]], 'purple', linewidth=3)  # Link CI


    ax.plot([I[0], B[0]], [I[1], B[1]], 'k--', linewidth=1)  # Link IB
    ax.plot([I[0], J[0]], [I[1], J[1]], 'yellow', linewidth=3)  # Link IJ
    ax.plot([J[0], K[0]], [J[1], K[1]], 'blue', linewidth=3)  # Link KJ
    ax.plot([B[0], K[0]], [B[1], K[1]], 'darkgreen', linewidth=3)  # Link KB
    ax.plot([B[0], C[0]], [B[1], C[1]], 'purple', linewidth=3)  # Link BC
    ax.plot([C[0], D[0]], [C[1], D[1]], 'red', linewidth=3)  # Link CD
    
    # # Add links to point A
    ax.plot([B[0], A[0]], [B[1], A[1]], 'darkgreen', linewidth=3)  # Link AB
    ax.plot([K[0], A[0]], [K[1], A[1]], 'darkgreen', linewidth=3)  # Link AK
    
    # Plot the points if no color is specified (to avoid cluttering with multiple positions)
    if color is None:
        for name, point in points.items():
            ax.plot(point[0], point[1], 'ko', markersize=8)
            ax.text(point[0]+0.05, point[1]+0.05, name, fontsize=12)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Mechanism Plot with F at Origin')
    
    # Set x-axis limits as requested: -200 to 5
    ax.set_xlim(-200, 5)
    
    # Calculate reasonable y-axis limits based on points
    max_y_dim = max([np.max(np.abs(point[1])) for point in points.values()]) * 1.2
    ax.set_ylim(-max_y_dim, max_y_dim)
    
    # Only show if explicitly requested
    if show:
        plt.show()
    
    # Verify closures of the loops
    kb_check = np.linalg.norm(K - B)
    ge_check = np.linalg.norm(G - E)
    ab_check = np.linalg.norm(A - B)
    ak_check = np.linalg.norm(A - K)
    # print(f"Expected KB distance: {values[L['KB']]}, Actual distance: {kb_check}")
    # print(f"Expected GE distance: {values[L['GE']]}, Actual distance: {ge_check}")
    # print(f"Expected AB distance: {values[L['AB']]}, Actual distance: {ab_check}")
    # print(f"Expected AK distance: {values[L['AK']]}, Actual distance: {ak_check}")
    
    return fig, ax

def position(DF):
    values[L["DF"]] += DF
    # Create equations based on the vector loop closure
    # Horizontal components (cosine equation)
    eq1 = L["GE"] * sp.cos(A["GE"]) - L["GD"] * sp.cos(A["GD"]) - L["DF"] * sp.cos(A["DF"]) - L["FE"] * sp.cos(A["FE"])
    
    # Vertical components (sine equation)
    eq2 = L["GE"] * sp.sin(A["GE"]) - L["GD"] * sp.sin(A["GD"]) - L["DF"] * sp.sin(A["DF"]) - L["FE"] * sp.sin(A["FE"])
    
    # Substitute known values
    eq1_subs = eq1.subs(values)
    eq2_subs = eq2.subs(values)
    
    # Create a system of equations to solve for GE and GD angles
    system = [eq1_subs, eq2_subs]
    
    # Variables to solve for
    unknowns = [A["GE"], A["GD"]]
    
    # Solve the system of equations
    solution1 = sp.solve(system, unknowns)
    # Choose the solution with the positive angle for GD
    solution = None

    # print("Stored solutions: ", stored_solutions)

    # print("Solution 1: ", solution1)

    if not stored_solutions:
        for sol in solution1:
            if sol[1] > 0:
                solution = sol
                stored_solutions.append([solution])
                break
    else:
        # Find the solution closest to the previous solution
        last_solution = stored_solutions[-1][0]  # Get the most recent solution
        closest_solution = None
        min_distance = float('inf')
        
        for sol in solution1:
            # Calculate Euclidean distance between current solution and last solution
            distance = np.sqrt(float((sol[0] - last_solution[0])**2))
            
            if distance < min_distance:
                min_distance = distance
                closest_solution = sol
        
        solution = closest_solution
        stored_solutions.append([solution])
    
    # If no solution was found, return False to indicate failure
    if solution is None:
        print(f"No valid solution found for extension {DF} — linkage 1")
        return False

    values[A["GD"]] = solution[1]
    values[A["GE"]] = solution[0]

    # Calculate GH and CG angles based on GE and GD
    values[A["GH"]] = values[A["GE"]] + values["A_HGE"]
    values[A["CG"]] = values[A["GD"]] + values["A_CGD"]
    
    # Create equations for IC and IH angles
    eq3 = L["IC"] * sp.cos(A["IC"]) - L["IH"] * sp.cos(A["IH"]) + L["GH"] * sp.cos(A["GH"]) - L["CG"] * sp.cos(A["CG"])
    eq4 = L["IC"] * sp.sin(A["IC"]) - L["IH"] * sp.sin(A["IH"]) + L["GH"] * sp.sin(A["GH"]) - L["CG"] * sp.sin(A["CG"])
    
    # Substitute known values
    eq3_subs = eq3.subs(values)
    eq4_subs = eq4.subs(values)
    
    # Create a system of equations to solve for IC and IH angles
    system2 = [eq3_subs, eq4_subs]
    
    # Variables to solve for
    unknowns2 = [A["IC"], A["IH"]]
    
    # Solve the system of equations
    solution2 = sp.solve(system2, unknowns2)
    # print("Solution 2: ", solution2)
    
    # Choose the solution based on continuity or positivity
    solution = None
    
    if len(stored_solutions) > 1:  # Not the first iteration
        # Find the solution closest to the previous solution
        last_solution = stored_solutions[-2][1]  # Get the most recent solution
        closest_solution = None
        min_distance = float('inf')
        
        for sol in solution2:
            # Calculate Euclidean distance between current solution and last solution
            distance = np.sqrt(float((sol[0] - last_solution[0])**2 + (sol[1] - last_solution[1])**2))
            
            if distance < min_distance:
                min_distance = distance
                closest_solution = sol
        
        solution = closest_solution
    else:  # First iteration
        # Choose the solution with positive IH angle
        for sol in solution2:
            if sol[1] > 0:  # Check if IH angle is positive
                solution = sol
                break
    
    # If no solution was found, use the first solution
    if solution is None:
        if not solution2:
            print(f"No valid solution for second equation system at extension {DF} — linkage 2")
            return False
        solution = solution2[0]
    
    # Apply the chosen solution
    values[A["IC"]] = solution[0]
    values[A["IH"]] = solution[1]

    # Store the solution for the next iteration
    if stored_solutions:
        stored_solutions[-1].append(solution)


    # Set BI and IJ angles based on known angles
    values[A["IB"]] = values[A["IC"]] + values["A_BIC"]
    values[A["IJ"]] = values[A["IH"]] + values["A_JIH"]
    
    # Create equations for KB and KJ angles
    eq5 = L["KB"] * sp.cos(A["KB"]) - L["IB"] * sp.cos(A["IB"]) + L["IJ"] * sp.cos(A["IJ"]) - L["KJ"] * sp.cos(A["KJ"])
    eq6 = L["KB"] * sp.sin(A["KB"]) - L["IB"] * sp.sin(A["IB"]) + L["IJ"] * sp.sin(A["IJ"]) - L["KJ"] * sp.sin(A["KJ"])
    
    # Substitute known values
    eq5_subs = eq5.subs(values)
    eq6_subs = eq6.subs(values)
    
    # Create a system of equations to solve for KB and KJ angles
    system3 = [eq5_subs, eq6_subs]
    
    # Variables to solve for
    unknowns3 = [A["KB"], A["KJ"]]
    
    # Solve the system of equations
    solution3 = sp.solve(system3, unknowns3)
    # print("Solution 3: ", solution3)
    
    # Choose the solution based on previous solutions or positive KJ angle
    solution = None
    
    if stored_solutions and len(stored_solutions) > 1:
        # Not the first iteration, choose solution closest to the last one
        last_solution = stored_solutions[-2][2]  # Get the last KB, KJ solution
        
        closest_solution = None
        min_distance = float('inf')
        
        for sol in solution3:
            # Calculate Euclidean distance between current solution and last solution
            # Normalize angles to 0-360 degrees
            sol_1_norm = sol[1] % (2 * np.pi)
            sol_0_norm = sol[0] % (2 * np.pi)
            last_1_norm = last_solution[1] % (2 * np.pi)
            last_0_norm = last_solution[0] % (2 * np.pi)
            distance = np.sqrt(float((sol_1_norm - last_1_norm) ** 2) + float((sol_0_norm - last_0_norm) ** 2))
            if distance < min_distance:
                min_distance = distance
                closest_solution = sol
        
        solution = closest_solution
    else:  # First iteration
        # Choose the solution with positive KJ angle
        for sol in solution3:
            if sol[1] > 0:  # Check if KJ angle is positive
                solution = sol
                break
    
    # If no solution was found, use the first solution
    if solution is None:
        if not solution3:
            print(f"No valid solution for third equation system at extension {DF} — linkage 3")
            return False
        solution = solution3[0]
    
    # Apply the chosen solution
    values[A["KB"]] = solution[0]
    values[A["KJ"]] = solution[1]
    
    # Store the solution for the next iteration
    if stored_solutions:
        stored_solutions[-1].append(solution)

    # Horizontal components (cosine equation)
    eq1 = L["AB"] * sp.cos(A["AB"]) - L["AK"] * sp.cos(A["AK"]) - L["KB"] * sp.cos(A["KB"])
    
    # Vertical components (sine equation)
    eq2 = L["AB"] * sp.sin(A["AB"]) - L["AK"] * sp.sin(A["AK"]) - L["KB"] * sp.sin(A["KB"])
    
    # Substitute known values
    eq1_subs = eq1.subs(values)
    eq2_subs = eq2.subs(values)
    
    # Create a system of equations to solve for AB and AK angles
    system = [eq1_subs, eq2_subs]
    
    # Variables to solve for
    unknowns = [A["AB"], A["AK"]]
    
    # Solve the system of equations
    solution1 = sp.solve(system, unknowns)
    # Choose the solution with the positive angle for AK
    solution = None

    # print("Stored solutions: ", stored_solutions)
    # print("Solution 1: ", solution1)

    if not stored_solutions:
        for sol in solution1:
            if sol[1] > 0:  # Check if AK angle is positive
                solution = sol
                stored_solutions.append([solution])
                break
    else:
        # Find the solution closest to the previous solution
        last_solution = stored_solutions[-1][0]  # Get the most recent solution
        closest_solution = None
        min_distance = float('inf')
        
        for sol in solution1:
            # Calculate Euclidean distance between current solution and last solution
            # Normalize angles to 0-360 degrees
            sol_0_norm = sol[0] % (2 * np.pi)
            sol_1_norm = sol[1] % (2 * np.pi)
            last_0_norm = last_solution[0] % (2 * np.pi)
            last_1_norm = last_solution[1] % (2 * np.pi)
            distance = np.sqrt(float((sol_0_norm - last_0_norm) ** 2) + float((sol_1_norm - last_1_norm) ** 2))
            
            if distance < min_distance:
                min_distance = distance
                closest_solution = sol
        
        solution = closest_solution
    
    # If no solution was found, return False to indicate failure
    if solution is None:
        print(f"No valid solution found for extension {DF} — linkage 1")
        return False

    values[A["AB"]] = solution[0]
    values[A["AK"]] = solution[1]

    # Store the solution for the next iteration
    if stored_solutions:
        stored_solutions[-1].append(solution)

    return True

def velocity(DF_dot):
    global values
    global A_dot
    global L_dot

    L_dot["DF"] = DF_dot

    eq1 = L["GE"] * A_dot["GE"] * sp.cos(A["GE"]) - L["GD"] * A_dot["GD"] * sp.cos(A["GD"]) - L_dot["DF"] * sp.sin(A["DF"])
    
    # Vertical components (sine equation)
    eq2 = -L["GE"] * A_dot["GE"] * sp.sin(A["GE"]) + L["GD"] * A_dot["GD"] * sp.sin(A["GD"]) - L_dot["DF"] * sp.cos(A["DF"])
    
    # Substitute known values
    eq1_subs = eq1.subs(values)
    eq2_subs = eq2.subs(values)
    
    # Create a system of equations to solve for GE and GD velocity angles
    system = [eq1_subs, eq2_subs]
    
    # Variables to solve for
    unknowns = [A_dot["GE"], A_dot["GD"]]
    
    # Solve the system of equations
    solution1 = sp.solve(system, unknowns)
    
    # If no solution was found, return False to indicate failure
    if solution1 is None:
        print(f"No valid solution found for velocity {DF_dot} — linkage 1")
        return False

    values[A_dot["GD"]] = solution1[A_dot["GD"]]
    values[A_dot["GE"]] = solution1[A_dot["GE"]]

def force(DF):
    global values
    global spring_rate_N_per_mm

    F_spring = spring_rate_N_per_mm * (spring_free_length - (DF + spring_fully_compressed_length))

    # Calculate points using the updated values
    points = calculate_points(values)
    
    # Unpack all points
    A, B, C, D, E, F, G, H, I, J, K = [points[key] for key in 'ABCDEFGHIJK']

    # Calculate spring force vector in FD direction
    FD_vector = D - F
    FD_unit = FD_vector / np.linalg.norm(FD_vector)
    F_spring_vector = F_spring * FD_unit

    print("F_spring_vector: ", F_spring_vector)
    
    # Calculate vector GD
    GD_vector = D - G
    print("GD_vector: ", GD_vector)
    
    # Calculate cross product between spring force and GD vector
    # For 2D vectors, cross product gives scalar: a×b = a_x*b_y - a_y*b_x
    cross_product_spring_GD = F_spring_vector[0] * GD_vector[1] - F_spring_vector[1] * GD_vector[0]
    print("cross_product_spring_GD: ", cross_product_spring_GD)
    
    # Calculate vector GA
    GA_vector = A - G
    print("GA_vector: ", GA_vector)
    
    # Calculate vector AK
    AK_vector = K - A
    AK_unit = AK_vector / np.linalg.norm(AK_vector)
    print("AK_unit: ", AK_unit)
    
    # Calculate perpendicular unit vector to AK (rotate 90 degrees)
    # Calculate perpendicular unit vector to AK (rotate 90 degrees counterclockwise)
    # The direction matters for the cross product calculation - this gives a consistent perpendicular direction
    AK_perp = np.array([-AK_unit[1], AK_unit[0]])
    print("AK_perp: ", AK_perp)
    
    # Calculate force at perpendicular to AK by crossing with vector GA
    cross_product_GA_AKperp = GA_vector[0] * AK_perp[1] - GA_vector[1] * AK_perp[0]
    print("cross_product_GA_AKperp: ", cross_product_GA_AKperp)
    # Store results in values dictionary for potential use elsewhere
    values["cross_product_spring_GD"] = cross_product_spring_GD
    values["cross_product_GA_AKperp"] = cross_product_GA_AKperp
    values["F_spring_magnitude"] = F_spring

    # Calculate the force at perpendicular to AK that creates zero torque at G
    # For zero torque at G: cross_product_spring_GD + F_AK_perp * cross_product_GA_AKperp = 0
    # Therefore: F_AK_perp = -cross_product_spring_GD / cross_product_GA_AKperp
    
    if abs(cross_product_GA_AKperp) > 1e-10:  # Avoid division by zero
        F_AK_perp = -cross_product_spring_GD / cross_product_GA_AKperp
        values["F_AK_perp"] = F_AK_perp
        print(f"Force perpendicular to AK for zero torque at G: {F_AK_perp:.3f} N")
    else:
        values["F_AK_perp"] = 0
        print("Warning: GA vector is parallel to AK perpendicular - cannot calculate force")

    return F_AK_perp

def animate_mechanism(start=0, end=20, step=1, fps=5):
    # Calculate number of frames
    num_frames = int((end - start) / step) + 1
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Save a deep copy of the original values
    original_values = copy.deepcopy(values)
    
    # Function to update the plot for each frame
    def update_frame(frame_num):
        ax.clear()
        
        # Calculate the extension for this frame
        extension = start + frame_num * step
        print("Extension: ", extension)
        
        try:
            # Completely reset values to their original state
            values = copy.deepcopy(original_values)
            
            # Calculate position for this extension
            success = position(extension)
            
            if success:
                # Calculate points using the updated values
                points = calculate_points(values)
                
                # Unpack all points
                A, B, C, D, E, F, G, H, I, J, K = [points[key] for key in 'ABCDEFGHIJK']
                
                # Plot links
                ax.plot([F[0], E[0]], [F[1], E[1]], 'b-', linewidth=2)  # Link FE
                ax.plot([F[0], D[0]], [F[1], D[1]], 'g-', linewidth=2)  # Link DF
                ax.plot([D[0], G[0]], [D[1], G[1]], 'r-', linewidth=2)  # Link GD
                ax.plot([G[0], E[0]], [G[1], E[1]], 'orange', linewidth=2)  # Link GE
                ax.plot([G[0], C[0]], [G[1], C[1]], 'c-', linewidth=2)  # Link CG
                ax.plot([G[0], H[0]], [G[1], H[1]], 'm-', linewidth=2)  # Link GH
                ax.plot([H[0], I[0]], [H[1], I[1]], 'y-', linewidth=2)  # Link IH
                ax.plot([I[0], B[0]], [I[1], B[1]], 'k-', linewidth=2)  # Link IB
                ax.plot([I[0], J[0]], [I[1], J[1]], 'b-', linewidth=2)  # Link IJ
                ax.plot([J[0], K[0]], [J[1], K[1]], 'g-', linewidth=2)  # Link KJ
                ax.plot([B[0], K[0]], [B[1], K[1]], 'r-', linewidth=2)  # Link KB
                # ax.plot([B[0], A[0]], [B[1], A[1]], 'darkgreen', linewidth=2)  # Link AB
                # ax.plot([K[0], A[0]], [K[1], A[1]], 'darkblue', linewidth=2)  # Link AK
                
                # Plot points
                for name, point in points.items():
                    ax.plot(point[0], point[1], 'ko', markersize=8)
                    ax.text(point[0]+0.05, point[1]+0.05, name, fontsize=12)
                
                ax.set_title(f'Mechanism Plot - Extension: {extension:.1f} mm')
            else:
                ax.text(0.5, 0.5, f"No valid solution for extension {extension:.1f}mm", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Mechanism Plot - No Solution: {extension:.1f} mm')
                
        except Exception as e:
            # If there's an error, show it on the plot
            ax.text(0.5, 0.5, f"Error at extension {extension:.1f}mm: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Mechanism Plot - Error at Extension: {extension:.1f} mm')
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-200, 5)
        ax.set_ylim(-150, 150)
        
        return []
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=True, interval=1000/fps)
    
    # Save the animation as a GIF
    ani.save('mechanism_animation.gif', writer='pillow', fps=fps)
    
    # Reset values to original state after animation is complete
    values.update(original_values)
    
    print(f"Animation saved as 'mechanism_animation.gif' with {num_frames} frames at {fps} fps")
    
    return ani

def plot_all_positions():
    global values
     # Create a single figure for all positions
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define a colormap for the range of positions
    colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    # Save a deep copy of the original values
    original_values = copy.deepcopy(values)
    
    # Plot all positions in a single figure
    for i, extension in enumerate(np.arange(0, 20, 0.1)):
        # try:
        # Completely reset values to their original state
        values = copy.deepcopy(original_values)
        
        # Calculate position for this extension
        success = position(extension)
        
        # Only plot if the position calculation was successful
        if success:
            # Plot with the corresponding color, don't show yet
            color_str = f'C{i%10}'  # Using matplotlib's default color cycle
            fig, ax = plot_mechanism(values, fig, ax, show=False, color=color_str)
        else:
            print(f"Skipping position for extension {extension} due to invalid solution")
                
        # except Exception as e:
        #     print(f"Error processing extension {extension}: {str(e)}")
    
    # Reset values to original state after all iterations
    values = copy.deepcopy(original_values)
    
    # Add a title for the combined plot
    ax.set_title('Mechanism Positions for Valid Extensions 0-19')
    
    # Now show the complete figure with all positions
    plt.show()

def plot_gif():
    # Save a list of all joint angles for each extension
    all_joint_angles = []
    global values
    # Create a directory for gif images if it doesn't exist
    if not os.path.exists('gif'):
        os.makedirs('gif')
    
    # Save a deep copy of the original values
    original_values = copy.deepcopy(values)
    
    # Generate images for each extension
    for extension in np.arange(0, 15, 0.1):
        # Create a new figure for each position
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Completely reset values to their original state
        values = copy.deepcopy(original_values)
        
        # Calculate position for this extension
        success = position(extension)
        
        # Only plot if the position calculation was successful
        if success:
            # Save all joint angles for this extension
            joint_angles = {
                'extension': extension,
                'AB': np.rad2deg(float(values[A["AB"]])),
                'AK': np.rad2deg(float(values[A["AK"]])),
                'GD': np.rad2deg(float(values[A["GD"]])),
                'GE': np.rad2deg(float(values[A["GE"]])),
                'CG': np.rad2deg(float(values[A["CG"]])),
                'GH': np.rad2deg(float(values[A["GH"]])),
                'IH': np.rad2deg(float(values[A["IH"]])),
                'IB': np.rad2deg(float(values[A["IB"]])),
                'IJ': np.rad2deg(float(values[A["IJ"]])),
                'KJ': np.rad2deg(float(values[A["KJ"]]))
            }
            all_joint_angles.append(joint_angles)
            # Plot the mechanism for this extension
            fig, ax = plot_mechanism(values, fig, ax, show=False)
            
            # Set title and y limits
            ax.set_title(f'Mechanism Position - Extension: {extension:.1f} mm')
            ax.set_ylim(-80, 40)  # Set y limits as requested
            
            # Save the figure as an image
            filename = f'gif/extension_{extension:.1f}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"Saved {filename}")
            
            # Close the figure to free memory
            plt.close(fig)
        else:
            print(f"Skipping position for extension {extension:.1f} due to invalid solution")
    
    # Reset values to original state after all iterations
    values = copy.deepcopy(original_values)

    # Create a comprehensive plot of all joint angles vs extension
    if all_joint_angles:
        plt.figure(figsize=(15, 10))
        
        # Extract data for plotting
        extensions = [data['extension'] for data in all_joint_angles]
        
        # Plot each joint angle
        plt.subplot(2, 2, 1)
        plt.plot(extensions, [data['AB'] for data in all_joint_angles], 'b-', linewidth=2, label='AB')
        plt.plot(extensions, [data['AK'] for data in all_joint_angles], 'r-', linewidth=2, label='AK')
        plt.xlabel('Extension (mm)')
        plt.ylabel('Angle (degrees)')
        plt.title('Triangle AKB Joint Angles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(extensions, [data['GD'] for data in all_joint_angles], 'g-', linewidth=2, label='GD')
        plt.plot(extensions, [data['GE'] for data in all_joint_angles], 'orange', linewidth=2, label='GE')
        plt.plot(extensions, [data['CG'] for data in all_joint_angles], 'purple', linewidth=2, label='CG')
        plt.plot(extensions, [data['GH'] for data in all_joint_angles], 'm-', linewidth=2, label='GH')
        plt.xlabel('Extension (mm)')
        plt.ylabel('Angle (degrees)')
        plt.title('Point G Connected Angles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(extensions, [data['IH'] for data in all_joint_angles], 'y-', linewidth=2, label='IH')
        plt.plot(extensions, [data['IB'] for data in all_joint_angles], 'k-', linewidth=2, label='IB')
        plt.plot(extensions, [data['IJ'] for data in all_joint_angles], 'c-', linewidth=2, label='IJ')
        plt.xlabel('Extension (mm)')
        plt.ylabel('Angle (degrees)')
        plt.title('Point I Connected Angles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(extensions, [data['KJ'] for data in all_joint_angles], 'darkblue', linewidth=2, label='KJ')
        plt.xlabel('Extension (mm)')
        plt.ylabel('Angle (degrees)')
        plt.title('Joint KJ Angle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('joint_angles_vs_extension.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved comprehensive joint angles plot as 'joint_angles_vs_extension.png'")
    
    print(f"Generated {len(os.listdir('gif'))} images in the 'gif' folder")
    print("You can use these images to create an animated GIF using external tools")

def plot_angles_vs_extension():
    global values
    # Save original values
    original_values = copy.deepcopy(values)
    
    # Define extension range
    extensions = np.arange(0, 20, 0.1)  # Same range as plot_gif for consistency
    ab_angles = []
    bc_angles = []
    cd_angles = []
    valid_extensions = []
    
    # Calculate angles for each extension
    for extension in extensions:
        # Reset values to original state
        values = copy.deepcopy(original_values)
        
        # Calculate position for the given extension
        success = position(extension)
        
        if success:
            # Calculate points to get BC and CD angles
            points = calculate_points(values)
            
            # Calculate the angles in degrees
            ab_angle = np.rad2deg(float(values[A["AB"]]))
            bc_angle = np.rad2deg(float(values["A_BC"]))
            cd_angle = np.rad2deg(float(values["A_CD"]))
            
            # Store the results
            valid_extensions.append(extension)
            ab_angles.append(ab_angle)
            bc_angles.append(bc_angle)
            cd_angles.append(cd_angle)
        else:
            print(f"No valid solution found for {extension:.1f}mm extension")
    
    # Reset values to original state
    values = copy.deepcopy(original_values)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(valid_extensions, ab_angles, 'b-', linewidth=2, label='AB angle')
    plt.plot(valid_extensions, bc_angles, 'r-', linewidth=2, label='BC angle')
    plt.plot(valid_extensions, cd_angles, 'g-', linewidth=2, label='CD angle')
    
    plt.xlabel('Extension (mm)')
    plt.ylabel('Angle (degrees)')
    plt.title('Link Angles vs Spring Extension')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_force_vs_extension():
    global values
    # Save original values
    original_values = copy.deepcopy(values)
    
    # Define extension range
    extensions = np.arange(0, 16, 0.1)  # Same range as plot_gif for consistency
    forces = []
    
    # Calculate angles for each extension
    for extension in extensions:
        # Reset values to original state
        values = copy.deepcopy(original_values)
        
        # Calculate position for the given extension
        success = position(extension)
        
        if success:
            # Calculate points to get BC and CD angles
            forces.append(force(extension))
        else:
            print(f"No valid solution found for {extension:.1f}mm extension")
    
    # Reset values to original state
    values = copy.deepcopy(original_values)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(extensions, forces, 'b-', linewidth=2, label='Force')
    
    plt.xlabel('Extension (mm)')
    plt.ylabel('Force (N)')
    plt.title('Force Tip vs Spring Extension')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Plot angles as a function of extension
    # plot_angles_vs_extension()
    
    # Uncomment any of these if you want to run them
    # animate_mechanism(0, 20, 1, 5)
    # success = position(2.6)
    # plot_mechanism(values)
    # plot_all_positions()
    # plot_gif()
    # position(0.0)
    # velocity(0.1)
    plot_force_vs_extension()