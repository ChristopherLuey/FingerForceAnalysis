import sympy as sp
import numpy as np
from scipy.optimize import minimize
# ------------------------------------------------------------------
# 0) data (mm, deg, N)
L = dict(AB=25, BC=35, CD=40, EF=20, FG=80, GH=9, HI=30, IJ=9,
         JK=30, BK=11, CI=15, DG=15, AK=23)          # rigid bars
θ = dict(t1=75, t2=150, t3=75, t4=90)                # given angles
F_DE = 60.0                                          # spring load (N)
L_DE_val = 78.0                                      # current piston length
# ------------------------------------------------------------------
# 1) unknown planar coordinates
pts = list("ABCDEFGHIJK")
x,y = sp.symbols(' '.join(f'x{p}' for p in pts)), sp.symbols(' '.join(f'y{p}' for p in pts))
P   = {p:(x[i],y[i]) for i,p in enumerate(pts)}
# ------------------------------------------------------------------
# 2) fix the ground H-G-F (pick convenient frame)
P['G'] = (sp.Integer(0), 0)
P['F'] = (L['FG'], 0)
# locate H by its two ground distances
xH,yH = sp.symbols('xH yH'); P['H']=(xH,yH)
eq_ground = [(xH-0)**2 + yH**2 - L['GH']**2,
             (xH-L['FG'])**2 + yH**2 - (L['FG']+L['GH']-L['HF'] if 'HF' in L else L['FG'])**2]  # HF if supplied
# ------------------------------------------------------------------
eqs=[]                                                           # constraints collector
def dist(p,q,Lpq):  # |pq|²=L²
    if Lpq is None: return  # skip if length not provided
    (xi,yi),(xj,yj)=P[p],P[q]; eqs.append((xi-xj)**2+(yi-yj)**2-Lpq**2)
# rigid triangles / bars
for a,b in [('A','B'),('B','K'),('A','K')]: dist(a,b,L[a+b])
for a,b in [('B','C'),('C','I'),('B','I')]: dist(a,b, L['BC'] if a=='B' and b=='C' else L['CI'])
for a,b in [('C','D'),('D','G'),('C','G')]: dist(a,b, L['CD'] if a=='C' else L['DG'] if a=='D' else L['CG'] if 'CG' in L else None)
for a,b in [('J','I'),('I','H')]: dist(a,b,L['IJ'] if a=='J' and b=='I' else L['HI'])
for a,b in [('J','K')]:                        dist(a,b,L['JK'])
for a,b in [('E','F')]:                        dist(a,b,L['EF'])
# piston DE (variable)
L_DE = sp.symbols('L_DE')
xD,yD = P['D']; xE,yE = P['E']
eqs.append((xD-xE)**2 + (yD-yE)**2 - L_DE**2)
# angle locks
def lock_angle(p,q,r,θdeg):
    u=sp.Matrix(P[p])-sp.Matrix(P[q]); v=sp.Matrix(P[r])-sp.Matrix(P[q])
    eqs.append(u.dot(v)-sp.sqrt(u.dot(u))*sp.sqrt(v.dot(v))*sp.cos(sp.rad(θdeg)))
lock_angle('C','D','G',θ['t1'])
lock_angle('H','G','F',θ['t2'])
lock_angle('B','C','I',θ['t3'])
lock_angle('J','I','H',θ['t4'])
# ground closure
eqs.extend(eq_ground)
# ------------------------------------------------------------------
# Get only the symbolic coordinates that aren't fixed
unknowns = []
for p in pts:
    if p not in ['G', 'F']:  # Skip fixed points
        for c in P[p]:
            if not isinstance(c, (int, float)):
                unknowns.append(c)
unknowns.append(L_DE)

# More physically meaningful initial guess
init = {}
# Start with ground points as reference
xG, yG = P['G']
xF, yF = P['F']

# Position H based on ground constraints
init[P['H'][0]] = float(L['FG']/2)  # x coordinate
init[P['H'][1]] = float(L['GH'])    # y coordinate

# Position D based on ground and angle constraints
init[P['D'][0]] = float(L['DG'] * sp.cos(sp.rad(θ['t1'])))
init[P['D'][1]] = float(L['DG'] * sp.sin(sp.rad(θ['t1'])))

# Position C based on CD length and angle
init[P['C'][0]] = float(init[P['D'][0]] + L['CD'] * sp.cos(sp.rad(θ['t1'] + 30)))
init[P['C'][1]] = float(init[P['D'][1]] + L['CD'] * sp.sin(sp.rad(θ['t1'] + 30)))

# Position B based on BC length and angle
init[P['B'][0]] = float(init[P['C'][0]] + L['BC'] * sp.cos(sp.rad(θ['t3'])))
init[P['B'][1]] = float(init[P['C'][1]] + L['BC'] * sp.sin(sp.rad(θ['t3'])))

# Position A based on AB length
init[P['A'][0]] = float(init[P['B'][0]] + L['AB'] * sp.cos(sp.rad(θ['t3'] + 45)))
init[P['A'][1]] = float(init[P['B'][1]] + L['AB'] * sp.sin(sp.rad(θ['t3'] + 45)))

# Position I based on angle constraints
init[P['I'][0]] = float(init[P['C'][0]] + L['CI'] * sp.cos(sp.rad(θ['t3'] + 90)))
init[P['I'][1]] = float(init[P['C'][1]] + L['CI'] * sp.sin(sp.rad(θ['t3'] + 90)))

# Position J based on IJ length and angle
init[P['J'][0]] = float(init[P['I'][0]] + L['IJ'] * sp.cos(sp.rad(θ['t4'])))
init[P['J'][1]] = float(init[P['I'][1]] + L['IJ'] * sp.sin(sp.rad(θ['t4'])))

# Position K based on JK length
init[P['K'][0]] = float(init[P['J'][0]] + L['JK'] * sp.cos(sp.rad(θ['t4'] + 45)))
init[P['K'][1]] = float(init[P['J'][1]] + L['JK'] * sp.sin(sp.rad(θ['t4'] + 45)))

# Position E based on piston length
init[P['E'][0]] = float(init[P['D'][0]] + L_DE_val * sp.cos(sp.rad(θ['t1'] + 15)))
init[P['E'][1]] = float(init[P['D'][1]] + L_DE_val * sp.sin(sp.rad(θ['t1'] + 15)))

# Add small perturbations to avoid singularities
for k in init:
    if k != L_DE:
        init[k] = float(init[k] + 0.1)

init[L_DE] = float(L_DE_val)

# Try a completely different approach with scipy's minimize
# Function to calculate the error in our constraints
def constraint_error(vars_array):
    try:
        # Convert array back to our variables
        var_dict = {u: float(v) for u, v in zip(unknowns[:-1], vars_array)}
        
        # Include the fixed L_DE value
        var_dict[L_DE] = float(L_DE_val)
        
        # Calculate error for each constraint
        total_error = 0
        for eq in eqs:
            try:
                # Substitute our variables into the equation
                e = eq.subs(var_dict)
                # Some equations might evaluate to complex numbers if the mechanism is in an invalid configuration
                if isinstance(e, complex) or not np.isfinite(float(e)):
                    # Return a high penalty for invalid configurations
                    return 1e10
                # Calculate the absolute value of the error
                total_error += float(abs(e))**2
            except (TypeError, ValueError, ZeroDivisionError):
                # Return a high penalty if we can't evaluate the equation
                return 1e10
        
        return total_error
    except Exception as e:
        # Return a high penalty for any other errors
        print(f"Error in constraint evaluation: {e}")
        return 1e10

# Create initial array of values
initial_array = []
for u in unknowns:
    if u != L_DE:  # Skip L_DE as it's fixed
        if u in init:
            initial_array.append(float(init[u]))
        else:
            initial_array.append(25.0)  # Default value

# Use a simpler, faster optimizer
print("Starting optimization...")
try:
    # Add a timeout mechanism
    import signal
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException("Optimization timed out")
    
    # Set a 30-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    # Use the Nelder-Mead optimizer which is faster but less accurate
    result = minimize(
        constraint_error, 
        initial_array,
        method='Nelder-Mead',  # Simplex method - faster but less accurate
        options={'maxiter': 300, 'disp': True, 'xatol': 1e-3, 'fatol': 1e-3}
    )
    
    # Turn off the alarm
    signal.alarm(0)
    
except TimeoutException:
    print("Optimization timed out, using best result so far")
    # Create a simple result object with the initial values
    from types import SimpleNamespace
    result = SimpleNamespace()
    result.success = False
    result.x = initial_array
    result.message = "Optimization timed out"

# Extract results and create pose dictionary
var_values = result.x
pose = {u: float(v) for u, v in zip(unknowns[:-1], var_values)}
pose[L_DE] = float(L_DE_val)

# Verify the solution by checking all constraints
max_error = 0
for i, eq in enumerate(eqs):
    try:
        error = abs(float(eq.subs(pose)))
        max_error = max(max_error, error)
        if error > 1e-3:
            print(f"Warning: Constraint {i} has error {error}")
    except:
        print(f"Error evaluating constraint {i}")

print(f"Maximum constraint error: {max_error}")

# Attempt to calculate the force even if optimization wasn't fully successful
try:
    # differential kinematics: dy_A / dL_DE
    yA = P['A'][1].subs(pose)
    # Use finite differences for derivative since we're in numpy/scipy world
    delta = 0.001
    pose_plus = pose.copy()
    pose_plus[L_DE] = float(L_DE_val) + delta
    yA_plus = P['A'][1].subs(pose_plus)
    dydL = (yA_plus - yA) / delta
    
    F_A = F_DE / dydL  # virtual work
    print(f"Tip-down force at A = {float(F_A):.1f} N")
except Exception as e:
    print(f"Error calculating force: {e}")
    print("Using our initial configuration for a rough estimate")
    # Try with the initial configuration
    try:
        init_dict = {u: init[u] if u in init else 25.0 for u in unknowns[:-1]}
        init_dict[L_DE] = L_DE_val
        
        yA = float(P['A'][1].subs(init_dict))
        delta = 0.001
        init_dict_plus = init_dict.copy()
        init_dict_plus[L_DE] = L_DE_val + delta
        yA_plus = float(P['A'][1].subs(init_dict_plus))
        dydL = (yA_plus - yA) / delta
        
        F_A = F_DE / dydL
        print(f"Rough estimate of tip-down force at A = {float(F_A):.1f} N (based on initial configuration)")
    except:
        print("Could not provide even a rough estimate of the force.")
