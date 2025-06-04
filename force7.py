#!/usr/bin/env python3
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1. Symbolic setup
xg, yg, xh, yh, xi, yi, xc, yc, xj, yj, xk, yk, xb, yb, xa, ya = sp.symbols(
    'xg yg xh yh xi yi xc yc xj yj xk yk xb yb xa ya', real=True)
ext = sp.symbols('ext', real=True)
L_AB, L_BK, L_AK, L_BC, L_KJ, L_JI, L_CI, L_HI, L_CD, L_HG, L_DG, L_GE, L_EF = sp.symbols(
    'L_AB L_BK L_AK L_BC L_KJ L_JI L_CI L_HI L_CD L_HG L_DG L_GE L_EF', positive=True)
θ_JIH, θ_BCI, θ_HGE, θ_CDG = sp.symbols('θ_JIH θ_BCI θ_HGE θ_CDG', real=True)

# 2. Fixed points
D = sp.Point(ext, 0)
E = sp.Point(0, L_EF)
F = sp.Point(0, 0)

# 3. Symbolic joint points
A = sp.Point(xa, ya)
B = sp.Point(xb, yb)
K = sp.Point(xk, yk)
J = sp.Point(xj, yj)
I = sp.Point(xi, yi)
C = sp.Point(xc, yc)
H = sp.Point(xh, yh)
G = sp.Point(xg, yg)

# 4. Link-length constraints
eqs = []
links = [
    (A, B, L_AB), (B, K, L_BK), (A, K, L_AK),
    (B, C, L_BC), (K, J, L_KJ), (J, I, L_JI),
    (C, I, L_CI), (H, I, L_HI), (C, D, L_CD),
    (H, G, L_HG), (D, G, L_DG), (G, E, L_GE)
]
for P, Q, Lp in links:
    eqs.append(P.distance(Q)**2 - Lp**2)

# 5. Angle constraints helper
def angle_eq(P, Q, R, theta):
    v1 = P - Q
    v2 = R - Q
    return (v1.dot(v2)/(sp.sqrt(v1.dot(v1))*sp.sqrt(v2.dot(v2)))) - sp.cos(theta)

# 4 angle constraints
eqs.append(angle_eq(J, I, H, θ_JIH))
eqs.append(angle_eq(B, C, I, θ_BCI))
eqs.append(angle_eq(H, G, E, θ_HGE))
eqs.append(angle_eq(C, D, G, θ_CDG))

# 6. Variable vector and Jacobian
X = sp.Matrix([xg, yg, xh, yh, xi, yi, xc, yc, xj, yj, xk, yk, xb, yb, xa, ya])
J = sp.Matrix(eqs).jacobian(X)

# 7. Lambdify for numeric evaluation
f_num = sp.lambdify(
    (X, ext, L_AB, L_BK, L_AK, L_BC, L_KJ, L_JI,
     L_CI, L_HI, L_CD, L_HG, L_DG, L_GE, L_EF,
     θ_JIH, θ_BCI, θ_HGE, θ_CDG),
    sp.Matrix(eqs), 'numpy')
J_num = sp.lambdify(
    (X, ext, L_AB, L_BK, L_AK, L_BC, L_KJ, L_JI,
     L_CI, L_HI, L_CD, L_HG, L_DG, L_GE, L_EF,
     θ_JIH, θ_BCI, θ_HGE, θ_CDG),
    J, 'numpy')

# 8. Newton–Raphson solver

def newton_raphson(f, J, x0, args, tol=1e-8, maxiter=50):
    x = x0.astype(float).copy()
    for i in range(maxiter):
        F = np.array(f(x, *args)).flatten()
        Jm = np.array(J(x, *args))
        dx = np.linalg.solve(Jm, F)
        x = x - dx
        if np.linalg.norm(dx) < tol:
            return x
    raise RuntimeError("Newton–Raphson did not converge")

# 9. Numeric solve_positions using analytic Jacobian
def solve_positions(ext_val, L_vals, θ_vals, x0):
    args = [ext_val] + [L_vals[k] for k in
            ('AB','BK','AK','BC','KJ','JI','CI','HI','CD','HG','DG','GE','EF')] + \
           [θ_vals[k] for k in ('JIH','BCI','HGE','CDG')]
    sol = newton_raphson(f_num, J_num, x0, args)
    names = ['G','H','I','C','J','K','B','A']
    pts = {n: sol[2*i:2*i+2] for i, n in enumerate(names)}
    pts['D'] = np.array([ext_val, 0.0])
    pts['E'] = np.array([0.0, L_vals['EF']])
    pts['F'] = np.array([0.0, 0.0])
    return pts

# 10. Plot and validate (same as before)
def plot_linkage(pts):
    links = [('A','B'),('B','K'),('A','K'),('B','C'),('K','J'),('J','I'),
             ('C','I'),('H','I'),('C','D'),('H','G'),('D','G'),('G','E')]
    fig, ax = plt.subplots()
    for u,v in links:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], '-o')
        ax.text(*pts[u], u)
    ax.set_aspect('equal','box'); ax.grid(True)
    return fig, ax


def validate_linkages(pts, L_vals):
    print("\nLinkage validation:")
    print(f"{'Link':<5}{'Calc':>12}{'Exp':>12}{'Err':>10}")
    links = [
       ('A','B','AB'),('B','K','BK'),('A','K','AK'),
       ('B','C','BC'),('K','J','KJ'),('J','I','JI'),
       ('C','I','CI'),('H','I','HI'),('C','D','CD'),
       ('H','G','HG'),('D','G','DG'),('G','E','GE')
    ]
    for u,v,key in links:
        dist = np.linalg.norm(pts[u]-pts[v])
        print(f"{key:<5}{dist:12.6f}{L_vals[key]:12.6f}{dist-L_vals[key]:10.6f}")

# 11. Example usage
if __name__ == '__main__':
    L = {
        'AB':25, 'BK':11, 'AK':23, 'BC':35,
        'KJ':78, 'JI':9, 'CI':15, 'HI':30,
        'CD':40, 'HG':9, 'DG':10, 'GE':50,
        'EF':20
    }
    θ = {key:np.deg2rad(val) for key,val in
         [('JIH',90), ('BCI',75), ('HGE',136.955998), ('CDG',75)]}
    x0 = np.array([
        85.76427, -7.09483,   # G
        94.08472, -3.76172,   # H
        121.96907, -15.92347, # I
        124.43758, -1.30800,  # C
        125.14311, -7.78200,  # J
        150.20776, -24.98900, # K
        156.27349, -15.69858, # B
        173.00700, -41.12248  # A
    ])
    P = solve_positions(70, L, θ, x0)
    validate_linkages(P, L)
    fig, ax = plot_linkage(P)
    plt.show()
