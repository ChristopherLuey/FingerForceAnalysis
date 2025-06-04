import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def solve_positions(extension, L, θ):
    # Ground pins
    F = np.array([0.0, 0.0])
    E = np.array([0.0, L['EF']])
    D = np.array([extension, 0.0])

    # Hard-coded initial guess
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

    def angle_poly(u, v, θ0):
        d = np.dot(u, v)
        return d*d - (np.dot(u,u)*np.dot(v,v)*(np.cos(θ0)**2))

    def eqs(x):
        xg,yg, xh,yh, xi,yi, xc,yc, xj,yj, xk,yk, xb,yb, xa,ya = x
        eq = []

        # Link-length constraints (FG removed)
        links = [
            ('A','B','AB'), ('B','K','BK'), ('A','K','AK'),
            ('K','J','JK'), ('J','I','IJ'), ('B','C','BC'),
            ('C','I','CI'), ('C','G','GC'), ('G','H','GH'),
            ('H','I','HI'), ('G','D','DG'), ('G','E','GE')
        ]
        pts = {
            'A':(xa,ya), 'B':(xb,yb), 'C':(xc,yc),
            'D':(D[0],D[1]), 'E':(E[0],E[1]), 'F':(F[0],F[1]),
            'G':(xg,yg), 'H':(xh,yh), 'I':(xi,yi),
            'J':(xj,yj), 'K':(xk,yk)
        }
        for u,v,key in links:
            ux,uy = pts[u]
            vx,vy = pts[v]
            eq.append((ux-vx)**2 + (uy-vy)**2 - L[key]**2)

        # Angle constraints
        eq.append(angle_poly([xj-xi, yj-yi], [xh-xi, yh-yi], θ['JIH']))
        eq.append(angle_poly([xb-xc, yb-yc], [xi-xc, yi-yc], θ['BCI']))
        eq.append(angle_poly([xh-xg, yh-yg], [E[0]-xg, E[1]-yg], θ['HGE']))
        eq.append(angle_poly([xc-D[0], yc-D[1]], [xg-D[0], yg-D[1]], θ['CDG']))

        return eq

    sol, info, ier, msg = fsolve(eqs, x0, full_output=True)
    # if ier != 1:
    #     raise RuntimeError(f"fsolve failed to converge: {msg}")

    names = ['G','H','I','C','J','K','B','A']
    pts = {n: sol[2*i:2*i+2] for i, n in enumerate(names)}
    pts.update({'D':D, 'E':E, 'F':F})
    return pts


def plot_linkage(pts, links=None, annotate=True):
    if links is None:
        links = [
            ('A','B'), ('B','K'), ('A','K'),
            ('K','J'), ('J','I'), ('B','C'),
            ('C','I'), ('C','G'), ('G','H'),
            ('H','I'), ('G','D'), ('G','E'), ('E','F'), ('F','D'), ('D','C')
        ]
    fig, ax = plt.subplots()
    for u,v in links:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], '-o')
    if annotate:
        for name,(x,y) in pts.items(): ax.text(x, y, name)
    ax.set_aspect('equal','box')
    ax.grid(True)
    return fig, ax


# Example usage
if __name__=='__main__':
    L = {
        'EF':18.80418,'DG':10,'FG':80,'CI':15,'HI':30,
        'GH':9,'GC':40,'BK':11,'KI':30.5,'IJ':9,
        'JK':78,'AB':25,'AK':23,'BC':35,'GE':50.0
    }
    θ = {key:np.deg2rad(val) for key,val in
         [('JIH',90), ('BCI',75), ('HGE',136.955998), ('CDG',75)]}
    extension = 67.46580
    extension_max = 100
    P = solve_positions(extension_max, L, θ)
    fig,_ = plot_linkage(P)
    plt.show()
