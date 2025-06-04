import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def solve_positions(extension, L, θ):
    # Ground pins
    F = np.array([0.0, 0.0])
    E = np.array([0.0, L['EF']])
    D = np.array([extension, 0.0])

    # Hard-coded initial guess for G,H,I,C,J,K,B,A
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

        # 12 link-length constraints
        links = [
            ('A','B','AB'), ('B','K','BK'), ('A','K','AK'),
            ('B','C','BC'), ('K','J','KJ'), ('J','I','JI'),
            ('C','I','CI'), ('H','I','HI'), ('C','D','CD'),
            ('H','G','HG'), ('D','G','DG'), ('G','E','GE')
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

        # 4 angle constraints
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
            ('B','C'), ('K','J'), ('J','I'),
            ('C','I'), ('H','I'), ('C','D'),
            ('H','G'), ('D','G'), ('G','E')
        ]
    fig, ax = plt.subplots()
    for u,v in links:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], '-o')
    if annotate:
        for name,(x,y) in pts.items(): ax.text(x, y, name)
    ax.set_aspect('equal','box')
    ax.grid(True)
    return fig, ax


def validate_linkages(pts, L):
    """Compare calculated linkage lengths with design specifications"""
    links = [
        ('A','B','AB'), ('B','K','BK'), ('A','K','AK'),
        ('B','C','BC'), ('K','J','KJ'), ('J','I','JI'),
        ('C','I','CI'), ('H','I','HI'), ('C','D','CD'),
        ('H','G','HG'), ('D','G','DG'), ('G','E','GE')
    ]
    
    print("\nLinkage validation:")
    print(f"{'Link':<5} {'Calculated':>12} {'Expected':>12} {'Error':>10}")
    for u, v, key in links:
        # Calculate distance between points
        dist = np.linalg.norm(pts[u] - pts[v])
        # Get expected length from L
        expected = L[key]
        error = dist - expected
        print(f"{key:<5} {dist:12.6f} {expected:12.6f} {error:10.6f}")


# Example usage
if __name__=='__main__':
    L = {
        'AB':25, 'BK':11, 'AK':23, 'BC':35,
        'KJ':78, 'JI':9, 'CI':15, 'HI':30,
        'CD':40, 'HG':9, 'DG':10, 'GE':50,
        'EF':20
    }
    θ = {key:np.deg2rad(val) for key,val in
         [('JIH',90), ('BCI',75), ('HGE',136.955998), ('CDG',75)]}
    P = solve_positions(67.46580, L, θ)
    print(P)
    validate_linkages(P, L)
    fig,_ = plot_linkage(P)
    plt.show()