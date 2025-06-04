import numpy as np
from scipy.optimize import fsolve

def solve_positions(extension, L, θ):
    # 1) ground pins
    F = np.array([0.0, 0.0])
    E = np.array([0.0, L['EF']])
    D = np.array([extension, 0.0])

    # 2) initial guess...
    # x0 = np.zeros(16)
    #   (you should pick a non-zero spread as before)

        # 2) intelligent initial guess instead of zeros:
    #    – G halfway between D and F, lifted by DG/2
    # mid_DF = (D + F) / 2
    # xg0 = mid_DF[0]
    # yg0 = mid_DF[1] + L['DG']/2

    # #    – H directly above G by roughly HG
    # xh0 = xg0
    # yh0 = yg0 + L['GH']

    # #    – I to the right of H by HI
    # xi0 = xh0 + L['HI']
    # yi0 = yh0

    # #    – C further right of I by CI
    # xc0 = xi0 + L['CI']
    # yc0 = yi0

    # #    – J above I by IJ
    # xj0 = xi0
    # yj0 = yi0 + L['IJ']

    # #    – K left of I by KI
    # xk0 = xi0 - L['KI']
    # yk0 = yi0

    # #    – B left of J by JB
    # xb0 = xj0 - L['JB']
    # yb0 = yj0

    # #    – A left of B by AB
    # xa0 = xb0 - L['AB']
    # ya0 = yb0

    # x0 = np.array([
    #   xg0,yg0,
    #   xh0,yh0,
    #   xi0,yi0,
    #   xc0,yc0,
    #   xj0,yj0,
    #   xk0,yk0,
    #   xb0,yb0,
    #   xa0,ya0
    # ])

    x0 = np.array([
      85.76427,-7.09483,
      94.08472,-3.76172,
      121.96907,-15.92347,
      124.43758,-1.308,
      125.14311,-7.782,
      150.20776,-24.989,
      156.27349,-15.69858,
      173.007,-41.12248
    ])

    print(x0)

    def angle_poly(u, v, θ0):
        d = np.dot(u, v)
        return d*d - (np.dot(u,u)*np.dot(v,v)*(np.cos(θ0)**2))

    # 3) build exactly your 12 link‐length + 4 angle equations
    def eqs(x):
        xg,yg, xh,yh, xi,yi, xc,yc, xj,yj, xk,yk, xb,yb, xa,ya = x
        eq = []

        # — link‐lengths —
        links = [
            ('A','B','AB'),
            ('B','K','BK'),
            ('A','K','AK'),
            ('K','J','JK'),
            ('J','I','IJ'),
            ('B','C','BC'),
            ('C','I','CI'),
            ('C','G','GC'),
            ('G','H','GH'),
            ('H','I','HI'),
            ('G','D','DG'),
            ('G','E','GE'),
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

        # — angle constraints (polynomial form) —
        # ∠J–I–H
        eq.append(angle_poly(
            [xj-xi, yj-yi], [xh-xi, yh-yi], θ['JIH']
        ))
        # ∠B–C–I
        eq.append(angle_poly(
            [xb-xc, yb-yc], [xi-xc, yi-yc], θ['BCI']
        ))
        # ∠H–G–E
        eq.append(angle_poly(
            [xh-xg, yh-yg], [E[0]-xg, E[1]-yg], θ['HGE']
        ))
        # ∠C–D–G
        eq.append(angle_poly(
            [xc-D[0], yc-D[1]], [xg-D[0], yg-D[1]], θ['CDG']
        ))

        return eq

    sol, info, ier, msg = fsolve(eqs, x0, full_output=True)
    # if ier != 1:
    #     raise RuntimeError("fsolve failed: " + msg)

    names = ['A','B','C','G','H','I','J','K']
    pts = { n: sol[2*i:2*i+2] for i,n in enumerate(names) }
    pts.update({ 'D':D, 'E':E, 'F':F })
    return pts


import matplotlib.pyplot as plt

def plot_linkage(pts, links=None, annotate=True):
    if links is None:
        links = [
            ('A','B'),('B','K'),('A','K'),
            ('K','J'),('J','I'),('B','C'),
            ('C','I'),('C','G'),('G','H'),
            ('H','I'),('G','D'),('G','E')
        ]
    fig, ax = plt.subplots()
    for u,v in links:
        x_vals = [pts[u][0], pts[v][0]]
        y_vals = [pts[u][1], pts[v][1]]
        ax.plot(x_vals, y_vals, '-o', linewidth=2)
    if annotate:
        for name,(x,y) in pts.items():
            ax.text(x, y, name)
    ax.set_aspect('equal','box')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True)
    return fig, ax

def measure_angle(P, u, v, w):
    """
    Returns the angle (in degrees) at joint v formed by links v–u and v–w.
    P : dict of points
    u,v,w : labels, e.g. 'J','I','H'
    """
    pu, pv, pw = P[u], P[v], P[w]
    a = pu - pv
    b = pw - pv
    cosθ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosθ, -1,1)))

# example
if __name__=='__main__':
    L = {
      'EF':18.80418,'DG':10,'FG':80,'CI':15,'HI':30,
      'GH':9,'GC':40,'BK':11,'KI':30.5,'IJ':9,
      'JK':78,'AB':25,'AK':23,'BC':35,
      # you must supply G–E:
      'GE':50.0
    }
    θ = {
      'JIH':np.deg2rad(90),
      'BCI':np.deg2rad(75),
      'HGE':np.deg2rad(136.955998),
      'CDG':np.deg2rad(75)
    }
    P = solve_positions(67.46580, L, θ)
    print("∠J–I–H =",  measure_angle(P,'J','I','H'),  "target", np.degrees(θ['JIH']))
    print("∠B–C–I =",  measure_angle(P,'B','C','I'),  "target", np.degrees(θ['BCI']))
    print("∠H–G–E =",  measure_angle(P,'H','G','E'),  "target", np.degrees(θ['HGE']))
    print("∠C–D–G =",  measure_angle(P,'C','D','G'),  "target", np.degrees(θ['CDG']))
    for k in ['A','B','C','D','E','F','G','H','I','J','K']:
        print(f"{k}: {P[k]}")
    fig, ax = plot_linkage(P)
    plt.show()