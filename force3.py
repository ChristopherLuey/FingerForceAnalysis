import numpy as np

# --- your existing helpers ---
def deg2rad(d): return d*np.pi/180
def rad2deg(r): return r*180/np.pi

def circle_intersect(p0, r0, p1, r1):
    """ two‐circle intersection, clamped for numeric robustness """
    d = np.linalg.norm(p1-p0)
    d = max(abs(r0-r1), min(d, r0+r1))
    a = (r0*r0 - r1*r1 + d*d)/(2*d)
    h = np.sqrt(max(r0*r0 - a*a, 0.0))
    p2 = p0 + a*(p1-p0)/d
    offset = np.array([ (p1[1]-p0[1])*h/d, -(p1[0]-p0[0])*h/d ])
    return np.vstack([p2+offset, p2-offset])

# --- your original forward_kin & reaction_at_A, but made re-entrant on DF ---
def forward_kin_spring(L, A, DF_override, pos0):
    """
    Like your forward_kin, but:
      - H,G,F are locked to pos0['H'],pos0['G'],pos0['F']
      - instead of computing DF from law‐of‐cosines, we take DF=DF_override
    """
    # 1) bake in the fixed block HGF
    H, G, F = pos0['H'], pos0['G'], pos0['F']

    # 2) find D as the intersection of circles G–D (radius DG) and F–D (radius DF_override)
    ptsD = circle_intersect(G, L['DG'], F, DF_override)
    # pick the one closest to the original D
    D = ptsD[np.argmin([np.linalg.norm(pt-pos0['D']) for pt in ptsD])]

    # 3) rebuild the rest of the chain exactly as forward_kin does,
    #    but always anchoring to this new D, C, … back to A.
    #
    #  a) C via fixed ∠CDG = theta1  
    DG_dir = (G - D)/np.linalg.norm(G - D)
    θ1 = deg2rad(A['theta1'])
    # two possible C’s; pick closest to original
    candC = [
      D + L['CD']*np.array([ 
        DG_dir[0]*np.cos(+θ1) - DG_dir[1]*np.sin(+θ1),
        DG_dir[0]*np.sin(+θ1) + DG_dir[1]*np.cos(+θ1),
      ]),
      D + L['CD']*np.array([ 
        DG_dir[0]*np.cos(-θ1) - DG_dir[1]*np.sin(-θ1),
        DG_dir[0]*np.sin(-θ1) + DG_dir[1]*np.cos(-θ1),
      ]),
    ]
    C = candC[np.argmin([np.linalg.norm(c-pos0['C']) for c in candC])]

    #  b) I via ∠BCI = theta3
    DC_dir = (C - D)/np.linalg.norm(C - D)
    θ3 = deg2rad(A['theta3'])
    candI = [
      C + L['CI']*np.array([ 
        -DC_dir[0]*np.cos(+θ3) - (-DC_dir[1])*np.sin(+θ3),
        -DC_dir[0]*np.sin(+θ3) + (-DC_dir[1])*np.cos(+θ3),
      ]),
      C + L['CI']*np.array([ 
        -DC_dir[0]*np.cos(-θ3) - (-DC_dir[1])*np.sin(-θ3),
        -DC_dir[0]*np.sin(-θ3) + (-DC_dir[1])*np.cos(-θ3),
      ]),
    ]
    I = candI[np.argmin([np.linalg.norm(i-pos0['I']) for i in candI])]

    #  c) the rest you can recompute exactly as in your forward_kin,
    #     using circle_intersect for KJ, then JIH, then HGF (but HGF is fixed),
    #     then BCI, ABK back to A.  For brevity I’ll call your original:

    # -- assume you refactored your old forward_kin into a function
    #    forward_kin_from_C(L, A, fixed=(C,I,H,G,D,F)) that
    #    completes J, K, B, A --
    #    I’m going to just *patch* pos0 for everything up to D,C
    #    and then call your original forward_kin, overriding its H,G,F,D,C:

    new_pos = forward_kin(L, A)        # rebuild everything...
    # but overwrite the bits we just solved:
    new_pos.update({'D':D,'C':C,'I':I,'H':H,'G':G,'F':F})
    # now everything downstream will be consistent
    return new_pos

def forward_kin(L, A):
    # 1) A–K–B triangle (ABK)
    A0 = np.array([0.0,0.0])
    K  = A0 + np.array([L['AK'],0.0])
    φ_KB = np.pi - deg2rad(A['alpha2'])
    B  = K + L['BK'] * np.array([np.cos(φ_KB), np.sin(φ_KB)])

    # 2) BCI block
    φ_BA = np.arctan2(A0[1]-B[1], A0[0]-B[0])
    φ_BC = φ_BA + deg2rad(A['alpha1'])
    C  = B + L['BC'] * np.array([np.cos(φ_BC), np.sin(φ_BC)])
    φ_CB = φ_BC + np.pi
    φ_CI = φ_CB + deg2rad(A['theta3'])
    I  = C + L['CI'] * np.array([np.cos(φ_CI), np.sin(φ_CI)])

    # 3) K–J–I block (KJ)
    ptsJ = circle_intersect(K, L['JK'], I, L['IJ'])
    J   = ptsJ[np.argmax(ptsJ[:,1])]   # pick the “upper” solution

    # 4) JIH block
    φ_IJ = np.arctan2(J[1]-I[1], J[0]-I[0])
    φ_IH = φ_IJ + deg2rad(A['theta4'])
    H   = I + L['HI'] * np.array([np.cos(φ_IH), np.sin(φ_IH)])

    # 5) HGF block
    φ_HI = φ_IH + np.pi
    φ_HG = φ_HI + deg2rad(A['theta2'])
    G   = H + L['GH'] * np.array([np.cos(φ_HG), np.sin(φ_HG)])

    # 6) CDG block → find D
    φ_GC = np.arctan2(C[1]-G[1], C[0]-G[0])
    φ_GD = φ_GC + deg2rad(A['theta1'])
    D   = G + L['DG'] * np.array([np.cos(φ_GD), np.sin(φ_GD)])

    # 7) compute DF via law-of-cosines (∠GFE = A['GFE'])
    DF = np.sqrt(
      L['DE']**2 + L['EF']**2
      - 2*L['DE']*L['EF']*np.cos(deg2rad(A['GFE']))
    )

    # 8) intersect for F, then pick the branch where
    #    1) EF ⟂ FG and 2) DE matches
    candF = circle_intersect(D, DF, G, L['FG'])
    best=None; err_best=1e9
    for f in candF:
        # two possible EF directions ⟂ GF
        uGF = (G-f)/np.linalg.norm(G-f)
        perps = [ np.array([-uGF[1], uGF[0]]),
                  np.array([ uGF[1],-uGF[0]]) ]
        for perp in perps:
            E = f + L['EF']*perp
            err = abs(np.linalg.norm(E-D) - L['DE'])
            if err<err_best:
                err_best=err
                best=(f,E)
    F,E = best

    return dict(A=A0, B=B, K=K, C=C, I=I, J=J, H=H, G=G, D=D, E=E, F=F)


def reaction_at_A(pos, Fmag):
    """As before."""
    D, A0, F = pos['D'], pos['A'], pos['F']
    u = (F-D)/np.linalg.norm(F-D)
    return -Fmag * u[1]



if __name__=='__main__':
    # --- data as before ---
    lengths = {
      'AB':25,'BC':35,'CD':40,'DE':78,'EF':20,'FG':80,'GH':9,
      'HI':30,'IJ':9,'JK':30,'BK':11,'CI':15,'DG':15,'AK':23
    }
    angles = {
      'alpha1': 66.762, 'alpha2':87.168,
      'theta1':  75.0,  'theta2':150.0,
      'theta3':  75.0,  'theta4': 90.0,
      'GFE':     90.0
    }
    k     = 5e3      # N/m
    pos0  = forward_kin(lengths, angles)
    DF0   = np.linalg.norm(pos0['D'] - pos0['F'])

    # sweep 0→10 mm extension
    exts = np.linspace(0, 10, 51)
    out = []
    for x in exts:
        DFx = DF0 + x/1000          # mm→m if your k is N/m
        Fx  = k*(x/1000)            # spring force
        posx= forward_kin_spring(lengths, angles, DFx, pos0)
        Ay  = reaction_at_A(posx, Fx)
        out.append((x, DFx, Fx, Ay))

    # print a few
    print("ext(mm)   DF(m)    Fspring(N)    A_y(N)")
    for x,DFx,Fx,Ay in out[::10]:
        print(f"{x:6.2f}    {DFx:7.4f}     {Fx:8.2f}     {Ay:8.2f}")
