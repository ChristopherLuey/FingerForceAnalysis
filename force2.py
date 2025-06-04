import numpy as np

def deg2rad(d): return d*np.pi/180
def rad2deg(r): return r*180/np.pi

def circle_intersect(p0, r0, p1, r1):
    # two‐circle intersection, clamped to avoid numeric errors
    d = np.linalg.norm(p1-p0)
    d = max(abs(r0-r1), min(d, r0+r1))
    a = (r0*r0 - r1*r1 + d*d) / (2*d)
    h = np.sqrt(max(r0*r0 - a*a, 0.0))
    p2 = p0 + a*(p1-p0)/d
    offset = np.array([ (p1[1]-p0[1])*h/d, -(p1[0]-p0[0])*h/d ])
    return np.vstack([p2+ offset, p2- offset])

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

def angle_between(u,v):
    return rad2deg(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))

def hinge_angles(pos):
    A,B,K,C,I,J,H,G,D,F = (pos[k] for k in ('A','B','K','C','I','J','H','G','D','F'))
    return {
      'hinge@B (ABK↔BCI)': angle_between(A-B, C-B),
      'hinge@K (ABK↔KJ)' : angle_between(A-pos['K'], J-pos['K']),
      'hinge@I (BCI↔JIH)': angle_between(C-I, H-I),
      'hinge@H (JIH↔HGF)': angle_between(I-H, G-H),
      'hinge@D (HGF↔DF)' : angle_between(G-D, F-D),
    }

def reaction_at_A(pos, Fmag):
    # downward force at A = negative vertical component of Fmag@D→F
    D,F = pos['D'], pos['F']
    u   = (F-D)/np.linalg.norm(F-D)
    return -Fmag * u[1]

if __name__=='__main__':
    lengths = {
      'AB':25,'BC':35,'CD':40,'DE':78,'EF':20,'FG':80,'GH':9,
      'HI':30,'IJ':9,'JK':30,'BK':11,'CI':15,'DG':15,'AK':23
    }
    angles = {
      'alpha1': 66.762,  # ABK @ B
      'alpha2': 87.168,  # ABK @ K
      'theta1': 75.0,    # CDG
      'theta2':150.0,    # HGF
      'theta3': 75.0,    # BCI
      'theta4': 90.0,    # JIH
      'GFE':   90.0,     # angle at F between GF & EF
    }
    F_D = 100.0  # N

    pos = forward_kin(lengths, angles)

    print("Hinge angles:")
    for name, val in hinge_angles(pos).items():
        print(f"  {name:25s} = {val:6.2f}°")

    print(f"\nDownward reaction at A = {reaction_at_A(pos, F_D):6.2f} N")

        # reaction_at_A(pos, F) is linear in F, so get the per-Newton gain:
    gain = reaction_at_A(pos, 1.0)

    # solve  gain * F_D = 50  ⇒  F_D = 50 / gain
    F_D_required = 50.0 / gain

    print(f"Required F_D = {F_D_required:.2f} N")
    print(f"Check: reaction_at_A = {reaction_at_A(pos, F_D_required):.2f} N")
