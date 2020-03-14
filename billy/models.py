import numpy as np

def sinusoid_model(params, t):
    A = params[0]
    ω = params[1]
    φ_0 = params[2]
    return A * np.sin(ω*t + φ_0)

