import numpy as np
from scipy.optimize import minimize

def mve(A, b):
    m, p = A.shape
    margin = 1e-6  # Small positive margin
    
    def objective(x_flat):
        x = x_flat.reshape(p, p+1)
        B = x[:, :p]
        try:
            val = -np.log(np.linalg.det(B))  # maximizing log(det(B^-1)) is same as minimizing -log(det(B))
        except np.linalg.LinAlgError:
            return np.inf
        return val
    
    def constraint(x_flat):
        x = x_flat.reshape(p, p+1)
        B = x[:, :p]
        d = x[:, p]
        vals = []
        for i in range(m):
            a = A[i]
            bi = b[i]
            val = a @ d + np.linalg.norm(B.T @ a) - (bi - margin)
            vals.append(val)
        return -np.array(vals)  # must be <= 0

    x0 = np.hstack((np.eye(p), np.zeros((p, 1)))).flatten()
    bounds = [(None, None)] * len(x0)
    cons = {'type': 'ineq', 'fun': constraint}

    result = minimize(
        objective, x0, constraints=cons, bounds=bounds,
        method='SLSQP', options={'ftol': 1e-9, 'maxiter': 2000, 'disp': True}
    )
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    x_opt = result.x.reshape(p, p+1)
    B = x_opt[:, :p]
    d = x_opt[:, p]
    return B, d
