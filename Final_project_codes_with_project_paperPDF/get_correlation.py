def get_correlation(pred, target):
    from scipy.stats import spearmanr, kendalltau
    import numpy as np
    
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    rho, _ = spearmanr(pred, target)
    tau, _ = kendalltau(pred, target)
    return rmse, rho, tau