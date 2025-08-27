# %%
# Truth models to generate signal for calibration data 
# in contaminant transport results.
import h5py
from generalizedADE import *

# %%
class generalizedADETruthModel(generalizedADE):
    def __init__(self, scenario=10, t_spatial=1.5, x_qoi=2.5):
        super().__init__(t_spatial, x_qoi)

        self.u = 1.05
        self.IC_mode = 0.9
        self.nu_p = 0.0095

        self.eigenvalues[1:21] = self.import_DNS_eigenvalue_means(scenario)

    def import_DNS_eigenvalue_means(self, scenario):
        h5f = h5py.File('all_eigenvalue_endpoints.h5', 'r')
        eigenvalues = np.zeros(20, dtype='complex64')
        for i in range(eigenvalues.size):
            eigenvalues[i] = np.mean(h5f[f'scenario_{scenario}/lambda_{i+1}'])
        return eigenvalues    
    
class FRADETruthModel(FRADE):
    def __init__(self, nu_m = 0.15, alpha=1.7, t_spatial=1.5, x_qoi=2.5):
        super().__init__(t_spatial, x_qoi)

        self.u = 1.05
        self.IC_mode = 0.9
        self.nu_p = 0.0095

        self.nu_m = nu_m
        self.alpha = alpha
        FRADE_params = np.array([self.nu_m, self.alpha])
        self.set_eigenvalue_params(FRADE_params)
