# %%
import os, sys
from generalizedADE import *
import scipy.stats as ss
import scipy.optimize as so
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Convenience methods for defining priors

def logn_hyperparams_from_CDF_constraints(constraint1, constraint2):
    # This method will apply two CDF constraints to derive the
    # hyperparameters of a log-normally distributed random variable.
    # It is assumed that constraints are passed as tuples where the 
    # first entry is the variable value of the constraint and the 
    # second entry is the specified probability at that variable value.
    # For example, if P(X <= 1) = 0.5, the tuple would be (1,0.5).

    # Given these constraints where we denote x1 and x2 the rv values
    # and P1, P2 the assigned probabilities, 
    #
    # sigma = ( ln(x2)-ln(x1) ) / ( PhiInv(P2) - PhiInv(P1) ),
    # where PhiInv is the inverse CDF of a standard normal RV
    #
    # mu = ln(x1) + PhiInv(P1) * sigma
    
    x1, P1 = constraint1
    x2, P2 = constraint2

    sn = ss.norm()

    sigma = (np.log(x2) - np.log(x1)) / (sn.ppf(P2) - sn.ppf(P1))
    mu = np.log(x1) - sn.ppf(P1) * sigma
    return mu, sigma

def logn_hyperparams_from_mode_and_quantile(mode, constraint_tuple):
    # This method will apply a constraint on the mode of a 
    # log-normally distributed random variable, as well as one 
    # cosntraint on the CDF, to derive the hyperparameters of the 
    # random variable. The constraint should be passed as a tuple
    # where the variable value is the first entry and the probability
    # value is the second. 
    # For example, if P(X <= 1) = 0.5, the tuple would be (1,0.5).

    # Given these constraints where we denote M the mode, xc the 
    # rv value, and Pc is the probability, 
    #
    # sigma = 0.5 * [ -PhiInv(Pc) + sqrt(PhiInv^2(Pc) - 4(ln(M)-ln(xc)))],
    # where PhiInv is the inverse CDF of a standard normal RV
    #
    # mu = ln(M)+sigma^2
    
    xc, Pc = constraint_tuple

    # sigma^2 + b sigma + c = 0 where b = PhiInv(Pc), c=ln(M)-ln(xc)
    b = ss.norm().ppf(Pc)
    c = np.log(mode) - np.log(xc)

    sigma = 0.5 * (-b + np.sqrt(b**2 - 4*c))
    mu = np.log(mode) + sigma**2
    return mu, sigma

# %%
class Bayes:
    def __init__(self, lhood_sd=1e-4, lhood_seed=20241029, lhood_type='additive'):

        self.instantiate_prior()
        self.lhood_sd = lhood_sd
        self.lhood_type = lhood_type
        self.instantiate_likelihood(lhood_seed)
        self.create_results_dir()
        
    def create_results_dir(self):
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)
        self.figdir = f"{self.results_dir}/figs/"
        if not os.path.exists(self.figdir): os.makedirs(self.figdir)

    # Abstract methods to be defined by child classes
    def instantiate_prior(self):
        return
    
    def instantiate_likelihood(self, lhood_seed):
        return
    
    def log_prior(self, theta):
        return
    
    def log_lhood(self, theta):
        return
    
    def get_prior_mean(self):
        return
    
    def get_prior_bounds(self):
        return
    
    ##############################################
    # MCMC Methods
    ##############################################
    def log_post(self, theta):
        lp = self.log_lhood(theta) + self.log_prior(theta)
        if not np.isfinite(lp): 
            return -np.inf
        return lp
    
    def run_mcmc(self, N=1000, theta0=None, restart_from_previous_run=False, seed=20240919):
        if theta0 is None:
            theta0 = self.get_prior_mean()
    
        nwalkers = 2*self.n_params
    
        # Create/load an HDF5 archive of the chain for checkpointing
        chain_archive=f'{self.results_dir}/mcmc_chain.h5'
        backend = emcee.backends.HDFBackend(chain_archive)
        
        pos = None # default is None so it uses restart.
        if not restart_from_previous_run:
            np.random.seed(seed)
            pos = theta0 + 1e-3 * np.random.randn(nwalkers, theta0.size)
            backend.reset(pos.shape[0], pos.shape[1])
    
        self.sampler = emcee.EnsembleSampler(nwalkers, theta0.size, self.log_post, backend=backend)
    
        self.sampler.run_mcmc(pos, N, progress=True);
        self.chains = self.sampler.get_chain(flat=True)
    
    def load_mcmc_chain(self, filename=None):
        if filename is None:
            filename=f'{self.results_dir}/mcmc_chain.h5'
        self.sampler = emcee.backends.HDFBackend(filename) 
        self.chains = self.sampler.get_chain(flat=True)
    
    def optimize_for_mle(self, theta0=None):
        if theta0 is None:
            theta0 = self.get_prior_mean()
        
        loss = lambda theta: - self.log_lhood(theta)

        bounds = self.get_prior_bounds()
        return so.minimize(loss, theta0, bounds=bounds)
    
    def optimize_for_map(self, theta0=None):
        if theta0 is None:
            theta0 = self.get_prior_mean()
        
        loss = lambda theta: - self.log_lhood(theta) - self.log_prior(theta)

        bounds = self.get_prior_bounds()
        return so.minimize(loss, theta0, bounds=bounds)
    
    #====================================================================
    # Plotting things 
    #====================================================================
        
    def plot_mcmc_chains(self, chains=None, return_fig=False):
        if chains is None:
            chains = self.chains
        Nrows = int(np.ceil(self.n_params/3))
        fig = plt.figure(figsize=(7,1.5*Nrows))
        axs = fig.subplots(Nrows, 3, sharex=True).flatten()
        for label, samples, ax in zip(self.parameter_names, chains.T, axs):
            ax.plot(samples)
            ax.set_xlabel('MCMC Iteration')
            ax.set_ylabel(label, rotation=0, ha='right')
            ax.spines[['top','right']].set_visible(False)

        if not self.n_params == axs.size: 
            for ax in axs[self.n_params - axs.size:]:
                fig.delaxes(ax)  
        fig.tight_layout()
        if return_fig: return fig

    def plot_marginal_histograms(self, samples, vs_prior=False, return_fig=False):
        # Inputs: 
        #   samples [N x Nparams]
        Nrows = int(np.ceil(self.n_params/3))
        fig = plt.figure(figsize=(7,1.5*Nrows))
        axs = fig.subplots(Nrows,3).flatten()

        if vs_prior:
            X = self.get_prior_samples(1000)
        for i, (label, samples, ax) in enumerate(zip(self.parameter_names, samples.T, axs)):
            ax.hist(samples, histtype='step', density=True)
            if vs_prior:
                ax.hist(X[:,i], histtype='step', density=True, label='Prior')
                if i==0:
                    ax.legend(loc='best')
            ax.set_xlabel(label)
            if i % 3 == 0: 
                ax.set_ylabel('Probability\ndensity', rotation=0, ha='right')
            ax.spines[['top','right']].set_visible(False)
        if not self.n_params == axs.size: 
            for ax in axs[self.n_params - axs.size:]:
                fig.delaxes(ax)  
        fig.tight_layout()

        if return_fig: return fig

    def plot_2d_contours(self, samples, other_samples=None, vs_prior=False, label='', return_fig=False):
        # Inputs: 
        #   samples [N x Nparams]
        #   other_samples [N x Nparams]: samples to compare to
        #   vs_prior [True/False]: if True, sets other_samples to samples from the prior.

        dim = 1.25*(self.n_params-1)
        fig = plt.figure(figsize=(dim, dim))
        axs = fig.subplots(self.n_params-1, self.n_params-1)

        if vs_prior:
            other_samples = self.get_prior_samples(1000)
            label='Posterior (black) vs Prior (blue)'
        for i, (sample1, label1) in enumerate(zip(samples[:,:-1].T, self.parameter_names[:-1])):
            for j, (sample2, label2) in enumerate(zip(samples[:,1:].T, self.parameter_names[1:])):

                ax = axs[j,i]
                if i > j:
                    plt.delaxes(ax)
                if not other_samples is None:
                    ax.plot(other_samples[:,i], other_samples[:,j+1], '.', ms=1)
                ax.plot(sample1, sample2, 'k.', ms=1)
                ax.set_xlabel(label1)
                ax.set_ylabel(label2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_box_aspect(1)
                ax.spines[['top','right']].set_visible(False)

        fig.tight_layout()

        if return_fig: return fig

    def plot_spatial_pushforward(self, input_samples, 
                                 t=None, 
                                 individual_samples=False, 
                                 with_noise=True, 
                                 label='',
                                 return_fig=False):
        # Inputs:
        #   input_samples [N x Nparams]
        #   t [float]: time to compute the pushforward for (default = self.t)
        #   individual_samples [True/False]: whether to plot individual samples or the 
        #   confidence interval
        #   with_noise [True/False]: whether to sample the likelihood
        #   label: the label to apply to the figure.

        if t is None:
            t = self.t

        pushforward_samples = self.get_spatial_pushforward_samples(input_samples, t, with_noise)
        N = pushforward_samples.shape[0]

        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)

        if individual_samples:
            for j in range(N):
                ax.plot(self.x, pushforward_samples[j,:], 'C0')
        else:
            q5, q95 = np.nanquantile(pushforward_samples, q=[0.05,0.95], axis=0)
            mean = np.nanmean(pushforward_samples, axis=0)
            ax.fill_between(self.x, q5, q95, color='#b4c5d1')
            ax.plot(self.x, mean)

        c_true = self.truth_model.f_field_from_eigenvalues(t=t)        
        ax.plot(self.x, c_true, 'k')

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\langle c \rangle$,'+f'\nt={t}', 
                        rotation=0, ha='right')
        ax.set_title(label)
            
        ax.spines[['top','right']].set_visible(False)

        fig.tight_layout()

        if return_fig: return fig

    def plot_prior_spatial_pushforward(self, N=10, t=None, individual_samples=False, with_noise=True, return_fig=False):
        X = self.get_input_samples(N)
        self.plot_spatial_pushforward(X, t=t, individual_samples=individual_samples, with_noise=True, label='Prior Pushforward', return_fig=return_fig)

    def plot_qoi_pushforward(self, input_samples, other_samples=None, vs_prior=False, label='',  with_noise=True, return_fig=False):
        # Inputs:
        #   input_samples [N x Nparams]
        #   label: the label to apply to the figure.

        N = input_samples.shape[0]

        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)

        qoi_samples = self.get_qoi_pushforward_samples(input_samples, with_noise)
    
        ax.hist(qoi_samples, bins=30, density=True, histtype='step')

        true_qoi = self.truth_model.f_field_from_eigenvalues()[-1]
        ax.vlines(true_qoi, 0, ax.get_ylim()[1], 'k')

        if vs_prior:
            other_samples = self.get_prior_samples(N)
        
        other_label='Prior' if vs_prior else 'Other'
        if not other_samples is None:
            other_qois = self.get_qoi_pushforward_samples(other_samples, with_noise)
            ax.hist(other_qois, bins=30, density=True, histtype='step', label=other_label)
            ax.legend(loc='best')

        ax.set_xlabel(f'Concentration at outflow boundary,\n $t=${self.t:.2f}')
        ax.set_ylabel('Probability\ndensity', rotation=0, ha='right')
        ax.set_title(label)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()

        if return_fig: return fig
    
    def plot_prior_qoi_pushforward(self, N=10, return_fig=False):
        X = self.get_input_samples(N)
        self.plot_qoi_pushforward(X, label='Prior Pushforward', return_fig=return_fig)
    
    def plot_time_pushforward(self, input_samples, 
                              tvec=None, x=None, 
                              individual_samples=False, 
                              with_noise=True, 
                              vs_data=True, label='', return_fig=False):
        # Inputs:
        #   input_samples [N x Nparams]
        #   tvec [numpy array]: time to compute the pushforward for (default = self.calibration_tvec)
        #   x [float]: the x location to compute the time history at
        #   individual_samples [True/False]: whether to plot individual samples or the confidence interval
        #   with_noise [True/False]: whether to sample the likelihood
        #   label: the label to apply to the figure.

        if x is None:
            x = self.x_calibration
            # TODO: add a loop here if x is more than a scalar?
        if tvec is None:
            tvec = self.calibration_tvec.copy()

        N = input_samples.shape[0]
        pushforward_samples = self.get_time_pushforward_samples(input_samples,tvec,x,with_noise)

        c_true = self.truth_model.f_time_field_from_eigenvalues(tvec=tvec, x=x)

        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)
            
        if individual_samples:
            for j in range(N):
                ax.plot(tvec, pushforward_samples[j,:], 'C0')
        else:
            q1,q99 = np.nanquantile(pushforward_samples, q=[0.01,0.99], axis=0)
            mean = np.nanmean(pushforward_samples, axis=0)
            ax.fill_between(tvec, q1, q99, color='#b4c5d1')
            ax.plot(tvec, mean)

        if vs_data:        
            ax.plot(tvec, c_true, 'k')

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\langle c \rangle$,'+f'\nx={x:.2f}', 
                        rotation=0, ha='right')
        ax.set_title(label)

        ax.spines[['top','right']].set_visible(False)

        fig.tight_layout()

        if return_fig: return fig
    
    def plot_prior_time_pushforward(self, N, tvec=None, x=None, 
                                    individual_samples=False, with_noise=True, vs_data=True, return_fig=False):
        X = self.get_input_samples(N)
        self.plot_time_pushforward(X, tvec=tvec, x=x, individual_samples=individual_samples, with_noise=with_noise, vs_data=vs_data, label='Prior Pushforward', return_fig=return_fig)

    #=============================================================
    # Predictive sampling 
    #=============================================================

    def get_spatial_pushforward_samples(self, 
                                        input_samples, 
                                        t=None, 
                                        with_noise=True):
           
        if t is None:
            t = self.t

        N = input_samples.shape[0]
        pushforward_samples = np.zeros((N, self.Nx)) 
        for j in range(N):
            pushforward_samples[j,:] = self.f_field(input_samples[j,:], t=t)
        if with_noise:
            if self.lhood_type=='multiplicative':
                pushforward_samples *= np.exp(ss.multivariate_normal(np.zeros_like(self.x), 
                                                             cov=self.lhood_sd**2.).rvs(N))
            else:
                pushforward_samples += ss.multivariate_normal(np.zeros_like(self.x), 
                                                             cov=self.lhood_sd**2.).rvs(N)
        return pushforward_samples
    
    def get_time_pushforward_samples(self, 
                                     input_samples, 
                                     tvec=None, x=None, 
                                     with_noise=True):
        
        if x is None:
            x = self.x_calibration 
            self.check_x_location(x)
        if tvec is None:
            tvec = self.calibration_tvec.copy()

        N = input_samples.shape[0]
        pushforward_samples = np.zeros((N,tvec.size))
        for j in range(N):
            pushforward_samples[j,:] = self.f_time_field(input_samples[j,:], tvec=tvec, x=x)

        if with_noise:
            noise_rv = ss.multivariate_normal(np.zeros(tvec.size), cov=self.lhood_sd**2.)
            noise_samples = noise_rv.rvs(N)
            if self.lhood_type=='multiplicative':
                pushforward_samples *= np.exp(noise_samples)
            else:
                pushforward_samples += noise_samples
        return pushforward_samples

    def get_qoi_pushforward_samples(self, 
                                    input_samples, 
                                    with_noise=True):
        N = input_samples.shape[0]
        qoi_samples = np.zeros(N)
        for j in range(N):
            qoi_samples[j] = self.f(input_samples[j,:]) 

        if with_noise:
            noise_rv = ss.norm(0,scale=self.lhood_sd)
            noise_samples = noise_rv.rvs(N)
            if self.lhood_type=='multiplicative':
                qoi_samples *= np.exp(noise_samples)
            else:
                qoi_samples += noise_samples
        return qoi_samples

    #=============================================================
    # Other postprocessing
    #=============================================================
    def compute_correlations(self, samples):
        df = pd.DataFrame(data=samples, columns=self.parameter_names)
        return df.corr()


# %%
# Hierarchical FRADE, model parameters fixed at truth values.
class onlyFRADEMFU(FRADE, Bayes):
    # Calibration happens for a short time series at a specific x location.
    # The prediction QoI is the breakthrough time at a downstream x location.

    def __init__(self, truth_model, 
                 x_calibration=1.3984375, 
                 t_calibration=0.2, 
                 Nt_calibration=10, 
                 lhood_sd=1e-4, 
                 lhood_seed=20241029,
                 lhood_type='multiplicative',
                 results_dir='results/FRADEMFU_Only'):

        self.truth_model = truth_model
        FRADE.__init__(self, truth_model.t, truth_model.x_qoi) # Instantiates numerical solver

        # Setting the model parameters to the true values for now.
        # Later perhaps set to nominals and see what happens.
        self.IC_mode = 1#self.truth_model.IC_mode
        self.u = 1#self.truth_model.u
        self.nu_p = .01#self.truth_model.nu_p
        self.theta = np.array([self.IC_mode, self.u, self.nu_p, 0, 0])

        # The x location at which the breakthrough time
        # is computed
        self.x_qoi = truth_model.x_qoi

        # The x location and time horizon of the time
        # sequence used for calibration
        self.x_calibration = x_calibration
        self.t_calibration = t_calibration
        self.Nt_calibration = Nt_calibration

        self.define_params()

        self.sample_from_posterior = False
        self.results_dir=results_dir
        Bayes.__init__(self, lhood_sd, lhood_seed, lhood_type)

#===========================================================
# Class instantiation things
#===========================================================
    def define_params(self):
        #model_params = ["IC_mode", "u", "nu_p"]
        MFU_params = [ "nu", "alpha" ]
        MFU_hyperparams = ["nu_m", "alpha_m","nu_s",  "alpha_s" ]

        self.parameter_names = MFU_hyperparams+MFU_params
        self.n_MFU_params = len(MFU_params)
        self.n_MFU_hyperparams = len(MFU_hyperparams)
        self.n_params = self.n_MFU_params + self.n_MFU_hyperparams

        # Used later for indexing parameters for model evaluation
        self.hyperparam_inds = np.arange(self.n_MFU_hyperparams)

    def instantiate_prior(self):
        
        self.nu_prior = lambda m,s: ss.lognorm(scale=np.exp(m), s=s)
        self.alpha_prior = lambda m,s: ss.truncnorm(a=(1-m)/s, b=(2-m)/s, loc=m, scale=s)

        self.MFU_param_priors = [self.nu_prior, self.alpha_prior]

        # Setting hyperpriors for mu and sigma of nu
        mu_nu, sigma_nu = logn_hyperparams_from_CDF_constraints((0.1,0.1),(0.5,0.99))
        self.nu_m_prior = ss.norm(loc=mu_nu, scale=.5*np.abs(mu_nu))
        mu, sigma = logn_hyperparams_from_mode_and_quantile(sigma_nu, (1.5*sigma_nu,0.99))
        self.nu_s_prior = ss.lognorm(scale=np.exp(mu), s=sigma)
        
        self.alpha_m_prior = ss.norm(loc=1.5, scale=0.25)
        self.alpha_s_prior = ss.expon()
        
        self.MFU_hyperpriors = [ self.nu_m_prior, 
                                 self.alpha_m_prior, 
                                 self.nu_s_prior, 
                                 self.alpha_s_prior ]
        
        self.priors=self.MFU_hyperpriors + self.MFU_param_priors

    def instantiate_likelihood(self, lhood_seed):
        # Here we reduce to the x location and time series that we want
        # to use for calibration.

        self.calibration_tvec = np.linspace(0, self.t_calibration, self.Nt_calibration+1, endpoint=True)[1:]

        if self.lhood_type=='multiplicative':
            self.calibration_data = np.log(self.truth_model.f_time_field_from_eigenvalues(tvec=self.calibration_tvec, 
                                                                                          x=self.x_calibration).flatten(order='F'))
        else:
            self.calibration_data = self.truth_model.f_time_field_from_eigenvalues(tvec=self.calibration_tvec, 
                                                                                   x=self.x_calibration).flatten(order='F')
        self.likelihood = ss.multivariate_normal(np.zeros_like(self.calibration_data), cov=self.lhood_sd**2.)

        np.random.seed(lhood_seed)
        self.true_evolution = self.calibration_data.copy()
        self.calibration_data += self.likelihood.rvs()

#====================================================================
# Sampling methods
#====================================================================

    def get_prior_samples(self, N):
        X = np.zeros((N,self.n_params))
        for i, prior in enumerate(self.priors[:-self.n_MFU_params]):     
            X[:,i] = prior.rvs(N)
        hyperparam_samples = X[:,:-self.n_MFU_params]
        X[:,-self.n_MFU_params:] = self.get_hierarchical_samples(hyperparam_samples)
        return X
    
    def get_hierarchical_samples(self, hyperparam_samples):
        # Given samples of hyperparameters, sample the parameter distributions
        X = np.zeros((hyperparam_samples.shape[0],self.n_MFU_params))
        mus = hyperparam_samples[:,:self.n_MFU_hyperparams//2]
        sigmas = hyperparam_samples[:,self.n_MFU_hyperparams//2:]
        for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
            for j, (prior, m, s) in enumerate(zip(self.MFU_param_priors, mu, sigma)):
                X[i,j] = prior(m,s).rvs()
        return X 
    
    def get_input_samples( self, N ):
        if not self.sample_from_posterior:
            return self.get_prior_samples(N)
        elif not hasattr(self, 'posterior_kde'):
            print("ERROR, you have specified to sample from the posterior, but you have not defined a posterior KDE") 
        else:
            return self.posterior_kde.resample(N).T

    def get_posterior_pushforward_samples(self, full_chain_samples):
        # Given the samples from the full joint posterior (including MFU parameters), 
        # replace the MFU parameter samples in the chain with resampled values

        temp = full_chain_samples.copy()
        hyperparam_samples = full_chain_samples[:, :-self.n_MFU_params]
        temp[:,-self.n_MFU_params:] = self.get_hierarchical_samples(hyperparam_samples)
        return temp

#====================================================================
# Model evaluation methods
#====================================================================
    def f_field(self, theta, t=None):
        self.theta[-2:] = theta[-2:]
        return super().f_field(self.theta, t)
    
    def f_time_field(self, theta, tvec=None, x=None):
        self.theta[-2:] = theta[-2:]
        return super().f_time_field(self.theta, tvec, x)
    
    def f(self, theta):
        self.theta[-2:] = theta[-2:]
        return super().f(self.theta)

#====================================================================
# Bayesian things
#====================================================================
    def log_prior(self, theta):
    
            # model parameters and hyperparameters
            log_prior = 0
            for prior, t in zip(self.priors, theta[:-self.n_MFU_params]):
                log_prior += prior.logpdf(t)
    
            # hierarchical coefficient priors
            hyperparams = theta[:-self.n_MFU_params]
            mus = hyperparams[:self.n_MFU_hyperparams//2]
            sigmas = hyperparams[self.n_MFU_hyperparams//2:]
            for prior, m, s, t in zip(self.MFU_param_priors, 
                                      mus, sigmas,
                                      theta[-self.n_MFU_params:]):
                log_prior += prior(m,s).logpdf(t)
            
            return log_prior
    
    def log_lhood(self, theta):
        try:
            if self.lhood_type=='multiplicative':
                f = np.log(self.f_time_field(theta, self.calibration_tvec, self.x_calibration).flatten(order='F'))
            else:
                f = self.f_time_field(theta, self.calibration_tvec, self.x_calibration).flatten(order='F')
            log_lhood = self.likelihood.logpdf(self.calibration_data-f)
            return log_lhood 
        except:
            return -np.infty
    
    def get_prior_mean(self):
        theta0 = np.zeros(self.n_params)
        # Getting mean of model parameter priors and hyperpriors
        for i, prior in enumerate(self.priors[:-self.n_MFU_params]):
            theta0[i] = prior.mean()
        # Given the mean of the hyperpriors, getting the mean of the coefficient
        # priors
    
        hyperparameter_means = theta0[:-self.n_MFU_params]
        mu_means = hyperparameter_means[:self.n_MFU_hyperparams//2]
        sigma_means = hyperparameter_means[self.n_MFU_hyperparams//2:]
        for i, (prior, m, s) in enumerate(zip(self.priors[-self.n_MFU_params:], 
                                              mu_means, 
                                              sigma_means)):
            theta0[-self.n_MFU_params+i] = prior(m,s).mean()
        return theta0
    
    def get_prior_bounds(self):
        # For model parameter priors and hyperpriors
        bounds = [ (prior.ppf(0.05), prior.ppf(0.95)) for prior in self.priors[:-self.n_MFU_params] ]

        # hardcoding the ones for the MFU parameters for now
        bounds += [ (0,np.inf), (1,2)]
        return bounds 
