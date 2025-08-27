# %%
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
import emcee

class LinearHierarchicalModel:
    """
    A linear hierarchical model which implements Bayesian inference
    on the hyperparameters of the slope. Contains an implementation
    of a nonlinear data-generating function y= c x^alpha
    """

    def __init__(self, m_mu=3., m_s=0.6, s_lam=6., 
                 n_m=1000, n_s=1000, n=300, 
                 m_lin=[0., 10.], s_lin=[0., 4.], 
                 dat_coeff=3., dat_expon=2.):
        """
        Inputs:
            m_mu (float): mu hyperprior mean
            m_s (float): mu hyperprior std dev
            s_lam (float): sigma hyperprior shape parameter
            n_m (int): number of samples for integration of posterior normalizing constant
            n_s (int): number of samples for integration of posterior normalizing constant
            n (int): linspace refinement for grid evaluation of posterior
            m_lin (list): mu lower and upper bound for grid evaluation of posterior
            s_lin (list): sigma lower and upper bound for grid evaluation of posterior
            dat_coeff (float): coefficient on nonlinear data-generating function
            dat_expon (float): power on nonlinear data-generating function
        """ 
        self.seed = 20230103

        ###########################
        # Likelihood instantiation
        ###########################
        self.s_noise = 0.1 # noise standard deviation
        self.dat_expon = dat_expon
        self.dat_coeff = dat_coeff
        self.x_dat = np.linspace(0, 3, 10)
        self.x = np.linspace(0,3,1000) # fine refinement of domain
        self.measurement_rv = sps.norm(0,self.s_noise)
        self.y = self.generate_data(theta=self.dat_coeff, x=self.x_dat)

        ###############################
        # Hyperparameter distributions
        ###############################
        # Gaussian on mu
        self.m_mu = m_mu #hyperprior mean
        self.m_s = m_s #hyperprior std
        self.p_m =  sps.norm(self.m_mu, self.m_s) #hyperprior density
        self.n_m = n_m #number of samples to draw for integration of normalizing constant
        self.n = n #linspace refinement
        self.m_linspace = np.linspace(m_lin[0], m_lin[1], self.n)

        # Exponential on sigma
        if np.size(s_lam) == 1:
            self.s_lam = s_lam #exponential parameter
            self.p_s = sps.expon(loc=0, scale=1/self.s_lam) #hyperprior density
        else: 
            a = s_lam[0]
            b = s_lam[1]
            self.p_s = sps.beta(a=a, b=b)

        self.n_s = n_s #number of samples to draw for integration of normalizing constant
        self.s_linspace = np.linspace(s_lin[0], s_lin[1], self.n)
        
        ##create meshgrid for approximating the denominator (normalizing constant)
        np.random.seed(self.seed)
        self.m_samps = self.p_m.rvs(self.n_m)
        self.s_samps = self.p_s.rvs(self.n_s)
        S, M = np.meshgrid(self.s_samps, self.m_samps)
        self.S = S.flatten()
        self.M = M.flatten()
        prior_samps = np.array([self.M, self.S])
        self.prior_samps = prior_samps.T
    
    def get_prior_samples(self, n=1000):
        samples = np.zeros((n,2))
        samples[:,0] = self.p_m.rvs(n)
        samples[:,1] = self.p_s.rvs(n) 
        return samples 

    def f_data(self, theta=None, x=1):
        if theta is None:
            theta = self.dat_coeff
        f_d = theta * x**self.dat_expon
        return f_d
    
    def generate_data(self, theta=None, x=1):
        f = self.f_data(theta=theta, x=x)
        return f + self.measurement_rv.rvs(size=f.size)

    def f(self, theta, x=1): 
        return theta * x
    
    def pf_mod(self, theta=None, x=1):
        return self.f(theta=theta, x=x)
        
    def predictive_mod(self, theta=None, x=1):
        return self.pf_mod(theta, x) + self.measurement_rv.rvs(size=np.array(x).size)
    
    def eval_full_log_pri(self, theta):
        z, mu, sigma = theta
        z_rv = sps.norm(loc=mu, scale=sigma)
        return z_rv.logpdf(z) + self.p_m.logpdf(mu) + self.p_s.logpdf(sigma)
       
    def eval_full_log_like(self, theta):
        return self.measurement_rv.logpdf(self.y - theta[0]*self.x_dat).sum()
    
    def eval_neg_log_like(self, theta):
        return -self.eval_full_log_like(theta)
    
    def eval_full_log_post(self, theta):
        lp = self.eval_full_log_pri(theta) + self.eval_full_log_like(theta)

        # Guarding against something returning NaN.
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp

    def run_mcmc(self, N=5000):
        """
        Performs MCMC sampling 
        """
        # Get an ensemble of starting points for the walkers in the emcee algorithm
        theta0 = np.array([self.p_m.mean(), self.p_m.mean(), self.p_s.mean()])
        nwalkers = 2*theta0.size
        np.random.seed(self.seed) # set seed for reproducibility
        pos = theta0 + 1e-4 * np.random.randn(nwalkers, theta0.size)

        self.sampler = emcee.EnsembleSampler(nwalkers, theta0.size, self.eval_full_log_post)
        self.sampler.run_mcmc(pos, nsteps=N, progress=True)

    def get_chains(self, burnin=0):
        samples = self.sampler.get_chain(discard=burnin, flat=True) # Gives you one MCMC chain that combines the walkers
        return samples
    
    def Sample_Bayes_post_MCMC(self, burnin=5000):
        self.run_mcmc()
        return self.get_chains(burnin)
    
    def get_MLE(self):
        """"
        This function gets the maximum likelihood estimate using python's scipy optimize 
        """
        initial_guess = [self.m_mu]
        result = minimize(self.eval_neg_log_like, initial_guess)

        MLE = result.x

        return MLE
    
    def get_pf_samples(self, samples):
        pf = np.zeros((samples.shape[0], self.x.size))
        for i, (m,s) in enumerate(samples):
            theta_samp = sps.norm.rvs(m,s)
            pf[i,:] = self.pf_mod(theta_samp, x=self.x)
        return pf

    def get_predictive_samples(self, samples):
        pf = np.zeros((samples.shape[0], self.x.size))
        for i, (m,s) in enumerate(samples):
            theta_samp = sps.norm.rvs(m,s)
            pf[i,:] = self.predictive_mod(theta_samp, x=self.x)
        return pf

    def get_prior_pf_samples(self, n=1000):
        return self.get_pf_samples(self.get_prior_samples(n))
    
    def get_prior_predictive_samples(self, n=1000):
        return self.get_predictive_samples(self.get_prior_samples(n))
    
    def get_theta_samples(self, hp_samples):
        theta_samples = np.zeros(hp_samples.shape[0])
        for i in range(theta_samples.size):
            theta_samples[i] = sps.norm.rvs(hp_samples[i,0], hp_samples[i,1])
        return theta_samples
    
    ################## Plots ######################
    
    ################## Plots ######################

    def plot_model_versus_data(self):
        plt.figure(figsize=(4,3))
        plt.plot(self.x, self.f(theta=self.dat_coeff, x=self.x), '-', color='royalblue', label='inv mod')
        plt.plot(self.x_dat, self.y, '.', color='darkorange', label='noisy data')
        plt.legend()
        plt.title('Data versus model')
        plt.show()
        plt.close()
        return
    
    # @profile
    def plot_pri_pf(self, ylim=None):
        #prior quantile
        pf = self.get_prior_pf_samples(self.n_s)
        pp_q5, pp_q95 = np.quantile(pf, q=(0.025, 0.975), axis=0)
        
        plt.figure(figsize=(4,3))
        plt.fill_between(self.x, pp_q5, pp_q95, color='darkorange', alpha=0.5, label='pri')
        plt.plot(self.x_dat, self.y, '.', markersize=10, color='black', label='noisy data')
        plt.title('Push-forward versus data')
        if ylim is not None:
            plt.ylim([ylim[0], ylim[1]])
        plt.ylabel('response')
        plt.xlabel('x')
        plt.legend()
        plt.show()
        plt.close()
        return

    def plot_compare_pf_v_data(self, samps, ylim=None):
        #prior quantile
        pf = self.get_prior_pf_samples(self.n_s)
        pp_q5, pp_q95 = np.quantile(pf, q=(0.025, 0.975), axis=0)
        
        #post quantile
        postpf = self.get_pf_samples(samps)
        postp_q5, postp_q95 = np.quantile(postpf, q=(0.025, 0.975), axis=0)

        plt.figure(figsize=(4,3))
        plt.fill_between(self.x, pp_q5, pp_q95, color='darkorange', alpha=0.5, label='pri')
        plt.fill_between(self.x, postp_q5, postp_q95, color='royalblue', alpha=0.3, label='post')
        plt.plot(self.x_dat, self.y, '.', markersize=10, color='black', label='noisy data')
        plt.title('Comparison of push-forward versus data')
        if ylim is not None:
            plt.ylim([ylim[0], ylim[1]])
        plt.ylabel('response')
        plt.xlabel('x')
        # plt.xticks([])
        plt.legend()
        plt.show()
        plt.close()
        return
    
    def plot_compare_MLE_pf_v_data(self, samps, MLE, ylim=None):
        pf = self.get_prior_pf_samples()
        pp_q5, pp_q95 = np.quantile(pf, q=(0.025, 0.975), axis=0)

        postpf = self.get_pf_samples(samps)
        postp_q5, postp_q95 = np.quantile(postpf, q=(0.025, 0.975), axis=0)
        postp_q50 = np.quantile(postpf, q=0.5, axis=0)

        plt.figure(figsize=(4,3))
        plt.fill_between(self.x, pp_q5, pp_q95, color='darkorange', alpha=0.5, label='pri')
        plt.fill_between(self.x, postp_q5, postp_q95, color='royalblue', alpha=0.3, label='post')
        plt.plot(self.x_dat, self.y, '.', markersize=10, color='black', label='noisy data')
        plt.plot(self.x, self.f(MLE, x=self.x), '-', color='lime', label='MLE')
        plt.plot(self.x, postp_q50, ':', color='darkblue', label='post-pred')
        plt.title('Push-forward versus data')
        if ylim is not None:
            plt.ylim([ylim[0], ylim[1]])
        plt.ylabel('response')
        plt.xlabel('x')
        # plt.xticks([])
        plt.legend()
        plt.show()
        plt.close()
        return
        
    def plot_m_prior(self):
        plt.plot(self.m_linspace, self.p_m.pdf(self.m_linspace), '-', color='darkorange')
        plt.title(r'Prior on  $\mu$')
        plt.show()
        plt.close
        return
    
    def plot_s_prior(self):
        plt.plot(self.s_linspace, self.p_s.pdf(self.s_linspace), '-', color='darkorange')
        plt.title(r'Prior on  $\sigma$')
        plt.show()
        plt.close
        return
    
    def plot_chains(self, samps):
        fig = plt.figure(figsize=(6,2))
        axs = fig.subplots(1,3)
        for ax, s, label in zip(axs, samps.T, [r'$\theta$', r'$\mu$', r'$\sigma$']):
            ax.plot(s)
            ax.set_ylabel(label, rotation=0, ha='right', va='center')
        fig.tight_layout()
 
    def plot_KDE_posterior(self, samps, xlim=None, ylim=None, max_lev=None):
        """ this function plots the 2D KDE of the standard Bayesian posterior samples """

        x = samps[:, 0]
        y = samps[:, 1]
        # Define the borders
        deltaX = (max(x) - min(x))/2
        deltaY = (max(y) - min(y))/2

        if xlim is None:
            xmin = min(x) - deltaX
            xmax = max(x) + deltaX
        else: 
            xmin = xlim[0]
            xmax = xlim[1]

        if ylim is None:
            ymin = min(y) - deltaY
            ymax = max(y) + deltaY
        else: 
            ymin = ylim[0]
            ymax = ylim[1]

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = sps.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        fig = plt.figure(figsize=(4,3))
        ax = fig.gca()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if max_lev is None:
            max_lev = np.max(f)
        levels = np.linspace(0, max_lev, 21)
        cset = ax.contourf(xx, yy, f, levels=levels)
        plt.colorbar(cset)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\sigma$')
        plt.title('Sample-approximated Posterior Contours')
        plt.show()
        plt.close()
        return
    
    def plot_KDE_prior(self, samps=None, xlim=None, ylim=None):
        """ this function plots the 2D KDE of the prior samples using a Gaussian approximation rather than the density function"""
        if samps is None:
            x = self.m_samps[:]
            y = self.s_samps[:]
        else:
            x = samps[:, 0]
            y = samps[:, 1]
        # Define the borders
        deltaX = (max(x) - min(x))/2
        deltaY = (max(y) - min(y))/2

        if xlim is None:
            xmin = min(x) - deltaX
            xmax = max(x) + deltaX
        else: 
            xmin = xlim[0]
            xmax = xlim[1]

        if ylim is None:
            ymin = min(y) - deltaY
            ymax = max(y) + deltaY
        else: 
            ymin = ylim[0]
            ymax = ylim[1]

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = sps.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        fig = plt.figure(figsize=(4, 3))
        ax = fig.gca()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        print('prior max level is', np.max(f))
        levels = np.linspace(0, np.max(f), 21)
        cset = ax.contourf(xx, yy, f, levels=levels)
        plt.colorbar(cset)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\sigma$')
        plt.title('Sample-approximated Prior Contours')
        plt.show()
        plt.close()
        return
    
    def plot_marg_samp_hist(self, samps, s_linspace=None, m_linspace=None):

        if s_linspace is None:
            s_linspace = self.s_linspace
        if m_linspace is None:
            m_linspace = self.m_linspace

        fig = plt.figure(figsize=(6, 3))
        axs = fig.subplots(1, 2).flatten()

        axs[0].hist(samps[:, 0], histtype = 'step', bins=30, density=True, label = 'post')
        axs[0].plot(m_linspace, self.p_m.pdf(m_linspace), label='prior')
        axs[0].set_xlim(m_linspace[0], m_linspace[-1])
        axs[0].set_xlabel(r'$\mu$', fontsize=14)

        axs[1].hist(samps[:, 1], histtype = 'step', bins=30, density=True, label = 'post')
        axs[1].plot(s_linspace, self.p_s.pdf(s_linspace), label='prior')
        axs[1].set_xlim(s_linspace[0], s_linspace[-1])
        axs[1].set_xlabel(r'$\sigma$', fontsize=14)
        axs[1].legend()

        fig.suptitle('Histogram of marginal posterior samples')
        fig.tight_layout()
        return 
