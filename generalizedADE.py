# %%
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# %%
class generalizedADE:
  """
  Generalized linear operator ADE, 
    c' + u dc/dx = nu_p d^2c/dx^2 + L c
  """
  def __init__(self, t=1.496, x_qoi=2):
    self.t = t
    self.tvec = np.linspace(0,t,100)

    self.Nx = 512
    self.Lx = 4
    # You want to sample the periodic signal up through the point just before the boundary
    # on the RHS (to not repeat)
    self.x = np.linspace( 0, self.Lx, self.Nx+1 )[:-1] 
    self.x_qoi = x_qoi
    self.x_qoi_ind = self.get_x_ind(x_qoi)

    # Nominal values from DNS, potential UQ params 
    self.IC_mode = 1
    self.IC_sd = 0.1
    self.nu_p = 0.01
    self.u = 1.0

    self.get_IC_fourier_coeffs( self.IC_mode, self.IC_sd )
    
    self.Nk = self.IC_coeffs.size
    self.k_vec = np.arange(0,self.Nk)
    self.k_vec[-1] = 0 # Zero out Nyquist mode
    self.kx = 2. * np.pi * self.k_vec / self.Lx
    self.ikx = 1j * self.kx

    self.eigenvalues = np.zeros((self.Nk),dtype='complex')

    # The QoI returned by f(x) is divided by this value.
    self.qoi_rescale_factor = 1

  def check_x_location(self, x):
        if not x in self.x:
            print(f"ERROR: {x} isn't in the x vector"\
            "\nfor the true Gaussian evolution. True x locations:")
            print(self.x)

  def get_IC_fourier_coeffs( self, IC_mode, IC_sd ):
    # Assumes a Gaussian IC
    self.IC = np.exp( -0.5 * ( self.x - IC_mode)**2. / IC_sd**2. )
    self.IC_coeffs = np.fft.rfft( self.IC )

  def get_x_ind(self, x):
    if isinstance(x,list) or isinstance(x,np.ndarray):
      ind=[]
      for xi in x:
        ind.append(self.get_x_ind(xi))
    else:
      self.check_x_location(x)
      ind = np.argmin(np.abs(self.x - x))
    return ind

  def set_eigenvalue_params(self, theta):
    self.eigenvalues[1:theta[3:].size+1] = theta[3:]

  def f_field(self, theta, t=None):
    # Assumes a vector of inputs [ IC_mode, u, nu_p, eigenvalues ]
    # If you want to use default values for any of the first 3 parameters, set their 
    # values to None in the input array.
    # If t not passed as argument, defaults to self.t
    self.set_non_eigenvalue_params(theta)
    self.set_eigenvalue_params(theta)  
    return self.f_field_from_eigenvalues(t)

  def f_field_from_eigenvalues( self, t=None):
    if t is None: t = self.t
    return np.fft.irfft( self.IC_coeffs * np.exp( t * 
      ( -self.nu_p * self.kx**2. + self.eigenvalues - self.u * self.ikx ) ) )

  def f_time_field(self, theta, tvec=None, x=None):
    self.set_non_eigenvalue_params(theta)
    self.set_eigenvalue_params(theta)
    return self.f_time_field_from_eigenvalues(tvec, x)

  def f_time_field_from_eigenvalues(self, tvec=None, x=None):
      if tvec is None: tvec = self.tvec
      c = np.fft.irfft( self.IC_coeffs * np.exp( tvec[:,np.newaxis] * 
                        ( -self.nu_p * self.kx**2. + self.eigenvalues - self.u * self.ikx ) ) )
      
      ind = self.x_qoi_ind if x is None else self.get_x_ind(x)      
      return c[:,ind]

  def set_non_eigenvalue_params( self, theta):
    if not theta[0] is None: # Override to allow for non-Gaussian IC unit tests
      self.get_IC_fourier_coeffs(theta[0].real, self.IC_sd)
    if not theta[1] is None:
      self.u = theta[1].real
    if not theta[2] is None:
      self.nu_p = theta[2].real

  def f(self, theta, t=None):
    return generalizedADE.f_field(self,theta,t)[0]/self.qoi_rescale_factor # Concentration at outflow boundary at time t

  def fX(self, X, t=None):
    # Loops over array of inputs of dimension Nsamples x Ninputs and 
    # returns an array of outputs.
    fX = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
      fX[i] = self.f(X[i,:],t)
    return fX
  
  def f_fieldX(self, X, t=None):
    # Loops over array of inputs of dimension Nsamples x Ninputs and returns an array of outputs
    fX = np.zeros((X.shape[0],self.Nx))
    for i in range(X.shape[0]):
      fX[i,:] = self.f_field(X[i,:],t)
    return fX

class FRADE(generalizedADE):
  """
  Fractional ADE, 
    c' + u dc/dx = nu_p d^2c/dx^2 + nu_m d^alpha c /dx^alpha
  """
  # Inheriting __init__ from generalizedADE

  def set_eigenvalue_params(self, theta):
    # Sets eigenvalues via nu_m d^alpha c /dx^alpha 
    self.eigenvalues = theta[-2] * self.ikx**theta[-1]

class ADE(generalizedADE):
  """
  ADE, 
    c' + u dc/dx = nu_p d^2c/dx^2
  """

  # Inheriting __init__ from generalizedADE

  def set_eigenvalue_params(self, theta):
    return
  
class cFRADE(generalizedADE):
  """
  Fractional ADE

    We're going to try calibrating a linear operator of the following form in spectral space:

    Le^(ikx) = ( nu_mr k^alpha_r + i nu_mi k^alpha_i )e^(ikx)


    c' + u dc/dx = nu_p d^2c/dx^2 + Lc
  """
  # Inheriting __init__ from generalizedADE

  def set_eigenvalue_params(self, theta):
    # Sets eigenvalues via nu_r k^alpha_r + i nu_i k^alpha_i

    nur, nui, alphar, alphai = theta[-4:]
    self.eigenvalues = -nur * self.kx**alphar + 1j * nui * self.kx**alphai

# %%
def main(args):

    model = FRADE()
    params = np.array( [model.IC_mode, model.u, model.nu_p, 0.1, 1.5])
    c = model.f_field(params)

    plt.plot(model.x, c)
    plt.show()

if __name__=="__main__":
  main(sys.argv[1:])
# %%
