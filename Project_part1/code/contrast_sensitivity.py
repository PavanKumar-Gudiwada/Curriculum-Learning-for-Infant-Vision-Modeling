import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# Parameters for each age (these are estimates to match the image)
parameters = {
    1: {"gamma_max": 9, "f_max": 0.5, "delta": 0.4, "beta": 0.6},
    3: {"gamma_max": 15, "f_max": 0.8, "delta": 0.4, "beta": 1.8},
    8: {"gamma_max": 31, "f_max": 0.9, "delta": 0.4, "beta": 1.5},
    48: {"gamma_max": 100, "f_max": 3.0, "delta": 0.4, "beta": 4.0},
}

def compute_S_prime(log_f, gamma_max, log_f_max, beta):
  
  """CSF subpart 1: 
            
      Computes the S' part of the CSF formula
      
      Args: 
        gamma_max (𝛄max): peak sensitivity of the CSF.
        log_f_max: frequency where the peak sensitivity occurs. It also indicates the center of the log-contrast sensitivity function.
        beta (𝛽): width of the sensitivity function, defined at half the maximum sensitivity (𝛄max).
          
      Returns: 
        returns the S' curve which is used when the f>fmax
  """

  # Constants
  kappa = np.log10(2)
  beta_prime = np.log10(2 * beta)
  # Calculate S'(f) in log scale
  S_prime = np.log10(gamma_max) - kappa * ((log_f - log_f_max) / (beta_prime / 2))**2
  return S_prime


def compute_CSF(frequencies, gamma_max, log_f_max, delta, beta):

  """CSF subpart 2: 
      
      Computes the CSF curve using the truncated log parameter model for 
      applying it to the inages
      
      Args: 
        gamma_max (𝛄max): peak sensitivity of the CSF.
        log_f_max: frequency where the peak sensitivity occurs. It also indicates the center of the log-contrast sensitivity function.
        beta (𝛽): width of the sensitivity function, defined at half the maximum sensitivity (𝛄max).
        delta (𝛿): Ensures the asymmetry of the CSF
          
      Returns: 
        calculated linear CSF curve
  """
  # Convert linear frequency to log scale
  log_f = np.log10(frequencies + 1e-5)
  # Calculate S'(log_f)
  S_prime = compute_S_prime(log_f, gamma_max, log_f_max, beta)
  S = np.where((log_f < log_f_max) & (S_prime < np.log10(gamma_max) - delta), np.log10(gamma_max) - delta, S_prime)
  return 10**S  


def get_contrast_sensitivity_transform(age: float,
                                       image_path: str) -> np.array:

  """Gets the contrast sensitivity transform

  Converts the image to spatial frequency domain, 
  Applies CSF using CSF supbart 1(compute_S_prime) and CSF subpart 2(compute_CSF),
  Reconstructs the image

  Args:
    age: age in months, only possible values: 1, 3, 8, 48
    image_path: image path as string

  Returns:
    adjusted image after applying contrast sensitivity function as np.array
  """
    
  try:
    # map age from calling function to age of the parameter curves available
    if age > 12.0:
      age = 48
    elif (age >6.0) & (age <=12):
      age = 8
    elif (age >2.5) & (age <=6.0):
      age = 3
    elif (age >0.0) & (age <=2.5):
      age = 1
      
    params = parameters[age]
  except:
    print("Age in months can only be integer, with possible values 1, 3, 8, 48 only!")
    print("Root directory path should be valid!")
    return
  
  gamma_max, f_max, delta, beta = params.values()

  # Load the image
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
  image = image / 255.0  # Normalize to [0, 1] for easier manipulation

  # Initialize an empty list to store the processed color channels
  adjusted_channels = []

  # Create a frequency grid
  rows, cols = image.shape[:2]
  u = np.arange(-rows // 2, rows // 2)
  v = np.arange(-cols // 2, cols // 2)
  U, V = np.meshgrid(v, u)
  frequencies = np.sqrt(U**2 + V**2)  # Calculate radial frequency

  # Process each color channel separately
  for i in range(3):  # Loop over the R, G, and B channels
    # Apply FFT to the current channel
    fft_channel = fft2(image[:, :, i])
    fft_channel_shifted = fftshift(fft_channel)  # Shift zero frequency to the center
    
    csf = compute_CSF(frequencies, gamma_max, f_max, delta, beta)

    # Apply the CSF to the FFT coefficients
    csf = csf / np.max(csf) #normalise the CSF map
    adjusted_fft_channel = fft_channel_shifted * csf

    # Inverse FFT to get the adjusted image channel
    adjusted_fft_channel = ifftshift(adjusted_fft_channel)  # Shift back
    adjusted_channel = np.real(ifft2(adjusted_fft_channel))  # Take the real part
    adjusted_channel = np.clip(adjusted_channel, 0, 1)  # Clip to valid range

    # Add the adjusted channel to the list
    adjusted_channels.append(adjusted_channel)

  # Stack the adjusted channels along the third axis to form an RGB image
  adjusted_image = np.stack(adjusted_channels, axis=2)

  return adjusted_image
