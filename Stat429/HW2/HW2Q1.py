import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import librosa
from scipy.special import softmax


data, sample_rate = librosa.load('AudioNote.mp3', sr=None)


#----- Question 1 Part A -----

n_samples = len(data)
time = np.arange(n_samples) / sample_rate

# Plot
plt.figure(figsize=(12, 4))
plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform of AudioNote.mp3')


#----- Question 1 Part B -----

t = np.arange(n_samples)/sample_rate
y = data
X = np.zeros((n_samples, 3))


fft_vals = np.fft.fft(data)
fft_vals = np.abs(fft_vals)
freqs = np.fft.fftfreq(n_samples, d=1/sample_rate)
mask = freqs > 0 
freq_idx = np.argmax(fft_vals)

pos_fftvals = [fft_vals[i] for i in range(len(fft_vals)) if mask[i]]
pos_freqs = [freqs[i] for i in range(len(freqs)) if mask[i]]

idx = np.argmax(pos_fftvals)
f = pos_freqs[idx]


X[:,0] = 1
X[:,1] = np.cos(2*np.pi*f*t)
X[:,2] = np.sin(2*np.pi*f*t)

model1 = sm.OLS(y,X).fit()
rss = np.sum(model1.resid ** 2)
sign, logdet = np.linalg.slogdet(X.T @ X)


def grid_posterior(fgrid):
    log_post = np.zeros(len(fgrid))
    for i, fg in enumerate(fgrid):
        X = np.zeros((n_samples, 3))
        X[:,0] = 1
        X[:,1] = np.cos(2*np.pi*fg*t)
        X[:,2] = np.sin(2*np.pi*fg*t)
        model1 = sm.OLS(y, X).fit()
        rss = np.sum(model1.resid ** 2)
        sign, logdet = np.linalg.slogdet(X.T @ X)
        log_post[i] = -(len(data)-3)/2*np.log(rss) - (1/2)*logdet
    return softmax(log_post)

fgrid = np.linspace(pos_freqs[idx-1], pos_freqs[idx+1], 200)
probs = grid_posterior(fgrid)
peak = np.argmax(probs)


fgrid2 = np.linspace(fgrid[peak-1], fgrid[peak+1], 200)
probs2 = grid_posterior(fgrid2)

plt.figure()
plt.plot(fgrid2, probs2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Posterior probability')
plt.title('Posterior of f (zoomed grid)')

samples = np.random.choice(fgrid2, p=probs2, size=1000)
f_post = fgrid2[np.argmax(probs2)]
lower, upper = np.percentile(samples, [2.5, 97.5])

print(f"FFT estimate: {f}")
print(f"Posterior point estimate: {f_post}")
print(f"95% interval: [{lower}, {upper}]")



#----- Question 1 Part C -----

print(f"Best estimate of the frequency (posterior peak): {f_post} Hz")
print(f"95% uncertainty interval: [{lower}, {upper}] Hz")



#----- Question 1 Part D -----

print(f"The estimated frequency is {f_post:.4f} Hz, which corresponds to the musical note A4 (440 Hz).")

plt.show()