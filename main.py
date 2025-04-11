# main.py (FastAPI Backend)

import fastapi
import uvicorn
import tensorflow as tf
import numpy as np
import io
import os
import math

# Import necessary functions from your feature extraction code
# (You might need to put these functions in a separate file, e.g., features.py
# and import them: from features import extract_enhanced_features, runs_test)
# --- OR copy the functions directly into this file ---
from scipy.stats import chisquare, entropy, kstest, skew, kurtosis, norm
from scipy.fftpack import fft
from scipy.signal import periodogram

# ----------- Helper Functions (runs_test) -----------
# PASTE your runs_test function definition here...
def runs_test(sequence):
    # Ensure sequence is numpy array of uint8
    if isinstance(sequence, bytes):
        seq_int = np.frombuffer(sequence, dtype=np.uint8)
    elif isinstance(sequence, np.ndarray) and sequence.dtype == np.uint8:
        seq_int = sequence
    elif isinstance(sequence, np.ndarray):
         seq_int = sequence.astype(np.uint8) # Attempt conversion if needed
    else: # Assume list-like
         seq_int = np.array(sequence, dtype=np.uint8)

    n = len(seq_int) # Get length *after* conversion
    if n < 2: return 0.5 # Handle sequences too short for runs test

    median = np.median(seq_int)

    # Handle cases where all values are the same (median might not separate)
    if np.all(seq_int == seq_int[0]): return 0.0

    signs = np.sign(seq_int - median)
    # Propagate sign for values equal to median
    for i in range(n):
      if signs[i] == 0:
          # Try propagating from previous non-zero
          if i > 0 and signs[i-1] != 0:
              signs[i] = signs[i-1]
          # If previous was also zero or it's the first element, find next non-zero
          else:
              for k in range(i + 1, n):
                  if signs[k] != 0:
                      signs[i] = signs[k] # Assign the same sign as the next different value
                      break
          # If still zero (all remaining elements are median), assign arbitrarily based on what's common
          if signs[i] == 0:
              non_zero_signs = signs[signs != 0]
              signs[i] = 1 if len(non_zero_signs) == 0 or np.sum(non_zero_signs > 0) >= np.sum(non_zero_signs < 0) else -1


    if np.all(signs == 0): return 0.0 # Should not happen with above logic

    runs = 1
    for i in range(1, n):
      # Count run if signs differ and neither is zero
      if signs[i] != 0 and signs[i-1] != 0 and signs[i] != signs[i-1]:
          runs += 1

    n1 = np.sum(signs > 0); n2 = np.sum(signs < 0)

    # If only one type of sign exists (e.g., all above median after propagation), it's non-random
    if n1 == 0 or n2 == 0: return 0.0

    N = n1 + n2 # Total count of non-median values
    if N <= 1: return 0.5 # Avoid division by zero

    runs_exp = ((2.0 * n1 * n2) / N) + 1.0
    std_dev_sq_numerator = (2.0 * n1 * n2 * (2.0 * n1 * n2 - N))
    std_dev_sq_denominator = (N**2 * (N - 1.0))

    if std_dev_sq_denominator == 0:
        return 0.0 if runs != runs_exp else 0.5

    std_dev_sq = std_dev_sq_numerator / std_dev_sq_denominator
    if std_dev_sq <= 0: return 0.0 if runs != runs_exp else 0.5

    std_dev_runs = np.sqrt(std_dev_sq)
    if std_dev_runs == 0:
         return 0.0 if runs != runs_exp else 0.5

    z = (runs - runs_exp) / std_dev_runs
    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    return p_value if not math.isnan(p_value) else 0.5

# ----------- Feature Extraction Function -----------
# PASTE your extract_enhanced_features function definition here...
def extract_enhanced_features(sequence_bytes):
    # Ensure input is bytes
    if not isinstance(sequence_bytes, bytes):
        raise TypeError("Input sequence must be bytes")

    seq_int = np.frombuffer(sequence_bytes, dtype=np.uint8)
    n = len(seq_int)
    num_fft_coeffs = 64
    num_psd_coeffs = 32
    num_bins = 16
    num_lags = 8
    num_basic_stats = 11
    expected_len = num_basic_stats + num_lags + num_fft_coeffs + num_psd_coeffs + num_bins

    if n == 0: return np.zeros(expected_len)

    mean_val = np.mean(seq_int); std_dev = np.std(seq_int)
    if n > 3 and std_dev > 1e-6:
        skew_val = skew(seq_int); kurtosis_val = kurtosis(seq_int)
    else: skew_val = 0.0; kurtosis_val = -3.0 # Kurtosis of normal distribution is 3 (or 0 if excess kurtosis)

    counts = np.bincount(seq_int, minlength=256)
    # Ensure counts sum > 0 for entropy and chi-square
    if np.sum(counts) == 0: return np.zeros(expected_len) # Or handle as error
    freq = counts / n; ent_val = entropy(freq, base=2)

    # Check expected frequencies for chi-square
    expected_freq = n / 256.0
    # Apply chi-square test only if expected frequency is reasonable (e.g., >= 5)
    if expected_freq >= 5:
        chi_stat, chi_p = chisquare(counts)
    else:
        chi_p = 1.0 # Cannot reliably compute chi-square, assume pass

    ks_stat, ks_p = kstest(seq_int / 255.0, 'uniform')
    autocorrs = []; lags = [1, 2, 3, 5, 8, 13, 21, 34][:num_lags]
    for lag in lags:
        if lag < n:
             # Ensure variance is not zero before calculating correlation
             if np.std(seq_int[:-lag]) > 1e-9 and np.std(seq_int[lag:]) > 1e-9:
                 corr = np.corrcoef(seq_int[:-lag], seq_int[lag:])[0, 1]
                 autocorrs.append(corr if not math.isnan(corr) else 0.0)
             else:
                 autocorrs.append(0.0) # Assign 0 if variance is zero
        else: autocorrs.append(0.0)

    runs_pval = runs_test(seq_int) # Use the corrected runs_test

    # Add check for FFT/PSD on sequence length
    if n > num_fft_coeffs:
        fft_coeffs = np.abs(fft(seq_int - mean_val))
        fft_features = np.log(fft_coeffs[1:num_fft_coeffs+1] + 1e-10)
    else:
        fft_features = np.zeros(num_fft_coeffs)
    # Ensure correct length if shorter sequence processed
    if len(fft_features) < num_fft_coeffs: fft_features = np.pad(fft_features, (0, num_fft_coeffs - len(fft_features)))


    psd_features = np.zeros(num_psd_coeffs)
    if n > 1:
        # Check if sequence is constant
        if np.all(seq_int == seq_int[0]):
             psd_features = np.zeros(num_psd_coeffs) # Assign zeros or specific value
        else:
             try:
                 freqs, psd = periodogram(seq_int)
                 num_psd_to_take = min(len(psd), num_psd_coeffs + 1)
                 if num_psd_to_take > 1:
                      psd_f = np.log(psd[1:num_psd_to_take] + 1e-10)
                      if len(psd_f) < num_psd_coeffs:
                           psd_features = np.pad(psd_f, (0, num_psd_coeffs - len(psd_f)))
                      else:
                           psd_features = psd_f[:num_psd_coeffs]
                 else:
                      psd_features = np.zeros(num_psd_coeffs)
             except ValueError: # Handle potential errors in periodogram if sequence is ill-conditioned
                 psd_features = np.zeros(num_psd_coeffs)

    high_bytes = np.sum(seq_int >= 128) / n if n > 0 else 0.0
    byte_transitions = np.mean(np.abs(np.diff(seq_int))) if n > 1 else 0.0
    monotonic_runs = np.sum(np.diff(np.sign(np.diff(seq_int))) != 0) + 1 if n > 2 else 1.0
    bins = np.histogram(seq_int, bins=num_bins, range=(0, 255))[0] / n if n > 0 else np.zeros(num_bins)

    feature_vector = np.array([
        mean_val / 255.0, std_dev / 128.0, ent_val / 8.0, chi_p, ks_p, skew_val, kurtosis_val,
        runs_pval, high_bytes, byte_transitions / 255.0, monotonic_runs / n
    ] + autocorrs + list(fft_features) + list(psd_features) + list(bins))

    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6) # Handle potential NaNs/Infs

    # Final check for expected length
    if len(feature_vector) != expected_len:
         # Pad if too short (might happen with very short input n), though ideally should match
         if len(feature_vector) < expected_len:
              feature_vector = np.pad(feature_vector, (0, expected_len - len(feature_vector)))
         # Truncate if too long (should not happen)
         else:
              feature_vector = feature_vector[:expected_len]
         print(f"Warning: Feature vector length adjusted to {expected_len}")


    return feature_vector # Return vector directly

# ----------- Constants -----------
EXPECTED_FEATURE_LENGTH = 131 # Verify this matches your model input
SEQ_LENGTH = 1024
# --- Paths relative to this main.py file when deployed ---
# --- Ensure these files are included in your Git repo! ---
MODEL_PATH = 'rng_health_model' # Keras SavedModel directory
SCALER_MEAN_PATH = 'rng_scaler_mean.npy'
SCALER_SCALE_PATH = 'rng_scaler_scale.npy'

# ----------- Load Model and Scaler (Globally) -----------
model = None
scaler_params = None
model_load_error = None
scaler_load_error = None

try:
    # Use TFSMLayer for inference from SavedModel directory in Keras 3+
    model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
    print("Keras TFSMLayer loaded successfully.")
except Exception as e:
    model_load_error = f"Error loading Keras model: {e}"
    print(model_load_error)

try:
    scaler_params = {
        "mean": np.load(SCALER_MEAN_PATH),
        "scale": np.load(SCALER_SCALE_PATH)
    }
    print("Scaler parameters loaded successfully.")
except Exception as e:
    scaler_load_error = f"Error loading scaler parameters: {e}"
    print(scaler_load_error)

# ----------- FastAPI App Definition -----------
app = fastapi.FastAPI()

@app.get("/") # Basic root endpoint
def read_root():
    return {"message": "RNG Strength Predictor API"}

@app.post("/predict/")
async def predict_rng(file: fastapi.UploadFile = fastapi.File(...)):
    """
    Receives RNG byte data, extracts features, scales, predicts,
    and returns the result.
    """
    if model_load_error:
        raise fastapi.HTTPException(status_code=500, detail=f"Model loading failed: {model_load_error}")
    if scaler_load_error:
        raise fastapi.HTTPException(status_code=500, detail=f"Scaler loading failed: {scaler_load_error}")
    if model is None or scaler_params is None:
         raise fastapi.HTTPException(status_code=500, detail="Model or Scaler not loaded on server.")

    # Read bytes from uploaded file
    sequence_bytes = await file.read()

    # Validate length
    if len(sequence_bytes) != SEQ_LENGTH:
        raise fastapi.HTTPException(status_code=400, detail=f"Invalid file size: Received {len(sequence_bytes)} bytes, expected {SEQ_LENGTH} bytes.")

    try:
        # 1. Extract Features
        features = extract_enhanced_features(sequence_bytes)

        if features.shape[0] != EXPECTED_FEATURE_LENGTH:
             raise fastapi.HTTPException(status_code=500, detail=f"Feature extraction error: wrong feature count ({features.shape[0]}).")

        # 2. Scale Features
        scaled_features = (features.reshape(1, -1) - scaler_params['mean']) / scaler_params['scale']
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32) # Ensure float32

        # 3. Predict
        prediction_tensor = model(scaled_features) # Call TFSMLayer
        probability_flawed = prediction_tensor['output_0'].numpy()[0][0] # Use the correct output key

        # 4. Determine Result
        is_flawed = probability_flawed > 0.5
        result_text = "Flawed" if is_flawed else "Healthy"

        # 5. Return JSON Response
        return {
            "filename": file.filename,
            "prediction": result_text,
            "probability_flawed": float(probability_flawed) # Ensure JSON serializable
        }

    except Exception as e:
        print(f"Error during prediction processing: {e}") # Log server-side error
        raise fastapi.HTTPException(status_code=500, detail=f"Error processing file: {e}")

# --- Run Command (for local testing, Render uses start command) ---
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000) # Port 8000 for local test