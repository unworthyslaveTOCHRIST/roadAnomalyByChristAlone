# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE
# BY GOD'S GRACE ALONE

import os
import sys
import django
import pandas as pd
import numpy as np
import ahrs

from scipy.stats import skew, kurtosis, entropy
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

from rest_framework.response import Response
from rest_framework import status


from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
import math



# Step 1: Add the project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Step 2: Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_for_christ.settings')  # <-- adjust to your real settings path

# Step 3: Setup Django
django.setup()


# Actual Utilities 

def align_to_global_frame(df):
    # Correcting the axes along which the sensor readings were taken
    df["acc_x_corrected"] = -df["acc_y"]
    df["acc_y_corrected"] = df["acc_x"]
    df["acc_z_corrected"] = df["acc_z"]

    df["rot_x_corrected"] = -df["rot_y"]
    df["rot_y_corrected"] = df["rot_x"]
    df["rot_z_corrected"] = df["rot_z"]
    df["batch"] = "inference"

    df.drop(columns=["acc_x","acc_y","acc_z","rot_x","rot_y","rot_z"],inplace=True)

    # Example: assume `df` has columns: ['acc_x_corrected',..., 'rot_x_corrected',...]
    ## Ensure original acceleration and rotation readings are **pseudo-aligned with the global frame 

    # === 2. Extract raw values ===
    accel = df[['acc_x_corrected', 'acc_y_corrected', 'acc_z_corrected']].to_numpy()
    gyro = df[['rot_x_corrected', 'rot_y_corrected', 'rot_z_corrected']].to_numpy()
    sample_period = 1/40  # 40 Hz

    # === 3. Estimate orientation using Madgwick filter ===
    madgwick = Madgwick(sampleperiod=sample_period)
    quaternions = np.zeros((len(df), 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])

    for i in range(len(df)):
        q = madgwick.updateIMU(q, gyr=gyro[i], acc=accel[i])
        quaternions[i] = q

    # === 4. Rotate sensor-frame acceleration into global frame ===
    accel_global = np.zeros_like(accel)
    for i in range(len(df)):
        q = quaternions[i]
        rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy expects [x, y, z, w]
        accel_global[i] = rotation.apply(accel[i])

    # === 5. Combine into DataFrame and export ===
    df_global = pd.DataFrame(accel_global, columns=['acc_x_global', 'acc_y_global', 'acc_z_global'])
    df_combined = pd.concat([df, df_global], axis=1)

    # Save to CSV
    df_combined.to_csv("imu_with_global_accel_2.csv", index=False)
    print("Saved processed IMU data with global acceleration as 'imu_with_global_accel_2.csv'")
    # df_combined.head()

    return df_combined



# def data_preproc_pipeline(df):
def fix_batches(df_anomaly_type,batch_col = 'batch', expected_size = 100, min_batch_size = 30):
    new_rows = [] # Graciously to be used to store all the fixed-size rows to be elicited
    
    #Graciously regrouping rows in df_anomaly_type by common batch ids
    for batch_id, group in df_anomaly_type.groupby(batch_col):
        for i in range(0,len(group),expected_size):
            chunk = group.iloc[i:i+expected_size].copy()
            #Graciously assigning a new unique batch name to new chunk
            
            chunk[batch_col] = f"{batch_id}_{i//expected_size}"
            new_rows.append(chunk)

    # Graciously concantenating all chunks back together
    fixed_df = pd.concat(new_rows, ignore_index=True)

    newer_rows = []
    for batch_id, group in fixed_df.groupby(batch_col):
        if len(group) <= min_batch_size:
            continue
        newer_rows.append(group)

    fixed_df = pd.concat(newer_rows, ignore_index=True)
    return fixed_df


#BY GOD'S GRACE ALONE
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft


def apply_filter(data, fs, cutoff, filter_type='lowpass', order=4):
    """
    Apply an IIR filter (Butterworth) to the input signal.
    """
    nyq = 0.5 * fs
    if filter_type in ['lowpass', 'highpass']:
        norm_cutoff = cutoff / nyq
    elif filter_type in ['bandpass', 'bandstop']:
        norm_cutoff = [c / nyq for c in cutoff]
    else:
        raise ValueError("Invalid filter_type. Choose from 'lowpass', 'highpass', 'bandpass', 'bandstop'.")

    b, a = butter(order, norm_cutoff, btype=filter_type)
    return filtfilt(b, a, data)


def extract_fft_features(signal, fs):
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1/fs)
    spectrum = np.abs(fft(signal))[:N//2]
    freqs = freq[:N//2]

    # Power spectrum
    power = spectrum ** 2
    power_norm = power / np.sum(power) if np.sum(power) != 0 else power

    dom_freq = freqs[np.argmax(power)]
    spec_entropy = entropy(power_norm)
    spec_centroid = np.sum(freqs * power_norm)

    # Band power in bins (e.g. 0–5 Hz, 5–10 Hz, ..., up to Nyquist)
    bands = [(i, i + 5) for i in range(0, int(fs // 2), 5)]
    band_powers = [np.sum(power[(freqs >= low) & (freqs < high)]) for low, high in bands]

    # Energy in specific bands
    energy = np.sum(power)
    spec_skew = skew(power)
    spec_kurt = kurtosis(power)

    return {
        "dominant_frequency": dom_freq,
        "spectral_entropy": spec_entropy,
        "spectral_centroid": spec_centroid,
        **{f"band_power_{i}-{j}Hz": p for (i, j), p in zip(bands, band_powers)},
        "spectral_energy": energy,
        "spectral_skewness": spec_skew,
        "spectral_kurtosis": spec_kurt,
    }


def extract_features_windowed(df, fs, window_size, stride, filter_cfg=None):
    
    features = []
    signal_columns = ['acc_x_global', 'acc_y_global', 'acc_z_global', 'rot_x_corrected', 'rot_y_corrected', 'rot_z_corrected']
    start_idxs = np.arange(0, len(df) - window_size + 1, stride)

    for start in start_idxs:
        window = df.iloc[start:start + window_size]
        feature_vector = {}
        center_row = window.iloc[window_size // 2]
        feature_vector["latitude"] = round(center_row["latitude"],2)
        feature_vector["longitude"] = round(center_row["longitude"],2)
        feature_vector["inference_start_time"] = center_row["timestamp"]

        for col in signal_columns:
            signal = window[col].values

            if filter_cfg:
                signal = apply_filter(
                    signal,
                    fs=fs,
                    cutoff=filter_cfg.get('cutoff', 10),
                    filter_type=filter_cfg.get('type', 'lowpass'),
                    order=filter_cfg.get('order', 4)
                )

            # Time-domain features
            feature_vector[f"{col}_mean"] = np.mean(signal)
            feature_vector[f"{col}_std"] = np.std(signal)
            feature_vector[f"{col}_min"] = np.min(signal)
            feature_vector[f"{col}_max"] = np.max(signal)
            feature_vector[f"{col}_rms"] = np.sqrt(np.mean(signal**2))
            feature_vector[f"{col}_range"] = np.ptp(signal)

            # Frequency-domain features
            fft_feats = extract_fft_features(signal, fs)
            for k, v in fft_feats.items():
                feature_vector[f"{col}_{k}"] = v

        features.append(feature_vector)

        features_df = pd.DataFrame(features)
 
    return features_df



def apply_feature_extraction_across_all_identical_anomaly_batches(entire_df,window_size = 20,window_stride = 5, cutoff_freq = 20):
    combined_df = pd.DataFrame()

    for batch_id, group in entire_df.groupby("batch"):
        signal_length = len(group)
        fs = signal_length
        filter_cfg = {
            "cutoff" : min(cutoff_freq, len(group)),                  # Cutoff frequency of 20 Hz
            "type"   : "lowpass",
            "order"  : 4
        }
        window_size = window_size
        stride = window_stride

        if signal_length < 2:
            print("Graciously skipping too short batches")
            continue # Graciously skipping too short batches
            

        # Dynamically constraining window size and stride, so that sample-length of signal is not exceeded
        dynamic_window = min(window_size, signal_length )
        dynamic_stride = min(stride, dynamic_window)

        if signal_length < dynamic_window:
            print("Not enough data to form a window")
            continue # Not enough data to form a window

        num_windows = 1 + (signal_length - dynamic_window) // dynamic_stride

        if num_windows <= 0:
            print("Zero or negative strides are disregarded")
            continue # Zero or negative strides are disregarded

         # Extract features for current batch
        features_df = extract_features_windowed(group, fs, window_size, stride, filter_cfg = filter_cfg)
        combined_df = pd.concat([combined_df, features_df], ignore_index=True)

    return combined_df


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'RoadAnomalyPredictionOutput', 'road_anomaly_model_2.pkl')
import joblib



def ml_pipeline(feature_engineered_df): 
    # Spliting feature-engineered inference data  

    if not os.path.exists(MODEL_PATH):
        print("🚫 Model file not found.")
        return Response("Model file missing.", status= status.HTTP_500_INTERNAL_SERVER_ERROR)


    model = joblib.load(MODEL_PATH)

    latitude = feature_engineered_df["latitude"]
    longitude = feature_engineered_df["longitude"]
    inference_start_time = feature_engineered_df["inference_start_time"]
    feature_engineered_df.drop(columns=["latitude","longitude","inference_start_time"], inplace = True)
    inf_data = feature_engineered_df
    print(inf_data.head())
    # # print("Graciously before dropping incomplete rows",inf_data.info())
    inf_data.dropna(inplace = True)
    # # print("Graciously before dropping incomplete rows",inf_data.info())
    model = joblib.load(MODEL_PATH)
    predictions     =    pd.Series(model.predict(inf_data))
    predictions_df  =    pd.DataFrame()
    predictions_df["predictions"] = predictions
    predictions_df["latitude"]  = latitude
    predictions_df["longitude"] = longitude
    predictions_df["inference_start_time"] = inference_start_time

    df_final = pd.DataFrame()
    for location_group_id, location_group in predictions_df.groupby(["latitude","longitude"]):
        df_per_location = pd.DataFrame()
        location_group.reset_index()

        prediction_per_location_group = location_group["predictions"].value_counts(normalize=True)
        prediction_name = prediction_per_location_group.index
        prediction_value = prediction_per_location_group.values
        df_per_location["predictions"] = prediction_name
        df_per_location["confidence_in_%"] = prediction_value * 100
        df_per_location["latitude"]  = location_group["latitude"].iloc[0]
        df_per_location["longitude"]  = location_group["longitude"].iloc[0]
        df_per_location["inference_start_time"]  = location_group["inference_start_time"].iloc[0]

        df_final = pd.concat([df_final,df_per_location], ignore_index=True)

    print(type(df_final))
    df_final.to_csv("predictions.csv", index=False)
    print("Saved  Gracious predictions 'predictions.csv'")


    return df_final


# Now imports will work
from RoadAnomalyInferenceLogs.models import RoadAnomalyInferenceLogs
from RoadAnomalyManualDataCollection.models import RoadAnomalyManualDataCollection
from RoadAnomalyInput.models import RoadAnomalyInput


     
# Graciously working purely with RoadAnomalyManualData
def update_input_main():
    data = RoadAnomalyManualDataCollection.objects.all().values("latitude", "longitude", "anomaly")
    df = pd.DataFrame(data)
    if df.empty:
        return

    grouped = df.groupby(["latitude", "longitude"]).agg(lambda x : x.mode()[0]).reset_index()
    grouped = df.groupby(["latitude", "longitude"]).agg(lambda x : x.mode()[0]).reset_index()
    for _,row in grouped.iterrows():
        RoadAnomalyInput.objects.get_or_create(
            latitude = row.latitude,
            longitude = row.longitude,
            anomaly = row.anomaly

        )

if __name__ == "__main__":
    # inference_data = RoadAnomalyInferenceLogs.objects.order_by("id").values()
    # df = pd.DataFrame(inference_data)
    # print(df.columns)
    # data_globally_aligned = align_to_global_frame(df)
    # batched_df = fix_batches(data_globally_aligned)
    # engineered_df = apply_feature_extraction_across_all_identical_anomaly_batches(batched_df)
    # predictions_df = ml_pipeline(engineered_df)
    # print(predictions_df.head())

    update_input_main()
    