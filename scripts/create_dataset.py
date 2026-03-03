import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from vis import load_signal, load_events

def bandpass_filter(signal, fs):
    low = 0.17
    high = 0.4

    nyq = 0.5 * fs 
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    filtered = filtfilt(b, a, signal)

    return filtered

def create_windows(df, events, fs):
    window_sec = 30
    step_sec = 15  # 50% overlap

    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    values = df['value'].values
    timestamps = df['timestamp']

    dataset = []

    for start in range(0, len(values) - window_samples, step_samples):
        end = start + window_samples

        window_signal = values[start:end]
        window_start_time = timestamps.iloc[start]
        window_end_time = timestamps.iloc[end-1]

        label = "Normal"
        best_overlap = 0
        best_label = "Normal"

        for _, ev in events.iterrows():
            overlap_start = max(window_start_time, ev['start'])
            overlap_end = min(window_end_time, ev['end'])

            overlap = (overlap_end - overlap_start).total_seconds()

            if overlap > best_overlap  :  # the event with maximum overlap 
                best_overlap = overlap
                best_label = ev['event_type']
        
        if best_overlap > 0.5*window_sec:
            label = best_label

        dataset.append({
            'start_time': window_start_time,
            'end_time': window_end_time,
            'signal': window_signal.tolist(),
            'label': label
        })

    return pd.DataFrame(dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', required=True)
    parser.add_argument('-out_dir', required=True)
    parser.add_argument('--force', action='store_true') # use this flag to regenerate all datasets
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for participant in os.listdir(args.in_dir):
        folder = os.path.join(args.in_dir, participant)

        if not os.path.isdir(folder):
            continue

        print(f"Processing {participant}...")

        # finding participant data files
        flow_file = None
        event_file = None

        for f in os.listdir(folder):
            if 'flow' in f.lower() and 'event' not in f.lower():
                flow_file = os.path.join(folder, f)
            if 'event' in f.lower():
                event_file = os.path.join(folder, f)

        if flow_file is None or event_file is None:
            print("Missing files, skipping.")
            continue

        df = load_signal(flow_file)
        events = load_events(event_file)

        dt = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
        fs = 1 / dt # estimating sampling frequency

        df['value'] = bandpass_filter(df['value'].values, fs) # filter

        out_path = os.path.join(args.out_dir, f"{participant}_dataset.csv")

        if os.path.exists(out_path) and not args.force:
            print(f"Skipping {participant} (already processed)") # only generated datasets which don't exist
            continue

        dataset = create_windows(df, events, fs) 
        dataset.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()