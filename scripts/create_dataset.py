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

            overlap = max(0,(overlap_end - overlap_start).total_seconds())

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
    parser.add_argument(
        '-name',
        nargs='+',
        help='Participant names to process (eg. AP01 AP02)'
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_participants = sorted(os.listdir(args.in_dir))

    if args.name:
        participants_to_process = args.name
    else:
        participants_to_process = all_participants

    for participant in participants_to_process:

        folder = os.path.join(args.in_dir, participant)

        if not os.path.isdir(folder):
            print(f"{participant} not found, skipping.")
            continue

        print(f"Processing {participant}...")

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
        fs = 1 / dt

        df['value'] = bandpass_filter(df['value'].values, fs)

        out_path = os.path.join(args.out_dir, f"{participant}_dataset.csv")

        dataset = create_windows(df, events, fs)
        dataset.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()