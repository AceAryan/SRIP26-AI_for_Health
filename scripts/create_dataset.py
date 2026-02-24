import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def load_signal(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'Data:':
            data_start = i + 1
            break

    timestamps = []
    values = []

    for line in lines[data_start:]:
        if not line.strip():
            continue
        ts_str, val_str = line.split(';')
        ts_str = ts_str.strip().replace(',', '.')
        timestamps.append(pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S.%f'))
        values.append(float(val_str.strip()))

    return pd.DataFrame({'timestamp': timestamps, 'value': values})

def load_events(filepath):
    events = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if ';' not in line or '-' not in line:
            continue

        parts = line.split(';')
        time_range = parts[0].strip()
        event_type = parts[2].strip()

        date_str, times = time_range.split(' ')
        start_time, end_time = times.split('-')

        start = pd.to_datetime(f"{date_str} {start_time}".replace(',', '.'),
                               format='%d.%m.%Y %H:%M:%S.%f')
        end = pd.to_datetime(f"{date_str} {end_time}".replace(',', '.'),
                             format='%d.%m.%Y %H:%M:%S.%f')

        if end < start:
            end += pd.Timedelta(days=1)

        events.append({'start': start, 'end': end, 'label': event_type})

    return pd.DataFrame(events)

def bandpass_filter(signal, fs):
    low = 0.17
    high = 0.5

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
        window_end_time = timestamps.iloc[end]

        label = "Normal"

        for _, ev in events.iterrows():
            overlap_start = max(window_start_time, ev['start'])
            overlap_end = min(window_end_time, ev['end'])

            overlap = (overlap_end - overlap_start).total_seconds()

            if overlap > 15:  # more than 50% of 30 sec
                label = ev['label']
                break

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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for participant in os.listdir(args.in_dir):
        folder = os.path.join(args.in_dir, participant)

        if not os.path.isdir(folder):
            continue

        print(f"Processing {participant}...")

        # find airflow file
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

        # estimate sampling frequency
        dt = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
        fs = 1 / dt

        # filter
        df['value'] = bandpass_filter(df['value'].values, fs)

        # create dataset
        dataset = create_windows(df, events, fs)

        # save
        out_path = os.path.join(args.out_dir, f"{participant}_dataset.csv")
        dataset.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()