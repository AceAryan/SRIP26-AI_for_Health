import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
from matplotlib.backends.backend_pdf import PdfPages

# helper fxn for locating data files
def find_file(folder, keywords, exclude=None):
    for file in os.listdir(folder):
        name_lower = file.lower() # to generalize file name, removing difference cause due to 
        if all(k.lower() in name_lower for k in keywords): # check if file contains that keyword
            if exclude and any(e.lower() in name_lower for e in exclude): # to distinguish between flow and flow events, as they both contain a common keyword - flow
                continue
            return os.path.join(folder, file)
    return None

# helper fxn to parse data from signal files (flow, spO2, thorac), converting it into a pandas dataframe
def load_signal(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'Data:':
            data_start = i + 1
            break

    # data format = dd.mm.yyyy h:m:s,ms; flow_value

    timestamps, values = [], []
    for line in lines[data_start:]:
        line = line.strip()
        if not line: # skip any empty line in data
            continue
        ts_str, val_str = line.split(';')
        ts_str = ts_str.strip().replace(',', '.')
        timestamps.append(pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S.%f'))
        values.append(float(val_str.strip()))

    return pd.DataFrame({'timestamp': timestamps, 'value': values})

# helper fxn to parse data from flow events file, converting to a pandas dataframe
def load_events(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'Data:' or (i > 0 and ';' in line and '-' in line.split(';')[0]):
            if ';' in line and '-' in line.split(';')[0]: # finding first data line
                data_start = i
                break
        if line.strip() == 'Data:':
            data_start = i + 1
            break

    # data format == dd.mm.yy h1:m1:s1,ms1-h2:m2:s2,ms2; ;Event_type; Sleep_Event

    events = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or ';' not in line:
            continue
        parts = line.split(';')
        if len(parts) < 3:
            continue
        try:
            time_range = parts[0].strip()   # e.g. 30.05.2024 01:47:49,666-01:48:06,770
            event_type = parts[2].strip()   # e.g. Hypopnea

            date_str, times_str = time_range.split(' ') # splitting date from duration
            start_time_str, end_time_str = times_str.split('-') # splitting start time and end time from duration

            start_str = f"{date_str} {start_time_str}".replace(',', '.') # h:m:s,ms to h:m:s.ms
            end_str   = f"{date_str} {end_time_str}".replace(',', '.')

            start = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S.%f')
            end   = pd.to_datetime(end_str,   format='%d.%m.%Y %H:%M:%S.%f')

            if end < start: # if end time goes past midnight implies end is on the next day as compared to start
                end += pd.Timedelta(days=1)

            events.append({'start': start, 'end': end, 'event_type': event_type})
        except Exception as e:
            continue

    return pd.DataFrame(events) if events else None

# color per event type
EVENT_COLORS = {
    'obstructive apnea': 'red',
    'mixed apnea':       'red',
    'central apnea':     'red',
    'hypopnea':          'yellow',
    'body event':        'lightgreen',
}

def get_event_color(event_type):
    return EVENT_COLORS.get(event_type.lower(), 'lightgreen') # if can't find any specific event type, default = lightgreen (can classify as body event) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', required=True, help='Give Path to Data Folder of a Participant e.g. Data/AP05')
    args = parser.parse_args()

    folder = args.name
    participant_name = os.path.basename(folder)

    if not os.path.exists(folder):
        print(f"Error: folder not found: {folder}")
        return

    signals = {} # setting up variables for plot
    for key, keywords, exclude, color, label, ylabel in [
        ('airflow',  ['flow'],   ['event'], 'blue',      'Nasal Flow',     'Nasal Flow (L/min)'),
        ('thoracic', ['thorac'], [],        'orange', 'Thoracic Movement',  'Resp. Amplitude'),
        ('spo2',     ['spo2'],   ['event'], 'green',       'SpO2',              'SpO2 (%)'),
    ]: # loading signal files (flow, spO2, thorac)
        path = find_file(folder, keywords, exclude=exclude)
        if path:
            print(f"Found {key}: {os.path.basename(path)}")
            signals[key] = (load_signal(path), color, label, ylabel)
        else:
            print(f"Warning: no file found for {key}")

    if not signals:
        print("No signal files found.")
        return

    events_path = find_file(folder, ['event'], exclude=None) # loading flow events file
    events = load_events(events_path) if events_path else None
    if events is not None:
        print(f"Found flow events: {os.path.basename(events_path)} ({len(events)} events)")

    all_times = []
    for df, *_ in signals.values():
        all_times += [df['timestamp'].min(), df['timestamp'].max()]
    t_start, t_end = min(all_times), max(all_times)

    os.makedirs('Visualizations', exist_ok=True)
    out = os.path.join('Visualizations', f'{participant_name}_visualization.pdf')

    window_minutes = 5 
    window = pd.Timedelta(minutes=window_minutes) 
    total_duration = t_end - t_start
    total_pages = int(total_duration / window) + 1

    with PdfPages(out) as pdf:

        current_start = t_start
        page_count = 0

        while current_start < t_end:
            current_end = current_start + window

            fig, axes = plt.subplots(len(signals), 1, figsize=(20, 10), sharex=True)

            if len(signals) == 1:
                axes = [axes]

            page_title = f"{participant_name} - {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}"
            fig.suptitle(page_title, fontsize=13)

            for ax, (key, (df, color, label, ylabel)) in zip(axes, signals.items()):
                df_window = df[
                    (df['timestamp'] >= current_start) &
                    (df['timestamp'] <= current_end)
                ]

                if df_window.empty:
                    continue

                ax.plot(df_window['timestamp'], df_window['value'],
                        linewidth=0.6, color=color, label=label)

                ax.set_ylabel(ylabel, fontsize=8)

                ymin, ymax = df_window['value'].min(), df_window['value'].max() # dynamic y limits
                pad = max((ymax - ymin) * 0.1, 0.5)
                ax.set_ylim(ymin - pad, ymax + pad)

                if key == 'airflow' and events is not None:
                    events_window = events[
                        (events['end'] >= current_start) &
                        (events['start'] <= current_end)
                    ]

                    for _, ev in events_window.iterrows():
                        ec = get_event_color(ev['event_type'])

                        ax.axvspan(ev['start'], ev['end'],
                                alpha=0.30, color=ec)

                        ax.text(ev['start'], # event label text
                                ax.get_ylim()[1] * 0.95,
                                ev['event_type'],
                                fontsize=8,
                                verticalalignment='top')

                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=5)) # x axis major ticks every 5 seconds
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M:%S'))
                ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1)) # x axis minor ticks every 1 second

                ax.grid(which='major', linestyle='-', linewidth=0.6, alpha=0.8) 
                ax.grid(which='minor', linestyle=':', linewidth=0.2, alpha=0.4)

                ax.legend(loc='upper right', fontsize=8)

            axes[-1].set_xlabel('Time', fontsize=10)

            plt.setp(axes[-1].xaxis.get_majorticklabels(),
                    rotation=90, ha='right', fontsize=10)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            current_start = current_end
            page_count += 1
            print(f"\rGenerated page {page_count} out of {total_pages}", end="", flush=True)
        print()  

    print(f"Saved multi-page PDF ({page_count} pages): {out}")

if __name__ == '__main__':
    main()