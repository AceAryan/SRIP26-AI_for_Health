# SRIP26-AI_for_Health

This Repository contains the solutions to the AI for Health Task to be done for applying to Sustainibility Lab for SRIP 26.

**Directory Structure** :

SRIP26-AI_for_Health/
│
├── Data/
│ ├── AP01/
│ │ ├── nasal_airflow.txt
│ │ ├── thoracic_movement.txt
│ │ ├── spo2.txt
│ │ ├── flow_events.txt
│ │ └── sleep_profile.txt
│ ├── AP02/
│ └── ...
│
├── Visualizations/
│ ├── AP01_visualization.pdf
│ └── ...
│
├── Dataset/
│ ├── breathing_dataset.csv
│ └── sleep_stage_dataset.csv
│
├── models/
│ └── cnn_model.py
│
├── scripts/
│ ├── vis.py
│ ├── create_dataset.py
│ └── train_model.py
│
├── report.pdf
├── requirements.txt
└── README.md

**How to Run** :

(Use the following command from root directory)
For Generating Sleep Visualization PDF:
```bash 
python Scripts/vis.py -name "Data/AP05"
```
For Generating Dataset after filtering and processing data into labeled windows:
```bash
python Scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

For Training the 1D CNN Model:
```bash
python scripts/train_model.py -data_dir "Dataset"
```

**Acknowledgement** : I have used the help of Google to search upon the documentations of libraries and help to find out what tool I can use for performing a certain action. I have also used Github Copilot for Identifying Syntax Errors and Chatgpt to find edgecases and make the code better. I have tried my best to do the Task myself and limit the usage of these tools.

**Submitted By** :
Aryan Kumar
24110055
