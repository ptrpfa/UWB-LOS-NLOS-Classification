### Table of Contents
1. [Team](#csc3102-data-analytics-team-2)
2. [Overview](#overview)
5. [Dataset](#dataset)
2. [Methodology](#methodology)
3. [Repository Structure](#repository-structure)
4. [Program Usage](#program-usage)

### CSC3102 Data Analytics Team 2
---
- Peter Febrianto Afandy (*2200959*) 
- Adrian Pang Zi Jian (*2200692*)
- Ryan Lai Wei Shao (*2201159*) 
- Tan Yu Jie (*2201782*)
- Ashley Tay Yong Jun (*2200795*)

### Overview
---
The goal of this project is to analyze the provided *UWN-LOS-NLOS-Dataset* dataset for the binary classification of Ultra-Wideband (UWB) wireless signals into within `Line of Sight (LOS)` or `Non-Line of Sight (NLOS)` classes.

### Dataset
---
The UWN-LOS-NLOS-Dataset dataset consists of 42,000 Channel Impulse Response (CIR) samples taken within seven different indoor environments, captured using Decaware’s DWM1000 UWB transceivers. These CIR samples measure the signal propagation between anchors and tags deployed in these environments. The original dataset can be obtained [here](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set).

More stuff here... (distribution of data etc)

### Methodology
---
1. Data Preprocessing
2. Classification
3. Data Visualisation and Interpretation

```
<ADD MORE INFO HERE>
```

### Repository Structure
---
```
/archive (previous experimentation programs included just for reference)

/dataset (contains UWB dataset)

/docs (contains documentation images)

/export (contains exported intermediary files)

/models (contains exported models)

config.py (configuration file)

utils.py (utility functions)

requirements.txt (Python dependencies for training/testing the UWB model)

README.md (this file)
```

### Program Usage
---
The final fine-tuned UWB model is available in the `/export` folder. However, you can train or test the model by following the instructions below.

1. Create a Python `virtualenv` on your local environment:
    ```
    python3 -m venv .venv
    ```
2. Install the necessary project dependencies:
    ```
    pip3 install -r requirements.txt
    ```
3. Run the interactive Python notebook to train/test the UWB model, ensuring that you've linked the notebook to the correct Python `virtualenv`.