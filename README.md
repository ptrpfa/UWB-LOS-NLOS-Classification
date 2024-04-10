### Table of Contents
1. [Team](#csc3102-data-analytics-team-2)
2. [Overview](#overview)
5. [Dataset](#dataset)
2. [Methodology](#methodology)
3. [Repository Structure](#repository-structure)
4. [Program Usage](#program-usage)

### CSC3102 Data Analytics Team 2
---
- Peter Febrianto Afandy 
- Adrian Pang Zi Jian 
- Ryan Lai Wei Shao 
- Tan Yu Jie 
- Ashley Tay Yong Jun 

### Overview
---
The goal of this project is to analyze the provided *UWN-LOS-NLOS-Dataset* dataset for the binary classification of Ultra-Wideband (UWB) wireless signals into within `Line of Sight (LOS)` or `Non-Line of Sight (NLOS)` classes. Classification of UWB signals into LOS or NLOS is an important problem to solve as it plays a crucial role for obtaining the precise localization of devices within indoor environments.

### Dataset
---
The UWN-LOS-NLOS-Dataset dataset consists of 42,000 Channel Impulse Response (CIR) samples taken within seven different indoor environments, captured using Decaware’s DWM1000 UWB transceivers. These CIR samples measure the signal propagation between anchors and tags deployed in these environments. The original dataset can be obtained [here](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set).

A brief overview of all initial features is detailed below:

<ins>CIR Features</ins>
- `CIR0` to `CIR1015` (1016 samples with 1 nano-second resolution)

<ins>Non-CIR Features</ins>
- Class Feature
    - `NLOS` 
        - Indicates LOS/NLOS Class (0: LOS, 1: NLOS)
- Frame Quality Indicators (used to assess the quality of messages received, and any related timestamps)
    - `RANGE`
        - Measured range (time of flight)
    - `FP_IDX` (First Power Index)
        - Index of detected first path element in CIR accumulator (**16-bit value** reporting the position within the accumulator that the Leading Edge (LDE) algorithm has determined to be the first path)
    - `FP_AMP1`
        - The amplitude of the sample reported is the magnitude of the accumulator tap at the **index 1** beyond the integer portion of the rising edge `FP_IDX`.
    - `FP_AMP2`
        - The amplitude of the sample reported is the magnitude of the accumulator tap at the **index 2** beyond the integer portion of the rising edge `FP_IDX`.
    - `FP_AMP3`
        - The amplitude of the sample reported is the magnitude of the accumulator tap at the **index 3** beyond the integer portion of the rising edge `FP_IDX`.
    - `STDEV_NOISE` ($\sigma\ CIREN)$
        - Standard Deviation of CIR Estimate Noise (**16-bit value** reporting the standard deviation of the noise level seen during the LDE algorithm’s analysis of the accumulator data)
    - `CIR_PWR`
        - Total Channel Impulse Response Power (**16-bit value** reporting the sum of the squares of the magnitudes of the accumulator from the estimated highest power portion of the channel, which is related to the receive signal power)
    - `MAX_NOISE`
        - Maximum value of noise detected
    - `RXPACC`
        - Received RX preamble symbols (Preamble Accumulation Count)
    - `CH`
        - Channel value
    - `FRAME_LEN`
        - Length of frame
    - `PREAM_LEN`
        - Length of preamble
    - `BITRATE`
        - Message's bit rate
    - `PRFR`
        - Pulse Repetition Frequency Rate (Mhz)

After pre-processing and dataset preparations, we derived six variants of Non-CIR Feature subsets and nine variants of CIR Feature subsets, amounting to a total of 72 possible dataset variants.
- Non-CIR Features:
    - `df_noncir`: Non-CIR
    - `df_noncir_mm_scaled`: Non-CIR (Min-Max Scaled)
    - `df_noncir_ss_scaled`: Non-CIR (Standard Scaled)
    - `df_noncir_trimmed`: Non-CIR (after feature selection)
    - `df_noncir_mm_scaled_trimmed`: Non-CIR (Min-Max Scaled, after feature selection)
    - `df_noncir_ss_scaled_trimmed`: Non-CIR (Standard Scaled, after feature selection)

- CIR Features: (Does not contain `NLOS` class feature!)
    - `df_cir`: CIR
    - `df_cir_mm_scaled`: CIR (Min-Max Scaled)
    - `df_cir_ss_scaled`: CIR (Standard Scaled)
    - `df_cir_stats`: CIR Statistical Measures
    - `df_cir_stats_mm_scaled`: CIR Statistical Measures (Min-Max Scaled)
    - `df_cir_stats_ss_scaled`: CIR Statistical Measures (Standard Scaled)
    - `df_cir_stats_trimmed`: CIR Statistical Measures (after feature selection)
    - `df_cir_stats_mm_scaled_trimmed`: CIR Statistical Measures (Min-Max Scaled, after feature selection)
    - `df_cir_stats_ss_scaled_trimmed`: CIR Statistical Measures (Standard Scaled, after feature selection)
    - `df_pca`: PCA(CIR)
    - `df_pca_mm_scaled`: PCA(CIR) (Min-Max Scaled)
    - `df_pca_ss_scaled`: PCA(CIR) (Standard Scaled)

After model training and testing, we derived on **three** optimal dataset combinations, which are:
1. `noncir_ss_scaled_trimmed_cir_pca_ss_scaled.pkl`
    - Non-CIR: Standard scaled non-CIR features after feature selection process
    - CIR: Standard scaled CIR features after Principal Component Analysis

2. `noncir_ss_scaled_trimmed_cir_ss_scaled.pkl`
    - Non-CIR: Standard scaled non-CIR features after feature selection process
    - CIR: Standard scaled original CIR features

3. `noncir_ss_scaled_trimmed_cir_pca.pkl`
    - Non-CIR: Standard scaled non-CIR features after feature selection process
    - CIR features after Principal Component Analysis

### Methodology
---
The following methodology was employed for data analysis of the UWB Dataset for LOS/NLOS classification:
1. Data Pre-processing and Preparation 
    - Dataset Creation
    - Preliminary Exploratory Data Analysis
    - Feature Engineering
        - Outlier Management
        - Feature Creation
        - Feature Reduction
        - Feature Analysis
        - Feature Selection
        - Dataset Subsetting
2. Data Mining
    - Initial Data Mining
    -  Hyper-parameter Fine-tuning
3. Analysis
    - Model Performance Comparison and Analysis


We utilised three different types of models, **Support Vector Machines (SVM)**, **Multi-Layer Perceptron (MLP)**, and **Random Forest (RF)**. After training and tuning the models, we derived the following optimal models:
- **SVM**: kernel = ‘rbf’, C = 0.1, gamma = ‘auto’
    ```
    # Create a SVM Classifier with the kernel of linear for linear hyperplane
    clf = SVC(kernel = 'rbf', C = 0.1, gamma = 'auto', random_state = RANDOM_STATE)
    
    clf.fit(x_train, y_train)
    
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    
    # Export model
    save_to_pickle(f'{MODEL_FOLDER}/svm_non_linear_70_30_dataset_1.pkl', clf, complete_path=False)
    ```

- **MLP**: hidden_layer_size = (5, 5, 5), activation = ‘relu’, solver = ‘adam’, max_iter = 1000
    ```
    # Create a MLP Classifier with the kernel of linear for linear hyperplane
    clf = MLPClassifier(hidden_layer_sizes = (5, 5, 5), activation = 'relu', learning_rate = 'constant', solver = 'adam', max_iter = 1000, random_state = RANDOM_STATE)
    
    clf.fit(x_train, y_train)
    
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    
    # Export model
    save_to_pickle(f'{MODEL_FOLDER}/mlp_70_30_dataset_1.pkl', clf, complete_path=False)
    ```

- **RF**: max_depth = 10, criterion = ‘gini’
    ```
    clf = RandomForestClassifier(max_depth = 10, criterion = ‘gini’, random_state = RANDOM_STATE)

    clf.fit(x_train, y_train)

    save_to_pickle(f'{MODEL_FOLDER}/rf_70_30_dataset_1.pkl', clf, complete_path=False)
    ```

### Repository Structure
---
```
/archive (previous experimentation programs included just for reference)

/dataset (contains UWB dataset)

/docs (contains documentation images)

/export (used to store exported intermediary files)

/models (contains exported models)

config.py (configuration file)

utils.py (utility functions)

requirements.txt (Python dependencies for training/testing the UWB model)

README.md (this file)
```

### Program Usage
---
The final fine-tuned UWB models are available in the `/export` folder. However, you can train or test the model by following the instructions below. Do note that data processing and model training/testing may take a while to run due to the complexities and nature of the dataset.

1. Create a Python `virtualenv` on your local environment:
    ```
    python3 -m venv .venv
    ```
2. Install the necessary project dependencies:
    ```
    pip3 install -r requirements.txt
    ```
3. Run the interactive Python notebook to train/test the UWB model, ensuring that you've linked the notebook to the correct Python `virtualenv`. The following notebooks have been prepared:
    - Data Preprocessing
        - [preprocessing.ipynb ](/preprocessing.ipynb)
    - Data Mining
        - [data_mining_mlp.ipynb](/data_mining_mlp.ipynb)
        - [data_mining_rf.ipynb](/data_mining_rf.ipynb) 
        - [data_mining_svm.ipynb](/data_mining_svm.ipynb) 
        - [data_mining_dbscan.ipynb](/data_mining_dbscan.ipynb) 
    - Analysis
        - [data_mining_train_test_time.ipynb](/data_mining_train_test_time.ipynb) 
        - [evaluation.ipynb](/evaluation.ipynb) 
