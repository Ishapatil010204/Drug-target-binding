<<<<<<< HEAD
# 🧬 DrugLikeliness12

GPU-accelerated Drug–Target Binding Prediction using **XGBoost (CUDA)**. This project trains models on the **BindingDB dataset** to predict binding affinity (regression) and binding likelihood (classification). It includes a Streamlit application for inference and is optimized for NVIDIA GPUs with **CUDA 12.8+**.

---

## 🚀 Features

* ⚡ **GPU-Accelerated Training:** Utilizes XGBoost built with CUDA 12.8 support via the NVIDIA PyPI index.
* 🧠 **Dual-Task Prediction:** Trains both a classifier (`y_class`) and a regressor (`y_reg`).
* 🧩 **Data Pipeline:** Preprocessing and feature extraction steps handled within the training notebooks. Includes skip-logic to reuse processed data.
* 💾 **Model Persistence:** Saves trained models (`.pkl`), scalers, and feature extractors.
* 🖥️ **Interactive App:** A Streamlit app (`src/app.py`) for loading models and making predictions.

---

```
## 🏗️ Project Structure
.
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
├── data/                   # Datasets
│   ├── raw/                # Original data files
│   │   └── BindingDB_All.tsv
│   └── processed/          # Cleaned and preprocessed data
│       ├── cleaned_bindingdb_data.csv
│       ├── X_features.npy
│       ├── y_class.npy
│       └── y_reg.npy
│
├── models/                 # Trained models and preprocessing tools
│   ├── classifier.pkl
│   ├── feature_extractor.pkl
│   ├── regressor.pkl
│   └── scaler.pkl
│
├── notebooks/              # Jupyter notebooks for exploration and training
│   ├── dataset.ipynb
│   ├── train.ipynb
│   └── trainv2.ipynb
│
└── src/                    # Source code
└── app.py              # Main application script
```

## ⚙️ Setup Instructions

**Prerequisites:**
* NVIDIA GPU with **CUDA 12.8** or later installed.
* Matching NVIDIA drivers.
* Conda package manager.

**Steps:**

1.  **Create and activate Conda environment:**
    ```bash
    conda create -n dti_gpu python=3.11
    conda activate dti_gpu
    ```

2.  **Install dependencies:**
    * Install PyTorch with CUDA 12.8 support:
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
        ```
    * Install the NVIDIA-built XGBoost:
        ```bash
        pip install xgboost==2.1.4 --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com)
        ```
    * Install remaining requirements from the file:
        ```bash
        pip install -r requirements.txt
        ```
    * **Important:** Downgrade NumPy due to a dependency conflict (likely with RDKit if used, or another library):
        ```bash
        pip install "numpy<2"
        ```

---

## ▶️ Usage

1.  **Prepare Data:**
    * Place the `BindingDB_All.tsv` file into the `data/raw/` directory.

2.  **Train Models:**
    * Choose one of the training notebooks:
        * `notebooks/train.ipynb`: Run cells sequentially from top to bottom.
        * `notebooks/trainv2.ipynb`: Run cells step-by-step.
    * Execute all cells within the chosen notebook. This will preprocess data, train models, and save outputs to `data/processed/` and `models/`.

3.  **Run Inference Application:**
    * Launch the Streamlit app from the project's **root directory**:
        ```bash
        streamlit run src/app.py
        ```
    * Access the app via the provided `localhost` URL in your browser.

---

## 📁 Outputs

* **Models (`models/`):**
    * `classifier.pkl`
    * `regressor.pkl`
    * `scaler.pkl`
    * `feature_extractor.pkl`
* **Processed Data (`data/processed/`):**
    * `cleaned_bindingdb_data.csv`
    * `X_features.npy`
    * `y_class.npy`
    * `y_reg.npy`

---

## 🛠️ Key Dependencies (from `requirements.txt`)

| Component      | Version          | Purpose                   |
| -------------- | ---------------- | ------------------------- |
| Python         | 3.11             | Core runtime              |
| PyTorch        | 2.9.0+cu128      | CUDA dependency / RDKit   |
| XGBoost        | 2.1.4 (NVIDIA)   | GPU Training              |
| scikit-learn   | 1.7.2            | Preprocessing & Metrics   |
| NumPy          | <2.0 (e.g., 1.26.4)| Data / Dependency Fix     |
| Pandas         | 2.2.3            | Data Handling             |
| Streamlit      | 1.39.0           | Web Application           |
| RDKit          | *(Implied)* | Cheminformatics (in App)  |
| TQDM           | 4.67.1           | Progress Bars             |
| Jupyter        | *(Implied)* | Notebook Environment      |

---

## 📝 Notes

* Ensure NVIDIA drivers and CUDA toolkit align with library requirements (CUDA 12.8).
* The `.gitignore` file should exclude large data (`.tsv`, `.csv`), features (`.npy`), and model (`.pkl`) files.
=======
# Drug-target-binding
>>>>>>> f37c5d3110611be2588c4bc5ef471b921d03ef1b
