# DurianLeafDisease

A modern deep learning codebase for **multi-task and single-task classification** of durian leaf diseases and severity levels. It combines custom-built convolutional neural networks (CNNs), Depth Estimation (Depth Anything V2), Geo-GradCAM 3D, Weather forecasting integration, and Large Language Model (Qwen3.6-27B via Groq) case-based treatment recommendations.
---

## Highlight Features

- **Multi-task & Single-task Learning**: Jointly predicts **disease type** and **severity level** simultaneously in one forward pass, or trains dedicated single-task networks.
- **Custom CNN Architectures from Scratch**: Supports 7 lightweight to medium architectures: MobileNetV2, MobileNetV3 (Small/Large), ResNet18, ResNet50, DenseNet121, and TickNet-Large.
- **Geo-GradCAM 3D Visualisation**: Reconstructs depth using *Depth Anything V2*, aligns it with Grad-CAM saliency maps, and renders an interactive 3D surface plot using Plotly.
- **Context-Aware Recommendations**: Leverages a local Case-Based Reasoning (CBR) database combined with real-time weather forecasts (via Open-Meteo) and geo-coordinates (via Nominatim reverse-geocoding) to generate customised treatment plans using Qwen3.6-27B.
- **Premium Streamlit Interface**: An interactive web app featuring browser geolocation, real-time weather risks, depth estimation overlays, interactive 3D graphics, and printable diagnostic summaries.
---
# Output Pipeline

The complete output pipeline of the proposed system is illustrated below.

---

### 1. Disease & Severity Prediction
Predict the disease category and severity level using the trained CNN-based multi-task model.

<p align="center">
  <img src="image.png" alt="Disease and Severity Prediction" width="800">
</p>

---

### 2. Weather Data Acquisition
Retrieve real-time weather information based on the user's location.

<p align="center">
  <img src="image-1.png" alt="Weather Data Acquisition" width="800">
</p>

---

### 3. Grad-CAM Visualization
Generate a 2D Grad-CAM heatmap to highlight the image regions that contribute most to the prediction.

<p align="center">
  <img src="image-2.png" alt="Grad-CAM Visualization" width="800">
</p>

---

### 4. Geo-GradCAM 3D Visualization
Visualize disease localization in a 3D representation for enhanced interpretability.

<p align="center">
  <img src="image-3.png" alt="Geo-GradCAM 3D Visualization" width="800">
</p>

---

### 5. Treatment Recommendation
Generate treatment recommendations based on the predicted disease, severity level, and current weather conditions.

<p align="center">
  <img src="image-4.png" alt="Treatment Recommendation" width="800">
</p>

---

### 6. Supporting References
Provide reliable references supporting the recommended treatment strategies.

<p align="center">
  <img src="image-5.png" alt="Treatment References" width="800">
</p>

## Project Directory Structure

```
DurianLeafDisease/
├── app/                          # Interactive Streamlit Interface
│   ├── app.py                    # Streamlit web app main script
│   └── __init__.py
│
├── pipeline/                     # Inference & Recommendation Engine
│   ├── pipeline.py               # Combined pipeline (CNN inference + Depth + GradCAM 3D + LLM)
│   ├── recommender.py            # Case-Based Reasoning (CBR) logic + Groq LLM client
│   ├── weather_kb.py             # Open-Meteo weather forecast client & rule-based weather risks
│   ├── knowledge_base/           # Treatment databases and weather risk templates (JSON)
│   │   ├── durian_leaf_case_based_recommendation_kb.json
│   │   └── durian_leaf_weather_risk_kb.json
│   └── __init__.py
│
├── models/                       # Custom CNN Architectures & Neural Modules
│   ├── MAFC.py                   # Multi-scale Attention Feature Calibration module
│   ├── mobilenetv3_custom.py     # MobileNetV2 / MobileNetV3 (Multi-task variants)
│   ├── mobilenetv3_single.py     # MobileNetV2 / MobileNetV3 (Single-task variants)
│   ├── resnet_custom.py          # ResNet18 / ResNet50 (Multi-task variants)
│   ├── resnet_single.py          # ResNet18 / ResNet50 (Single-task variants)
│   ├── densenet_custom.py        # DenseNet121 (Multi-task)
│   ├── densenet_single.py        # DenseNet121 (Single-task)
│   ├── ticknet_custom.py         # TickNet-Large (Multi-task)
│   ├── ticknet_single.py         # TickNet-Large (Single-task)
│   └── __init__.py
│
├── training/                     # Training & Evaluation Framework
│   ├── config.py                 # Global constants, hyperparameter defaults, dataset paths
│   ├── dataset.py                # Dataset loader for Multi-task learning
│   ├── dataset_single.py         # Dataset loader for Single-task learning
│   ├── train.py                  # Multi-task training orchestration script
│   ├── train_single.py           # Single-task training orchestration script
│   ├── evaluate.py               # Multi-task test evaluation & metric plotting
│   ├── evaluate_single.py        # Single-task test evaluation & metric plotting
│   ├── utils.py                  # Helper functions (profiling, parameter & FLOPs counting, seed)
│   └── __init__.py
│
├── experiments/                  # LLM & Knowledge Base Experiments
│   ├── llm_unconstrained_experiment.py  # Constrained vs. Unconstrained recommendation comparison
│   └── __init__.py
│
├── data/                         # Dataset Directory (Excluded from git)
│   └── Durian_Leaf_Diseases/
│       ├── train/                # Training images + metadata_train.csv
│       ├── val/                  # Validation images + metadata_val.csv
│       └── test/                 # Test images + metadata_test.csv
│
├── checkpoints/                  # Saved weights (.pth) & training logs (.json)
├── results/                      # Evaluation metrics and confusion matrices
├── Grad-CAM/                     # Local Grad-CAM 2D image overlays
├── output/                       # Saved Geo-GradCAM 3D HTMLs & output assets
└── .env                          # Local environment configurations (GROQ API key)
```

---

## Installation & Setup

1. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   *Required key packages: `torch`, `torchvision`, `transformers`, `streamlit`, `streamlit-geolocation`, `plotly`, `opencv-python`, `pandas`, `pillow`, `scikit-learn`, `matplotlib`, `tqdm`, `python-dotenv`.*

2. **Configure Groq API Key**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   *Note: Weather forecasting is powered by Open-Meteo and coordinates lookup by OpenStreetMap Nominatim, both of which are free and do not require API keys.*

3. **Data Preparation**:
   Place the **Durian Leaf Diseases Dataset (DLDD)** under `data/Durian_Leaf_Diseases/`. Ensure the directories `train/`, `val/`, and `test/` along with their corresponding CSV files are present.

---

## Running the Project

> [!IMPORTANT]
> **Always run the scripts from the project root directory** (e.g., `DurianLeafDisease/`) to ensure the modular imports (`models.*`, `training.*`, `pipeline.*`) work correctly.

### 1. Start the Streamlit Web Application
Run the interactive web app locally from the project root:
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` on your browser. 
- Adjust parameters (resolution, Depth Map resolution, Grad-CAM attention boost) from the sidebar.
- Enable Geolocation to automatically fetch current weather data and calculate treatment risk based on humidity and rainfall.

### 2. Run Training
To train a model, use the `python -m` syntax from the project root:

- **Multi-task Training (Disease + Severity)**:
  ```bash
  python -m training.train --model mobilenetv2 --epochs 30 --batch_size 16 --lr 1e-4
  ```
- **Single-task Training (Disease OR Severity)**:
  ```bash
  python -m training.train_single --model resnet18 --task disease --epochs 30 --batch_size 16
  ```

#### Command-line Options:
| Argument | Options / Defaults | Description |
|---|---|---|
| `--model` | `mobilenetv2`, `mobilenetv3_small`, `mobilenetv3_large`, `resnet18`, `resnet50`, `densenet121`, `ticknet_large` | Model architecture (Required) |
| `--task` | `disease`, `severity` | Prediction target (For single-task training only) |
| `--epochs` | Default: `60` | Number of training epochs |
| `--batch_size` | Default: `32` | Batch size |
| `--image_size` | Default: `224` | Input image size |
| `--lr` | Default: `1e-4` | Learning rate |
| `--use_class_weight` / `--use_severity_weight` | Flags | Enable class-weighted loss (for imbalanced classes) |

### 3. Model Evaluation
Evaluate a trained model checkpoint on the test set and generate classification reports:

- **Multi-task Evaluation**:
  ```bash
  python -m training.evaluate --model mobilenetv2 --run_dir mobilenetv2_img224_bs32_lr0.0001
  ```
- **Single-task Evaluation**:
  ```bash
  python -m training.evaluate_single --model resnet18 --task disease --run_dir resnet18_task_disease_img224_bs16_lr0.0001
  ```
Outputs (classification reports, metric JSONs, and confusion matrices) are stored in `results/`.

### 4. Direct Inference CLI & API
You can run the end-to-end inference pipeline on a single leaf image from the command line:
```bash
python -m pipeline.pipeline
```
*Note: The CLI is configured via the entry point inside `pipeline/pipeline.py`.*

To integrate inference inside your custom Python scripts:
```python
from pipeline.pipeline import run_pipeline

results = run_pipeline(
    image_path="data/Durian_Leaf_Diseases/test/algal/DLDD_TEST_000006.jpg",
    checkpoint_path="checkpoints/v2batch16_best/best_mobilenetv3_multitask.pth",
    tasks=("disease", "severity"),
    img_size=224,
    max_size=200,
    attention_boost=0.4,
    output_dir="output",
    save_2d_overlay=True,
    save_3d_html=True,
    use_weather=True
)

print(f"Disease Class : {results['disease_label']} ({results['disease_confidence']:.2%})")
print(f"Severity Level: {results['severity_label']} ({results['severity_confidence']:.2%})")
print(f"Treatment Recommendations: {results['recommendation']['recommendation_text']}")
```

### 5. LLM Recommendation Experiment (Constrained vs. Unconstrained)

To validate the hallucination reduction and safety benefits of our Case-Based Reasoning (CBR) Knowledge Base pipeline against unguided LLM generation, reviewers can reproduce the comparative experiment across all 78 KB disease cases:

```bash
# 1. Quick test run with 3 cases:
python -m experiments.llm_unconstrained_experiment --limit 3

# 2. Full experiment across all 78 cases (requires GROQ_API_KEY):
python -m experiments.llm_unconstrained_experiment --model qwen/qwen3.6-27b

# 3. Instant analysis from pre-computed cache (no API calls needed):
python -m experiments.llm_unconstrained_experiment --only-analysis

# 4. Long-running resilient execution with automatic rate-limit backoff:
powershell -ExecutionPolicy Bypass -File run_experiment_resume.ps1
```

#### Command-line Options for Experiment:
| Argument | Options / Defaults | Description |
|---|---|---|
| `--limit` | Default: `0` (all 78 cases) | Maximum number of cases to evaluate |
| `--model` | Default: `qwen/qwen3.6-27b` | Groq LLM model name (e.g. `qwen/qwen3.6-27b`) |
| `--only-analysis` | Flag | Run statistical analysis from cached `raw_cases.json` without API requests |
| `--force` | Flag | Ignore existing cache and re-run all API generations |
| `--sleep` | Default: `0.2` | Delay in seconds between API requests |
| `--temperature` | Default: `0.3` | LLM generation sampling temperature |

#### Experiment Outputs:
Generated results are stored in `results/llm_experiment/`:
- `cases_detail.csv`: Full side-by-side texts generated by both constrained and unconstrained modes, together with hallucination flags for each case.
- `statistics_summary.csv`: Aggregated comparison metrics (overall, broken down by disease type and severity level) covering pesticide mentions, specific chemical names, numeric dosage hallucination, and weather fabrication rates.

---

## Dataset Specifications

The Durian Leaf Diseases Dataset is structured into:
- **Disease types**: `healthy`, `algal` (tảo lục), `allocaridara_attack` (rầy nhảy), `blight` (thối lá), `phomopsis` (đốm lá Phomopsis).
- **Severity levels**: `severity_0` (Khỏe mạnh), `severity_1` (Nhẹ), `severity_2` (Trung bình), `severity_3` (Nặng).

---

## Checkpoint Format
Saved `.pth` model weights contain:
```python
{
    "model_name": str,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "epoch": int,
    "train_loss": float,
    "val_loss": float,
    "val_disease_acc": float,
    "val_severity_acc": float,
    "flops_readable": str,
    "params_readable": str,
    "image_size": int,
    "batch_size": int,
}
```

---
*Created for durian farm diagnostics.*
