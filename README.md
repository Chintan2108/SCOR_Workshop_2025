# SCOR Workshop 2025: Machine Learning for Aquatic Remote Sensing

**Event:** International Science Council (ISC) SCOR Workshop on Satellite Remote Sensing  
**Location:** Department of Marine Sciences, Berhampur University, India  
**Instructor:** Chintan B. Maniyar (PhD Candidate, UGA Geography)

---

## 🌊 About This Repository
This repository contains the official training materials, codebases, and presentation slides for the **"Machine Learning for Aquatic Remote Sensing"** sessions delivered at the SCOR 2025 Workshop.

The curriculum is designed to bridge the gap between **Optical Physics** and **Data Science**, moving beyond standard black-box implementation to **Scientific Machine Learning (SciML)**—models that are robust, explainable, and physically consistent.

### 📚 Sessions Covered
* **Session 1: The Landscape:** Theory, Optical Water Types (OWTs), and the taxonomy of ML algorithms (RF, SVM, NNs).
* **Session 2: Hands-on Implementation:** A complete Python workflow from data preprocessing to training a physics-informed Neural Network and generating water quality maps.


## 📂 Repository Structure

| Folder/File | Description |
| :--- | :--- |
| `notebooks/` | Jupyter Notebooks for the hands-on session. Start with `01_EDA`. |
| `data/` | Contains the dataset csvs. **Note:** Large satellite imagery is hosted externally (see `satellite_image_download_link.txt` inside this folder for the GDrive link). |
| `models/` | Pre-trained models (`.pth` for PyTorch, `.joblib` for sklearn) for quick inference. |
| `slides and brochures/` | PDFs of the lecture slides and workshop brochures. |
| `utils.py` | Helper functions for visualization and metrics (custom plotting scripts). |
| `scor_2025_python_env.yml` | Conda environment file for Linux/Mac. |
| `scor_2025_python_env_cross-plt.yml` | Cross-platform Conda environment file (Recommended for Windows). |

---

## 🚀 Getting Started

### 1. Prerequisites
You will need **Git** and **Conda** (Anaconda or Miniconda) installed on your system.

### 2. Installation
To run the materials locally, clone this repository and set up the Python environment.

```bash
# 1. Clone the repository
git clone [https://github.com/Chintan2108/SCOR_Workshop_2025.git](https://github.com/Chintan2108/SCOR_Workshop_2025.git)
cd SCOR_Workshop_2025

# 2. Create the environment
# For Windows users (recommended):
conda env create -f scor_2025_python_env_cross-plt.yml

# For Linux/Mac users:
conda env create -f scor_2025_python_env.yml

# 3. Activate the environment
conda activate scor_2025
```

### 3. Installation
Once the environment is active, launch Jupyter Lab to access the notebooks:
```bash
jupyter lab
```

Navigate to the ```notebooks/``` folder and open 01_EDA.ipynb.

## 🧪 Scientific Context
The code demonstrated here addresses specific challenges in **Case-2 (Complex) Waters**:
- Log-Normal Distributions: Handling the skewed distribution of Chlorophyll-a.
- Feature Engineering: Utilizing spectral indices rather than raw bands.
- Explainable AI: Using Feature Importance to validate physical drivers.

**Disclaimer**: These models are developed for educational demonstration. While they follow scientific best practices, rigorous validation is required before applying them to new geographic regions or operational monitoring.


## 🤝 Acknowledgments
- International Science Council (ISC) & SCOR: For organizing the workshop.
- Berhampur University: For hosting the event.
- NASA FINESST Program: For supporting the instructor's research.


📬 Contact
Chintan B. Maniyar: chintanmaniyar@uga.edu  |  chintanmaniyar@gmail.com


## 🎓 For Educators & Instructors
**Please steal this code!**

This workshop was built with the specific goal of democratizing access to scientific machine learning in aquatic remote sensing. We strongly encourage university professors, workshop organizers, and lab PIs to use these slides, notebooks, and datasets in their own curriculum.

* **You are free to:** Adapt, remix, and present these materials in your own classrooms or training sessions.
* **We ask that you:** Please provide a link back to this repository so students can access the latest updates and reference the original source.

**Suggested attribution for lecture notes:**
> *"Materials adapted from the SCOR 2025 Workshop on Machine Learning for Aquatic Remote Sensing developed by Chintan B. Maniyar (University of Georgia)."*

---

*Center for Geospatial Research, University of Georgia*
