📊 WESAD Multimodal Emotion Recognition
This repository contains implementations of several machine learning models (LDA, Random Forest, Decision Tree, AdaBoost, KNN) for emotion classification using the WESAD dataset. Each script evaluates various combinations of chest and wrist sensor modalities.

📁 Contents
File Name	Model Type
app_lda.py	Linear Discriminant Analysis (LDA)
app_rf.py	Random Forest
appdt.py	Decision Tree
appada.py	AdaBoost with max depth = 100
appknn.py	K-Nearest Neighbors

🧪 Dataset
Dataset used: WESAD - Wearable Stress and Affect Detection

Ensure the WESAD dataset folder is placed in the same directory as the scripts, with subject .pkl files properly organized.

🛠️ How to Run
Each script is standalone. You can run them via:

bash
Copy code
python app_lda.py
python app_rf.py
python appdt.py
python appada.py
python appknn.py
Results will be saved in:

CSV format: accuracy and F1-scores across folds

Text summary files for quick insight

📈 Output
wesad_*_modality_comparison_results.csv: detailed metrics across all modality combinations.

*_summary.txt: summary of the best-performing combinations.

📦 Dependencies
Install dependencies with:

bash
Copy code
pip install numpy pandas scikit-learn scipy tqdm joblib
🧠 Modalities Evaluated
Each script processes combinations of:

Chest sensors: ACC, ECG, EDA, EMG, Resp, Temp

Wrist sensors: ACC, BVP, EDA, TEMP

Modalities include:

Single sensors

All sensors (per device)

Physiological (excluding ACC)

Combined chest & wrist

🔼 How to Upload This Project to GitHub
Initialize Git:

bash
Copy code
git init
Add Files:

bash
Copy code
git add .
Commit Changes:

bash
Copy code
git commit -m "Initial commit - Added WESAD model evaluation scripts"
Create a GitHub Repo:

Go to GitHub.

Create a new repository (do NOT initialize with README if you already have one).

Push to GitHub:

bash
Copy code
git remote add origin https://github.com/your-username/your-repo-name.git
git branch -M main
git push -u origin main
