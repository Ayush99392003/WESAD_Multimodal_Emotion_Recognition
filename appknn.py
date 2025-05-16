import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler  
import joblib
import scipy.stats
import scipy.signal
from collections import defaultdict
import hashlib

class WESADProcessor:
    def __init__(self, dataset_path, cache_dir="processed_cache"):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define all available sensors
        self.chest_sensors = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensors = ['ACC', 'BVP', 'EDA', 'TEMP']
        # Sampling frequencies
        self.chest_fs = 700  # Chest device samples at 700Hz
        self.wrist_fs = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
        self.valid_labels = {
            1: 'baseline', 
            2: 'stress', 
            3: 'amusement'
        }

    def get_cache_key(self, subject, combination, window_sec, stride_sec):
        """Generate unique cache key for this processing configuration"""
        config_str = f"{subject}_{combination['name']}_{window_sec}_{stride_sec}"
        return hashlib.md5(config_str.encode()).hexdigest() + ".joblib"

    def load_subject(self, subject):
        """Load subject data with validation checks"""
        subject_path = os.path.join(self.dataset_path, subject)
        pkl_path = os.path.join(subject_path, f"{subject}.pkl")
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing data for {subject}")
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        # Validate data structure
        if not all(k in data for k in ['signal', 'label']):
            raise ValueError(f"Invalid data structure for {subject}")
            
        # Convert to numpy arrays
        for device in ['chest', 'wrist']:
            sensors = getattr(self, f'{device}_sensors')
            for sensor in sensors:
                if sensor in data['signal'][device]:
                    if sensor == 'ACC' and device == 'wrist':
                        # ACC is already 3D array (n_samples, 3)
                        data['signal'][device][sensor] = np.array(data['signal'][device][sensor])
                    else:
                        data['signal'][device][sensor] = np.array(data['signal'][device][sensor]).flatten()
        
        data['label'] = np.array(data['label'])
        return data

    def extract_features(self, signal):
        """Extract 12 statistical and spectral features"""
        with np.errstate(all='ignore'):
            features = [
                np.mean(signal),
                np.std(signal),
                np.median(signal),
                scipy.stats.skew(signal),
                scipy.stats.kurtosis(signal),
                np.max(signal) - np.min(signal),  # Range
                np.sum(np.abs(np.diff(signal))) / len(signal),  # Mean absolute difference
                len(scipy.signal.find_peaks(signal)[0]),  # Peak count
                np.quantile(signal, 0.95) - np.quantile(signal, 0.05),  # IQR95
                np.mean(np.abs(np.fft.fft(signal))[:len(signal)//2]),  # FFT magnitude mean
                np.percentile(signal, 25),  # Q1
                np.percentile(signal, 75)   # Q3
            ]
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    def process_subject(self, subject, combination, window_sec=4, stride_sec=2):
        """Process subject with caching support"""
        cache_file = os.path.join(self.cache_dir, self.get_cache_key(subject, combination, window_sec, stride_sec))
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                cached_data = joblib.load(cache_file)
                print(f"Loaded cached data for {subject} ({combination['name']})")
                return cached_data['X'], cached_data['y'], cached_data['groups']
            except:
                print(f"Cache corrupted for {subject}, reprocessing...")
                os.remove(cache_file)
        
        # Process if no cache exists
        data = self.load_subject(subject)
        chest_window_size = window_sec * self.chest_fs
        stride = stride_sec * self.chest_fs
        n_samples = len(data['signal']['chest']['ECG'])  # Reference signal
        
        X, y, groups = [], [], []
        
        for start in range(0, n_samples - chest_window_size + 1, stride):
            end = start + chest_window_size
            
            # Get labels in current window (must be homogeneous)
            window_labels = data['label'][start:end]
            valid_labels = window_labels[np.isin(window_labels, [1, 2, 3])]  # Only accept 1,2,3
            if len(np.unique(valid_labels)) != 1:
                continue  # Skip mixed-label windows
                
            label = valid_labels[0]
            features = []
            valid = True
            
            # Process chest sensors from combination
            for sensor in combination['chest']:
                if sensor not in data['signal']['chest']:
                    continue
                window = data['signal']['chest'][sensor][start:end]
                features.extend(self.extract_features(window))
            
            # Process wrist sensors from combination
            t_start = start / self.chest_fs
            t_end = end / self.chest_fs
            
            for sensor in combination['wrist']:
                if sensor not in data['signal']['wrist']:
                    continue
                    
                fs = self.wrist_fs[sensor]
                s_start = int(t_start * fs)
                s_end = int(t_end * fs)
                
                if sensor == 'ACC':
                    # Process each accelerometer axis separately
                    for axis in range(3):
                        axis_data = data['signal']['wrist']['ACC'][s_start:s_end, axis]
                        features.extend(self.extract_features(axis_data))
                else:
                    sensor_data = data['signal']['wrist'][sensor][s_start:s_end]
                    features.extend(self.extract_features(sensor_data))
            
            if valid and features:
                X.append(features)
                y.append(label)
                groups.append(hash(subject))  # Unique group per subject
        
        result = {
            'X': np.array(X),
            'y': np.array(y),
            'groups': np.array(groups)
        }
        
        # Save to cache
        joblib.dump(result, cache_file)
        print(f"Cached processed data for {subject} ({combination['name']})")
        
        return result['X'], result['y'], result['groups']

def evaluate_model(model, X_test, y_test, class_names):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                 target_names=class_names,
                                 output_dict=True,
                                 zero_division=0)
    
    # Calculate balanced accuracy
    bal_acc = sum(
        accuracy_score(y_test[y_test == cls], y_pred[y_test == cls])
        for cls in np.unique(y_test)
    ) / len(np.unique(y_test))
    
    overall_f1 = np.mean([report[cls]['f1-score'] for cls in class_names])

    return accuracy, bal_acc, overall_f1, report


def main():
    DATASET_PATH = "WESAD"
    SUBJECTS = [f'S{i}' for i in [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]]
    WINDOW_SEC = 4
    STRIDE_SEC = 2
    # Define all modality combinations to evaluate
    MODALITY_COMBINATIONS = [
        # Wrist individual modalities
        {'name': 'wrist_ACC', 'chest': [], 'wrist': ['ACC']},
        {'name': 'wrist_BVP', 'chest': [], 'wrist': ['BVP']},
        {'name': 'wrist_EDA', 'chest': [], 'wrist': ['EDA']},
        {'name': 'wrist_TEMP', 'chest': [], 'wrist': ['TEMP']},
        
        # Chest individual modalities
        {'name': 'chest_ACC', 'chest': ['ACC'], 'wrist': []},
        {'name': 'chest_ECG', 'chest': ['ECG'], 'wrist': []},
        {'name': 'chest_EDA', 'chest': ['EDA'], 'wrist': []},
        {'name': 'chest_EMG', 'chest': ['EMG'], 'wrist': []},
        {'name': 'chest_Resp', 'chest': ['Resp'], 'wrist': []},
        {'name': 'chest_Temp', 'chest': ['Temp'], 'wrist': []},
        
        # Device combinations
        {'name': 'wrist_all', 'chest': [], 'wrist': ['ACC','BVP','EDA','TEMP']},
        {'name': 'chest_all', 'chest': ['ACC','ECG','EDA','EMG','Resp','Temp'], 'wrist': []},
        
        # Physiological without ACC
        {'name': 'wrist_phys', 'chest': [], 'wrist': ['BVP','EDA','TEMP']},
        {'name': 'chest_phys', 'chest': ['ECG','EDA','EMG','Resp','Temp'], 'wrist': []},
        
        # Combined devices
        {'name': 'all_modalities', 'chest': ['ACC','ECG','EDA','EMG','Resp','Temp'], 
         'wrist': ['ACC','BVP','EDA','TEMP']},
        {'name': 'all_phys', 'chest': ['ECG','EDA','EMG','Resp','Temp'], 
         'wrist': ['BVP','EDA','TEMP']}
    ]
    
    processor = WESADProcessor(DATASET_PATH)
    class_names = list(processor.valid_labels.values())
    
    print("=== WESAD Modality Combination Evaluation (KNN) ===")
    print(f"Subjects: {len(SUBJECTS)} | Window: {WINDOW_SEC}s | Stride: {STRIDE_SEC}s")
    
    final_results = []
    
    for combo in tqdm(MODALITY_COMBINATIONS, desc="Evaluating Combinations"):
        print(f"\n=== Processing {combo['name']} ===")
        # Process all subjects with current combination
        combo_X, combo_y, combo_groups = [], [], []
        for subject in SUBJECTS:
            try:
                X, y, groups = processor.process_subject(subject, combo, WINDOW_SEC, STRIDE_SEC)
                combo_X.append(X)
                combo_y.append(y)
                combo_groups.append(groups)
            except Exception as e:
                print(f"Skipped {subject}: {str(e)}")
                continue
        
        if not combo_X:
            print(f"No valid data for {combo['name']}")
            continue
            
        X = np.vstack(combo_X)
        y = np.concatenate(combo_y)
        groups = np.concatenate(combo_groups)
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass Distribution for {combo['name']}:")
        for cls, count in zip(unique, counts):
            label_name = processor.valid_labels.get(cls, f'unknown_{cls}')
            print(f"{label_name}: {count} samples ({count/len(y):.1%})")
        
        # GroupKFold Cross-Validation
        gkf = GroupKFold(n_splits=5)
        fold_results = defaultdict(list)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Add feature scaling for KNN
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize KNN model
            model = KNeighborsClassifier(
                n_neighbors=5,       # Default neighbors
                weights='distance', # Weight by distance
                n_jobs=-1           # Parallel processing
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate with scaled features
            acc, bal_acc, overall_f1, report = evaluate_model(model, X_test_scaled, y_test, class_names)
            # Store results
            fold_results['combination'].append(combo['name'])
            fold_results['fold'].append(fold+1)
            fold_results['accuracy'].append(acc)
            fold_results['balanced_accuracy'].append(bal_acc)
            fold_results['overall_f1'].append(overall_f1)
            
            # Store class-wise metrics
            for cls in class_names:
                fold_results[f'{cls}_precision'].append(report[cls]['precision'])
                fold_results[f'{cls}_recall'].append(report[cls]['recall'])
                fold_results[f'{cls}_f1'].append(report[cls]['f1-score'])
        
        # Save results for this combination
        combo_df = pd.DataFrame(fold_results)
        final_results.append(combo_df)
    
    # Combine all results
    final_df = pd.concat(final_results)
    final_df.to_csv('wesad_modality_comparison_results.csv', index=False)
    
    # Generate summary statistics
    summary = final_df.groupby('combination').agg({
    'accuracy': ['mean', 'std'],
    'overall_f1': ['mean', 'std'],
    'balanced_accuracy': ['mean', 'std'],
    'baseline_f1': ['mean', 'std'],
    'stress_f1': ['mean', 'std'],
    'amusement_f1': ['mean', 'std']
})
    
    print("\n=== Final Results Summary ===")
    print(summary)
    
    # Save detailed report
    with open('modality_comparison_summary.txt', 'w') as f:
        f.write("=== WESAD Modality Combination Results ===\n\n")
        f.write(summary.to_string())
        
        # Add best performing combinations
        best_acc = summary[('accuracy', 'mean')].idxmax()
        best_bal_acc = summary[('balanced_accuracy', 'mean')].idxmax()
        best_f1 = summary[('overall_f1', 'mean')].idxmax()

        f.write("\n\n=== Best Performing Combinations ===\n")
        f.write(f"Highest Accuracy: {best_acc} ({summary.loc[best_acc, ('accuracy', 'mean')]:.4f})\n")
        f.write(f"Highest Balanced Accuracy: {best_bal_acc} ({summary.loc[best_bal_acc, ('balanced_accuracy', 'mean')]:.4f})\n")
        f.write(f"Highest Overall F1-Score: {best_f1} ({summary.loc[best_f1, ('overall_f1', 'mean')]:.4f})\n")

    print("\nAnalysis complete. Results saved to:")
    print("- wesad_modality_comparison_results.csv")
    print("- modality_comparison_summary.txt")
    print(f"- Processed data cache: {processor.cache_dir}")

if __name__ == "__main__":
    main()