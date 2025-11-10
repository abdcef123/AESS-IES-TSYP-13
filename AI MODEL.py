"""
CubeSat Complete ML Pipeline
Train AI models AND make predictions in one script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

# Fix encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("CubeSat Complete ML Pipeline - Training & Prediction")
print("=" * 70)

# ========== STEP 1: LOAD DATA ==========
print("\n[STEP 1] Loading telemetry data...")

import os
import glob
from pathlib import Path

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Try multiple locations
search_paths = [
    os.getcwd(),
    '.',
    script_dir,
    r'C:\Users\21650\Downloads\FDIR_Neural_Networks-main\FDIR_Neural_Networks-main\Cubesat_Simulator',
    r'C:\Users\21650\Downloads',
]

csv_file = None
all_csv_files = []

# Search all paths for CSV files
for search_path in search_paths:
    if os.path.exists(search_path):
        found_files = glob.glob(os.path.join(search_path, 'CubeSat_Telemetry_*.csv'))
        all_csv_files.extend(found_files)

# Remove duplicates
all_csv_files = list(set(all_csv_files))

if all_csv_files:
    # Sort by file SIZE (largest first) - newer data has more samples
    all_csv_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    csv_file = all_csv_files[0]
    
    file_size = os.path.getsize(csv_file)
    file_time = os.path.getmtime(csv_file)
    from datetime import datetime
    file_datetime = datetime.fromtimestamp(file_time)
    
    print(f"  [OK] Found {len(all_csv_files)} CSV file(s)")
    print(f"  [OK] Using LARGEST: {os.path.basename(csv_file)}")
    print(f"  [OK] Size: {file_size / 1024:.1f} KB")
    print(f"  [OK] Created: {file_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print(f"  [X] No CSV files found!")
    print(f"\n  Searched locations:")
    for path in search_paths:
        if os.path.exists(path):
            print(f"    - {path}")
    print(f"\n  Solutions:")
    print(f"    1. Run MATLAB script to generate CSV")
    print(f"    2. Move CSV to same folder as this Python script")
    exit(1)

try:
    df = pd.read_csv(csv_file)
    print(f"  [OK] Loaded {len(df)} samples, {len(df.columns)} features")
except Exception as e:
    print(f"  [X] Error loading CSV: {e}")
    exit(1)

# ========== STEP 2: DATA PREPROCESSING ==========
print("\n[STEP 2] Data preprocessing...")

features_to_use = [col for col in df.columns if col != 'Time']
X = df[features_to_use].values

# Detect anomalies using statistical method (per row)
from scipy import stats
z_scores = np.abs(stats.zscore(X, axis=0))  # Calculate z-score for each feature
anomaly_per_row = np.max(z_scores, axis=1)  # Get max z-score for each sample
threshold_z = 3
y = (anomaly_per_row > threshold_z).astype(int)

print(f"  [OK] Features: {len(features_to_use)}")
print(f"  [OK] Samples: {len(y)}")
print(f"  [OK] Anomalies detected: {np.sum(y)} ({100*np.sum(y)/len(y):.2f}%)")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data: 75% training, 25% temporary (for test+validation)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Split temporary 25% into: 12% test and 13% validation
# 12/25 = 0.48, 13/25 = 0.52
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.52, random_state=42, stratify=y_temp
)

print(f"  [OK] Training: {len(X_train)} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"  [OK] Testing: {len(X_test)} samples ({100*len(X_test)/len(X):.1f}%)")
print(f"  [OK] Validation: {len(X_val)} samples ({100*len(X_val)/len(X):.1f}%)")

# ========== STEP 3: TRAIN MODELS ==========
print("\n[STEP 3] Training AI models...\n")

models = {}

print("  Training Isolation Forest...", end=" ", flush=True)
models['isolation_forest'] = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
models['isolation_forest'].fit(X_train)
print("[OK]")

print("  Training One-Class SVM...", end=" ", flush=True)
models['ocsvm'] = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
models['ocsvm'].fit(X_train)
print("[OK]")

print("  Training Random Forest...", end=" ", flush=True)
models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
models['random_forest'].fit(X_train, y_train)
print("[OK]")

print("  Training Neural Network...", end=" ", flush=True)
models['neural_net'] = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), 
    max_iter=500, 
    random_state=42, 
    early_stopping=True,
    validation_fraction=0.13,  # Use validation set
    n_iter_no_change=10
)
models['neural_net'].fit(X_train, y_train)
print("[OK]")

# ========== STEP 4: EVALUATE MODELS ==========
print("\n[STEP 4] Model evaluation...\n")

results = {}

for model_name, model in models.items():
    print(f"  {model_name.upper()}")
    
    if model_name in ['isolation_forest', 'ocsvm']:
        y_pred_test = model.predict(X_test)
        y_pred_val = model.predict(X_val)
        y_pred_test = (y_pred_test == -1).astype(int)
        y_pred_val = (y_pred_val == -1).astype(int)
    else:
        y_pred_test = model.predict(X_test)
        y_pred_val = model.predict(X_val)
    
    test_acc = np.mean(y_pred_test == y_test)
    val_acc = np.mean(y_pred_val == y_val)
    cm = confusion_matrix(y_test, y_pred_test)
    
    print(f"    Test Accuracy: {test_acc:.4f}")
    print(f"    Validation Accuracy: {val_acc:.4f}")
    print(f"    Confusion Matrix: {cm.ravel()}")
    
    results[model_name] = {
        'test_acc': test_acc,
        'val_acc': val_acc,
        'y_pred': y_pred_test,
        'cm': cm
    }

# ========== STEP 5: SAVE MODELS ==========
print("\n[STEP 5] Saving trained models...")

for model_name, model in models.items():
    with open(f'cubesat_model_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"  [OK] cubesat_model_{model_name}.pkl")

with open('cubesat_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  [OK] cubesat_scaler.pkl")

with open('cubesat_features.pkl', 'wb') as f:
    pickle.dump(features_to_use, f)
print(f"  [OK] cubesat_features.pkl")

# ========== STEP 6: PREDICTIONS ON ALL DATA ==========
print("\n[STEP 6] Making predictions on all data...\n")

X_all_scaled = scaler.transform(X)
predictions = {}

for model_name, model in models.items():
    print(f"  {model_name}...", end=" ", flush=True)
    
    try:
        if model_name in ['isolation_forest', 'ocsvm']:
            y_pred = model.predict(X_all_scaled)
            y_pred = (y_pred == -1).astype(int)
        else:
            # Use batch prediction for neural network to avoid memory issues
            batch_size = 500
            y_pred_batches = []
            for i in range(0, len(X_all_scaled), batch_size):
                batch = X_all_scaled[i:i+batch_size]
                y_pred_batch = model.predict(batch)
                y_pred_batches.append(y_pred_batch)
            y_pred = np.concatenate(y_pred_batches)
        
        try:
            if hasattr(model, 'score_samples'):
                y_scores = -model.score_samples(X_all_scaled)
            elif hasattr(model, 'predict_proba'):
                # Batch prediction for proba too
                if model_name == 'neural_net':
                    y_scores_batches = []
                    for i in range(0, len(X_all_scaled), batch_size):
                        batch = X_all_scaled[i:i+batch_size]
                        y_proba_batch = model.predict_proba(batch)[:, 1]
                        y_scores_batches.append(y_proba_batch)
                    y_scores = np.concatenate(y_scores_batches)
                else:
                    y_scores = model.predict_proba(X_all_scaled)[:, 1]
            else:
                y_scores = y_pred.astype(float)
        except:
            y_scores = y_pred.astype(float)
        
        n_anom = np.sum(y_pred)
        print(f"Found {n_anom} anomalies")
        
        predictions[model_name] = {
            'labels': y_pred,
            'scores': y_scores
        }
    except Exception as e:
        print(f"Error: {e}")
        predictions[model_name] = {
            'labels': np.zeros(len(X_all_scaled), dtype=int),
            'scores': np.zeros(len(X_all_scaled))
        }

# ========== STEP 7: ENSEMBLE VOTING ==========
print("\n[STEP 7] Ensemble voting...")

ensemble_pred = np.zeros(len(X))
for model_name, pred_dict in predictions.items():
    ensemble_pred += pred_dict['labels']

ensemble_pred = (ensemble_pred >= 2).astype(int)
n_ensemble = np.sum(ensemble_pred)

print(f"  Ensemble detected: {n_ensemble} anomalies ({100*n_ensemble/len(ensemble_pred):.2f}%)")

# ========== STEP 8: SAVE RESULTS ==========
print("\n[STEP 8] Saving predictions...")

results_df = pd.DataFrame({
    'Time': df['Time'],
    'Ensemble_Anomaly': ensemble_pred,
})

for model_name in predictions.keys():
    results_df[f'{model_name}_Anomaly'] = predictions[model_name]['labels']

results_df.to_csv('cubesat_predictions.csv', index=False)
print(f"  [OK] cubesat_predictions.csv")

# ========== STEP 9: VISUALIZATION ==========
print("\n[STEP 9] Creating visualization...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Model accuracy comparison
    ax = axes[0, 0]
    model_names = list(results.keys())
    accs = [results[m]['test_acc'] for m in model_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax.bar(range(len(model_names)), accs, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy on Test Set')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ensemble predictions timeline
    ax = axes[0, 1]
    ensemble_scores = np.mean([predictions[m]['scores'] for m in model_names], axis=0)
    ax.plot(df['Time'], ensemble_scores, linewidth=1.5, color='blue', label='Score')
    anomaly_idx = np.where(ensemble_pred == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(df['Time'].iloc[anomaly_idx], ensemble_scores[anomaly_idx], 
                  color='red', s=100, zorder=5, label='Anomaly', marker='X')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    ax.set_title('Ensemble Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confusion matrix (best model)
    ax = axes[1, 0]
    best_model = max(results, key=lambda m: results[m]['test_acc'])
    cm = results[best_model]['cm']
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white', fontweight='bold', fontsize=14)
    ax.set_title(f'Confusion Matrix - {best_model.replace("_", " ").title()}')
    plt.colorbar(im, ax=ax)
    
    # Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
TRAINING SUMMARY

Total Samples: {len(df)}
Features: {len(features_to_use)}
Training Samples: {len(X_train)}
Test Samples: {len(X_test)}

BEST MODEL
{best_model.replace('_', ' ').upper()}
Accuracy: {results[best_model]['test_acc']:.4f}

ANOMALIES DETECTED
Ensemble: {n_ensemble} ({100*n_ensemble/len(ensemble_pred):.2f}%)

STATUS: TRAINING COMPLETE
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('cubesat_complete_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] cubesat_complete_analysis.png")
    plt.show()
except Exception as e:
    print(f"  [X] Visualization error: {e}")

# ========== FINAL SUMMARY ==========
print("\n" + "=" * 70)
print("COMPLETE PIPELINE FINISHED")
print("=" * 70)
print(f"\nModels Trained: {len(models)}")
print(f"Best Model: {best_model.replace('_', ' ').upper()}")
print(f"Test Accuracy: {results[best_model]['test_acc']:.4f}")
print(f"\nAnomalies Detected: {n_ensemble} / {len(df)} ({100*n_ensemble/len(df):.2f}%)")
print(f"\nOutput Files:")
print(f"  - cubesat_model_*.pkl (4 trained models)")
print(f"  - cubesat_scaler.pkl")
print(f"  - cubesat_features.pkl")
print(f"  - cubesat_predictions.csv")
print(f"  - cubesat_complete_analysis.png")
print("\n" + "=" * 70)