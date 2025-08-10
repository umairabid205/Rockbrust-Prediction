# Rockburst Prediction Model

## Random Forest Implementation with 24-Hour Auto-Retraining

This folder contains the organized Random Forest model implementation for rockburst intensity prediction.

## 📁 File Structure

### Core Model Files

- **`model.py`** - Random Forest model definition and configuration
- **`train_model.py`** - Model training script with 24-hour retraining logic
- **`test_model.py`** - Comprehensive model testing and validation
- **`model_eval.py`** - Detailed model evaluation and performance analysis
- **`feature_engineering.py`** - Feature engineering pipeline (preserved from original)

### Saved Model Artifacts (`../artifacts/models/`)

- **`rockburst_rf_model.pkl`** - Trained Random Forest model (1.6MB)
- **`feature_engineer.pkl`** - Feature engineering pipeline
- **`feature_scaler.pkl`** - Feature scaling transformer
- **`model_config.json`** - Model configuration and parameters
- **`training_log.json`** - Training timestamp and metadata

## 🚀 Quick Start

### 1. Train the Model

```bash
# Train with default settings (2000 samples)
python models/train_model.py

# Force retrain even if model is recent
python models/train_model.py --force-retrain

# Train with custom data
python models/train_model.py --data-path /path/to/data.csv
```

### 2. Test the Model

```bash
# Test with 500 generated samples
python models/test_model.py --n-samples 500

# Test with custom data
python models/test_model.py --test-data /path/to/test_data.csv
```

### 3. Evaluate the Model

```bash
# Generate comprehensive evaluation report
python models/model_eval.py --n-samples 1000

# Save results to custom directory
python models/model_eval.py --output-dir ./custom_eval_results
```

### 4. Run Complete Pipeline Demo

```bash
python demo_model_pipeline.py
```

## ⏰ 24-Hour Auto-Retraining

The model automatically retrains every 24 hours to ensure optimal performance.

### Manual Setup with Cron

```bash
# Make retraining script executable
chmod +x retrain_model.sh

# Add to crontab for daily retraining at 2 AM
crontab -e
# Add this line:
0 2 * * * /Users/umair/Downloads/projects/project_2/retrain_model.sh
```

### Check Retraining Status

```bash
# Check if model needs retraining
python models/train_model.py

# View retraining logs
ls -la logs/retraining/
```

## 📊 Model Performance

### Current Performance (as of last training)

- **Accuracy**: 96.75%
- **F1-Score**: 96.53%
- **Precision**: 96.89%
- **Recall**: 96.75%

### Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 585 engineered features (from 9 original)
- **Classes**: Low (0), Medium (1), High (2) intensity
- **Training Samples**: 1,600 (80% split)
- **Test Samples**: 400 (20% split)

## 🔧 Configuration

### Model Parameters

```python
{
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}
```

### Retraining Schedule

- **Interval**: 24 hours
- **Automatic**: Yes (when using cron)
- **Force Retrain**: Available via command line flag

## 🎯 Key Features

1. ✅ **Random Forest Model**: Optimized for rockburst prediction
2. ✅ **Auto-Retraining**: 24-hour schedule ensures model freshness
3. ✅ **Feature Engineering**: 585 features from geological data
4. ✅ **Model Persistence**: All components saved as .pkl files
5. ✅ **Comprehensive Testing**: Validation and integrity checks
6. ✅ **Performance Evaluation**: Detailed metrics and visualizations
7. ✅ **Clean Architecture**: Organized, modular code structure

## 🛠️ Usage Examples

### Load and Use Trained Model

```python
from models.model import RockburstRandomForestModel
import pandas as pd

# Load trained model
model = RockburstRandomForestModel()
model.load_model('./artifacts/models')

# Make predictions
sample_data = pd.DataFrame({
    'Duration_days': [15.5],
    'Energy_Unit_log': [4.2],
    'Energy_density_Joule_sqr': [250.0],
    # ... other features
})

X, _ = model.prepare_features(sample_data)
prediction = model.predict(X)
probabilities = model.predict_proba(X)

print(f"Predicted intensity: {['Low', 'Medium', 'High'][prediction[0]]}")
```

### Check Model Status

```python
from models.train_model import ModelTrainer

trainer = ModelTrainer()
trainer.load_existing_model()
status = trainer.get_model_status()

print(f"Needs retraining: {status['needs_retraining']}")
print(f"Last training: {status['last_training_time']}")
```

## 📈 Monitoring

### Training Logs

- Location: `./logs/retraining/`
- Format: `retraining_YYYYMMDD_HHMMSS.log`
- Content: Training progress, metrics, errors

### Model Files

- Total size: ~1.7MB
- Location: `./artifacts/models/`
- Backup: Automatic (previous versions preserved)

---

**Author**: Data Science Team GAMMA  
**Created**: August 10, 2025  
**Last Updated**: August 10, 2025
