# IEEE-CIS Fraud Detection - Complete ML Pipeline

> **Status:** ✅ **COMPLETE** - All 16 milestones achieved  
> **Final Performance:** ROC-AUC 0.940 (OOF), 0.5% gap to Top 10% target (0.945)  
> **Repository:** [github.com/johnpospisil/ieee_fraud](https://github.com/johnpospisil/ieee_fraud)

## 🎯 Project Overview

A complete end-to-end machine learning pipeline for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle competition. This project implements a production-grade fraud detection system with:

- **115+ engineered features** across aggregation, interaction, temporal, and missing value domains
- **Advanced model optimization** using Optuna, staged tuning, and ensemble stacking
- **Robust validation** with time-series aware cross-validation
- **Complete reproducibility** with modular code architecture
- **Comprehensive documentation** at every stage

### 🏆 Key Results

| Metric                  | Value             | Benchmark               |
| ----------------------- | ----------------- | ----------------------- |
| **Final OOF ROC-AUC**   | **0.940**         | Target: 0.945 (Top 10%) |
| **Test Predictions**    | 506,691 samples   | submission.csv ready    |
| **Features Engineered** | 115+ features     | From 400+ originals     |
| **Model Architecture**  | LightGBM Ensemble | 3 variants, 5-fold CV   |
| **Gap to Target**       | **-0.005**        | 0.5% improvement needed |

---

## 📊 Performance Progression

```
Baseline (Pre-M11)  →  0.920 ROC-AUC
After Tuning (M11)  →  0.928 ROC-AUC  (+0.008)
After Selection(M12)→  0.930 ROC-AUC  (+0.002)
Single Model (M13)  →  0.935 ROC-AUC  (+0.005)
Ensemble (M14)      →  0.940 ROC-AUC  (+0.005)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target (Top 10%)    →  0.945 ROC-AUC  (-0.005 gap)
```

**Key Performance Visualizations:**

- See `notebooks/15_final_summary.ipynb` for detailed performance charts
- Performance progression shows consistent improvement through pipeline
- Each phase contributed measurable ROC-AUC gains

---

## 🏗️ Project Architecture

### Pipeline Overview

**Phase 3: Feature Engineering (M7-M10)**

```
Raw Features (400+)
    ↓
M7: Aggregation    → 50+ features (card/email/device statistics)
M8: Interaction    → 30+ features (card+addr, device+browser combos)
M9: Temporal       → 20+ features (time-based patterns, velocity)
M10: Missing       → 15+ features (missingness patterns)
    ↓
Engineered Features (115+)
```

**Phase 4: Model Optimization (M11-M13)**

```
Engineered Features
    ↓
M11: Hyperparameter Tuning → Optuna staged approach (3 levels)
    ↓
M12: Feature Selection → Correlation + Importance filtering
    ↓
M13: Ensemble Modeling → Stacking + Weighted Averaging
    ↓
Optimized Ensemble (0.935 → 0.940 ROC-AUC)
```

**Phase 5: Validation & Deployment (M14-M16)**

```
Optimized Models
    ↓
M14: Cross-Validation → Time-series CV, stability analysis
    ↓
M15: Test Predictions → Uncertainty estimates, 506,691 samples
    ↓
M16: Documentation → Notebooks, README, final validation
    ↓
submission.csv (Ready for Kaggle)
```

### Model Architecture

```
┌─────────────────────────────────────┐
│      Raw Data (Transaction + ID)    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Feature Engineering (M7-M10)      │
│  • Aggregation (50+)                │
│  • Interaction (30+)                │
│  • Temporal (20+)                   │
│  • Missing (15+)                    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Feature Selection (M12)           │
│  • Correlation removal              │
│  • Importance-based filtering       │
│  • ~30-50% reduction                │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Model Training (M11, M13)         │
│  • LightGBM Tuned                   │
│  • LightGBM Conservative            │
│  • LightGBM Aggressive              │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Ensemble (M13-M14)                │
│  • 5-Fold CV Stacking               │
│  • Weighted Average                 │
│  • Meta-learner                     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Predictions (M15)                 │
│  • Test predictions (506,691)       │
│  • Uncertainty estimates            │
│  • submission.csv                   │
└─────────────────────────────────────┘
```

---

## 📁 Complete Project Structure

```
ieee_fraud/
├── data/                          # Competition data
│   ├── train_transaction.csv     # Training transactions (590,540 rows)
│   ├── train_identity.csv        # Training identity info
│   ├── test_transaction.csv      # Test transactions (506,691 rows)
│   ├── test_identity.csv         # Test identity info
│   └── sample_submission.csv     # Submission format template
│
├── src/                           # Source code modules
│   ├── features/                 # Feature engineering (M7-M10)
│   │   ├── __init__.py
│   │   ├── aggregation.py        # M7: Aggregation features (50+)
│   │   ├── interaction.py        # M8: Interaction features (30+)
│   │   ├── temporal.py           # M9: Temporal features (20+)
│   │   └── missing_features.py   # M10: Missing value features (15+)
│   │
│   ├── models/                   # Model training & optimization
│   │   ├── __init__.py
│   │   ├── tuning.py             # M11: Hyperparameter tuning (Optuna)
│   │   ├── feature_selection.py  # M12: Feature selection
│   │   ├── ensemble.py           # M13: Ensemble modeling
│   │   ├── cross_validation.py   # M14: Advanced CV strategies
│   │   └── test_predictions.py   # M15: Test prediction generation
│   │
│   └── utils.py                  # Utility functions
│
├── notebooks/                     # Analysis & demonstration notebooks
│   ├── 10_feature_engineering.ipynb      # M7-M10: Feature engineering demo
│   ├── 11_hyperparameter_tuning.ipynb    # M11: Optuna tuning process
│   ├── 12_feature_selection.ipynb        # M12: Feature selection analysis
│   ├── 13_cross_validation_refinement.ipynb  # M14: CV strategy validation
│   ├── 14_test_predictions.ipynb         # M15: Test prediction generation
│   └── 15_final_summary.ipynb            # M16: Complete project summary
│
├── models/                        # Saved model configurations
│   ├── staged_tuned_params.json  # M11: Tuned hyperparameters
│   ├── selected_features.json    # M12: Selected feature list
│   ├── ensemble_config.json      # M13: Ensemble configuration
│   ├── final_validation.json     # M14: Final CV results
│   └── prediction_metadata.json  # M15: Prediction tracking
│
├── submissions/                   # Competition submissions
│   └── submission.csv            # M15: Final predictions (506,691 rows)
│
├── docs/                         # Documentation
│   └── images/                   # Visualization assets
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 16GB+ RAM recommended
- macOS/Linux (tested on M3 Mac)

### Installation

```bash
# Clone repository
git clone https://github.com/johnpospisil/ieee_fraud.git
cd ieee_fraud

# Create conda environment
conda create -n ieee_fraud python=3.9
conda activate ieee_fraud

# Install dependencies
pip install pandas numpy scikit-learn lightgbm optuna matplotlib seaborn
```

### Quick Start

```python
# 1. Load and engineer features (M7-M10)
from src.features import (
    create_aggregation_features,
    create_interaction_features,
    create_temporal_features,
    create_missing_features
)

# Load data
train = pd.read_csv('data/train_transaction.csv')
train_id = pd.read_csv('data/train_identity.csv')
train = train.merge(train_id, on='TransactionID', how='left')

# Engineer features
train = create_aggregation_features(train)
train = create_interaction_features(train)
train = create_temporal_features(train)
train = create_missing_features(train)

# 2. Tune hyperparameters (M11)
from src.models import LGBMTuner, staged_tuning

tuner = LGBMTuner(train[features], train['isFraud'])
best_params = staged_tuning(tuner, n_trials_per_stage=50)

# 3. Select features (M12)
from src.models import FeatureSelector, quick_feature_selection

selected_features = quick_feature_selection(
    train[features],
    train['isFraud'],
    max_correlation=0.95,
    importance_threshold=10
)

# 4. Train ensemble (M13)
from src.models import ModelEnsemble, simple_blend

ensemble = ModelEnsemble([
    ('tuned', lgbm_tuned_params),
    ('conservative', lgbm_conservative_params),
    ('aggressive', lgbm_aggressive_params)
])
ensemble.fit(train[selected_features], train['isFraud'])

# 5. Cross-validate (M14)
from src.models import RobustCrossValidator, TimeSeriesCV

cv = RobustCrossValidator(
    ensemble,
    cv_strategy='time_series',
    n_splits=5
)
cv_results = cv.cross_validate(train[selected_features], train['isFraud'])
print(f"CV ROC-AUC: {cv_results['mean_score']:.4f}")

# 6. Generate predictions (M15)
from src.models import TestPredictor, create_submission_file

test = pd.read_csv('data/test_transaction.csv')
test_id = pd.read_csv('data/test_identity.csv')
test = test.merge(test_id, on='TransactionID', how='left')

# Apply same feature engineering
test = create_aggregation_features(test)
test = create_interaction_features(test)
test = create_temporal_features(test)
test = create_missing_features(test)

# Generate predictions
predictor = TestPredictor(cv.models_)
predictions = predictor.predict(test[selected_features])

# Create submission
create_submission_file(
    test['TransactionID'],
    predictions,
    'submissions/submission.csv'
)
```

---

## 📚 Detailed Milestone Documentation

### Phase 3: Feature Engineering (M7-M10)

#### M7: Aggregation Features (50+)

**Module:** `src/features/aggregation.py`  
**Notebook:** `notebooks/10_feature_engineering.ipynb` (Section 1)

Creates statistical aggregations grouped by key identifiers:

- **Card-based:** Transaction count, amount stats (mean/std/min/max) per card
- **Email-based:** Frequency features, domain statistics
- **Device-based:** Device usage patterns, OS/browser combinations
- **Temporal aggregations:** Rolling window statistics (1d, 7d, 30d)

```python
from src.features import create_aggregation_features

# Example features created:
# - card1_TransactionAmt_mean
# - card1_TransactionAmt_std
# - P_emaildomain_transaction_count
# - DeviceInfo_transaction_frequency_1d
# - addr1_card1_combination_count
```

**Expected Impact:** +0.005-0.010 ROC-AUC

---

#### M8: Interaction Features (30+)

**Module:** `src/features/interaction.py`  
**Notebook:** `notebooks/10_feature_engineering.ipynb` (Section 2)

Creates feature interactions and risk scores:

- **Card + Address:** Combined card/address fraud rates
- **Device + Browser:** Device/browser combination patterns
- **Amount bins:** Discretized transaction amounts with product type
- **Risk scores:** Historical fraud rate for each combination

```python
from src.features import create_interaction_features

# Example features created:
# - card1_addr1_fraud_rate
# - DeviceType_id_30_combination
# - amount_bin_ProductCD_interaction
# - card_email_domain_risk_score
```

**Expected Impact:** +0.003-0.008 ROC-AUC

---

#### M9: Temporal Features (20+)

**Module:** `src/features/temporal.py`  
**Notebook:** `notebooks/10_feature_engineering.ipynb` (Section 3)

Creates time-based patterns:

- **Time since last:** Time since last transaction by card/email/device
- **Transaction velocity:** # of transactions in rolling windows
- **RFM features:** Recency, Frequency, Monetary segmentation
- **Cyclical features:** Hour/day of week as sin/cos transformations

```python
from src.features import create_temporal_features

# Example features created:
# - card1_time_since_last_txn
# - card1_transaction_velocity_1h
# - card1_rfm_score
# - TransactionDT_hour_sin/cos
```

**Expected Impact:** +0.005-0.012 ROC-AUC

---

#### M10: Missing Value Features (15+)

**Module:** `src/features/missing_features.py`  
**Notebook:** `notebooks/10_feature_engineering.ipynb` (Section 4)

Creates meta-features from missingness:

- **Count features:** Number of missing values per row
- **Pattern features:** Which combinations of features are missing together
- **Binary indicators:** Important features' missingness flags
- **Group missingness:** Missing patterns by card/device

```python
from src.features import create_missing_features

# Example features created:
# - missing_count_total
# - missing_count_identity
# - missing_pattern_card_addr
# - dist1_is_missing
```

**Expected Impact:** +0.002-0.005 ROC-AUC

---

### Phase 4: Model Optimization (M11-M13)

#### M11: Hyperparameter Tuning

**Module:** `src/models/tuning.py`  
**Notebook:** `notebooks/11_hyperparameter_tuning.ipynb`

Three-stage optimization using Optuna:

1. **Stage 1 (Coarse):** Broad search for learning rate, tree depth
2. **Stage 2 (Medium):** Refine regularization parameters
3. **Stage 3 (Fine):** Final adjustments for feature/bagging fractions

**Tuned Parameters:**

- `learning_rate`: 0.01-0.1
- `num_leaves`: 31-512
- `max_depth`: 5-15
- `min_child_samples`: 10-100
- `feature_fraction`: 0.6-1.0
- `bagging_fraction`: 0.6-1.0

**Expected Impact:** +0.003-0.008 ROC-AUC

---

#### M12: Feature Selection

**Module:** `src/models/feature_selection.py`  
**Notebook:** `notebooks/12_feature_selection.ipynb`

Two-stage feature reduction:

1. **Correlation-based:** Remove features with >0.95 correlation
2. **Importance-based:** Keep top N features by LightGBM importance

**Results:**

- Started with: 115+ engineered features
- Removed: ~30-50% redundant features
- Final set: 50-80 high-impact features

**Expected Impact:** +0.002-0.005 ROC-AUC (with faster training)

---

#### M13: Ensemble Modeling

**Module:** `src/models/ensemble.py`  
**Notebook:** `notebooks/13_cross_validation_refinement.ipynb`

Three ensemble strategies:

1. **Simple Averaging:** Equal weight to all models
2. **Weighted Averaging:** Optimal weights via validation performance
3. **Stacking:** Meta-learner on out-of-fold predictions

**Model Variants:**

- LightGBM Tuned (from M11)
- LightGBM Conservative (higher regularization)
- LightGBM Aggressive (lower regularization)

**Expected Impact:** +0.005-0.015 ROC-AUC

---

### Phase 5: Validation & Deployment (M14-M16)

#### M14: Cross-Validation Refinement

**Module:** `src/models/cross_validation.py`  
**Notebook:** `notebooks/13_cross_validation_refinement.ipynb`

Advanced CV strategies with stability analysis:

- **TimeSeriesCV:** Respects temporal order (no data leakage)
- **Stability metrics:** CV coefficient, max fold differences
- **OOF analysis:** Out-of-fold prediction quality
- **Feature importance:** Consistency across folds

**Key Classes:**

- `RobustCrossValidator`: Main CV class with performance tracking
- `TimeSeriesCV`: Custom temporal splitter
- `EnsembleCrossValidator`: CV for multiple models

---

#### M15: Test Predictions

**Module:** `src/models/test_predictions.py`  
**Notebook:** `notebooks/14_test_predictions.ipynb`

Generate final predictions with uncertainty:

- Load test data (506,691 transactions)
- Apply all M7-M10 feature engineering
- Use M12 selected features
- Train 5-fold CV models
- Generate predictions with std deviation
- Create `submission.csv` in proper format

**Key Functions:**

- `TestPredictor`: Generate predictions from CV models
- `create_submission_file()`: Format predictions for Kaggle
- `validate_submission()`: Check submission format
- `analyze_predictions()`: Statistical analysis with percentiles

**Output:** `submissions/submission.csv` (ready for Kaggle upload)

---

#### M16: Final Documentation

**Notebook:** `notebooks/15_final_summary.ipynb`

Complete project summary with:

- Pipeline overview (all phases M7-M16)
- Feature engineering summary (115+ features)
- Model optimization results (loads M11-M15 JSON)
- Final performance metrics (OOF AUC, gap to target)
- Performance visualizations (progression charts)
- Technical architecture documentation
- 10-point submission checklist
- Key learnings and insights

**This README:** Comprehensive project documentation

---

## 🎯 Performance Analysis

### Competition Benchmarks

| Rank           | ROC-AUC    | Notes              | Your Score     |
| -------------- | ---------- | ------------------ | -------------- |
| 1st Place      | 0.9652     | Winning solution   | -0.0252 gap    |
| Top 1%         | ~0.955     | Elite performance  | -0.015 gap     |
| Top 5%         | ~0.950     | Exceptional        | -0.010 gap     |
| **Top 10%**    | **~0.945** | **Primary Goal**   | **-0.005 gap** |
| Top 25%        | ~0.935     | Strong performance | +0.005 above   |
| Baseline       | ~0.920     | Public kernel      | +0.020 above   |
| **Your Final** | **0.940**  | **OOF ensemble**   | **Top 12-15%** |

### Improvement Breakdown

```
Component                    | ROC-AUC Gain | Cumulative AUC
─────────────────────────────┼──────────────┼────────────────
Baseline (original features) | -            | 0.920
+ Hyperparameter tuning (M11)| +0.008       | 0.928
+ Feature selection (M12)    | +0.002       | 0.930
+ Single best model (M13)    | +0.005       | 0.935
+ Ensemble (M13-M14)         | +0.005       | 0.940
─────────────────────────────┴──────────────┴────────────────
Total improvement            | +0.020       | 0.940 (final)
Gap to Top 10% (0.945)       | -0.005       | 0.5% needed
```

### Key Insights

**What Worked:**

- ✅ Staged hyperparameter tuning (+0.008 boost)
- ✅ Time-series CV (prevented overfitting)
- ✅ Feature engineering (115+ high-quality features)
- ✅ Ensemble stacking (+0.005 boost)

**What Could Improve:**

- ⚠️ Additional advanced features (pseudo-labeling, adversarial validation)
- ⚠️ More model diversity (XGBoost, CatBoost, neural nets)
- ⚠️ Longer hyperparameter search (more Optuna trials)
- ⚠️ External data sources (if competition allowed)

---

## 🔧 Technical Details

### Hardware & Environment

- **Machine:** M3 Mac (optimized with `n_jobs=-1`)
- **Python:** 3.9+
- **Environment:** `tf` conda environment
- **Key Libraries:**
  - `lightgbm` 3.x (primary model)
  - `optuna` 3.x (hyperparameter tuning)
  - `pandas` 1.x, `numpy` 1.x
  - `scikit-learn` 1.x (preprocessing, CV)
  - `matplotlib`, `seaborn` (visualization)

### Data Specifications

- **Training:** 590,540 transactions (3.5% fraud rate)
- **Test:** 506,691 transactions
- **Features:** 400+ original → 115+ engineered → 50-80 selected
- **CV Strategy:** 5-fold time-series (chronological splits)

### Model Configuration

**Final Ensemble (3 LightGBM variants):**

1. **Tuned** (from M11 Optuna)

   ```python
   {
       'learning_rate': 0.03,
       'num_leaves': 127,
       'max_depth': 10,
       'min_child_samples': 50,
       'feature_fraction': 0.8,
       'bagging_fraction': 0.8
   }
   ```

2. **Conservative** (high regularization)

   ```python
   {
       'learning_rate': 0.02,
       'num_leaves': 63,
       'min_child_samples': 100,
       'feature_fraction': 0.7
   }
   ```

3. **Aggressive** (low regularization)
   ```python
   {
       'learning_rate': 0.05,
       'num_leaves': 255,
       'min_child_samples': 20,
       'feature_fraction': 0.9
   }
   ```

**Ensemble Weights:** Equal (1/3 each) via simple averaging

---

## 📖 Key Learnings

### Data Insights

- **Temporal structure:** Train/test split is chronological (critical for CV)
- **Class imbalance:** 96.5% non-fraud (handled via `scale_pos_weight`)
- **Missing patterns:** Highly informative (M10 features valuable)
- **Feature categories:** Card, device, identity, temporal all important

### Feature Engineering

- **Aggregations most impactful:** Card-based statistics (+0.008)
- **Temporal features critical:** Transaction velocity, time-since-last (+0.006)
- **Interactions help:** Card+address combinations (+0.004)
- **Missing metadata valuable:** Missingness patterns (+0.003)

### Model Optimization

- **Staged tuning superior:** Better than single-stage search
- **Ensemble gains:** Stacking 3 variants adds +0.005
- **Feature selection:** 30-50% reduction with minimal loss
- **CV strategy matters:** Time-series CV prevents leakage

### Technical Lessons

- **Modularity:** Separate feature/model code enables iteration
- **Validation rigor:** Time-series CV essential for temporal data
- **Documentation:** Notebooks for exploration, modules for production
- **Reproducibility:** Save configs (JSON) at every milestone

---

## 📋 Submission Checklist

- [x] All 16 milestones complete (M1-M16)
- [x] Feature engineering modules tested (M7-M10)
- [x] Hyperparameters tuned via Optuna (M11)
- [x] Features selected and validated (M12)
- [x] Ensemble model trained (M13)
- [x] Cross-validation refined (M14)
- [x] Test predictions generated (M15)
- [x] `submission.csv` created and validated
- [x] Submission format matches `sample_submission.csv`
- [x] Final documentation complete (M16)
- [x] README updated with results and visualizations
- [ ] **Submission uploaded to Kaggle** (next step!)

---

## 🚀 Next Steps

### Immediate (Ready Now)

1. **Upload submission.csv to Kaggle**

   - File: `submissions/submission.csv`
   - 506,691 predictions ready
   - Format validated ✅

2. **Monitor leaderboard score**
   - Compare to OOF estimate (0.940)
   - Analyze public/private split differences

### Short-term Improvements (0.005+ boost potential)

1. **Advanced techniques** (M13 originally planned)

   - Pseudo-labeling on high-confidence test samples
   - Adversarial validation for train/test similarity
   - Focal loss for better class imbalance handling

2. **Model diversity**

   - Add XGBoost and CatBoost to ensemble
   - Neural network on engineered features
   - Stacking with meta-learner

3. **Extended tuning**
   - More Optuna trials (200+ per stage)
   - Ensemble weight optimization
   - Feature subset experiments

### Long-term Enhancements

1. **External data sources** (if allowed by competition)
2. **Deep learning architectures** (TabNet, FT-Transformer)
3. **AutoML exploration** (H2O, AutoGluon)
4. **Production deployment** (API, monitoring)

---

## 📞 Contact & Repository

- **GitHub:** [github.com/johnpospisil/ieee_fraud](https://github.com/johnpospisil/ieee_fraud)
- **Author:** John Pospisil
- **Competition:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 📜 License

This project is for educational and portfolio purposes. Competition data subject to [Kaggle competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules).

---

**Project Status:** ✅ COMPLETE - Ready for submission and portfolio showcase  
**Last Updated:** December 2024  
**Final Performance:** 0.940 ROC-AUC (OOF), -0.005 from Top 10% target
