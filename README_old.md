# IEEE-CIS Fraud Detection Project 🎯

## Project Overview

Improving fraud detection models on the IEEE-CIS Fraud Detection dataset to achieve top 10% performance (0.945+ ROC-AUC).

**Competition:** https://www.kaggle.com/competitions/ieee-fraud-detection  
**Target Metric:** ROC-AUC  
**Baseline to Beat:** 0.92 ROC-AUC  
**Goal:** 0.945+ ROC-AUC (Top 10%)

---

## 📋 Project Roadmap

### **PHASE 1: FOUNDATION & EXPLORATION** (Week 1)

#### **Milestone 1: Initial Data Exploration**

**Prompt:** _"Create an EDA notebook to explore the IEEE fraud detection data. Show dataset shapes, data types, missing values, target distribution, and basic statistics for both transaction and identity files."_

**Deliverables:**

- `notebooks/01_initial_eda.ipynb`
- Understanding of data structure
- Documentation of data quality issues
- Initial observations about class imbalance

---

#### **Milestone 2: Deep Feature Analysis**

**Prompt:** _"Analyze the features in detail: categorize the 434 features by type (V columns, C columns, D columns, etc.), visualize missing value patterns, identify high-cardinality categoricals, and show distributions of key numerical features split by fraud vs legitimate."_

**Deliverables:**

- `notebooks/02_feature_analysis.ipynb`
- Feature taxonomy document
- Missing value heatmap
- Key insights about predictive features

---

#### **Milestone 3: Temporal Analysis**

**Prompt:** _"Perform temporal analysis on the transaction data. Analyze TransactionDT patterns, create time-based features (hour, day, week), check for time-based trends in fraud rates, and validate time-based split strategy for cross-validation."_

**Deliverables:**

- `notebooks/03_temporal_analysis.ipynb`
- Time-series visualizations
- CV strategy recommendation
- Temporal feature ideas

---

### **PHASE 2: BASELINE MODELS** (Week 2)

#### **Milestone 4: Data Preprocessing Pipeline**

**Prompt:** _"Create a preprocessing pipeline: handle missing values, encode categorical variables, normalize numerical features, merge transaction and identity datasets, and prepare train/validation splits using time-based splitting."_

**Deliverables:**

- `src/preprocessing.py`
- Clean, processed datasets
- Reusable preprocessing functions
- Data validation checks

---

#### **Milestone 5: Simple Baseline Model**

**Prompt:** _"Build a simple LightGBM baseline model using basic features. Train with proper cross-validation, evaluate ROC-AUC, create feature importance plots, and establish the baseline score to beat."_

**Deliverables:**

- `notebooks/04_baseline_model.ipynb`
- `src/models/lgbm_baseline.py`
- Baseline ROC-AUC score
- Feature importance analysis

---

#### **Milestone 6: Reproduce Top Public Kernel & Competition Benchmark**

**Prompt:** _"Reproduce the top public kernel approach (target: 0.92 ROC-AUC). Implement their feature engineering, train their model architecture, and validate the score. Then analyze competition leaderboard results: compare your score to 1st place (0.9652), Top 10% (0.945), and Top 25% (0.935). Document the performance gap and what techniques the top winners used."_

**Deliverables:**

- `notebooks/05_public_kernel_reproduction.ipynb`
- Validated 0.92+ ROC-AUC
- Competition benchmark comparison table
- Analysis of top solutions (1st-10th place)
- Gap analysis with specific improvement targets

---

### **PHASE 3: FEATURE ENGINEERING** (Week 3)

#### **Milestone 7: Aggregation Features**

**Prompt:** _"Create aggregation features: group by card, email domain, device info, and ProductCD. Calculate count, mean, std, min, max for transaction amounts. Add frequency features (transactions per card/email/device in last 1 day, 7 days, 30 days)."_

**Deliverables:**

- `src/features/aggregation.py`
- 50+ new aggregation features
- Feature validation notebook
- Expected AUC lift: +0.005-0.010

---

#### **Milestone 8: Interaction Features**

**Prompt:** _"Create interaction features: card + address combinations, card + email domain, device + browser combinations, amount bins + product type. Add risk scores based on historical fraud rates for each combination."_

**Deliverables:**

- `src/features/interactions.py`
- 30+ interaction features
- Risk score features
- Expected AUC lift: +0.003-0.008

---

#### **Milestone 9: Advanced Temporal Features**

**Prompt:** _"Build advanced time features: time since last transaction (by card/email/device), transaction velocity (# transactions in rolling windows), recency/frequency/monetary (RFM) features, and time-of-day risk patterns."_

**Deliverables:**

- `src/features/temporal.py`
- 40+ temporal features
- RFM segmentation
- Expected AUC lift: +0.005-0.012

---

#### **Milestone 10: Missing Value Features**

**Prompt:** _"Create meta-features from missing values: count of missing features per row, missing value patterns (which combinations of features are missing together), binary indicators for important missing features, and imputation quality metrics."_

**Deliverables:**

- `src/features/missing_features.py`
- 20+ missingness features
- Pattern analysis
- Expected AUC lift: +0.002-0.005

---

### **PHASE 4: MODEL IMPROVEMENT** (Week 4)

#### **Milestone 11: Hyperparameter Optimization**

**Prompt:** _"Optimize LightGBM hyperparameters using Optuna: tune learning_rate, num_leaves, max_depth, min_child_samples, feature_fraction, and bagging_fraction. Run 100+ trials with cross-validation. Save best parameters."_

**Deliverables:**

- `notebooks/06_hyperparameter_tuning.ipynb`
- `config/best_params.json`
- Optimization history plots
- Expected AUC lift: +0.003-0.008

---

#### **Milestone 12: Multi-Model Ensemble**

**Prompt:** _"Create an ensemble: train LightGBM, XGBoost, and CatBoost with optimized parameters. Train a simple neural network on engineered features. Ensemble predictions using weighted averaging and stacking. Compare single model vs ensemble performance."_

**Deliverables:**

- `src/models/ensemble.py`
- `notebooks/07_ensemble_models.ipynb`
- Ensemble weights
- Expected AUC lift: +0.005-0.015

---

#### **Milestone 13: Advanced Techniques**

**Prompt:** _"Implement advanced techniques: adversarial validation to check train/test similarity, pseudo-labeling on high-confidence test predictions, focal loss for handling class imbalance, and calibration of probability outputs."_

**Deliverables:**

- `src/models/advanced_techniques.py`
- `notebooks/08_advanced_methods.ipynb`
- Adversarial validation score
- Expected AUC lift: +0.003-0.010

---

### **PHASE 5: OPTIMIZATION & DOCUMENTATION** (Week 5)

#### **Milestone 14: Feature Selection**

**Prompt:** _"Perform feature selection: calculate feature importance across all models, remove redundant features using correlation analysis, use recursive feature elimination, and test reduced feature sets. Find optimal feature count for best AUC."_

**Deliverables:**

- `notebooks/09_feature_selection.ipynb`
- Final feature list
- Feature importance rankings
- Cleaned up feature set

---

#### **Milestone 15: Final Model Training & Competition Comparison**

**Prompt:** _"Train the final model: use all engineered features, best hyperparameters, and ensemble approach. Train on full dataset with proper validation. Generate final predictions. Document final ROC-AUC score and compare to: (1) your initial baseline, (2) public kernel baseline (0.92), (3) competition Top 25% (0.935), (4) competition Top 10% (0.945), and (5) 1st place (0.9652). Calculate where you would rank on the leaderboard."_

**Deliverables:**

- `src/models/final_model.py`
- `submissions/final_submission.csv`
- Final ROC-AUC score
- Competition ranking analysis
- Model artifacts saved

---

#### **Milestone 16: Project Documentation & Final Comparison**

**Prompt:** _"Create comprehensive project documentation: detailed README with setup instructions, results comparison notebook showing improvement from baseline to final with competition benchmarks, technical blog post explaining methodology and how you approached beating public kernels, comparison table showing your score vs all competition tiers (Top 1%, 10%, 25%, baseline), and requirements.txt with all dependencies."_

**Deliverables:**

- Updated `README.md`
- `notebooks/10_results_comparison.ipynb` (with competition benchmarks)
- `docs/technical_writeup.md` (with leaderboard analysis)
- `requirements.txt`
- Clean, reproducible codebase
- Final competition standing analysis

---

## 📁 Expected Project Structure

```
ieee_fraud/
├── data/
│   ├── raw/                    # Original data files
│   ├── processed/              # Processed datasets
│   └── features/               # Engineered features
├── notebooks/
│   ├── 01_initial_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_temporal_analysis.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_public_kernel_reproduction.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_ensemble_models.ipynb
│   ├── 08_advanced_methods.ipynb
│   ├── 09_feature_selection.ipynb
│   └── 10_results_comparison.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── aggregation.py
│   │   ├── interactions.py
│   │   ├── temporal.py
│   │   └── missing_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lgbm_baseline.py
│   │   ├── ensemble.py
│   │   ├── advanced_techniques.py
│   │   └── final_model.py
│   └── utils.py
├── config/
│   ├── best_params.json
│   └── feature_config.yaml
├── submissions/
│   └── final_submission.csv
├── docs/
│   └── technical_writeup.md
├── models/                     # Saved model artifacts
├── requirements.txt
└── README.md
```

---

## 🎯 Success Metrics

| Phase                 | Target ROC-AUC | Competition Standing | Status     |
| --------------------- | -------------- | -------------------- | ---------- |
| Initial Baseline      | 0.920          | Public Kernel Level  | ⏳ Pending |
| + Feature Engineering | 0.935          | Top 25%              | ⏳ Pending |
| + Model Optimization  | 0.945          | Top 10%              | ⏳ Pending |
| + Advanced Techniques | 0.950+         | Top 5%               | ⏳ Pending |

### 🏆 Competition Leaderboard Reference

| Rank      | ROC-AUC | Notes              |
| --------- | ------- | ------------------ |
| 1st Place | 0.9652  | Winning solution   |
| Top 1%    | ~0.955  | Elite performance  |
| Top 5%    | ~0.950  | Exceptional        |
| Top 10%   | ~0.945  | **Primary Goal**   |
| Top 25%   | ~0.935  | Strong performance |
| Baseline  | ~0.920  | Public kernel      |

---

## 🚀 Getting Started

Use the prompts above in order. Each milestone builds on the previous one and represents a complete, actionable task you can assign.

**To begin, use:**  
_"Create an EDA notebook to explore the IEEE fraud detection data. Show dataset shapes, data types, missing values, target distribution, and basic statistics for both transaction and identity files."_

---

## 📊 Progress Tracking

- [ ] Phase 1: Foundation & Exploration (Milestones 1-3)
- [ ] Phase 2: Baseline Models (Milestones 4-6)
- [ ] Phase 3: Feature Engineering (Milestones 7-10)
- [ ] Phase 4: Model Improvement (Milestones 11-13)
- [ ] Phase 5: Optimization & Documentation (Milestones 14-16)

---

**Current Phase:** Phase 1 - Foundation & Exploration  
**Next Milestone:** Milestone 1 - Initial Data Exploration  
**Estimated Completion:** 5 weeks

Good luck! 🎉
