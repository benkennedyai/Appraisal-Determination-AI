import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("APPRAISAL AWARD PREDICTION MODEL")
print("="*70)

# Load the dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv('appraisal_demands_synthetic.csv')
print(f"✓ Loaded {len(df)} appraisal demands")

# Display key statistics
print(f"\nAward Statistics:")
print(f"  Average Award: ${df['award_amount'].mean():,.0f}")
print(f"  Median Award: ${df['award_amount'].median():,.0f}")
print(f"  Min Award: ${df['award_amount'].min():,.0f}")
print(f"  Max Award: ${df['award_amount'].max():,.0f}")

print(f"\nComplexity Distribution:")
print(df['complexity_class'].value_counts().sort_index())

# ==================== FEATURE ENGINEERING ====================
print("\n[2/7] Preparing features...")

# Select features for modeling
feature_cols = [
    # Carrier factors
    'carrier_philosophy',
    'adjuster_type',
    'ia_skill_level',
    'carrier_engineer',
    
    # Property/Loss
    'property_age_years',
    'loss_type',
    'is_catastrophe',
    'geographic_setting',
    'policy_type',
    
    # Claim history
    'prior_claims_count',
    'supplements_issued',
    
    # Financial dispute - KEY FEATURES
    'carrier_estimate',
    'demand_estimate',
    'dispute_amount',
    'dispute_percentage',
    
    # Parties
    'pa_involved',
    'pa_attorney',
    'pa_firm_type',
    
    # Complexity indicators
    'is_roof_dispute',
    'coverage_dispute',
    'line_items_disputed',
    'trades_involved',
    
    # Mitigation
    'mitigation_performed',
    'mitigation_cost',
    'mitigation_disputed_pct'
]

# Targets
award_target = 'award_amount'
complexity_target = 'complexity_class'

# Create feature dataframe
X = df[feature_cols].copy()
y_award = df[award_target].copy()
y_complexity = df[complexity_target].copy()

# Handle categorical variables
categorical_cols = ['carrier_philosophy', 'adjuster_type', 'ia_skill_level', 
                   'loss_type', 'geographic_setting', 'policy_type', 'pa_firm_type']

# Fill NaN in ia_skill_level (staff adjusters don't have this)
X['ia_skill_level'] = X['ia_skill_level'].fillna('staff')
X['pa_firm_type'] = X['pa_firm_type'].fillna('none')

# Label encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode complexity target
complexity_encoder = LabelEncoder()
y_complexity_encoded = complexity_encoder.fit_transform(y_complexity)

print(f"✓ Prepared {len(feature_cols)} features")

# ==================== TRAIN-TEST SPLIT ====================
print("\n[3/7] Splitting data...")
X_train, X_test, y_award_train, y_award_test, y_complexity_train, y_complexity_test = train_test_split(
    X, y_award, y_complexity_encoded, test_size=0.2, random_state=42, stratify=y_complexity_encoded
)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# ==================== AWARD AMOUNT PREDICTION (REGRESSION) ====================
print("\n[4/7] Training Award Amount Predictor...")

# Random Forest Regressor
print("\n  Training Random Forest Regressor...")
rf_regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_regressor.fit(X_train, y_award_train)
rf_pred = rf_regressor.predict(X_test)

rf_mae = mean_absolute_error(y_award_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_award_test, rf_pred))
rf_r2 = r2_score(y_award_test, rf_pred)

print(f"  ✓ Random Forest MAE: ${rf_mae:,.0f}")
print(f"  ✓ Random Forest RMSE: ${rf_rmse:,.0f}")
print(f"  ✓ Random Forest R²: {rf_r2:.3f}")

# Gradient Boosting Regressor
print("\n  Training Gradient Boosting Regressor...")
gb_regressor = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
gb_regressor.fit(X_train, y_award_train)
gb_pred = gb_regressor.predict(X_test)

gb_mae = mean_absolute_error(y_award_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_award_test, gb_pred))
gb_r2 = r2_score(y_award_test, gb_pred)

print(f"  ✓ Gradient Boosting MAE: ${gb_mae:,.0f}")
print(f"  ✓ Gradient Boosting RMSE: ${gb_rmse:,.0f}")
print(f"  ✓ Gradient Boosting R²: {gb_r2:.3f}")

# Select best regressor
if rf_r2 > gb_r2:
    best_regressor = rf_regressor
    best_regressor_name = 'Random Forest'
    best_r2 = rf_r2
    best_mae = rf_mae
    best_rmse = rf_rmse
    award_predictions = rf_pred
else:
    best_regressor = gb_regressor
    best_regressor_name = 'Gradient Boosting'
    best_r2 = gb_r2
    best_mae = gb_mae
    best_rmse = gb_rmse
    award_predictions = gb_pred

print(f"\n✓ Best Award Predictor: {best_regressor_name}")
print(f"  Mean Absolute Error: ${best_mae:,.0f}")
print(f"  R² Score: {best_r2:.3f}")

# Calculate prediction accuracy within ranges
within_10k = np.abs(y_award_test - award_predictions) <= 10000
within_20k = np.abs(y_award_test - award_predictions) <= 20000
within_30k = np.abs(y_award_test - award_predictions) <= 30000

print(f"\nPrediction Accuracy:")
print(f"  Within $10K: {within_10k.mean()*100:.1f}%")
print(f"  Within $20K: {within_20k.mean()*100:.1f}%")
print(f"  Within $30K: {within_30k.mean()*100:.1f}%")

# ==================== COMPLEXITY CLASSIFICATION ====================
print("\n[5/7] Training Complexity Classifier...")

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_classifier.fit(X_train, y_complexity_train)
complexity_pred = rf_classifier.predict(X_test)
complexity_accuracy = (complexity_pred == y_complexity_test).mean()

print(f"✓ Complexity Classifier Accuracy: {complexity_accuracy:.3f}")

# ==================== FEATURE IMPORTANCE ====================
print("\n[6/7] Analyzing feature importance...")

# Award prediction importance
award_importance = best_regressor.feature_importances_
award_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': award_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 Features for Award Prediction:")
print(award_importance_df.head(10).to_string(index=False))

# ==================== SAVE MODELS ====================
print("\n[7/7] Saving model artifacts...")

# Save award predictor
joblib.dump(best_regressor, 'award_predictor_model.pkl')
print("✓ Saved award predictor: award_predictor_model.pkl")

# Save complexity classifier
joblib.dump(rf_classifier, 'complexity_classifier_model.pkl')
print("✓ Saved complexity classifier: complexity_classifier_model.pkl")

# Save encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(complexity_encoder, 'complexity_encoder.pkl')
print("✓ Saved encoders")

# Save feature columns
joblib.dump(feature_cols, 'feature_columns.pkl')
print("✓ Saved feature columns")

# Save metadata
metadata = {
    'award_predictor_model': best_regressor_name,
    'award_mae': best_mae,
    'award_rmse': best_rmse,
    'award_r2': best_r2,
    'complexity_accuracy': complexity_accuracy,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'feature_count': len(feature_cols)
}
joblib.dump(metadata, 'model_metadata.pkl')
print("✓ Saved metadata")

# ==================== VISUALIZATIONS ====================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Actual vs Predicted Awards
plt.figure(figsize=(10, 8))
plt.scatter(y_award_test, award_predictions, alpha=0.5)
plt.plot([y_award_test.min(), y_award_test.max()], 
         [y_award_test.min(), y_award_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Award Amount ($)')
plt.ylabel('Predicted Award Amount ($)')
plt.title(f'Award Prediction: Actual vs Predicted\nR² = {best_r2:.3f}, MAE = ${best_mae:,.0f}')
plt.tight_layout()
plt.savefig('award_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot: award_prediction_scatter.png")

# Feature importance
plt.figure(figsize=(10, 8))
top_features = award_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title(f'Top 15 Features for Award Prediction - {best_regressor_name}')
plt.tight_layout()
plt.savefig('award_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot: award_feature_importance.png")

# Residuals distribution
plt.figure(figsize=(10, 6))
residuals = y_award_test - award_predictions
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Prediction Error ($)')
plt.ylabel('Frequency')
plt.title(f'Prediction Error Distribution\nMean Error: ${residuals.mean():,.0f}')
plt.axvline(0, color='r', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig('award_residuals.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot: award_residuals.png")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n✅ Award Predictor: {best_regressor_name}")
print(f"   Mean Absolute Error: ${best_mae:,.0f}")
print(f"   R² Score: {best_r2:.3f}")
print(f"   Within $20K: {within_20k.mean()*100:.1f}%")
print(f"\n✅ Complexity Classifier: Random Forest")
print(f"   Accuracy: {complexity_accuracy:.1%}")
print("\nModel files ready for Streamlit demo!")
print("="*70)