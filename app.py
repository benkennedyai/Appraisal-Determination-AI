import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Appraisal Award Predictor",
    page_icon="⚖️",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    award_model = joblib.load('award_predictor_model.pkl')
    complexity_model = joblib.load('complexity_classifier_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    complexity_encoder = joblib.load('complexity_encoder.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return award_model, complexity_model, label_encoders, complexity_encoder, feature_columns, metadata

try:
    award_model, complexity_model, label_encoders, complexity_encoder, feature_columns, metadata = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Error loading models: {e}")
    st.info("Please run the training script first to generate model files.")

# Header
st.title("⚖️ Insurance Appraisal Award Predictor")
st.markdown("Predict appraisal award amounts and complexity based on claim characteristics at demand receipt.")

if not models_loaded:
    st.stop()

# Sidebar - Model Stats
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Award Prediction MAE", f"${metadata['award_mae']:,.0f}")
    st.metric("Award R² Score", f"{metadata['award_r2']:.3f}")
    st.metric("Complexity Accuracy", f"{metadata['complexity_accuracy']:.1%}")
    st.metric("Training Samples", f"{metadata['training_samples']:,}")

# Main Input Form
st.header("📝 Enter Claim Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Financial")
    carrier_estimate = st.number_input(
        "Carrier Estimate ($)",
        min_value=5000,
        max_value=500000,
        value=50000,
        step=5000
    )
    
    demand_estimate = st.number_input(
        "Demand Amount ($)",
        min_value=carrier_estimate,
        max_value=1000000,
        value=int(carrier_estimate * 2.5),
        step=5000
    )
    
    dispute_amount = demand_estimate - carrier_estimate
    dispute_percentage = (dispute_amount / carrier_estimate) * 100
    
    st.info(f"**Gap:** ${dispute_amount:,.0f} ({dispute_percentage:.0f}%)")

with col2:
    st.subheader("Carrier & Loss")
    
    carrier_philosophy = st.selectbox(
        "Carrier Type",
        options=['liberal', 'moderate', 'tight'],
        index=1,
        help="Tight carriers (Allstate-like) defend aggressively"
    )
    
    loss_type = st.selectbox(
        "Loss Type",
        options=['hail', 'wind', 'water', 'fire', 'hurricane', 'vehicle_impact'],
        index=0
    )
    
    is_roof_dispute = st.checkbox("Roof Repair/Replace Dispute", value=False)
    
    carrier_engineer = st.checkbox("Engineer Retained", value=False)

with col3:
    st.subheader("Representation & Issues")
    
    pa_involved = st.checkbox("Public Adjuster", value=True)
    
    pa_attorney = st.checkbox("PA Attorney", value=False) if pa_involved else False
    
    coverage_dispute = st.checkbox("Coverage Issues", value=False)
    
    line_items_disputed = st.slider(
        "Disputed Line Items",
        min_value=5,
        max_value=200,
        value=30
    )

# Advanced options (collapsible)
with st.expander("⚙️ Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        adjuster_type = st.selectbox("Adjuster Type", ['staff', 'independent'], index=1)
        ia_skill_level = st.selectbox("IA Skill", ['high', 'medium', 'low'], index=1) if adjuster_type == 'independent' else 'staff'
        property_age_years = st.slider("Property Age", 0, 100, 20)
        prior_claims_count = st.selectbox("Prior Claims", [0, 1, 2, 3], index=0)
    
    with adv_col2:
        geographic_setting = st.selectbox("Location", ['suburban', 'urban', 'rural'], index=0)
        policy_type = st.selectbox("Policy Type", ['RCV', 'ACV'], index=0)
        is_catastrophe = st.checkbox("CAT Claim", value=False)
        pa_firm_type = st.selectbox("PA Firm", ['local', 'regional', 'national'], index=0) if pa_involved else 'none'
        
    mitigation_col1, mitigation_col2 = st.columns(2)
    with mitigation_col1:
        mitigation_performed = st.checkbox("Mitigation Services", value=False)
    with mitigation_col2:
        if mitigation_performed:
            mitigation_cost = st.number_input("Mitigation Cost", 0, 100000, 10000, step=1000)
            mitigation_disputed_pct = st.slider("% Disputed", 0, 100, 30) / 100
        else:
            mitigation_cost = 0
            mitigation_disputed_pct = 0
    
    supplements_issued = st.slider("Supplements Issued", 0, 10, 1)
    trades_involved = st.slider("Trades Involved", 1, 12, 3)

# Predict button
if st.button("🔮 Predict Appraisal Award", type="primary", use_container_width=True):
    
    # Prepare input
    input_data = pd.DataFrame([{
        'carrier_philosophy': carrier_philosophy,
        'adjuster_type': adjuster_type,
        'ia_skill_level': ia_skill_level,
        'carrier_engineer': carrier_engineer,
        'property_age_years': property_age_years,
        'loss_type': loss_type,
        'is_catastrophe': is_catastrophe,
        'geographic_setting': geographic_setting,
        'policy_type': policy_type,
        'prior_claims_count': prior_claims_count,
        'supplements_issued': supplements_issued,
        'carrier_estimate': carrier_estimate,
        'demand_estimate': demand_estimate,
        'dispute_amount': dispute_amount,
        'dispute_percentage': dispute_percentage,
        'pa_involved': pa_involved,
        'pa_attorney': pa_attorney,
        'pa_firm_type': pa_firm_type,
        'is_roof_dispute': is_roof_dispute,
        'coverage_dispute': coverage_dispute,
        'line_items_disputed': line_items_disputed,
        'trades_involved': trades_involved,
        'mitigation_performed': mitigation_performed,
        'mitigation_cost': mitigation_cost,
        'mitigation_disputed_pct': mitigation_disputed_pct
    }])
    
    # Encode categoricals
    categorical_cols = ['carrier_philosophy', 'adjuster_type', 'ia_skill_level', 
                       'loss_type', 'geographic_setting', 'policy_type', 'pa_firm_type']
    
    for col in categorical_cols:
        if col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
            except:
                input_data[col] = 0
    
    # Predict award
    predicted_award = award_model.predict(input_data)[0]
    
    # Predict complexity
    complexity_pred = complexity_model.predict(input_data)[0]
    complexity_class = complexity_encoder.inverse_transform([complexity_pred])[0]
    
    # Timeline estimates
    timeline_map = {
        'Simple': '30-60 days',
        'Moderate': '60-90 days',
        'Complex': '90-180 days',
        'High-Complexity': '180+ days'
    }
    timeline = timeline_map.get(complexity_class, '60-120 days')
    
    # Calculate carrier's additional payment
    carrier_additional = predicted_award - carrier_estimate
    carrier_additional_pct = (carrier_additional / carrier_estimate) * 100
    
    # Display results
    st.markdown("---")
    st.header("📊 Prediction Results")
    
    # Main metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "Predicted Award",
            f"${predicted_award:,.0f}",
            help="Expected appraisal award amount"
        )
    
    with metric_col2:
        delta_color = "inverse" if carrier_additional > 0 else "normal"
        st.metric(
            "Carrier Must Pay",
            f"${carrier_additional:,.0f}",
            f"{carrier_additional_pct:+.0f}%",
            delta_color=delta_color,
            help="Additional payment beyond carrier estimate"
        )
    
    with metric_col3:
        st.metric(
            "Complexity",
            complexity_class,
            help="Operational complexity level"
        )
    
    with metric_col4:
        st.metric(
            "Timeline",
            timeline,
            help="Expected duration"
        )
    
    # Breakdown
    st.subheader("💰 Award Breakdown")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        st.metric("Carrier Estimate", f"${carrier_estimate:,.0f}")
    
    with breakdown_col2:
        st.metric("Predicted Award", f"${predicted_award:,.0f}")
    
    with breakdown_col3:
        st.metric("Policyholder Demand", f"${demand_estimate:,.0f}")
    
    # Award position
    award_position_pct = ((predicted_award - carrier_estimate) / (demand_estimate - carrier_estimate)) * 100
    
    st.progress(award_position_pct / 100)
    st.caption(f"Award lands at **{award_position_pct:.0f}%** between carrier position (0%) and demand (100%)")
    
    # Key factors
    st.subheader("🎯 Key Factors")
    
    factors = []
    
    if coverage_dispute:
        factors.append("⚠️ Coverage disputes add $15K-25K to awards")
    
    if pa_attorney:
        factors.append("⚠️ Attorney involvement increases award by ~12%")
    
    if carrier_philosophy == 'tight':
        factors.append("⚠️ Tight carrier: awards typically 20-30% higher")
    
    if is_roof_dispute and not carrier_engineer:
        factors.append("✅ Roof dispute without engineer: carrier will likely lose")
    
    if carrier_engineer:
        factors.append("✅ Engineer retained: reduces award by ~$15K average")
    
    if mitigation_disputed_pct > 0.3:
        factors.append("⚠️ High mitigation dispute: umpires tend to side with contractors")
    
    if line_items_disputed > 50:
        factors.append(f"⚠️ {line_items_disputed} line items: increases award complexity")
    
    if factors:
        for factor in factors:
            st.markdown(factor)
    
    # Recommendation
    st.subheader("💡 Recommendation")
    
    settlement_target = carrier_estimate + (carrier_additional * 0.85)
    
    if carrier_additional > 30000:
        st.warning(f"""
        **High exposure case** - Carrier faces ${carrier_additional:,.0f} additional payment.
        
        **Settlement Strategy:** Consider settling at ${settlement_target:,.0f} (15% discount from predicted award) 
        to avoid {timeline} appraisal process and uncertainty.
        """)
    elif carrier_additional > 15000:
        st.info(f"""
        **Moderate exposure** - ${carrier_additional:,.0f} additional payment expected.
        
        Consider settlement at ${settlement_target:,.0f} to avoid appraisal costs and delays.
        """)
    else:
        st.success(f"""
        **Low exposure** - Predicted award close to carrier estimate.
        
        Defensible position. Proceed to appraisal if settlement cannot be reached at ${predicted_award:,.0f}.
        """)

# Footer
st.markdown("---")
st.caption(f"Model: {metadata['award_predictor_model']} | MAE: ${metadata['award_mae']:,.0f} | R²: {metadata['award_r2']:.2f} | Trained on {metadata['training_samples']:,} appraisals")