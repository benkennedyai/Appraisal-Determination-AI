import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from anthropic import Anthropic

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Appraisal Award Predictor",
    page_icon="⚖️",
    layout="wide"
)

# ============================================================
# ANTHROPIC CLIENT
# ============================================================
api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY not found. Add it to .streamlit/secrets.toml")
    st.stop()
client = Anthropic(api_key=api_key)

# ============================================================
# LOAD ML MODELS
# ============================================================
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
    st.error(f"Error loading models: {e}")
    st.info("Please run the training script first to generate model files.")

# ============================================================
# HEADER
# ============================================================
st.title("⚖️ Insurance Appraisal Award Predictor")
st.markdown("Predict appraisal award amounts and complexity based on claim characteristics at demand receipt.")

if not models_loaded:
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Award Prediction MAE", f"${metadata['award_mae']:,.0f}")
    st.metric("Award R² Score", f"{metadata['award_r2']:.3f}")
    st.metric("Complexity Accuracy", f"{metadata['complexity_accuracy']:.1%}")
    st.metric("Training Samples", f"{metadata['training_samples']:,}")
    st.divider()
    st.header("🤖 AI Analysis")
    st.caption(
        "After running a prediction, an LLM will generate a plain-English "
        "appraisal review summary and answer follow-up questions about the claim."
    )

# ============================================================
# MAIN INPUT FORM
# ============================================================
st.header("📝 Enter Claim Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Financial")
    carrier_estimate = st.number_input(
        "Carrier Estimate ($)", min_value=5000, max_value=500000, value=50000, step=5000
    )
    demand_estimate = st.number_input(
        "Demand Amount ($)", min_value=carrier_estimate, max_value=1000000,
        value=int(carrier_estimate * 2.5), step=5000
    )
    dispute_amount = demand_estimate - carrier_estimate
    dispute_percentage = (dispute_amount / carrier_estimate) * 100
    st.info(f"**Gap:** ${dispute_amount:,.0f} ({dispute_percentage:.0f}%)")

with col2:
    st.subheader("Carrier & Loss")
    carrier_philosophy = st.selectbox(
        "Carrier Type", options=['liberal', 'moderate', 'tight'], index=1,
        help="Tight carriers (Allstate-like) defend aggressively"
    )
    loss_type = st.selectbox(
        "Loss Type", options=['hail', 'wind', 'water', 'fire', 'hurricane', 'vehicle_impact'], index=0
    )
    is_roof_dispute = st.checkbox("Roof Repair/Replace Dispute", value=False)
    carrier_engineer = st.checkbox("Engineer Retained", value=False)

with col3:
    st.subheader("Representation & Issues")
    pa_involved = st.checkbox("Public Adjuster", value=True)
    pa_attorney = st.checkbox("PA Attorney", value=False) if pa_involved else False
    coverage_dispute = st.checkbox("Coverage Issues", value=False)
    line_items_disputed = st.slider("Disputed Line Items", min_value=5, max_value=200, value=30)

with st.expander("⚙️ Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        adjuster_type = st.selectbox("Adjuster Type", ['staff', 'independent'], index=1)
        ia_skill_level = (
            st.selectbox("IA Skill", ['high', 'medium', 'low'], index=1)
            if adjuster_type == 'independent' else 'staff'
        )
        property_age_years = st.slider("Property Age", 0, 100, 20)
        prior_claims_count = st.selectbox("Prior Claims", [0, 1, 2, 3], index=0)
    with adv_col2:
        geographic_setting = st.selectbox("Location", ['suburban', 'urban', 'rural'], index=0)
        policy_type = st.selectbox("Policy Type", ['RCV', 'ACV'], index=0)
        is_catastrophe = st.checkbox("CAT Claim", value=False)
        pa_firm_type = (
            st.selectbox("PA Firm", ['local', 'regional', 'national'], index=0)
            if pa_involved else 'none'
        )
    mit_col1, mit_col2 = st.columns(2)
    with mit_col1:
        mitigation_performed = st.checkbox("Mitigation Services", value=False)
    with mit_col2:
        if mitigation_performed:
            mitigation_cost = st.number_input("Mitigation Cost", 0, 100000, 10000, step=1000)
            mitigation_disputed_pct = st.slider("% Disputed", 0, 100, 30) / 100
        else:
            mitigation_cost = 0
            mitigation_disputed_pct = 0
    supplements_issued = st.slider("Supplements Issued", 0, 10, 1)
    trades_involved = st.slider("Trades Involved", 1, 12, 3)


# ============================================================
# HELPER: FORMAT CURRENCY FOR PROMPTS
# Streamlit's markdown renderer treats $number as LaTeX and
# mangles it. We write "USD X,XXX" in prompts so Claude
# outputs amounts in a format that renders cleanly.
# ============================================================
def fmt(amount):
    return "USD {:,.0f}".format(amount)


# ============================================================
# HELPER: BUILD SYSTEM PROMPT
# ============================================================
def build_system_prompt(claim_inputs, model_outputs):

    if claim_inputs['mitigation_performed']:
        mit_line = "Yes ({}, {:.0f}% disputed)".format(
            fmt(claim_inputs['mitigation_cost']),
            claim_inputs['mitigation_disputed_pct'] * 100
        )
    else:
        mit_line = "No"

    if claim_inputs['adjuster_type'] == 'independent':
        adjuster_line = "Independent (Skill: {})".format(claim_inputs['ia_skill_level'])
    else:
        adjuster_line = "Staff"

    pa_line = "Yes" if claim_inputs['pa_involved'] else "No"
    if claim_inputs['pa_attorney']:
        pa_line += " + Attorney"

    pa_firm_line = claim_inputs['pa_firm_type'].title() if claim_inputs['pa_involved'] else "N/A"

    lines = [
        "You are a senior property claims analyst with 20+ years of experience in insurance "
        "appraisal proceedings. You are assisting an adjuster reviewing a specific claim that "
        "has received an appraisal demand.",
        "",
        "Your role is to explain the ML model's prediction in plain English, highlight the key "
        "risk factors, and help the adjuster think through their options. You are a tool to "
        "support human decision-making, not to replace it.",
        "",
        "IMPORTANT: When writing dollar amounts, always write them as 'USD X,XXX' format "
        "(e.g. 'USD 75,000' not '$75,000'). Never use a dollar sign followed immediately by a number.",
        "",
        "## CLAIM FACTS",
        "- Carrier Estimate: {}".format(fmt(claim_inputs['carrier_estimate'])),
        "- Policyholder Demand: {}".format(fmt(claim_inputs['demand_estimate'])),
        "- Dispute Amount: {} ({:.0f}% above carrier)".format(
            fmt(claim_inputs['dispute_amount']), claim_inputs['dispute_percentage']
        ),
        "- Loss Type: {}".format(claim_inputs['loss_type'].replace('_', ' ').title()),
        "- Carrier Philosophy: {}".format(claim_inputs['carrier_philosophy'].title()),
        "- Adjuster: {}".format(adjuster_line),
        "- Roof Dispute: {}".format("Yes" if claim_inputs['is_roof_dispute'] else "No"),
        "- Engineer Retained: {}".format("Yes" if claim_inputs['carrier_engineer'] else "No"),
        "- Public Adjuster: {}".format(pa_line),
        "- PA Firm Type: {}".format(pa_firm_line),
        "- Coverage Dispute: {}".format("Yes" if claim_inputs['coverage_dispute'] else "No"),
        "- Line Items Disputed: {}".format(claim_inputs['line_items_disputed']),
        "- Prior Claims: {}".format(claim_inputs['prior_claims_count']),
        "- Catastrophe Claim: {}".format("Yes" if claim_inputs['is_catastrophe'] else "No"),
        "- Property Age: {} years".format(claim_inputs['property_age_years']),
        "- Policy Type: {}".format(claim_inputs['policy_type']),
        "- Mitigation Performed: {}".format(mit_line),
        "- Geographic Setting: {}".format(claim_inputs['geographic_setting'].title()),
        "",
        "## MODEL PREDICTIONS",
        "- Predicted Award: {}".format(fmt(model_outputs['predicted_award'])),
        "- Award Position: {:.0f}% between carrier estimate (0%) and demand (100%)".format(
            model_outputs['award_position_pct']
        ),
        "- Carrier Additional Exposure: {} ({:+.0f}%)".format(
            fmt(model_outputs['carrier_additional']), model_outputs['carrier_additional_pct']
        ),
        "- Complexity Class: {}".format(model_outputs['complexity_class']),
        "- Estimated Timeline: {}".format(model_outputs['timeline']),
        "- Model Margin of Error: +/- {}".format(fmt(model_outputs['model_mae'])),
        "",
        "## GUARDRAILS",
        "1. Only discuss THIS specific claim.",
        "2. Do NOT provide legal advice. If asked, say: That is a legal question - consult coverage counsel.",
        "3. Do NOT speculate about causation, fraud, or bad faith.",
        "4. Always anchor to the model's predicted award with its margin of error.",
        "5. Be concise. Use short paragraphs and bullet points.",
        "6. If something is outside your scope, say so plainly.",
        "7. End substantive recommendations with a reminder that final decisions rest with the adjuster and their supervisor.",
    ]

    return "\n".join(lines)


# ============================================================
# HELPER: GENERATE INITIAL NARRATIVE
# ============================================================
def generate_narrative(claim_inputs, model_outputs):
    system_prompt = build_system_prompt(claim_inputs, model_outputs)

    user_message = "\n".join([
        "Please provide a concise appraisal review summary for this demand.",
        "Structure it with these four bold headers:",
        "",
        "**Situation Summary**",
        "What this claim is and where the key dispute lies. 2-3 sentences.",
        "",
        "**Predicted Outcome**",
        "What the model predicts and why, in plain English. 2-3 sentences.",
        "",
        "**Top Risk Factors**",
        "The 2-3 factors most likely to move the award against the carrier. Use bullet points.",
        "",
        "**Recommended Action**",
        "Settlement vs appraisal recommendation with a target figure.",
        "",
        "Formatting rules:",
        "- Write all dollar amounts as 'USD X,XXX' (never use a dollar sign before a number)",
        "- Do not use italic formatting",
        "- Bold headers only, plain text for body",
        "- Keep under 300 words",
        "- Write for a claims adjuster, not a data scientist",
    ])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text


# ============================================================
# HELPER: CHAT RESPONSE
# ============================================================
def get_chat_response(claim_inputs, model_outputs, chat_history):
    system_prompt = build_system_prompt(claim_inputs, model_outputs)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system_prompt,
        messages=chat_history
    )
    return response.content[0].text


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
if 'narrative' not in st.session_state:
    st.session_state.narrative = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'claim_inputs' not in st.session_state:
    st.session_state.claim_inputs = None
if 'model_outputs' not in st.session_state:
    st.session_state.model_outputs = None


# ============================================================
# PREDICT BUTTON
# ============================================================
if st.button("🔮 Predict Appraisal Award", type="primary", use_container_width=True):

    st.session_state.chat_history = []
    st.session_state.narrative = None

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

    categorical_cols = [
        'carrier_philosophy', 'adjuster_type', 'ia_skill_level',
        'loss_type', 'geographic_setting', 'policy_type', 'pa_firm_type'
    ]
    for col in categorical_cols:
        if col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
            except Exception:
                input_data[col] = 0

    predicted_award = award_model.predict(input_data)[0]
    complexity_pred = complexity_model.predict(input_data)[0]
    complexity_class = complexity_encoder.inverse_transform([complexity_pred])[0]

    timeline_map = {
        'Simple': '30-60 days',
        'Moderate': '60-90 days',
        'Complex': '90-180 days',
        'High-Complexity': '180+ days'
    }
    timeline = timeline_map.get(complexity_class, '60-120 days')

    carrier_additional = predicted_award - carrier_estimate
    carrier_additional_pct = (carrier_additional / carrier_estimate) * 100
    award_position_pct = (
        (predicted_award - carrier_estimate) / (demand_estimate - carrier_estimate) * 100
    )

    st.session_state.claim_inputs = {
        'carrier_estimate': carrier_estimate,
        'demand_estimate': demand_estimate,
        'dispute_amount': dispute_amount,
        'dispute_percentage': dispute_percentage,
        'carrier_philosophy': carrier_philosophy,
        'loss_type': loss_type,
        'adjuster_type': adjuster_type,
        'ia_skill_level': ia_skill_level,
        'is_roof_dispute': is_roof_dispute,
        'carrier_engineer': carrier_engineer,
        'pa_involved': pa_involved,
        'pa_attorney': pa_attorney,
        'pa_firm_type': pa_firm_type if pa_involved else 'none',
        'coverage_dispute': coverage_dispute,
        'line_items_disputed': line_items_disputed,
        'prior_claims_count': prior_claims_count,
        'is_catastrophe': is_catastrophe,
        'property_age_years': property_age_years,
        'policy_type': policy_type,
        'mitigation_performed': mitigation_performed,
        'mitigation_cost': mitigation_cost,
        'mitigation_disputed_pct': mitigation_disputed_pct,
        'geographic_setting': geographic_setting,
        'supplements_issued': supplements_issued,
        'trades_involved': trades_involved,
    }

    st.session_state.model_outputs = {
        'predicted_award': predicted_award,
        'complexity_class': complexity_class,
        'timeline': timeline,
        'carrier_additional': carrier_additional,
        'carrier_additional_pct': carrier_additional_pct,
        'award_position_pct': award_position_pct,
        'model_mae': metadata['award_mae'],
    }

    st.session_state.prediction_run = True

    with st.spinner("Generating Appraisal Review Summary..."):
        st.session_state.narrative = generate_narrative(
            st.session_state.claim_inputs,
            st.session_state.model_outputs
        )


# ============================================================
# RESULTS
# ============================================================
if st.session_state.prediction_run and st.session_state.model_outputs:

    outputs = st.session_state.model_outputs
    inputs = st.session_state.claim_inputs

    st.markdown("---")
    st.header("📊 Prediction Results")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Predicted Award", f"${outputs['predicted_award']:,.0f}")
    with m2:
        delta_color = "inverse" if outputs['carrier_additional'] > 0 else "normal"
        st.metric(
            "Increase From Carrier Estimate",
            f"${outputs['carrier_additional']:,.0f}",
            f"{outputs['carrier_additional_pct']:+.0f}%",
            delta_color=delta_color
        )
    with m3:
        st.metric("Complexity", outputs['complexity_class'])
    with m4:
        st.metric("Timeline", outputs['timeline'])

    st.subheader("💰 Award Breakdown")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Carrier Estimate", f"${inputs['carrier_estimate']:,.0f}")
    with b2:
        st.metric("Predicted Award", f"${outputs['predicted_award']:,.0f}")
    with b3:
        st.metric("Policyholder Demand", f"${inputs['demand_estimate']:,.0f}")

    st.progress(min(outputs['award_position_pct'] / 100, 1.0))
    st.caption(
        f"Award lands at **{outputs['award_position_pct']:.0f}%** "
        "between carrier position (0%) and demand (100%)"
    )

    st.subheader("🎯 Key Factors")
    factors = []
    if inputs['coverage_dispute']:
        factors.append("⚠️ Coverage disputes add $15K-25K to awards")
    if inputs['pa_attorney']:
        factors.append("⚠️ Attorney involvement increases award by ~12%")
    if inputs['carrier_philosophy'] == 'tight':
        factors.append("⚠️ Tight carrier: awards typically 20-30% higher")
    if inputs['is_roof_dispute'] and not inputs['carrier_engineer']:
        factors.append("✅ Roof dispute without engineer: carrier will likely lose")
    if inputs['carrier_engineer']:
        factors.append("✅ Engineer retained: reduces award by ~$15K average")
    if inputs['mitigation_disputed_pct'] > 0.3:
        factors.append("⚠️ High mitigation dispute: umpires tend to side with contractors")
    if inputs['line_items_disputed'] > 50:
        factors.append(f"⚠️ {inputs['line_items_disputed']} line items: increases award complexity")
    for factor in factors:
        st.markdown(factor)

    # Appraisal Review Summary
    st.markdown("---")
    st.header("📋 Appraisal Review Summary")
    if st.session_state.narrative:
        # Disable LaTeX rendering by wrapping in a container
        st.markdown(st.session_state.narrative)

    # Follow-up chat
    st.markdown("---")
    st.header("💬 Ask a Follow-Up Question")
    st.caption(
        "Ask about specific risk factors, settlement strategy, "
        "what happens in appraisal, or anything else about this claim."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask about this claim..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(
                    st.session_state.claim_inputs,
                    st.session_state.model_outputs,
                    st.session_state.chat_history
                )
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    f"Model: {metadata['award_predictor_model']} | "
    f"MAE: ${metadata['award_mae']:,.0f} | "
    f"R\u00b2: {metadata['award_r2']:.2f} | "
    f"Trained on {metadata['training_samples']:,} appraisals | "
    
)
