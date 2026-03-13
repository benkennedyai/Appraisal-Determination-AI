import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_appraisal_dataset(n_samples=1000):
    """
    Generate synthetic appraisal demand dataset.
    
    Scope: Features available AT THE MOMENT appraisal demand is received.
    Goal: Predict complexity/timeline for resource allocation and settlement decisions.
    
    Key Insight: This represents the 1-15% of claims that reach appraisal demand.
    Most claims (98%+) resolve without appraisal.
    """
    
    data = []
    
    # Carrier loss adjustment philosophies
    carrier_types = {
        'tight': {'weight': 0.25, 'appraisal_rate': 0.12, 'win_rate': 0.05},  # Allstate-like
        'moderate': {'weight': 0.50, 'appraisal_rate': 0.03, 'win_rate': 0.30},
        'liberal': {'weight': 0.25, 'appraisal_rate': 0.01, 'win_rate': 0.50}
    }
    
    for i in range(n_samples):
        # ==================== CARRIER FACTORS ====================
        
        # Carrier loss adjustment philosophy - CRITICAL
        carrier_philosophy = np.random.choice(
            list(carrier_types.keys()),
            p=[carrier_types[k]['weight'] for k in carrier_types.keys()]
        )
        
        # Staff vs Independent adjuster
        # Tight carriers use more IAs (cheaper), liberal use more staff (better quality)
        if carrier_philosophy == 'tight':
            adjuster_type = np.random.choice(['staff', 'independent'], p=[0.30, 0.70])
        elif carrier_philosophy == 'moderate':
            adjuster_type = np.random.choice(['staff', 'independent'], p=[0.50, 0.50])
        else:  # liberal
            adjuster_type = np.random.choice(['staff', 'independent'], p=[0.70, 0.30])
        
        # IA firm skill level (only if independent adjuster)
        if adjuster_type == 'independent':
            # Tight carriers tend to use cheaper (lower skill) IA firms
            if carrier_philosophy == 'tight':
                ia_skill = np.random.choice(['low', 'medium', 'high'], p=[0.50, 0.35, 0.15])
            elif carrier_philosophy == 'moderate':
                ia_skill = np.random.choice(['low', 'medium', 'high'], p=[0.25, 0.50, 0.25])
            else:  # liberal
                ia_skill = np.random.choice(['low', 'medium', 'high'], p=[0.10, 0.40, 0.50])
        else:
            ia_skill = None  # Staff adjuster
        
        # ==================== PROPERTY & LOSS CHARACTERISTICS ====================
        
        # Property type - RESIDENTIAL ONLY
        property_type = 'residential'
        
        # Loss type (peril)
        loss_type = np.random.choice(
            ['hail', 'wind', 'hurricane', 'fire', 'water', 'vehicle_impact'],
            p=[0.35, 0.20, 0.10, 0.15, 0.15, 0.05]
        )
        
        # CAT vs daily claim
        is_cat = loss_type == 'hurricane' or (loss_type in ['hail', 'wind'] and np.random.random() < 0.15)
        
        # Geographic setting
        geographic_setting = np.random.choice(
            ['urban', 'suburban', 'rural'],
            p=[0.35, 0.45, 0.20]
        )
        
        # Property age
        property_age_years = int(np.random.gamma(3, 8))  # Skewed toward older properties
        property_age_years = min(property_age_years, 100)
        
        # Policy type
        policy_type = np.random.choice(['RCV', 'ACV'], p=[0.75, 0.25])
        
        # Total insured value - RESIDENTIAL ONLY
        tiv = np.random.uniform(200000, 600000)
        
        # Prior claim history (reduces carrier's trust, harder negotiations)
        prior_claims = np.random.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.10, 0.05])
        has_prior_claims = prior_claims > 0
        
        # Emergency/mitigation services performed
        # More common for water/fire, opportunistic billing creates disputes
        if loss_type in ['water', 'fire']:
            mitigation_performed = np.random.random() < 0.70
        else:
            mitigation_performed = np.random.random() < 0.20
        
        if mitigation_performed:
            # Mitigation costs - often inflated, creates disputes
            mitigation_cost = np.random.uniform(2000, 25000)
            # Disputed mitigation percentage
            mitigation_disputed_pct = np.random.uniform(0, 0.60)  # 0-60% of mitigation disputed
        else:
            mitigation_cost = 0
            mitigation_disputed_pct = 0
        
        # ==================== DISPUTE CHARACTERISTICS ====================
        
        # Roof replacement vs repair dispute
        # Tight carriers fight repairs more, liberal carriers total roofs readily
        if carrier_philosophy == 'tight':
            roof_dispute_prob = 0.75
        elif carrier_philosophy == 'moderate':
            roof_dispute_prob = 0.55
        else:  # liberal
            roof_dispute_prob = 0.35
        
        is_roof_dispute = loss_type in ['hail', 'wind', 'hurricane'] and np.random.random() < roof_dispute_prob
        
        # Carrier estimate
        # RESIDENTIAL: Typical claims are 5-20% of home value
        damage_severity = np.random.uniform(0.05, 0.20)
        base_carrier_estimate = tiv * damage_severity
        
        # LARGE LOSSES get better adjusters and more resources
        # Large losses = smaller disputes relative to estimate
        if base_carrier_estimate > 250000:
            # High-quality adjuster handling (assumed for large losses)
            # Force to staff or high-skill IA
            if adjuster_type == 'independent':
                ia_skill = 'high'  # Large losses = high skill IA only
            adjuster_type = np.random.choice(['staff', 'independent'], p=[0.70, 0.30])
            
            # Large losses also have consultants, engineers, etc.
            # Carrier position is more defensible
            carrier_estimate = base_carrier_estimate * np.random.uniform(0.95, 1.05)  # Very accurate
            
        elif base_carrier_estimate > 100000:
            # Medium losses - medium or high skill only
            if adjuster_type == 'independent' and ia_skill == 'low':
                ia_skill = 'medium'  # No low-skill on 100k+ claims
            carrier_estimate = base_carrier_estimate * np.random.uniform(0.90, 1.00)
            
        else:
            # Small losses - original logic applies
            # Low-skill IA = worse estimates = bigger disputes
            if adjuster_type == 'independent' and ia_skill == 'low':
                carrier_estimate = base_carrier_estimate * np.random.uniform(0.65, 0.85)  # Lowball by 15-35%
            elif adjuster_type == 'independent' and ia_skill == 'medium':
                carrier_estimate = base_carrier_estimate * np.random.uniform(0.80, 0.95)  # Lowball by 5-20%
            else:
                carrier_estimate = base_carrier_estimate
        
        # PA involvement (higher for larger claims and tight carriers)
        if carrier_estimate > 15000:
            if carrier_philosophy == 'tight':
                pa_involved = np.random.random() < 0.85
            elif carrier_philosophy == 'moderate':
                pa_involved = np.random.random() < 0.70
            else:  # liberal
                pa_involved = np.random.random() < 0.55
        else:
            pa_involved = np.random.random() < 0.30
        
        # PA demand/estimate
        if not pa_involved:
            # Policyholder self-represented - less sophisticated
            demand_estimate = carrier_estimate * np.random.uniform(1.1, 1.5)
        else:
            # PA involved - demand depends on carrier philosophy and dispute type
            # LARGE LOSSES: PA demands are more reasonable (everyone has experts)
            if carrier_estimate > 250000:
                # Large losses = smaller multipliers (both sides have experts)
                if is_roof_dispute:
                    demand_estimate = carrier_estimate * np.random.uniform(1.2, 1.8)
                else:
                    demand_estimate = carrier_estimate * np.random.uniform(1.1, 1.6)
            
            elif carrier_estimate > 100000:
                # Medium losses
                if is_roof_dispute:
                    if carrier_philosophy == 'tight':
                        demand_estimate = carrier_estimate * np.random.uniform(2.0, 3.0)
                    elif carrier_philosophy == 'moderate':
                        demand_estimate = carrier_estimate * np.random.uniform(1.5, 2.5)
                    else:  # liberal
                        demand_estimate = carrier_estimate * np.random.uniform(1.2, 1.8)
                else:
                    if carrier_philosophy == 'tight':
                        demand_estimate = carrier_estimate * np.random.uniform(1.5, 2.5)
                    elif carrier_philosophy == 'moderate':
                        demand_estimate = carrier_estimate * np.random.uniform(1.3, 2.0)
                    else:  # liberal
                        demand_estimate = carrier_estimate * np.random.uniform(1.1, 1.6)
            
            else:
                # SMALL LOSSES: Original wild multipliers apply
                if is_roof_dispute:
                    if carrier_philosophy == 'tight':
                        demand_estimate = carrier_estimate * np.random.uniform(3.0, 5.5)
                    elif carrier_philosophy == 'moderate':
                        demand_estimate = carrier_estimate * np.random.uniform(2.0, 3.5)
                    else:  # liberal
                        demand_estimate = carrier_estimate * np.random.uniform(1.3, 2.0)
                else:
                    if carrier_philosophy == 'tight':
                        demand_estimate = carrier_estimate * np.random.uniform(2.0, 3.5)
                    elif carrier_philosophy == 'moderate':
                        demand_estimate = carrier_estimate * np.random.uniform(1.5, 2.5)
                    else:  # liberal
                        demand_estimate = carrier_estimate * np.random.uniform(1.2, 1.8)
        
        # Add mitigation cost to demand if disputed
        if mitigation_performed:
            demand_estimate += mitigation_cost * (1 - mitigation_disputed_pct)
        
        dispute_amount = demand_estimate - carrier_estimate
        dispute_pct = (dispute_amount / carrier_estimate) * 100
        
        # PA attorney involvement (higher for larger disputes and prior claims)
        if pa_involved:
            attorney_prob = 0.15
            if dispute_amount > 50000:
                attorney_prob += 0.25
            if has_prior_claims:
                attorney_prob += 0.15  # Prior claims = contentious relationship
            pa_attorney = np.random.random() < min(attorney_prob, 0.70)
        else:
            pa_attorney = False
        
        # PA firm type
        if pa_involved:
            if pa_attorney:
                # Attorney PAs tend to be more sophisticated
                pa_firm_type = np.random.choice(['local', 'regional', 'national'], p=[0.30, 0.35, 0.35])
            else:
                pa_firm_type = np.random.choice(['local', 'regional', 'national'], p=[0.50, 0.30, 0.20])
        else:
            pa_firm_type = None
        
        # Carrier engineering involvement
        # More common for tight carriers defending positions, roof disputes >$20k
        if is_roof_dispute and carrier_estimate > 20000:
            if carrier_philosophy == 'tight':
                engineer_prob = 0.25
            elif carrier_philosophy == 'moderate':
                engineer_prob = 0.12
            else:
                engineer_prob = 0.05
            carrier_engineer = np.random.random() < engineer_prob
        else:
            carrier_engineer = False
        
        # Coverage considerations/disputes
        # More common with prior claims (suspicion), mitigation disputes, older properties
        coverage_dispute_prob = 0.10
        if has_prior_claims:
            coverage_dispute_prob += 0.10
        if mitigation_performed and mitigation_disputed_pct > 0.30:
            coverage_dispute_prob += 0.08
        if property_age_years > 40:
            coverage_dispute_prob += 0.05
        
        coverage_dispute = np.random.random() < coverage_dispute_prob
        
        # Number of line items in dispute
        if is_roof_dispute:
            line_items = np.random.randint(5, 25)
        elif loss_type == 'fire':
            line_items = np.random.randint(30, 200)
        elif loss_type == 'water':
            line_items = np.random.randint(15, 80)
        else:
            line_items = np.random.randint(10, 60)
        
        # Number of trades involved
        if loss_type == 'fire':
            trades_involved = np.random.randint(4, 12)
        elif loss_type in ['hail', 'wind', 'hurricane']:
            trades_involved = np.random.randint(1, 4)
        else:
            trades_involved = np.random.randint(2, 6)
        
        # Prior supplements issued (contentious history)
        supplements = np.random.poisson(1.2) if pa_involved else np.random.poisson(0.3)
        
        # ==================== COMPLEXITY SCORING ====================
        # COMPLEXITY = OPERATIONAL DIFFICULTY, NOT JUST DOLLAR AMOUNT
        # Focus on: # of issues, parties involved, legal complexity, process friction
        
        complexity_score = 0
        
        # CARRIER PHILOSOPHY - process difficulty
        if carrier_philosophy == 'tight':
            complexity_score += 25  # Difficult to work with
        elif carrier_philosophy == 'moderate':
            complexity_score += 10
        else:  # liberal
            complexity_score -= 5  # Cooperative
        
        # IA SKILL - process friction
        if adjuster_type == 'independent':
            if ia_skill == 'low':
                complexity_score += 15  # More back-and-forth, corrections needed
            elif ia_skill == 'medium':
                complexity_score += 6
            else:  # high
                complexity_score -= 3  # Smooth process
        
        # ROOF DISPUTES - actually LESS operationally complex (predictable outcome)
        if is_roof_dispute and not carrier_engineer:
            complexity_score -= 12  # Straightforward, predictable
        elif is_roof_dispute and carrier_engineer:
            complexity_score += 18  # Engineering reports = more work
        
        # CAT CLAIMS - actually smoother (less scrutiny, established processes)
        if is_cat:
            complexity_score -= 12
        
        # FIRE/WATER = genuinely more complex (multiple systems, causation)
        if loss_type == 'fire':
            complexity_score += 15  # Multiple trades, causation issues
        elif loss_type == 'water':
            complexity_score += 10  # Hidden damage, mitigation disputes
        
        # ATTORNEY INVOLVEMENT - major process complexity
        if pa_attorney:
            complexity_score += 25  # Legal process, formal discovery, delays
        
        # LINE ITEMS - operational complexity (each item needs review)
        if line_items > 100:
            complexity_score += 20
        elif line_items > 50:
            complexity_score += 12
        elif line_items > 25:
            complexity_score += 6
        
        # TRADES INVOLVED - coordination complexity
        if trades_involved > 6:
            complexity_score += 15  # Many experts needed
        elif trades_involved > 3:
            complexity_score += 8
        
        # COVERAGE DISPUTES - MAJOR complexity driver (legal, not just scope)
        if coverage_dispute:
            complexity_score += 35  # Appraisers can't resolve, adds legal layer
        
        # PRIOR CLAIMS - process friction (suspicion, extra scrutiny)
        if prior_claims >= 2:
            complexity_score += 15
        elif prior_claims == 1:
            complexity_score += 8
        
        # MITIGATION DISPUTES - genuine operational complexity
        if mitigation_performed and mitigation_disputed_pct > 0.40:
            complexity_score += 20  # Multiple vendors, billing disputes
        elif mitigation_performed and mitigation_disputed_pct > 0.20:
            complexity_score += 12
        
        # SUPPLEMENTS - history of contention
        if supplements > 3:
            complexity_score += 15
        elif supplements > 1:
            complexity_score += 8
        
        # NATIONAL PA FIRMS - more sophisticated process
        if pa_firm_type == 'national':
            complexity_score += 12
        
        # GEOGRAPHIC - rural = logistical complexity
        if geographic_setting == 'rural':
            complexity_score += 10  # Harder to get experts, travel time
        
        # POLICY TYPE - ACV depreciation = complexity
        if policy_type == 'ACV' and property_age_years > 20:
            complexity_score += 12  # Depreciation arguments
        
        # LARGE DISPUTES - some operational impact but not primary driver
        # De-emphasize financial size in complexity
        if dispute_pct > 200:
            complexity_score += 10  # Some added scrutiny
        elif dispute_pct > 150:
            complexity_score += 5
        
        # Add randomness
        complexity_score += np.random.randint(-8, 8)
        
        # NATIONAL PA FIRMS (more sophisticated)
        if pa_firm_type == 'national':
            pa_firm_impact = np.random.normal(6, 5)
            award_position_pct += pa_firm_impact
        
        # CLASSIFY BASED ON OPERATIONAL COMPLEXITY SCORE
        # NOT primarily driven by dollar amount
        if complexity_score < 5:
            complexity_class = 'Simple'
            timeline_days = int(np.random.normal(45, 10))
        elif complexity_score < 25:
            complexity_class = 'Moderate'
            timeline_days = int(np.random.normal(75, 15))
        elif complexity_score < 50:
            complexity_class = 'Complex'
            timeline_days = int(np.random.normal(135, 30))
        else:
            complexity_class = 'High-Complexity'
            timeline_days = int(np.random.normal(210, 45))
        
        timeline_days = max(30, min(timeline_days, 365))
        
        # ==================== APPRAISAL AWARD CALCULATION ====================
        # Award = where umpire lands between carrier estimate and demand
        # Expressed as % movement from carrier position toward demand
        
        # Base award position (50% = split the difference)
        award_position_pct = 50.0
        
        # ENGINEERING REDUCES DELTA (engineer provides defensible position)
        if carrier_engineer:
            # Significant variance but generally helps carrier
            engineering_impact = np.random.normal(-15, 8)  # Avg -15%, StdDev 8%
            award_position_pct += engineering_impact
        
        # IA QUALITY MATTERS
        if adjuster_type == 'independent':
            if ia_skill == 'low':
                # Low skill = indefensible position = award closer to demand
                ia_quality_impact = np.random.normal(25, 10)  # Move toward demand
                award_position_pct += ia_quality_impact
            elif ia_skill == 'medium':
                # Medium = slight move toward demand
                ia_quality_impact = np.random.normal(8, 8)
                award_position_pct += ia_quality_impact
            else:  # high skill
                # High skill = defensible position = carrier wins more
                ia_quality_impact = np.random.normal(-15, 10)  # Stronger carrier position
                award_position_pct += ia_quality_impact
        else:
            # STAFF ADJUSTER = between bad and average IA
            # Staff tend to be better than low-skill IA but worse than high-skill
            staff_impact = np.random.normal(12, 10)  # Worse than medium IA
            award_position_pct += staff_impact
        
        # LARGE LOSSES: Better adjusters = carrier wins more
        if carrier_estimate > 250000:
            # Large losses have quality adjusters, consultants, experts
            # Carrier position is more defensible
            large_loss_impact = np.random.normal(-12, 8)
            award_position_pct += large_loss_impact
        
        # PRIOR CLAIMS REDUCE AWARD (casts doubt on insured credibility)
        if prior_claims > 0:
            prior_claims_impact = np.random.normal(-8 * prior_claims, 5)
            award_position_pct += prior_claims_impact
        
        # MITIGATION DISPUTES INCREASE AWARD (umpires side with contractors)
        if mitigation_performed and mitigation_disputed_pct > 0.20:
            # Higher disputed % = more umpire sides with contractor
            mitigation_impact = mitigation_disputed_pct * np.random.uniform(20, 40)
            award_position_pct += mitigation_impact
        
        # ROOF DISPUTES (carriers lose 95% unless engineer)
        if is_roof_dispute and not carrier_engineer:
            if carrier_philosophy == 'tight':
                # Tight carriers on roof disputes = almost always lose position
                roof_impact = np.random.normal(40, 8)  # 95% loss rate
                award_position_pct += roof_impact
            else:
                roof_impact = np.random.normal(35, 8)  # Still high loss rate
                award_position_pct += roof_impact
        
        # CARRIER PHILOSOPHY BASELINE
        # Tight carriers take indefensible positions = lose MUCH more
        # Liberal carriers take fact-based positions = better outcomes
        # TIGHT CARRIERS SHOULD LOSE 2X AS OFTEN AS LIBERAL
        if carrier_philosophy == 'tight':
            philosophy_impact = np.random.normal(25, 10)  # Much worse outcomes
            award_position_pct += philosophy_impact
        elif carrier_philosophy == 'moderate':
            philosophy_impact = np.random.normal(8, 8)
            award_position_pct += philosophy_impact
        else:  # liberal
            philosophy_impact = np.random.normal(-8, 8)  # Better outcomes
            award_position_pct += philosophy_impact
        
        # COVERAGE DISPUTES (appraisers write damage regardless, but complicates)
        # Slight variance but generally neutral
        if coverage_dispute:
            coverage_impact = np.random.normal(0, 10)
            award_position_pct += coverage_impact
        
        # CAT CLAIMS (carriers less aggressive, more reasonable)
        if is_cat:
            cat_impact = np.random.normal(-10, 8)
            award_position_pct += cat_impact
        
        # ATTORNEY INVOLVEMENT (better documentation, more aggressive)
        # Attorneys DO increase awards, PAs alone are modest
        if pa_attorney:
            attorney_impact = np.random.normal(12, 8)  # Attorneys matter
            award_position_pct += attorney_impact
        elif pa_involved and not pa_attorney:
            # PA without attorney - modest effect (most PAs are bullshitters)
            pa_only_impact = np.random.normal(2, 6)  # Small effect
            award_position_pct += pa_only_impact
        
        # NATIONAL PA FIRMS (more sophisticated)
        if pa_firm_type == 'national':
            pa_firm_impact = np.random.normal(6, 5)
            award_position_pct += pa_firm_impact
        
        # Bound the award position between 0-100%
        award_position_pct = max(0, min(award_position_pct, 100))
        
        # Calculate actual award amount
        award_amount = carrier_estimate + (dispute_amount * (award_position_pct / 100))
        
        # SPECIAL CASE: Roof disputes have capped financial consequences
        # Carriers lose position (95%), but net cost is only 10-15k after deductible
        if is_roof_dispute and not carrier_engineer:
            typical_roof_net = np.random.uniform(8000, 18000)
            carrier_to_award_delta_raw = award_amount - carrier_estimate
            
            # If calculated delta exceeds typical roof cost, cap it
            if carrier_to_award_delta_raw > typical_roof_net * 1.5:
                # Recalculate award at capped amount
                award_amount = carrier_estimate + typical_roof_net
                # Recalculate position percentage based on capped amount
                award_position_pct = ((award_amount - carrier_estimate) / dispute_amount * 100) if dispute_amount > 0 else 50
                award_position_pct = max(0, min(award_position_pct, 100))
        
        # Calculate deltas
        carrier_to_award_delta = award_amount - carrier_estimate
        carrier_to_award_delta_pct = (carrier_to_award_delta / carrier_estimate) * 100
        demand_to_award_delta = demand_estimate - award_amount
        demand_to_award_delta_pct = (demand_to_award_delta / demand_estimate) * 100
        
        # Carrier "win" if award < 40% toward demand
        # Policyholder "win" if award > 60% toward demand
        # Split if 40-60%
        if award_position_pct < 40:
            appraisal_outcome = 'Carrier Favorable'
        elif award_position_pct > 60:
            appraisal_outcome = 'Policyholder Favorable'
        else:
            appraisal_outcome = 'Split Decision'
        
        # ==================== BUILD RECORD ====================
        
        data.append({
            # Carrier factors
            'carrier_philosophy': carrier_philosophy,
            'adjuster_type': adjuster_type,
            'ia_skill_level': ia_skill,
            'carrier_engineer': carrier_engineer,
            
            # Property/Loss
            'property_type': property_type,
            'property_age_years': property_age_years,
            'loss_type': loss_type,
            'is_catastrophe': is_cat,
            'geographic_setting': geographic_setting,
            'policy_type': policy_type,
            'total_insured_value': round(tiv, 2),
            
            # Claim history
            'prior_claims_count': prior_claims,
            'supplements_issued': supplements,
            
            # Financial dispute
            'carrier_estimate': round(carrier_estimate, 2),
            'demand_estimate': round(demand_estimate, 2),
            'dispute_amount': round(dispute_amount, 2),
            'dispute_percentage': round(dispute_pct, 2),
            
            # Parties
            'pa_involved': pa_involved,
            'pa_attorney': pa_attorney,
            'pa_firm_type': pa_firm_type,
            
            # Complexity indicators
            'is_roof_dispute': is_roof_dispute,
            'coverage_dispute': coverage_dispute,
            'line_items_disputed': line_items,
            'trades_involved': trades_involved,
            
            # Mitigation
            'mitigation_performed': mitigation_performed,
            'mitigation_cost': round(mitigation_cost, 2) if mitigation_performed else 0,
            'mitigation_disputed_pct': round(mitigation_disputed_pct, 2) if mitigation_performed else 0,
            
            # Target
            'complexity_class': complexity_class,
            'estimated_timeline_days': timeline_days,
            
            # Award/Outcome
            'award_amount': round(award_amount, 2),
            'award_position_pct': round(award_position_pct, 2),
            'carrier_to_award_delta': round(carrier_to_award_delta, 2),
            'carrier_to_award_delta_pct': round(carrier_to_award_delta_pct, 2),
            'demand_to_award_delta': round(demand_to_award_delta, 2),
            'demand_to_award_delta_pct': round(demand_to_award_delta_pct, 2),
            'appraisal_outcome': appraisal_outcome
        })
    
    df = pd.DataFrame(data)
    return df

# ==================== GENERATE & ANALYZE ====================

df = generate_appraisal_dataset(1000)

print("="*70)
print("APPRAISAL DEMAND COMPLEXITY DATASET")
print("="*70)
print(f"\nDataset: {len(df)} appraisal demands")
print("Scope: Features available AT DEMAND RECEIPT")
print("Goal: Predict complexity/timeline for resource allocation\n")
print("NOTE: This represents 1-15% of claims that reach appraisal.")
print("Most claims (98%+) resolve without appraisal.\n")

print("="*70)
print("COMPLEXITY DISTRIBUTION")
print("="*70)
print(df['complexity_class'].value_counts().sort_index())
print(f"\nAverage Timeline: {df['estimated_timeline_days'].mean():.1f} days")

print("\n" + "="*70)
print("CARRIER PHILOSOPHY ANALYSIS")
print("="*70)
print("\nDistribution:")
print(df['carrier_philosophy'].value_counts())
print("\nComplexity by Carrier:")
print(pd.crosstab(df['carrier_philosophy'], df['complexity_class'], 
                  normalize='index').round(3) * 100)
print("\nAvg Timeline by Carrier:")
print(df.groupby('carrier_philosophy')['estimated_timeline_days'].mean().round(1))
print("\nAvg Dispute % by Carrier:")
print(df.groupby('carrier_philosophy')['dispute_percentage'].mean().round(1))

print("\n" + "="*70)
print("ADJUSTER TYPE & IA SKILL ANALYSIS")
print("="*70)
print("\nAdjuster Type:")
print(df['adjuster_type'].value_counts())
print("\nIA Skill Distribution (Independent only):")
print(df[df['adjuster_type']=='independent']['ia_skill_level'].value_counts())
print("\nComplexity by IA Skill (Independent only):")
ia_df = df[df['adjuster_type']=='independent']
print(pd.crosstab(ia_df['ia_skill_level'], ia_df['complexity_class'], 
                  normalize='index').round(3) * 100)

print("\n" + "="*70)
print("ROOF DISPUTE ANALYSIS")
print("="*70)
print(f"\nTotal Roof Disputes: {df['is_roof_dispute'].sum()} ({df['is_roof_dispute'].mean()*100:.1f}%)")
print("\nRoof Disputes by Carrier:")
print(df.groupby('carrier_philosophy')['is_roof_dispute'].sum())
print("\nEngineer Usage (Roof Disputes Only):")
roof_df = df[df['is_roof_dispute']]
print(f"Total: {roof_df['carrier_engineer'].sum()} ({roof_df['carrier_engineer'].mean()*100:.1f}%)")
print("By Carrier:")
print(roof_df.groupby('carrier_philosophy')['carrier_engineer'].mean().round(3) * 100)

print("\n" + "="*70)
print("KEY FEATURES SUMMARY")
print("="*70)
print(f"\nPA Involved: {df['pa_involved'].sum()} ({df['pa_involved'].mean()*100:.1f}%)")
print(f"PA Attorney: {df['pa_attorney'].sum()} ({df['pa_attorney'].mean()*100:.1f}%)")
print(f"Coverage Disputes: {df['coverage_dispute'].sum()} ({df['coverage_dispute'].mean()*100:.1f}%)")
print(f"Prior Claims (any): {df['prior_claims_count'].gt(0).sum()} ({df['prior_claims_count'].gt(0).mean()*100:.1f}%)")
print(f"Mitigation Performed: {df['mitigation_performed'].sum()} ({df['mitigation_performed'].mean()*100:.1f}%)")
print(f"CAT Claims: {df['is_catastrophe'].sum()} ({df['is_catastrophe'].mean()*100:.1f}%)")

print("\n" + "="*70)
print("SAMPLE RECORDS")
print("="*70)
print(df[['carrier_philosophy', 'adjuster_type', 'ia_skill_level', 'loss_type', 
          'is_roof_dispute', 'carrier_engineer', 'pa_involved', 'pa_attorney',
          'dispute_percentage', 'complexity_class', 'estimated_timeline_days']].head(20))

print("\n" + "="*70)
print("DISPUTE SIZE ANALYSIS")
print("="*70)
print(f"\nMedian Dispute Amount: ${df['dispute_amount'].median():,.0f}")
print(f"Average Dispute Amount: ${df['dispute_amount'].mean():,.0f}")
print(f"75th Percentile: ${df['dispute_amount'].quantile(0.75):,.0f}")
print(f"95th Percentile: ${df['dispute_amount'].quantile(0.95):,.0f}")

print("\nDispute Amount Distribution:")
print(f"  < $25k: {(df['dispute_amount'] < 25000).sum()} ({(df['dispute_amount'] < 25000).mean()*100:.1f}%)")
print(f"  $25k-$50k: {((df['dispute_amount'] >= 25000) & (df['dispute_amount'] < 50000)).sum()} ({((df['dispute_amount'] >= 25000) & (df['dispute_amount'] < 50000)).mean()*100:.1f}%)")
print(f"  $50k-$100k: {((df['dispute_amount'] >= 50000) & (df['dispute_amount'] < 100000)).sum()} ({((df['dispute_amount'] >= 50000) & (df['dispute_amount'] < 100000)).mean()*100:.1f}%)")
print(f"  $100k-$200k: {((df['dispute_amount'] >= 100000) & (df['dispute_amount'] < 200000)).sum()} ({((df['dispute_amount'] >= 100000) & (df['dispute_amount'] < 200000)).mean()*100:.1f}%)")
print(f"  > $200k: {(df['dispute_amount'] >= 200000).sum()} ({(df['dispute_amount'] >= 200000).mean()*100:.1f}%)")

print("\n" + "="*70)
print("LARGE LOSS ANALYSIS (Carrier Estimate > $250k)")
print("="*70)
large_losses = df[df['carrier_estimate'] > 250000]
print(f"Count: {len(large_losses)} ({len(large_losses)/len(df)*100:.1f}%)")
if len(large_losses) > 0:
    print(f"Avg Award Position: {large_losses['award_position_pct'].mean():.1f}%")
    print(f"Avg Dispute %: {large_losses['dispute_percentage'].mean():.1f}%")
    print(f"Carrier Win Rate: {(large_losses['appraisal_outcome'] == 'Carrier Favorable').mean()*100:.1f}%")
    print("\nAdjuster Quality on Large Losses:")
    print(large_losses['adjuster_type'].value_counts())
    if 'ia_skill_level' in large_losses.columns:
        print(large_losses[large_losses['adjuster_type']=='independent']['ia_skill_level'].value_counts())

print("\n" + "="*70)
print("PA vs ATTORNEY IMPACT")
print("="*70)
print("Award Position by Representation:")
print(f"  No PA: {df[~df['pa_involved']]['award_position_pct'].mean():.1f}%")
print(f"  PA Only (no attorney): {df[(df['pa_involved']) & (~df['pa_attorney'])]['award_position_pct'].mean():.1f}%")
print(f"  PA + Attorney: {df[(df['pa_involved']) & (df['pa_attorney'])]['award_position_pct'].mean():.1f}%")

print("\n" + "="*70)
print("AWARD DELTA ANALYSIS")
print("="*70)
print(f"\nOverall Average Award Position: {df['award_position_pct'].mean():.1f}%")
print("(0% = carrier estimate, 100% = demand estimate)")

print("\nAppraisal Outcome Distribution:")
print(df['appraisal_outcome'].value_counts())

print("\nAverage Award Position by Carrier Philosophy:")
print(df.groupby('carrier_philosophy')['award_position_pct'].mean().round(1))

print("\nAverage Award Position by Adjuster Type:")
adj_stats = df.groupby('adjuster_type')['award_position_pct'].agg(['mean', 'std']).round(1)
print(adj_stats)

print("\nAverage Award Position by IA Skill (Independent only):")
ia_stats = df[df['adjuster_type']=='independent'].groupby('ia_skill_level')['award_position_pct'].agg(['mean', 'std']).round(1)
print(ia_stats)

print("\nEngineering Impact (Roof Disputes Only):")
roof_df = df[df['is_roof_dispute']]
print(f"With Engineer: {roof_df[roof_df['carrier_engineer']]['award_position_pct'].mean():.1f}%")
print(f"Without Engineer: {roof_df[~roof_df['carrier_engineer']]['award_position_pct'].mean():.1f}%")
print(f"Delta: {roof_df[~roof_df['carrier_engineer']]['award_position_pct'].mean() - roof_df[roof_df['carrier_engineer']]['award_position_pct'].mean():.1f}% worse without engineer")

print("\nPrior Claims Impact:")
print(df.groupby('prior_claims_count')['award_position_pct'].mean().round(1))

print("\nMitigation Dispute Impact (when mitigation performed):")
mit_df = df[df['mitigation_performed']]
mit_df['high_mitigation_dispute'] = mit_df['mitigation_disputed_pct'] > 0.30
print(f"High Mitigation Dispute (>30%): {mit_df[mit_df['high_mitigation_dispute']]['award_position_pct'].mean():.1f}%")
print(f"Low Mitigation Dispute (≤30%): {mit_df[~mit_df['high_mitigation_dispute']]['award_position_pct'].mean():.1f}%")

print("\nRoof Dispute Analysis:")
roof_disputes = df[df['is_roof_dispute']]
print(f"Total Roof Disputes: {len(roof_disputes)}")
print(f"Avg Award Position: {roof_disputes['award_position_pct'].mean():.1f}%")
print(f"Avg Carrier Delta: ${roof_disputes['carrier_to_award_delta'].mean():,.0f}")
print(f"Median Carrier Delta: ${roof_disputes['carrier_to_award_delta'].median():,.0f}")
print(f"Win Rate for Insureds (>60%): {(roof_disputes['award_position_pct'] > 60).mean()*100:.1f}%")
print(f"\nWith Engineer:")
roof_eng = roof_disputes[roof_disputes['carrier_engineer']]
if len(roof_eng) > 0:
    print(f"  Avg Award Position: {roof_eng['award_position_pct'].mean():.1f}%")
    print(f"  Avg Carrier Delta: ${roof_eng['carrier_to_award_delta'].mean():,.0f}")
print(f"\nWithout Engineer:")
roof_no_eng = roof_disputes[~roof_disputes['carrier_engineer']]
if len(roof_no_eng) > 0:
    print(f"  Avg Award Position: {roof_no_eng['award_position_pct'].mean():.1f}%")
    print(f"  Avg Carrier Delta: ${roof_no_eng['carrier_to_award_delta'].mean():,.0f}")
    print(f"  Median Carrier Delta: ${roof_no_eng['carrier_to_award_delta'].median():,.0f}")
    print(f"  Win Rate for Insureds: {(roof_no_eng['award_position_pct'] > 60).mean()*100:.1f}%")

print("\n" + "="*70)
print("FINAL SUMMARY STATISTICS")
print("="*70)
print(f"\nAverage Award Amount: ${df['award_amount'].mean():,.0f}")
print(f"Median Award Amount: ${df['award_amount'].median():,.0f}")
print(f"Average Award Delta (Carrier to Award): ${df['carrier_to_award_delta'].mean():,.0f}")
print(f"Standard Deviation Award Delta: ${df['carrier_to_award_delta'].std():,.0f}")

print("\n" + "="*70)
print("CARRIER PHILOSOPHY WIN RATES")
print("="*70)
for carrier in ['tight', 'moderate', 'liberal']:
    carrier_df = df[df['carrier_philosophy'] == carrier]
    carrier_wins = (carrier_df['appraisal_outcome'] == 'Carrier Favorable').sum()
    carrier_losses = (carrier_df['appraisal_outcome'] == 'Policyholder Favorable').sum()
    total = len(carrier_df)
    print(f"\n{carrier.upper()}:")
    print(f"  Carrier Win Rate: {carrier_wins / total * 100:.1f}%")
    print(f"  Policyholder Win Rate: {carrier_losses / total * 100:.1f}%")
    print(f"  Split Decisions: {(total - carrier_wins - carrier_losses) / total * 100:.1f}%")
    print(f"  Avg Award Position: {carrier_df['award_position_pct'].mean():.1f}%")

# Save dataset
df.to_csv('appraisal_demands_synthetic.csv', index=False)
print("\n" + "="*70)
print("✓ Dataset saved to 'appraisal_demands_synthetic.csv'")
print("="*70)