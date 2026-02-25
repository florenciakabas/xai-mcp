# Breast Cancer Screening Protocol — Memorial Hospital

## Document ID: PROTO-2024-BC-001
## Effective Date: 2024-01-15
## Version: 2.1

## 1. Risk Classification Thresholds

All risk classifications are derived from the XGBoost/Random Forest ensemble
predictions trained on institutional data. Thresholds were calibrated during
the 2023 annual review (see Committee Minutes CM-2023-11-04).

- **High Risk**: Model probability >= 0.85 → Immediate biopsy referral
  - Patient must be contacted within 48 hours
  - Referring physician receives automated notification
  - Case is flagged for multidisciplinary tumor board review

- **Moderate Risk**: Model probability 0.60–0.84 → Additional imaging
  - Recommend contrast-enhanced MRI within 2 weeks
  - If MRI confirms suspicious findings, escalate to High Risk protocol
  - Patient receives standard anxiety-reduction counseling materials

- **Low Risk**: Model probability < 0.60 → Standard screening schedule
  - Continue annual mammography per age-appropriate guidelines
  - Re-screen in 12 months unless new symptoms emerge
  - No additional imaging required

## 2. Feature-Specific Clinical Guidelines

The following guidelines apply when specific features appear among the
top 3 SHAP contributors for a given prediction.

### mean_radius
- Values > 14.0 mm are clinically significant for mass detection
- Correlates with tumor staging: T1 (<=20mm), T2 (20-50mm), T3 (>50mm)
- When mean_radius is a top SHAP contributor AND value > 14.0:
  recommend ultrasound measurement verification before biopsy
- Values > 20.0 mm require urgent surgical consultation regardless of
  overall model probability

### texture_mean
- Elevated texture (> 20.0) suggests irregular cell architecture
- When combined with high mean_radius (> 14.0): increase urgency tier
  by one level (e.g., Moderate → High)
- Texture values alone (without elevated radius) are insufficient
  for escalation — document but do not change care pathway

### concave_points_mean
- Values > 0.05 are associated with irregular tumor margins
- Clinical action: recommend core needle biopsy over fine needle
  aspiration for improved diagnostic yield
- When concave_points_mean is top contributor: note potential for
  invasive ductal carcinoma in pathology referral

### worst_concavity
- Values > 0.30 suggest aggressive morphology
- Cross-reference with concave_points_mean for confirmation
- If both elevated: recommend PET scan to assess potential metastasis

## 3. Operational Rules

### Documentation Requirements
- All high-risk classifications must include:
  - Model confidence (probability)
  - Top 3 contributing features with SHAP values
  - Protocol version number (this document)
  - Reviewing clinician signature

### Audit Trail
- Every model-assisted classification must be logged with:
  - Timestamp, model version, data hash
  - Full SHAP explanation (archived for 7 years per HIPAA)
  - Any overrides by clinical judgment (with written rationale)

### Override Policy
- Clinicians may override model classifications with documented rationale
- Overrides require second-opinion review within 5 business days
- Override rate > 15% triggers model recalibration review

## 4. Escalation Contacts

- Tumor Board Coordinator: Dr. Sarah Chen, ext. 4421
- Radiology Lead: Dr. Marcus Webb, ext. 4450
- Patient Services: ext. 4400 (M-F 8am-5pm)
