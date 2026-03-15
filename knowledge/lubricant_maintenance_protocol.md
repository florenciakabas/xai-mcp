# Lubricant Quality Monitoring Protocol — Industrial Equipment Division

## Document ID: PROTO-2026-LQ-001
## Effective Date: 2026-01-10
## Version: 1.0

## 1. Quality Classification Thresholds

All quality classifications are derived from the LightGBM classifier trained on
historical oil analysis data from rotating equipment in Gulf Coast refinery
operations. Thresholds were calibrated during the Q4 2025 reliability review.

- **Degraded (probability >= 0.75)**: Immediate oil change required
  - Equipment must be taken offline within 72 hours for lubricant replacement
  - Maintenance supervisor notified automatically via CMMS work order
  - Root cause investigation initiated — cross-reference with vibration data
  - Sample retained for confirmatory lab analysis (ASTM D974, D6304)

- **Watch (probability 0.50–0.74)**: Increased monitoring frequency
  - Sampling interval reduced from 30 days to 7 days
  - Trend analysis required: compare last three samples for progression
  - If two consecutive samples remain in Watch zone, escalate to Degraded protocol
  - No immediate equipment downtime required

- **Acceptable (probability < 0.50)**: Normal operations
  - Continue standard 30-day sampling schedule
  - No additional actions required
  - Annual baseline comparison during turnaround inspections

## 2. Feature-Specific Maintenance Guidelines

The following guidelines apply when specific features appear among the top
SHAP contributors for a given prediction.

### total_acid_number
- Normal range: 0.2–0.8 mg KOH/g (per ASTM D974)
- Physical meaning: measures acidic byproducts of oxidation and contamination
- TAN > 2.0 mg KOH/g: lubricant has exceeded condemning limit — immediate drain
- TAN increase > 0.5 mg KOH/g between samples: accelerated oxidation detected
- Action: check seal integrity and breather systems for moisture or process gas ingress
- Cross-reference: oxidation_stability, water_content_ppm

### water_content_ppm
- Normal range: 50–120 ppm (per ASTM D6304, Karl Fischer method)
- Physical meaning: dissolved and free water accelerates corrosion and reduces film strength
- Water > 300 ppm: free water phase likely present — drain and dehydrate
- Water > 200 ppm: check cooler tube integrity and seal condition
- Seasonal note: summer humidity increases baseline by 30–50 ppm; adjust thresholds accordingly
- Action: inspect mechanical seals, heat exchanger tubes, breather desiccant
- Cross-reference: iron_ppm (corrosion indicator), demulsibility_min

### viscosity_40c
- Normal range: 61–76 cSt at 40°C (ISO VG 68 grade, per ASTM D445)
- Physical meaning: resistance to flow at operating temperature; primary indicator of film strength
- Viscosity decrease > 10%: fuel dilution or thermal cracking suspected
- Viscosity increase > 10%: oxidation polymerization or contamination with heavier product
- Seasonal note: temperature-sensitive — summer readings naturally lower by 2–5 cSt
- Action: verify operating temperature is within design spec; check for cross-contamination
- Cross-reference: viscosity_100c, flash_point_c

### particle_count_4um
- Normal range: 30–70 particles/mL at 4µm (per ISO 4406 cleanliness code)
- Physical meaning: abrasive wear particles that damage bearing surfaces
- Count > 100: filter bypass or catastrophic wear event
- Count > 150: shut down for inspection — bearing damage likely
- Seasonal note: summer operations generate 10–20% more particles due to thermal expansion wear
- Action: replace filters, inspect bearing clearances, check shaft alignment
- Cross-reference: iron_ppm, copper_ppm (wear metal confirmation)

### iron_ppm
- Normal range: 5–18 ppm (per ASTM D5185, ICP-OES)
- Physical meaning: ferrous wear from gears, bearings, and shafts
- Iron > 30 ppm: abnormal wear rate — schedule inspection at next planned outage
- Iron > 50 ppm: critical wear — immediate vibration analysis and borescope inspection
- Trending: rate of increase matters more than absolute value
- Action: correlate with vibration spectra; check gear mesh patterns
- Cross-reference: particle_count_4um, copper_ppm

### copper_ppm
- Normal range: 1–4 ppm (per ASTM D5185)
- Physical meaning: wear from bronze bushings, thrust washers, and cooler tubes
- Copper > 8 ppm: bushing or cooler tube erosion suspected
- Copper > 12 ppm: immediate cooler inspection — potential tube failure
- Action: inspect oil cooler tubes and bronze wear components
- Cross-reference: iron_ppm, water_content_ppm (cooler leak indicator)

### oxidation_stability
- Normal range: 250–350 minutes (per RPVOT / ASTM D2272)
- Physical meaning: remaining antioxidant life of the lubricant
- RPVOT < 150 min: lubricant approaching end of useful life
- RPVOT < 100 min: condemning limit — schedule oil change
- Rapid decline (> 50 min drop between samples): investigate contaminant ingress
- Action: consider antioxidant additive top-up if otherwise within spec
- Cross-reference: total_acid_number

### flash_point_c
- Normal range: 200–240°C (per ASTM D92, Cleveland Open Cup)
- Physical meaning: temperature at which vapors ignite — safety-critical parameter
- Flash point drop > 20°C: fuel or solvent contamination — fire hazard
- Flash point < 180°C: remove from service immediately, investigate contamination source
- Action: check process seal integrity; verify no cross-connection with fuel systems
- Cross-reference: viscosity_40c (dilution confirmation)

## 3. Operational Rules

### Sampling Requirements
- All samples must be taken from the same sample port using clean bottles
- Minimum 100 mL required for full analysis suite
- Record equipment runtime hours at time of sampling
- Label with equipment ID, date, runtime hours, and ambient temperature

### Seasonal Considerations
- Summer (May–September): adjust viscosity and water content thresholds upward
- Winter (October–April): watch for cold-start wear — first 30 minutes of operation
  generate higher particle counts that normalize after thermal equilibrium
- During seasonal transitions: take additional baseline sample within first week

### Override Policy
- Reliability engineers may override model classifications with documented rationale
- Overrides require review by Lubrication Program Manager within 5 business days
- Override rate exceeding 15% triggers model recalibration review
- All overrides logged in CMMS with reason code and approving engineer

## 4. Escalation Contacts

- Lubrication Program Manager: James Morales, ext. 3310
- Reliability Engineering Lead: Dr. Priya Venkatesh, ext. 3325
- Equipment Health Monitoring Center: ext. 3300 (24/7 coverage)
- Emergency Maintenance Coordinator: ext. 3399 (after-hours)
