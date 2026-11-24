# Phase 1: KNIGHT Dataset Inspection Report

- **Source file:** `X:\Metafusion\KNIGHT\knight\data\knight.json`
- **Patients:** `300` total (`300` unique `case_id`s, `case_00000` to `case_00299`)
- **Top-level JSON fields:** `41` total (`1` identifier, `1` target, `12` permitted clinical, `27` forbidden)
- **Unexpected target values:** `[]`

## 1. Target Distribution (`aua_risk_group`)

### Five-Class Task
| Raw Label | Class ID | Count | Percentage |
| :--- | :---: | :---: | :---: |
| `benign` | 0 | 25 | 8.33% |
| `low_risk` | 1 | 142 | 47.33% |
| `intermediate_risk` | 2 | 48 | 16.00% |
| `high_risk` | 3 | 40 | 13.33% |
| `very_high_risk` | 4 | 45 | 15.00% |

### Binary Task
| Binary Class | Count | Percentage |
| :--- | :---: | :---: |
| `0_non_high_risk (benign/low/intermediate)` | 215 | 71.67% |
| `1_high_or_very_high_risk` | 85 | 28.33% |

## 2. Detailed Inspection of the 12 Permitted Clinical Features

| # | Feature | Structure | Python Types | Nulls (%) | Numeric Parseable | Non-Numeric / Categories |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| 1 | `age_at_nephrectomy` | scalar | `int(300)` | 0 (0.0%) | 300 | — |
| 2 | `gender` | scalar | `str(300)` | 0 (0.0%) | 0 | 'male': 179, 'female': 120, 'transgender_male_to_female': 1 |
| 3 | `body_mass_index` | scalar | `float(300)` | 0 (0.0%) | 300 | — |
| 4 | `comorbidities` | nested_dict (19 keys) | `dict` | 0 (0.0%) | see subkeys below | see subkeys below |
| 5 | `smoking_history` | scalar | `str(300)` | 0 (0.0%) | 0 | 'never_smoked': 137, 'previous_smoker': 120, 'current_smoker': 43 |
| 6 | `age_when_quit_smoking` | scalar | `NoneType(35), str(265)` | 35 (11.7%) | 85 | 'not_applicable': 180 |
| 7 | `pack_years` | scalar | `NoneType(106), int(137), str(57)` | 106 (35.3%) | 194 | — |
| 8 | `chewing_tobacco_use` | scalar | `str(300)` | 0 (0.0%) | 0 | 'never_or_not_in_last_3mo': 295, 'quit_in_last_3mo': 5 |
| 9 | `alcohol_use` | scalar | `str(300)` | 0 (0.0%) | 0 | 'two_or_less_daily': 146, 'never_or_not_in_last_3mo': 131, 'more_than_two_daily': 17, '': 5, 'quit_in_last_3mo': 1 |
| 10 | `last_preop_egfr` | nested_dict (2 keys) | `dict` | 0 (0.0%) | see subkeys below | see subkeys below |
| 11 | `radiographic_size` | scalar | `float(300)` | 0 (0.0%) | 300 | — |
| 12 | `voxel_spacing` | nested_dict (3 keys) | `dict` | 0 (0.0%) | see subkeys below | see subkeys below |

### Numeric Summary (Scalar Permitted Features)

| Feature | Valid N | Min | P25 | Median | Mean | Std | P75 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `age_at_nephrectomy` | 300 | 1.0 | 51.0 | 60.5 | 58.8867 | 13.7737 | 68.0 | 90.0 |
| `body_mass_index` | 300 | 16.2 | 26.177500000000002 | 29.810000000000002 | 30.8583 | 6.6669 | 35.2575 | 55.39 |
| `age_when_quit_smoking` | 85 | 0.0 | 30.0 | 43.0 | 42.6941 | 15.1696 | 55.0 | 73.0 |
| `pack_years` | 194 | 0.0 | 0.0 | 0.0 | 9.768 | 23.0502 | 8.5 | 210.0 |
| `radiographic_size` | 300 | 1.2 | 2.6 | 4.1 | 5.03 | 3.1923 | 6.025 | 16.2 |

### Nested Dict Sub-Fields

#### `comorbidities` (19 sub-keys)
| Sub-key | Python Types | Nulls (%) | Summary |
| :--- | :--- | :---: | :--- |
| `myocardial_infarction` | `bool(300)` | 0 (0.0%) | True=13 (4.3%), False=287 |
| `congestive_heart_failure` | `bool(300)` | 0 (0.0%) | True=8 (2.7%), False=292 |
| `peripheral_vascular_disease` | `bool(300)` | 0 (0.0%) | True=13 (4.3%), False=287 |
| `cerebrovascular_disease` | `bool(300)` | 0 (0.0%) | True=8 (2.7%), False=292 |
| `dementia` | `bool(300)` | 0 (0.0%) | True=1 (0.3%), False=299 |
| `copd` | `bool(300)` | 0 (0.0%) | True=13 (4.3%), False=287 |
| `connective_tissue_disease` | `bool(300)` | 0 (0.0%) | True=1 (0.3%), False=299 |
| `peptic_ulcer_disease` | `bool(300)` | 0 (0.0%) | True=4 (1.3%), False=296 |
| `uncomplicated_diabetes_mellitus` | `bool(300)` | 0 (0.0%) | True=58 (19.3%), False=242 |
| `diabetes_mellitus_with_end_organ_damage` | `bool(300)` | 0 (0.0%) | True=2 (0.7%), False=298 |
| `chronic_kidney_disease` | `bool(300)` | 0 (0.0%) | True=24 (8.0%), False=276 |
| `hemiplegia_from_stroke` | `bool(300)` | 0 (0.0%) | True=1 (0.3%), False=299 |
| `leukemia` | `bool(300)` | 0 (0.0%) | True=1 (0.3%), False=299 |
| `malignant_lymphoma` | `bool(300)` | 0 (0.0%) | True=2 (0.7%), False=298 |
| `localized_solid_tumor` | `bool(300)` | 0 (0.0%) | True=43 (14.3%), False=257 |
| `metastatic_solid_tumor` | `bool(300)` | 0 (0.0%) | True=17 (5.7%), False=283 |
| `mild_liver_disease` | `bool(300)` | 0 (0.0%) | True=4 (1.3%), False=296 |
| `moderate_to_severe_liver_disease` | `bool(300)` | 0 (0.0%) | True=1 (0.3%), False=299 |
| `aids` | `bool(300)` | 0 (0.0%) | True=0 (0.0%), False=300 |

- Active comorbidities per patient distribution: `{0: 171, 1: 72, 2: 35, 3: 17, 4: 4, 5: 1}`

#### `last_preop_egfr` (2 sub-keys)
| Sub-key | Python Types | Nulls (%) | Summary |
| :--- | :--- | :---: | :--- |
| `value` | `NoneType(74), float(177), str(49)` | 74 (24.7%) | num=177 (min=17.0, med=71.0, max=90.0); non-num={"'>=90'": 49} |
| `days_before_nephrectomy` | `NoneType(74), int(226)` | 74 (24.7%) | num=226 (min=1.0, med=14.0, max=90.0) |

#### `voxel_spacing` (3 sub-keys)
| Sub-key | Python Types | Nulls (%) | Summary |
| :--- | :--- | :---: | :--- |
| `x_spacing` | `float(300)` | 0 (0.0%) | num=300 (min=0.4375, med=0.78125, max=1.0410000085830688) |
| `y_spacing` | `float(300)` | 0 (0.0%) | num=300 (min=0.4375, med=0.78125, max=1.0410000085830688) |
| `z_spacing` | `float(300)` | 0 (0.0%) | num=300 (min=0.5, med=3.0, max=5.0) |

## 3. Full Top-Level Schema & Forbidden Fields

| Field | Role | Present | Types | Nulls (%) |
| :--- | :--- | :---: | :--- | :---: |
| `case_id` | `identifier` | 300 | `str(300)` | 0 (0.0%) |
| `age_at_nephrectomy` | `permitted_clinical` | 300 | `int(300)` | 0 (0.0%) |
| `gender` | `permitted_clinical` | 300 | `str(300)` | 0 (0.0%) |
| `body_mass_index` | `permitted_clinical` | 300 | `float(300)` | 0 (0.0%) |
| `comorbidities` | `permitted_clinical` | 300 | `dict(300)` | 0 (0.0%) |
| `smoking_history` | `permitted_clinical` | 300 | `str(300)` | 0 (0.0%) |
| `age_when_quit_smoking` | `permitted_clinical` | 300 | `NoneType(35), str(265)` | 35 (11.7%) |
| `pack_years` | `permitted_clinical` | 300 | `NoneType(106), int(137), str(57)` | 106 (35.3%) |
| `chewing_tobacco_use` | `permitted_clinical` | 300 | `str(300)` | 0 (0.0%) |
| `alcohol_use` | `permitted_clinical` | 300 | `str(300)` | 0 (0.0%) |
| `intraoperative_complications` | `FORBIDDEN` | 300 | `dict(300)` | 0 (0.0%) |
| `hospitalization` | `FORBIDDEN` | 300 | `int(299), str(1)` | 0 (0.0%) |
| `ischemia_time` | `FORBIDDEN` | 300 | `NoneType(13), int(175), str(112)` | 13 (4.3%) |
| `radiographic_size` | `permitted_clinical` | 300 | `float(300)` | 0 (0.0%) |
| `pathologic_size` | `FORBIDDEN` | 300 | `float(300)` | 0 (0.0%) |
| `malignant` | `FORBIDDEN` | 300 | `bool(300)` | 0 (0.0%) |
| `pathology_t_stage` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `pathology_n_stage` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `pathology_m_stage` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `tumor_histologic_subtype` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `tumor_necrosis` | `FORBIDDEN` | 300 | `NoneType(34), bool(266)` | 34 (11.3%) |
| `tumor_isup_grade` | `FORBIDDEN` | 300 | `NoneType(56), int(244)` | 56 (18.7%) |
| `clavien_surgical_complications` | `FORBIDDEN` | 300 | `NoneType(1), str(299)` | 1 (0.3%) |
| `er_visit` | `FORBIDDEN` | 300 | `NoneType(3), bool(297)` | 3 (1.0%) |
| `readmission` | `FORBIDDEN` | 300 | `NoneType(3), bool(297)` | 3 (1.0%) |
| `estimated_blood_loss` | `FORBIDDEN` | 300 | `NoneType(1), int(299)` | 1 (0.3%) |
| `surgery_type` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `surgical_procedure` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `surgical_approach` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `operative_time` | `FORBIDDEN` | 300 | `NoneType(2), int(298)` | 2 (0.7%) |
| `cytoreductive` | `FORBIDDEN` | 300 | `bool(300)` | 0 (0.0%) |
| `positive_resection_margins` | `FORBIDDEN` | 300 | `bool(300)` | 0 (0.0%) |
| `last_preop_egfr` | `permitted_clinical` | 300 | `dict(300)` | 0 (0.0%) |
| `first_postop_egfr` | `FORBIDDEN` | 300 | `dict(300)` | 0 (0.0%) |
| `last_postop_egfr` | `FORBIDDEN` | 300 | `dict(300)` | 0 (0.0%) |
| `vital_status` | `FORBIDDEN` | 300 | `str(300)` | 0 (0.0%) |
| `vital_days_after_surgery` | `FORBIDDEN` | 300 | `int(300)` | 0 (0.0%) |
| `voxel_spacing` | `permitted_clinical` | 300 | `dict(300)` | 0 (0.0%) |
| `sarcomatoid_features` | `FORBIDDEN` | 300 | `bool(300)` | 0 (0.0%) |
| `rhabdoid_features` | `FORBIDDEN` | 300 | `bool(300)` | 0 (0.0%) |
| `aua_risk_group` | `target` | 300 | `str(300)` | 0 (0.0%) |

## 4. Local CT Imaging & Segmentation Mask Status

- Local CT volumes found (`imaging.nii.gz`): `0`
- Local segmentation masks found: `0`
- **Status:** No NIfTI CT volumes or KiTS21 segmentation masks are downloaded locally yet. `TUMOR_LABEL` and axis orientation cannot be verified on actual NIfTI files until data download is approved.
