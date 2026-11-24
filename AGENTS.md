# MetaFusion + KNIGHT — Agent Instructions & Standing Rules

This file defines mandatory guardrails for all agents working on the MetaFusion Kidney Tumor Risk Classification B.Tech research project.

## Source of Truth
- Proposal PDF: `docs/1 MetaFusion_KNIGHT_BTech_Proposal_Phase1.pdf`
- Dataset repository & permitted feature list: `KNIGHT/README.md` and `KNIGHT/knight/data/knight.json`
- If any instruction or implementation conflicts with the source-of-truth documents, **STOP and ask the user**.

## Standing Rules (Never Violate)
1. **Dataset boundary:** KNIGHT is the ONLY training/validation dataset. Do NOT touch KiTS23 until the user explicitly approves Phase 7.
2. **Permitted clinical features only:** Use ONLY the 12 permitted clinical fields listed in `KNIGHT/README.md`:
   - `age_at_nephrectomy`, `gender`, `body_mass_index`, `comorbidities`, `smoking_history`, `age_when_quit_smoking`, `pack_years`, `chewing_tobacco_use`, `alcohol_use`, `last_preop_egfr`, `radiographic_size`, `voxel_spacing`.
   - NEVER use pathology, surgical, postoperative, vital status, or any other `knight.json` field, and NEVER use `aua_risk_group`, as model inputs.
3. **Mask usage boundary:** The segmentation mask is used ONLY to select the axial slice (`z*`). It must NEVER be passed to the model as an input channel, crop mask, or feature.
4. **Zero data leakage:** All imputers, scalers, encoders, and class weights must be fit strictly on the training fold indices and applied to the validation fold.
5. **Frozen 5-fold CV splits:** One frozen set of 5 stratified folds (`StratifiedKFold`, `shuffle=True`, fixed seed, stratified on the five-class label) saved to `data/splits/folds.json` and reused by EVERY model and BOTH tasks (binary & five-class).
6. **Faithful Meta* implementations:** Do NOT substitute simple concatenation and label it `MetaFusion`, `MetaBlock`, or `MetaNet`. Implement the real mechanisms from the official papers/repos placed in `external/` and cite exact source files and line numbers. If an official implementation cannot be verified, STOP and inform the user.
7. **No fabrication:** Never invent results, URLs, label definitions, metrics, or feature names. Verify everything against local files or executed runs.
8. **Reporting standard:** Never report a single random split as a final result. All final metrics must be reported as `mean ± SD` across all 5 frozen folds.
9. **Reproducibility:** Set and log all random seeds; keep all experiments config-driven via YAML (`configs/`); avoid hardcoded magic numbers in code.
10. **Explicit permission for heavy/destructive actions:** Always ask before installing large dependencies, downloading datasets, deleting files, or running any mutating Git/GitHub operation (see `.agents/rules/git-approval.md`).

## Phase Gates
Stop at every phase gate, present the required artifact/report, and wait for explicit user approval before advancing:
- **Phase 0:** Setup (`tree`, `requirements.txt`, `AGENTS.md`, `.gitignore`, `configs/base.yaml`)
- **Phase 1:** Data inspection (`scripts/00_inspect_knight.py` + written report + mismatch audit)
- **Phase 2:** Frozen stratified 5-fold split (`data/splits/folds.json` + unit tests)
- **Phase 3:** Slice extraction + visual QA overlays (~20 random + extreme cases)
- **Phase 4:** Metadata pipeline + leakage unit tests
- **Phase 5:** All 7 model configs + shape unit tests + Meta* paper/repo mapping note
- **Phase 6:** 5-fold CV training (sanity baselines first, then fusion models) + results table & confusion matrices
- **Phase 7:** (Approval-gated) KiTS23 preprocessing & slice-selection verification only
