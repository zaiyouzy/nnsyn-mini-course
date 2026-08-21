# Fold-0 validation results

These files summarize our 300-epoch nnsyn reproduction on the public SynthRAD2025 Task 1 abdomen training data.

- `validation_mae.csv` contains masked MAE for all 35 held-out fold-0 cases.
- `validation_summary.json` contains the aggregate mean, median, standard deviation, minimum, and maximum.
- The published representative figure uses case `1ABA101`, selected because its MAE is closest to the cohort median.

The case identifiers are the de-identified identifiers supplied with SynthRAD2025. The validation records contain centers A, B, and C only; they do not contain Center D data. No raw medical volumes or trained weights are redistributed here.

The metrics and displayed image panels are derived from the SynthRAD2025 dataset and remain subject to its CC BY-NC 4.0 terms. We restored predicted values to Hounsfield units, displayed values outside the body mask as −1000 HU, calculated error inside the mask, and created the figure layout. Please cite:

Thummerer, A., et al. “SynthRAD2025 Grand Challenge dataset: Generating synthetic CTs for radiotherapy from head to abdomen.” *Medical Physics* 52(7), e17981 (2025). https://doi.org/10.1002/mp.17981

Dataset collection: https://doi.org/10.5281/zenodo.14918089
License: https://creativecommons.org/licenses/by-nc/4.0/