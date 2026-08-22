# nnsyn Mini-Course

A practical, repository-grounded introduction to **nnsyn**, a self-configuring framework for paired 3D medical image translation.

- **Interactive course:** https://zaiyouzy.github.io/nnsyn-mini-course/
- **Upstream framework:** https://github.com/aehrc/nnsyn
- **Technical reproduction guide:** [code/README.md](code/README.md)
- **Tested environment:** [code/ENVIRONMENT.md](code/ENVIRONMENT.md)

## What this course teaches

The course follows public SynthRAD2025 abdomen MRI–CT data through the same stages used by the repository. By the end, a learner should be able to explain:

1. why MRI-to-CT synthesis is a continuous regression problem rather than segmentation;
2. how nnsyn adapts nnU-Net planning, preprocessing, training, and inference;
3. how paired MRI, CT, and body-mask files become planned training arrays;
4. how a 3D encoder-decoder maps an MRI patch to a normalized CT patch;
5. what MSE, masked MSE, and Masked Anatomical Perception loss optimize;
6. how sliding-window predictions are combined and restored to Hounsfield units; and
7. how intensity, structural, anatomical, and visual evaluation complement one another.

## Course structure

| Chapter | Question |
|---|---|
| 01 · Clinical task | Why synthesize CT from MRI? |
| 02 · Framework | How does nnsyn adapt nnU-Net? |
| 03 · Data and plans | How does nnsyn build the training plan? |
| 04 · Architecture | How does one MRI patch become a CT patch? |
| 05 · Losses | What do MSE, masked MSE, and MAP loss optimize? |
| 06 · Training | What happens in one training iteration? |
| 07 · Inference | How is the full CT volume reconstructed? |
| 08 · Evaluation | How should synthetic CT be evaluated? |
| 09 · Hands-on | What did the full-data training and validation run show? |
| 10 · Summary | Which ideas connect the complete pipeline? |

## Repository contents

```text
index.html                         interactive course
styles.css                        main visual system
teaching-components.css           shared teaching components
figures/                          figures derived from public SynthRAD2025 data
code/README.md                    complete reproduction instructions
code/ENVIRONMENT.md               separate H100 and local smoke environments
code/full_run/                    data staging, 300-epoch trainer, runner, and Slurm jobs
code/evaluate_trained_validation.py  35-case HU restoration, masked MAE, and result figure
code/run_nnsyn_smoke_training.py  optional bounded installation check
code/run_smoke_inference.py       optional three-case inference check
code/make_course_figures.py       rebuilds the data-inspection figures
code/make_architecture_figure.py  rebuilds the network architecture figure
code/results/                     per-case and aggregate validation records
code/patches/                     compatibility patch and smoke trainer
LICENSE.md                        licensing scope for course, code, and figures
```

Raw medical images, trained weights, virtual environments, and machine-specific paths are not included.

## Verified training and validation run

The main result comes from our own full-data training run, not from the earlier smoke check.

| Item | Verified value |
|---|---|
| Data | 175 paired SynthRAD2025 Task 1 abdomen cases |
| Fold 0 | 140 training cases, 35 held-out validation cases |
| Network and loss | 3D PlainConvUNet with masked MSE |
| Training | 300 epochs; best validation checkpoint |
| Validation mean masked MAE | 105.0 HU |
| Validation median masked MAE | 102.0 HU |
| Validation range | 67.4–189.1 HU |
| Representative case | 1ABA101, masked MAE 102.0 HU |
| Training device | NVIDIA H100 80 GB GPU |

The representative figure uses the case nearest the cohort median, rather than the best-looking case. Predictions are restored to Hounsfield units before evaluation. Values outside the body mask are displayed as −1000 HU and excluded from masked MAE. These numbers describe one held-out validation fold; they are not challenge-test, five-fold, ensemble, external-site, or clinical-performance results.

For context, the KoalAI algorithm description reports `62.4335 ± 23.2705 HU` for its five-fold ResUNet-MAP ensemble. Our 105.0 HU baseline is not comparable performance and is not presented as SOTA. The course explains the winning recipe, while the released full-run code currently reproduces the simpler one-fold masked-MSE experiment.

The repository still includes the three-case smoke utilities as a quick installation and pipeline check. Their two-update output is not used as the course performance figure.

Per-case values and the aggregate summary are available in [`code/results/`](code/results/).

## Reproduce the walkthrough

Start with [code/README.md](code/README.md). It records the pinned nnsyn commit, data layout, compatibility notes, the optional local smoke check, the full-data validation evidence, and the claim boundary.

The walkthrough was tested against nnsyn commit `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695`. Use the tested commit because the included patch was created for that source version.

## AI assistance and human verification

| Course component | AI assistance | Human verification |
|---|---|---|
| Course structure and English text | Codex and Claude assisted with organization, drafting, and language revision | Zaiyou He reviewed and edited the final material alongside the cited sources and run records |
| Website and supporting scripts | Codex and Claude assisted with HTML/CSS, Python/PowerShell, and troubleshooting | The site was built locally; preprocessing, training, inference, and evaluation outputs were inspected |
| Figures | AI assisted with plotting code and layout | Figures were generated from public SynthRAD2025 data and checked against the source volumes and run records |
| Medical images and experimental measurements | Codex and Claude did not generate these materials | They come from SynthRAD2025 and the reported nnsyn run |

Zaiyou He ran the reported preprocessing, 300-epoch training, inference, and 35-case validation; inspected the resulting logs, checkpoints, images, and numerical values; and made the final technical and editorial decisions. Jun Ma provided supervision and course feedback. The authors take responsibility for the final material.

## Acknowledgements and provenance

We acknowledge the Australian e-Health Research Centre for releasing nnsyn, the nnU-Net authors for the underlying framework, and the SynthRAD2025 organizers for making the public dataset available.

| Material | Use in this course | Source and terms |
|---|---|---|
| [aehrc/nnsyn](https://github.com/aehrc/nnsyn) | Repository studied and executed; course code is tied to commit `c3ba6fd8` | Australian e-Health Research Centre; Apache License 2.0 |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Planning, preprocessing, training, and inference infrastructure inherited by nnsyn | Isensee et al.; Apache License 2.0 |
| [SynthRAD2025 Task 1](https://doi.org/10.5281/zenodo.14918089) | Public abdomen MRI–CT data and all medical-image figures | Thummerer et al.; public data and derived figures remain subject to CC BY-NC 4.0 |
| KoalAI algorithm description | External MAP-loss validation rows in Chapter 05 | Xin et al.; used only as a cited numerical comparison |
| Course scripts and figures | Reproduction, evaluation, and visualization | AI assistance and human verification are disclosed above |

No raw SynthRAD2025 scans are redistributed in this repository. See [`LICENSE.md`](LICENSE.md) for the licensing scope of the course text, original supporting code, upstream nnsyn material, and dataset-derived figures.

## Authors and contributions

- **Zaiyou He** — course design, repository study, public-data execution, supporting code, visualization, and writing. Molecular Imaging, University Health Network.
- **Jun Ma** — supervision, course feedback, and review. Princess Margaret Cancer Centre & AI Hub, University Health Network.

## References

- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., and Maier-Hein, K. H. “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.” *Nature Methods* 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
- Australian e-Health Research Centre. *nnsyn: Self-configured framework for medical image synthesis.* https://github.com/aehrc/nnsyn
- Thummerer, A., et al. “SynthRAD2025 Grand Challenge dataset: Generating synthetic CTs for radiotherapy from head to abdomen.” *Medical Physics* 52(7), e17981 (2025). https://doi.org/10.1002/mp.17981
- Xin, B., Sun, Z., Min, H., Belous, G., and Dowling, J. *Team KoalAI: Ensembled ResUnet with Masked Anatomical Perception Loss.* SynthRAD2025 algorithm description (2025).
