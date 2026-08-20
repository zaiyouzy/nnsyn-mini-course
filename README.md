# nnsyn Mini-Course

A practical, repository-grounded introduction to **nnsyn**, a self-configuring framework for paired 3D medical image translation.

- **Interactive course:** https://zaiyouzy.github.io/nnsyn-mini-course/
- **Upstream framework:** https://github.com/aehrc/nnsyn
- **Technical reproduction guide:** [code/README.md](code/README.md)
- **Tested environment:** [code/ENVIRONMENT.md](code/ENVIRONMENT.md)

## What this course teaches

The course follows one public SynthRAD2025 abdomen MRI–CT case through the same stages used by the repository. By the end, a learner should be able to explain:

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
| 03 · Data and plans | How do paired files become a training plan? |
| 04 · Architecture | How does one MRI patch become a CT patch? |
| 05 · Losses | What do MSE, masked MSE, and MAP loss optimize? |
| 06 · Training | What happens during one training iteration? |
| 07 · Inference | How do overlapping patches become one CT volume? |
| 08 · Evaluation | How should synthetic CT be evaluated? |
| 09 · Hands-on | What did the three-case execution check verify? |
| 10 · Summary | Which ideas connect the complete pipeline? |

## Repository contents

```text
index.html                         interactive course
styles.css                        main visual system
teaching-components.css           shared teaching components
figures/                          figures generated from the public teaching subset
code/README.md                    complete reproduction instructions
code/ENVIRONMENT.md               tested software and hardware versions
code/run_nnsyn_smoke_training.py  bounded training execution check
code/run_smoke_inference.py       inference, HU restoration, MAE, and Figure 06
code/make_course_figures.py       rebuilds Figures 01–05
code/patches/                     pinned compatibility patch and smoke trainer
```

Raw medical images, trained weights, virtual environments, and machine-specific paths are not included.

## Verified execution trace

The public code supports a deliberately small execution check, not a performance benchmark.

| Item | Verified value |
|---|---|
| Data | 3 public SynthRAD2025 abdomen cases |
| Local dataset identifier | Dataset501 |
| Fold 0 | 2 training cases, 1 held-out case |
| Training | 1 epoch, 2 optimizer updates, 1 validation iteration |
| Inference | 48 overlapping windows on case 1ABA033 |
| Observed masked MAE | 249.9 HU |
| Tested device | NVIDIA RTX 4070 Laptop GPU, 8 GB |

`Dataset501` is a local nnU-Net identifier created for the teaching subset; it is not an official SynthRAD2025 dataset number. The run verifies that preprocessing, loading, forward and backward passes, checkpointing, inference, HU restoration, masking, evaluation, and export connect end to end. It does not establish image quality, generalization, clinical suitability, or challenge performance.

## Reproduce the walkthrough

Start with [code/README.md](code/README.md). It records the pinned nnsyn commit, data layout, installation steps, compatibility patch, preprocessing command, bounded training command, inference command, observed trace values, and claim boundary.

The walkthrough was tested against nnsyn commit `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695`. Use the tested commit because the included patch was created for that source version.

## AI assistance and human verification

OpenAI Codex and Anthropic Claude were used to help organize the course, revise English text, develop and edit the HTML/CSS website and supporting Python/PowerShell scripts, troubleshoot the local Windows/CUDA workflow, and check consistency across the site, code, figures, and references.

Zaiyou He ran the reported preprocessing, smoke training, and inference steps; inspected the resulting logs, checkpoints, images, and numerical values; and made the final technical and editorial decisions. Codex and Claude did not generate the medical images or experimental measurements; these came from the public dataset and the reported nnsyn run. Jun Ma provided supervision and course feedback. The authors take responsibility for the final material.

## Data, software, and figure provenance

| Material | Use in this course | Source and terms |
|---|---|---|
| [aehrc/nnsyn](https://github.com/aehrc/nnsyn) | Repository studied and executed; course code is tied to commit `c3ba6fd8` | Australian e-Health Research Centre; Apache License 2.0 |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Planning, preprocessing, training, and inference infrastructure inherited by nnsyn | Isensee et al.; Apache License 2.0 |
| [SynthRAD2025 Task 1](https://doi.org/10.5281/zenodo.14918089) | Three public abdomen MRI–CT cases and all medical-image figures | Thummerer et al.; public data and derived figures remain subject to CC BY-NC 4.0 |
| KoalAI algorithm description | External MAP-loss validation rows in Chapter 05 | Xin et al.; these values are not results from the course run |
| Course scripts and figures | Teaching-scale execution, evaluation, and visualization | AI assistance and human verification are disclosed above |

No raw SynthRAD2025 scans are redistributed in this repository. The included `code/LICENSE-nnsyn.txt` is a copy of the upstream nnsyn license for reference. Course text and original tutorial layout are released under CC BY 4.0 unless otherwise stated; third-party material retains its original terms.

## Authors and contributions

- **Zaiyou He** — course design, repository study, public-data execution, supporting code, visualization, and writing. Molecular Imaging, University Health Network.
- **Jun Ma** — supervision, course feedback, and review. Princess Margaret Cancer Centre & AI Hub, University Health Network.

## References

- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., and Maier-Hein, K. H. “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.” *Nature Methods* 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
- Australian e-Health Research Centre. *nnsyn: Self-configured framework for medical image synthesis.* https://github.com/aehrc/nnsyn
- Thummerer, A., et al. “SynthRAD2025 Grand Challenge dataset: Generating synthetic CTs for radiotherapy from head to abdomen.” *Medical Physics* 52(7), e17981 (2025). https://doi.org/10.1002/mp.17981
- Xin, B., Sun, Z., Min, H., Belous, G., and Dowling, J. *Team KoalAI: Ensembled ResUnet with Masked Anatomical Perception Loss.* SynthRAD2025 algorithm description (2025).
