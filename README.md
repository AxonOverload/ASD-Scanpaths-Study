# Scanpath-Based Classification of ASD vs. Typical Development Using a CNN-LSTM

This project investigates whether a child's eye-movement pattern (scanpath) while freely viewing natural images can be used to
distinguish Autism Spectrum Disorder (ASD) from typical development (TD). A CNN extracts visual features from each stimulus
image, and an LSTM processes the sequence of fixations (in the order they occurred) to classify each scanpath as ASD or TD.

Beyond the main classification result, this project includes a small ablation study designed to test
what the model is actually learning from, specifically, whether temporal order of fixations matters,
whether fixation duration matters, and how a simple order-blind baseline compares.


## Motivation
Visual attention differences in ASD, including atypical allocation to social stimuli such as faces,
are well documented in the literature on covert/overt attention and priority-map construction.
This project tests whether such differences are learnable, in a purely data-driven way,
from raw gaze sequences on natural scenes (a free-viewing paradigm), without any task-specific instruction to the viewer.


## Dataset
[Saliency4ASD](https://zenodo.org/records/13960426)
- 300 natural scene images (people, animals, everyday scenes)
- Eye-tracking data from 14 children with ASD and 14 typically developing (TD) children, per image
- Provided as per-image scanpath files, each containing multiple participants' fixation sequences concatenated together (fixation index resets to 0 at each new participant)

Saliency4ASD does not provide individual subject IDs.
Following the convention established in prior work (Chen et al.; also used in subsequent papers, e.g. the 2025 Visual Attention Graph paper),
the i-th scanpath within each per-image file, for a given group, is treated as belonging to the same participant across images.
This project additionally validated the assumption empirically by checking how consistently each inferred participant index appeared across the 300 image files,
and filtered out sparse/inconsistent indices (appearing in fewer than 50 of 300 images) before use, recovering exactly 14 ASD + 14 TD consistent participant groups.


## Method 

### Pipeline
- Parse per-image scanpath files, splitting on fixation-index resets into individual participant scanpaths.
- Infer participant identity using positional consistency across files (see above), filtering unreliable indices.
- Extract spatial CNN features once per stimulus image using a pretrained ResNet-18, truncated before global average pooling (output: 512-channel, 7×7 spatial feature map).
- For each fixation, map its pixel coordinates to the corresponding 7×7 feature grid cell and sample the 512-dim feature vector at that location; concatenate fixation duration (513-dim per fixation).
- Feed the resulting variable-length sequence into an LSTM (packed/padded appropriately); classify using the final hidden state.

### Evaluation
Stratified, group-aware 7-fold cross-validation (StratifiedGroupKFold), grouped by inferred participant ID,
so no participant's data appears in both training testing within a fold.
This avoids the leakage risk of a naive random split, given 14 scanpaths per participant are highly correlated.


## Discussion 
The model learns real signal well above the naive baseline (≈62% vs ≈46%).

The ablation study produced a specific, non-obvious finding:
shuffling the temporal order of fixations does not meaningfully reduce accuracy, and removing fixation duration or switching LSTM→GRU also has negligible effect.
However, the LSTM does meaningfully outperform a simple order-blind mean-pooled baseline (≈62% vs. ≈58%, a gap larger than the folds' standard deviations).

This suggests the LSTM learns a non-trivial, non-linear aggregation of which locations were fixated and what visual content was present there,
but this aggregation does not depend on the sequence in which fixations occurred. The fact that it still outperforms the mean-pooled baseline (62% vs. 58%) proves that the interaction between the features (e.g. the specific combination of location (A) + visual content (B) + location (C)) matters immensely. 
This is consistent with the attention allocation literature's emphasis on spatial/priority-based differences (e.g. reduced attention to social regions) in a free-viewing paradigm with no task structure.


## References
- Duan, H. et al. "Saliency4ASD: Challenge, dataset and tools for visual attention modeling for autism spectrum disorder." Signal Processing: Image Communication, 2019 (ICME 2019 Grand Challenge).
- Visual Attention Graph (2025), arXiv:2503.08531, confirms the same participant-inference convention used here.
