class: middle, center, title-slide

# Segment Anything Model - SAM

.center[.width-45[![](./figures/Meta_lockup_positive primary_RGB.png)]]

Advanced Machine Learning

.footnote[Martin Dengis - Gilles Ooms]

---

class: middle

# Agenda

- Background Knowledge
- Paper Overview
- Promptable Segmentation Task
- Model Architecture
- Key Particularities
- Discussion

---
class: middle

# Background Knowledge

## Segmentation

.center.stretch[
     ![](figures/segmentation_masks_1.png)
     ![](figures/segmentation_masks_2.png)
     ![](figures/segmentation_masks_3.png)
]

.footnote[Adapted from [Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

---
class: middle

## Encoder-Decoder Architecture

## Zero-shot Learning

---

class: middle

# Paper Overview

The paper's value proposition is organized around 3 core components:

1. The promptable segmentation **task**
2. A **model** to predict segmentation masks
3. A diverse and large-scale **dataset (SA-1B)**

All combined, they gave rise to the .bold[*Segment Anything Model (SAM)*] - .italic[i.e.,] the (self-pronounced) first *foundation model* for image segmentation.

???

- Focus only on first 2 points in details (time constraints)
- For the SA-1B dataset and data engine, see additional slides
- Foundation model = promptable pre-trained model, that offers powerful generalization capabilities, to be used in a standalone manner or composed with other models for downstream tasks

---
class: middle

# Promptable Segmentation Task

---

class: middle

.center.width-65[![](./figures/prompt_examples.png)]

- **What**: Return a valid segmentation mask for *any* prompt
- **Prompt Types**: Points & bounding boxes (*sparse prompts*), masks (*dense prompts*), text (PoC).
- **Validity**: Even *ambiguous* prompts should yield reasonable masks

.footnote[[Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

???

This is the foundation of SAM's approach - taking various forms of input prompts and generating appropriate segmentation masks in response.
Heavily inspired from the next-token prediction task in NLP models.

---
class: middle

.question[Why Promptable Segmentation?]

<br/>

Adapting the *next-token prediction task* of NLP models to segmentation

1. naturally leads to **pre-training** as required for foundation models;
2. supports **zero-shot transfer**, .italic[i.e.,] adapting to new tasks via prompt engineering without requiring fine-tuning.

Finally, it also solves for ambiguity (more on that later).

???

This task design is critical because it allows the model to work as a foundation model rather than a task-specific one. The analogy with next-token prediction in language models helps understand how SAM generalizes beyond its training.

---
class: middle

## From Promptable Task to Zero-Shot Transfer

**Pre-training phase**

- Train with sequence of prompts
- Learn to predict valid masks
- Develop prompt-mask associations

.pull-right[
**Zero-shot transfer**
- Design appropriate prompts
- Integrate with other models
- .italic[Example: Object detector → box prompt → segmentation]
]

???

The key innovation here is how the task design enables the model to generalize.

- During pre-training, model learns a general relationship between prompts and valid masks.
- Then during inference, this learned relationship transfers to new data distributions and tasks by simply designing appropriate prompts.

---
class: middle

## Relation to Existing Tasks

- **Interactive segmentation**: SAM must produce valid masks with minimal guidance

- **Difference from multi-task models**: Not trained/tested on fixed task set

- **Composable with other models**:
  - Object detector + SAM = Instance segmentation
  - Text model + SAM = Text-guided segmentation
  - Manual clicks + SAM = Interactive segmentation

???

Slide is more for reference → skip it!

---

class: middle

# SAM Architecture

.center.width-65[![](figures/high-level-architecture.gif)]
.center.italic[High-level Overview of SAM]

.footnote[[Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

---
class: middle

Three main components:

  1. **Image Encoder**: Vision Transformer to create image embeddings
  2. **Prompt Encoder**: Converts different prompt types into embeddings
  3. **Mask Decoder**: Lightweight transformer that combines embeddings to produce masks

???

- The separation of image encoding from prompt encoding allows for computation reuse - crucial for interactive applications.
- It also ensures that embeddings are informationally rich as they are not aware of each others until decoding.

---

class: middle

## Image Encoder

- **Base Architecture**: Vision Transformer (ViT-H/16) pre-trained on MAE*
- *Enhancements*:
  - 14×14 windowed attention + 4 global attention blocks
  - Input resolution: 1024×1024 (rescaled and padded from original)
- *Output*:
  - 16× downscaled embedding (64×64)
  - Channel dimension reduced to 256 via convolutions <br/> (each followed by Layer Normalization)

.center.width-85[![](figures/architecture_flat_img_encoder.png)]

.footnote[Adapted from [Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

???

Vision Transformer (ViT-H/16) Pre-trained with MAE

- **ViT-H/16**:
  - "ViT" refers to **Vision Transformer**
  - "H" stands for **Huge**
  - "16" means the model **divides images into 16×16 patches** before processing.

- **Pre-trained with MAE***:
  - **MAE (Masked Autoencoder)** is a **self-supervised learning** method.
  - It **masks** random patches of an image and trains the model to **reconstruct** them.

---

class: middle

## Prompt Encoder

### Sparse Prompts (points & boxes)

- *Points*: positional encoding + learned embedding (foreground/background)
- *Boxes*: two points (top-left, bottom-right) with positional encoding
<!-- - Text: CLIP text encoder (proof-of-concept) -->

???

- Text prompts: CLIP text encoder (proof-of-concept)

### Dense Prompts (*masks*)

- Input at 1/4 resolution of image (256×256)
- Downscaled 4× via convolutions to 64×64
- Element-wise addition to image embedding

.center.width-85[![](figures/architecture_flat_prompt_encoder.png)]

.footnote[Adapted from [Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

???

- Mask is basically another image: so we encode then add to the existing image embedding
- The multi-modal prompt encoder is what makes SAM so flexible. By converting different types of prompts into a common embedding space, the model can adapt to various input modalities.

---

class: middle

## Mask Decoder

???

- A 2-layer modified transformer decoder
- Modified to be lightweight (real-time interaction)

---
class: middle

- **Process**:
  1. Image embedding (64×64, 256-dimensional)
  2. Prompt embeddings as tokens
  3. Self-attention among tokens
  4. Cross-attention between tokens and image embedding
  5. Point-wise MLP updates each token
  6. Cross-attention from image to tokens
- **Upsampling**: 4× via transposed convolutions

.center.width-90[![](figures/mask_decoder.png)]

.footnote[[Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

???

- The mask decoder is deliberately kept lightweight since it needs to run multiple times during interactive segmentation.
- The token embeddings are padded with **output tokens**, which are subject to the same processing workflow for them to acquire semantic information about the image and the prompts!

---
class: middle

## Handling Ambiguity

.red[Problem]: Single prompts often have multiple valid interpretations

1. *Naive Solution*: Averaging over multiple possible interpretations...
.center[but it would create blurry, invalid masks]
2. *Better Solution*:

???

Ambiguity handling is a key innovation in SAM. Rather than averaging over multiple possible interpretations (which would create blurry, invalid masks), the model explicitly predicts multiple possible masks and ranks them by confidence.

---
class: middle

.width-95[![](figures/architecture_flat_decoder_out_mask.png)]
.center.italic[Predict *multiple masks* simultaneously (default: 3)]

.footnote[[Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

---
class: middle
count: false

.center.stretch[![](figures/architecture_flat_decoder_out_tuple.png)]
.center.italic[Use *IoU* prediction head to rank masks]

.footnote[[Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

---
class: middle

.center.width-85[![](figures/iou.png)]
.center.italic[Intersection over Union (~confidence score)]

.footnote[[IdiotDeveloper](https://idiotdeveloper.com/what-is-intersection-over-union-iou/), 2023]

---

class: middle

# Key Particularities

## Promptable architecture

<!-- Promptable as required for a foundation model -->
<!-- 3D output to solve for ambiguity -->

## Dissociated Encoders

<!-- Encoders' embeddings are reconciled at decoder level (at runtime!) with 3 attention layers  -->
<!-- Image embedding is cached for amortizing the cost -->

---
class: middle

## Losses

<!-- TODO: add math formulations -->
<!-- 2 loss functions are combined in a 20:1 ratio : Focal (+def) & Dice (+def) losses + math formulation -->
<!-- SAM is class-agnostic, i.e. so we need to modify the classification Cross Entropy loss formulation -> Focal loss is a modified version thereof -->

- Focal Loss + Dice Loss (20:1 ratio)
  - Backpropagate only from the lowest-error mask
- MSE for IoU prediction

???

- SAM is class-agnostic, i.e. so we need to modify the classification Cross Entropy loss formulation -> Focal loss is a modified version thereof
- Focal loss = helps with class imbalance by focusing on hard-to-classify pixels
- Dice loss = optimizes the overlap between predicted and ground truth masks

---
class: middle

## Training Algorithm

- Simulates interactive segmentation
- 11 iterations per example:
  1. Initial prompt (point or box)
  2. 8 iterative points from error regions
  3. 2 iterations with previous mask only
- Lightweight decoder enables many iterations per batch
- *Multiple prompts*: Use dedicated output token to reduce ambiguity

???

- Skip slide?
- The training approach is designed to mimic real-world interactive segmentation workflows. By having the model learn from its own errors through iterative refinement, it becomes more robust to various prompt types and ambiguous situations.

---
class: middle

## Zero-shot learning

---

class: middle

# Discussion

## Results

## Limitations

<!-- The need for caching the image embedding due to the heavy processing thereof limits the ability to be used as a true foundation model -->

---
class: middle

## The Big Picture

- Meta is a company with plenty of resources
- While they advocate for open-sourcing everything (e.g., *SA-1B* is a big contribution in that sense), they most potentially also have a political agenda to serve
- SAM as the ".italic[first foundation model for segmentation]" is heavily emphasized. *Why is that?*
.pull-right[
$\Rightarrow$ Segmentation is a hot-topic right now
- from self-driving cars
- to Meta's push towards mixed augmented reality
  - SAM2 was released recently for video segmentation, which supports the point!
]

---
class: middle, center

# Thank you for your attention.

---
class: middle, black-slide, center
count: false

# Additional Slides

---
class: middle
count: false

<!-- Additional Slides -->
# IoU Score Approximation

.center.stretch[![](figures/mask_decoder.png)]

.footnote[[Kirillov et al.](https://arxiv.org/abs/2304.02643), 2023]

---
class: middle
count: false

<!-- Additional Slides -->
# SA-1B Dataset and Data Engine

---
class: middle
count: false

<!-- Additional Slides -->
# SAM 2
