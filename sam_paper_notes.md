# Segment Anything

## Abstract

- Segment Anything (SA) project
- task, model, and dataset for image segmentation
- model is designed and trained to be promptable so it can transfer zero-shot to new image distributions and tasks

## 1. Introduction

- Foundation models: can generalize to tasks and data distributions beyond those seen during training
- When scaled and trained with abundant data, these models’ zero and few-shot performance compares surprisingly well to fine-tuned models
- Once trained, engineered text prompts enable zero-shot generalization to novel concepts and data distributions. Such encoders also compose effectively with other modules to enable downstream tasks
- In this work, our goal is to build a foundation model for image segmentation. That is, we seek to develop a promptable model and pre-train it on a broad dataset using a task that enables powerful generalization. With this model, we aim to solve a range of downstream segmentation problems on new data distributions using prompt engineering. The success of this plan hinges on three components: task, model, and data. To develop them, we address the following questions about image segmentation:

1. What task will enable zero-shot generalization?
2. What is the corresponding model architecture?
3. What data can power this task and model?

### Task

- we propose the promptable segmentation task, where the goal is to return a valid segmentation mask given any segmentation prompt
- A prompt simply specifies what to segment in an image
- The requirement of a valid output mask means that even when a prompt is ambiguous and could refer to multiple objects, the output should be a reasonable mask for at least one of those objects

### Model

- The promptable segmentation task and the goal of real-world use impose that the model must support flexible prompts, needs to compute masks in amortized real-time, and must be ambiguity-aware.
- A simple design satisfies all constraints:

1. a powerful image encoder computes an image embedding,
2. a prompt encoder embeds prompts,
3. and then the two information sources are combined in a lightweight mask decoder that predicts segmentation masks.

- Separating image encoder and prompt encoder / mask decoder allows the same image embedding to be reused
- We focus on point, box, and mask prompts, and also present a POC with free-form text prompts. To make SAM ambiguity-aware, we design it to predict multiple masks for a single prompt

### Data Engine

- To achieve strong generalization to new data distributions,  necessary to train SAM on a large and diverse set of masks
=> Build a "data engine" (model-in-the-loop)
- Done in 3 stages:

1. assisted-manual
2. semi-automatic
3. fully automatic (final model trained on this dataset)

### Dataset

SA-1B, collected fully automatically using the final stage of our data engine, has 400× more masks than any existing segmentation dataset [66, 44, 117, 60], and as we verify extensively, the masks are of high quality and diversity.

## 2. Segment Anything Task

- We take inspiration from NLP: to build a foundation model for segmentation, we aim to define a task with analogous capabilities (as th next-token prediction task)

### Task

- **Prompt Concept in Segmentation:**
  - Adapts the NLP idea of a prompt to segmentation.
  - A prompt can be points, boxes, masks, text, or any info guiding segmentation.

- **Promptable Segmentation Task:**
  - The model must return a valid segmentation mask for any prompt.
  - A "valid" mask means it should reasonably correspond to at least one object, even if the prompt is ambiguous.
  - Analogous to how language models generate coherent responses to ambiguous text prompts.

- **Why This Task?**
  - Enables natural pre-training.
  - Supports zero-shot transfer by allowing adaptation to new segmentation tasks via prompts.

### Pre-training

- Uses a sequence of prompts (points, boxes, masks) to train the model.
- Unlike interactive segmentation, the goal is to always predict a valid mask, even for ambiguous prompts.
- Ensures effectiveness even for ambiguous prompts
- Requires specialized modeling and training loss strategies.

### Zero-shot Transfer

- The model learns to handle any prompt during inference.
- Downstream tasks can be solved by designing appropriate prompts.
- Example: A cat detector’s bounding box can serve as a prompt for instance segmentation.
- Many segmentation tasks can be reformulated as prompt-based tasks.

### Related Tasks

- Covers various segmentation types (interactive, edge detection, super-pixelization, etc.).
- Aim: Create a general segmentation model adaptable via prompts.
- Different from multi-task models, which are trained and tested on fixed tasks.
- Can integrate with existing models (e.g., object detector + segmentation model for instance segmentation).

### Discussion

- Prompting and composition enable flexible and extensible model use.
- Similar to how CLIP is used in DALL·E for text-image alignment.
- Composable design broadens application possibilities beyond fixed-task systems.
- Promptable segmentation can function within larger algorithmic systems, unlike purely interactive models.

## 3. Segment Anything Model

- SAM has 3 components
  - image encoder
  - flexible prompt encoder
  - fast mask decoder

1. Image Encoder

- Uses a **Vision Transformer (ViT-H/16)** pre-trained with **Masked Autoencoder (MAE)**.
- **14×14 windowed attention** and **4 global attention blocks**.
- Outputs a **16× downscaled embedding** of the input image.
- Input resolution: **1024×1024** (rescaled and padded).
- Embedding size: **64×64**.
- Channel dimension reduction:
  - **1×1 convolution** (256 channels).
  - **3×3 convolution** (256 channels).
  - Each convolution followed by **Layer Normalization**.

2. Prompt Encoder

2.1 **Sparse Prompts (i.e., points and boxes):**

- Mapped to **256-dimensional vectorial embeddings**.
- **Point representation**: Sum of **positional encoding** and one of two learned embeddings (foreground/background).
- **Box representation**: Two embeddings (top-left and bottom-right corners) with positional encoding.
- **Free-form text**: Uses **CLIP text encoder**.

2.2 **Dense Prompts (Masks):**

- Input masks at **1/4 resolution** of input image (so 1024/4 = 256²)
- Further downscaled **4×** using:
  - **Two 2×2 stride-2 convolutions** (output channels: **4, 16** respectively).
  - **1×1 convolution** (256 channels).
- Each layer separated by **GELU activation** and **Layer Normalization**.
- Mask embedding added **element-wise** to image embedding.
- If no mask prompt, a **learned "no mask" embedding** is added.

3. Lightweight Mask Decoder

- **Architecture**: Modified Transformer decoder inspired by segmentation models.
- **Input**:
  - Image embedding (64², 256-dimensional vectors).
  - Set of prompt embeddings (referred to as "tokens").
  - Learned output token embedding (analogous to a `[class]` token).
- **Decoder layers** (2 layers total, each performing 4 steps):
  1. **Self-attention** on tokens.
  2. **Cross-attention** (tokens as queries, image embedding as keys/values).
  3. **Point-wise MLP** updates each token.
  4. **Cross-attention** (image embedding as queries, tokens as keys/values).
- **Positional Encoding**:
  - Added to image embedding whenever used in attention.
  - Original prompt tokens (with positional encoding) are re-added during attention.
- **Transformer parameters**:
  - Embedding dimension: **256**.
  - MLP internal dimension: **2048** (applied only to prompt tokens, usually ≤ 20 tokens).
  - Cross-attention layers reduce query/key/value channel dimension to **128** for efficiency.
  - **8 attention heads** per layer.
- **Upsampling process**:
  - **4× upsampling** using two **transposed convolution layers** (input embedding initially downscaled 4× relative to input image).
  - **Transposed convolutions**:
    - **2×2 kernel, stride 2**.
    - Output channels: **64 and 32**.
    - **GELU activations** and **layer normalization**.
- **Final steps**:
  - Tokens attend to upsampled image embedding.
  - Output token embedding passed through a **3-layer MLP** (matches upscaled image embedding's channel dimension).
  - Final mask prediction via **point-wise spatial product** between the upscaled image embedding and MLP output.
- **Regularization**:
  - **Residual connections**, **layer normalization**, and **dropout (0.1 at training)** in all self/cross-attention and MLP layers.

### Resolving for ambiguity

- A single prompt can correspond to multiple valid masks; averaging over them is undesirable.
- Solution: Predict **multiple masks (default: 3)** using multiple output tokens.
  - These often represent **whole, part, and subpart** of an object.
- Loss computation:
  - Loss is calculated for each predicted mask.
  - **Backpropagation only uses the mask with the lowest loss** (common multi-output model technique).
- **Mask ranking**:
  - An additional small head predicts the **IoU** (Intersection over Union) between each mask and the object.
- **Multiple prompts case**:
  - Ambiguity is rarer, so predicting **three masks becomes unnecessary**.
  - A **fourth output token** predicts a single mask when multiple prompts are given.
  - This fourth mask is **never used for a single prompt** and **only used for multiple prompts**.

### Losses

- SAM is class-agnostic, so CrossEntropyLoss doesn't make sense here
- **Mask prediction loss** = **Focal Loss + Dice Loss (20:1 ratio)**.
- **IoU prediction head**:
  - Trained using **mean-square-error loss** (difference between predicted IoU and actual IoU).
  - Added to the mask loss with a **scaling factor of 1.0**.
- **No auxiliary deep supervision** after each decoder layer (unlike prior work).

### Training Algorithm

#### **Simulating Interactive Segmentation During Training**

- A **foreground point or bounding box** is randomly selected for the target mask (equal probability).
- **Point selection**:
  - Sampled uniformly from the ground truth mask.
- **Bounding box selection**:
  - Based on the ground truth mask’s bounding box.
  - **Random noise** added (std. deviation = **10% of box sidelength**, max **20 pixels**).
  - Balances **tight instance segmentation** and **loose interactive segmentation**.

#### **Iterative Mask Refinement**

- First prediction made using the initial prompt.
- **Subsequent points** selected from **error regions** (difference between predicted mask & ground truth):
  - **Foreground points** from false negatives.
  - **Background points** from false positives.
- The **previous mask prediction** is also used as an additional prompt.
- **Unthresholded mask logits** are passed instead of binarized masks to retain maximal information.
- If multiple masks are predicted, the one with the **highest predicted IoU** is used for the next step.

#### **Training Iterations**

- **Total of 11 iterations** per training example:
  1. **1 initial input prompt**.
  2. **8 iteratively sampled points** (from error regions).
  3. **2 extra iterations without new points** (to refine masks).
- One of the **two no-new-point iterations** is randomly inserted within the 8 iterations, and the other is always at the end.

#### **Efficiency Considerations**

- Using **many iterations is feasible** due to the **lightweight mask decoder**, which requires **<1% of the image encoder's compute**.
- Unlike previous interactive methods, this allows **many interactive steps per optimizer update**.

## 4. Segment Anything Data Engine

The data engine is designed to collect a dataset of 1.1 billion masks (SA-1B) through three stages: model-assisted manual annotation, semi-automatic annotation, and fully automatic annotation.

a) Assisted-manual stage:

- Professional annotators labeled masks using a browser-based interactive segmentation tool powered by SAM.
- Annotators clicked foreground/background object points and refined masks using pixel-precise tools.
- No semantic constraints were imposed; annotators labeled both "stuff" and "things."
- Annotators labeled objects they could name or describe but did not collect these names or descriptions.
- SAM was initially trained using common public segmentation datasets and retrained using newly annotated masks.
- Average annotation time per mask decreased from 34 to 14 seconds as the model improved.
- Collected 4.3 million masks from 120,000 images.

b) Semi-automatic stage:

- Aimed to increase mask diversity by focusing annotators on less prominent objects.
- Automatically detected confident masks and presented annotators with images prefilled with these masks.
- Annotators annotated additional unannotated objects.
- Trained a bounding box detector on first stage masks using a generic "object" category.
- Collected an additional 5.9 million masks in 180,000 images (total of 10.2 million masks).
- Average annotation time per mask increased to 34 seconds due to more challenging objects.
- Average number of masks per image increased from 44 to 72 masks.

c) Fully automatic stage:

- Annotation was fully automatic due to enhancements in the model.
- Developed an ambiguity-aware model to predict valid masks even in ambiguous cases.
- Prompted the model with a 32×32 regular grid of points and predicted a set of masks for each point.
- IoU prediction module used to select confident masks.
- Applied non-maximal suppression (NMS) to filter duplicates.
- Processed multiple overlapping zoomed-in image crops to improve smaller mask quality.
- Applied fully automatic mask generation to all 11 million images, producing a total of 1.1 billion high-quality masks.

## 5. SA-1B Dataset

### Key Points of Segment Anything Dataset (SA-1B)

- **Dataset Overview**:
  - SA-1B consists of **11M high-resolution**, **licensed**, and **privacy-protecting** images.
  - Contains **1.1B segmentation masks**, mostly generated automatically.
  - Released for **research use** with a favorable license.

### **Images**

- **Sourced from professional photographers** and licensed.
- **High resolution** (avg. **3300×4950 px**), but **downsampled** to 1500 px for accessibility.
- **Higher resolution** than many existing datasets (e.g., COCO ~480×640 px).
- **Privacy protections**: Faces and license plates are blurred.

### **Masks**

- **99.1% generated automatically**; evaluated for quality.
- **IoU comparison** with professional annotations:
  - **94%** of masks have **>90% IoU**.
  - **97%** have **>75% IoU**.
  - Comparable to human annotator consistency (**85-91% IoU**).
- Automatic masks are nearly as effective as fully annotated masks for training.

### **Mask Properties**

- **Spatial distribution**: Better coverage of image corners than **LVIS** and **ADE20K**.
- **Dataset size**:
  - **11× more images, 400× more masks** than Open Images.
  - **36× more masks per image** than Open Images, **3.5× more than ADE20K**.
- **Mask size & complexity**:
  - More **small & medium masks** due to high mask density.
  - **Mask concavity** (shape complexity) is similar to other datasets.
