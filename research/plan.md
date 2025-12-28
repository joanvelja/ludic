# ScaleRL, RL-ZVP, and DAPO Implementation Plan

This document outlines the plan to integrate features from **ScaleRL**, **RL-ZVP**, and **DAPO** into the `ludic` repository, adhering to the existing modular architecture.

## 1. Credit Assignment (`src/ludic/training/credit_assignment.py`)

**Goal:** Support new advantage estimation techniques without changing the core rollout loop.

### New Components:

*   **`BatchNormalizedReturn` (ScaleRL)**
    *   **Logic:** Computes group-centered advantages ($A_i = R_i - \mu_{group}$) but normalizes them using the **batch-wide** standard deviation ($\sigma_{batch}$) instead of the group standard deviation. This is preferred by ScaleRL for more robust normalization when reward distributions are highly concentrated.
    *   **Implementation:** A new `CreditAssigner` dataclass. It will collect all centered advantages in the batch first, compute $\sigma_{batch}$, and then assign final weights.

*   **`ZVPGroupNormalizedReturn` (RL-ZVP)**
    *   **Logic:** Detects "Zero-Variance Prompts" (ZVP) where $\sigma_{group} \approx 0$.
    *   **Action:**
        *   If ZVP: Marks the rollout metadata with `zvp_direction` (+1 if all correct, -1 if all wrong) and sets the scalar weight to `0.0`. This effectively signals the ZVP-aware loss function to switch modes for these samples.
        *   If Non-ZVP: Fallbacks to standard `GroupNormalizedReturn` logic.
    *   **Implementation:** A new `CreditAssigner` dataclass that injects this metadata.

## 2. Loss Functions (`src/ludic/training/loss.py`)

**Goal:** Integrate entropy-scaling and specific aggregation methods into the loss calculation.

### New Components:

*   **`ZVPCISPOLoss` (RL-ZVP)**
    *   **Logic:** Extends `CISPOLoss`.
        *   Checks `batch["meta"]["zvp_direction"]` (or similar metadata passed via `SAWItem`).
        *   **Non-ZVP:** Uses standard CISPO update.
        *   **ZVP:** Computes the entropy-guided gradient:
            *   $Weight = \alpha \cdot H(x)$ (for positive ZVP).
            *   $Weight = -\alpha \cdot (\max(H) - H(x))$ (for negative ZVP).
    *   **Implementation:** A new dataclass extending `CISPOLoss` that accepts an `alpha_zvp` parameter and branches logic based on metadata presence.

*   **`ScaleRLLoss` (ScaleRL)**
    *   **Logic:** Enforces **prompt-level** loss aggregation.
    *   **Implementation:** This is likely a specific configuration of `CISPOLoss` or `ReinforceLoss` that correctly handles normalization by `group_size` and `prompt_length`. We will verify if a new class is needed or if `length_normalize` flags are sufficient.

*   **DAPO Clipping**
    *   **Status:** `TokenClippedSurrogateLoss` already supports asymmetric `clip_eps_low` / `clip_eps_high`. We will ensure the presets in `algorithm.py` expose these correctly.

## 3. Data Curriculum & Sampling (`src/ludic/training/batching/`)

**Goal:** Filter prompts based on historical performance.

### New Components:

*   **`NoPositiveResampling` (ScaleRL)**
    *   **Logic:** "If a prompt is solved (pass rate > 0.9), never show it again."
    *   **Implementation:**
        *   We will implement this in `src/ludic/training/batching/requests_from_dataset.py`.
        *   Create a `HistoryFilter` wrapper or similar mechanism that maintains a persistent map of `{prompt_hash: pass_rate}`.
        *   It will filter items from the dataset queue before they are turned into `RolloutRequest`s.

*   **Dynamic Sampling (DAPO)**
    *   **Logic:** "If a batch has ZVP, drop them and sample *more* to fill the batch."
    *   **Implementation:** True dynamic *resampling* is complex in the decoupled `PipelineRL` architecture. We will implement the **Offline** variant (Zero-Variance Filtering) first, as used in ScaleRL. This acts as a `SampleFilter` in `RolloutEngine.generate_batch` that drops items with weight=0 (ZVP) before the update.

## 4. Algorithm Presets (`src/ludic/training/algorithm.py`)

**Goal:** Provide user-friendly entry points for the new recipes.

### New Factories:

*   **`make_scalerl(...)`**:
    *   Combines `BatchNormalizedReturn`.
    *   Uses `CISPOLoss` (with prompt-level aggregation).
    *   Configures `FP32` precision (via existing config or new args).
    *   Enables `NoPositiveResampling`.

*   **`make_rl_zvp(...)`**:
    *   Combines `ZVPGroupNormalizedReturn`.
    *   Uses `ZVPCISPOLoss`.

## Summary of Work

1.  **Step 1:** Implement `BatchNormalizedReturn` and `ZVPGroupNormalizedReturn` in `credit_assignment.py`.
2.  **Step 2:** Implement `ZVPCISPOLoss` in `loss.py`.
3.  **Step 3:** Implement `NoPositiveResampling` logic in `batching/`.
4.  **Step 4:** Create `make_scalerl` and `make_rl_zvp` in `algorithm.py`.
