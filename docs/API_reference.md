### **`pycemrg-image-analysis`: Orchestration Guide & API Reference**

This document outlines the architecture and intended use patterns for the `pycemrg-image-analysis` library. It is intended for developers writing orchestrator scripts (like ProjectScar) that consume this library.

#### **1. Core Philosophy: The Stateless Toolbox**

The fundamental design principle is **strict separation of concerns**.

*   **The Library (This Project):** The `pycemrg-image-analysis` library is a **stateless toolbox**. It contains a collection of powerful, single-purpose tools (Logic Engines, Utilities) that operate on in-memory data. The library has **zero knowledge** of file paths, project structures, or the sequence of a workflow. It never reads or writes files (except for the I/O utilities, which are called by the orchestrator).

*   **The Orchestrator (Your Script):** Your script is the **stateful orchestrator**. It is the "project manager." It is solely responsible for:
    1.  **File I/O:** All reading and writing of images and configuration files.
    2.  **Configuration Management:** Loading `labels.yaml`, `parameters.json`, and `semantic_maps`.
    3.  **Workflow Sequencing:** Defining the order of operations (e.g., "create aortic wall, then create RV myo, then push PA wall...").
    4.  **State Management:** Passing the output image from one step as the input to the next.

#### **2. The Canonical Orchestration Pattern**

Every workflow you build will follow these steps, executed by your orchestrator script.

**Step 1: Scaffold (First-Time Setup)**
Use the `ImageAnalysisScaffolder` to generate the necessary, correctly formatted configuration files for the components you need. This is the easiest way to get started.

**Step 2: Load All Configurations**
At the beginning of your script, load all required configurations into memory:
*   `labels.yaml` -> into a `pycemrg.data.labels.LabelManager` instance.
*   `parameters.json` -> into a simple Python `dict`.
*   The relevant `semantic_maps/*.json` files -> into Python `dict`s.
*   The initial input image -> into a `SimpleITK.Image` object.

**Step 3: Instantiate the Logic Engines**
Create instances of the logic classes you need (e.g., `MyocardiumLogic`, `ValveLogic`). These are lightweight objects with no state.

**Step 4: Execute the Workflow**
Loop through your defined sequence of steps. In each step:
1.  **Prepare the Inputs:** Parse the `semantic_map` dictionary and construct the strongly-typed `Rule` dataclass (e.g., `ValveRule`).
2.  **Build the Contract:** Create the final `...CreationContract` (e.g., `ValveCreationContract`), bundling the input image, rule, managers, and parameters.
3.  **Call the Logic:** Pass the contract to the appropriate logic engine method (e.g., `valve_logic.create_from_rule(contract)`).
4.  **Update State:** The logic method returns a new `sitk.Image`. Store this object to be used as the input for the next step in the sequence.

**Step 5: Persist the Final Result**
After the loop is complete, use the `utilities.save_image` function to write the final in-memory image to disk.

#### **3. The Toolbox: Key API Components**

##### **A. The Scaffolder (`pycemrg_image_analysis.ImageAnalysisScaffolder`)**
*   **Purpose:** Your entry point for new projects. Automates the creation of configuration files.
*   **Key Method:** `.scaffold_components(output_dir: Path, component_names: list[str])`
*   **Usage:**
    ```python
    from pycemrg_image_analysis import ImageAnalysisScaffolder
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components("./my_config", ["mitral_valve", "aortic_valve"])
    ```

##### **B. Logic Engines (`logic.MyocardiumLogic`, `logic.ValveLogic`)**
*   **Purpose:** Contain the core, high-level scientific workflows.
*   **Key Methods:**
    *   `myocardium_logic.create_from_semantic_map(contract: MyocardiumCreationContract)`
    *   `valve_logic.create_from_rule(contract: ValveCreationContract)`
    *   `myocardium_logic.push_structure(image: sitk.Image, contract: PushStructureContract)`
*   **Usage:** These are the main "verbs" of your workflow. The orchestrator's job is to prepare the `Contract` objects that these methods consume.

##### **C. Contracts & Rules (`logic.contracts`)**
*   **Purpose:** The strongly-typed dataclasses that form the safe, explicit API for the logic engines.
*   **Key Objects:**
    *   `MyocardiumRule`, `ValveRule`: You build these by parsing your `semantic_map.json`.
    *   `MyocardiumCreationContract`, `ValveCreationContract`: You build these to pass into the logic methods, containing the `Rule` and other necessary data.
    *   `PushStructureContract`: A simpler contract for direct logic calls.

##### **D. Utilities (`utilities.*`)**
*   **Purpose:** Low-level, pure functions. The orchestrator primarily uses the I/O functions.
*   **Key Methods for Orchestrator:**
    *   `utilities.load_image(path: Path) -> sitk.Image`
    *   `utilities.save_image(image: sitk.Image, path: Path)`

## `pycemrg_image_analysis.utilities.metrics`

Image quality metrics for volume comparison and validation. All functions expect pre-normalized `[0, 1]` data in `(Z, Y, X)` format.

### Quick Reference

| Function | Purpose | Key Parameters | Returns |
|----------|---------|----------------|---------|
| `compute_mse()` | Mean Squared Error | `predicted`, `ground_truth` | `float` (lower is better) |
| `compute_psnr()` | Peak Signal-to-Noise Ratio | `predicted`, `ground_truth`, `data_range=1.0` | `float` in dB (20-50 typical) |
| `compute_ssim()` | Structural Similarity Index | `predicted`, `ground_truth`, `win_size=7` | `float` in [0, 1] (>0.9 excellent) |
| `compute_gradient_error()` | Edge sharpness preservation | `predicted`, `ground_truth`, `axis=0` | `float` (lower is better) |
| `compare_volumes()` | Batch metric computation | `predicted`, `ground_truth`, `metrics=None` | `dict[str, float]` |

### Core Functions

#### `compute_mse(predicted, ground_truth) -> float`
Mean Squared Error between volumes. Returns 0.0 for identical volumes.

#### `compute_psnr(predicted, ground_truth, data_range=1.0) -> float`
Peak Signal-to-Noise Ratio in dB. Returns `inf` for identical volumes.

#### `compute_ssim(predicted, ground_truth, data_range=1.0, win_size=7, **kwargs) -> float`
3D Structural Similarity Index. Values: `>0.9` excellent, `0.7-0.9` good, `<0.7` poor.

**Note:** For thin slices with anisotropic spacing, use smaller `win_size=3`.

#### `compute_gradient_error(predicted, ground_truth, axis=0) -> float`
Mean absolute gradient error along specified axis (0=Z, 1=Y, 2=X). Useful for interpolation quality assessment.

#### `compare_volumes(predicted, ground_truth, data_range=1.0, metrics=None) -> Dict[str, float]`
Compute multiple metrics efficiently.

**Default metrics:** `['mse', 'psnr', 'ssim', 'gradient']`

**Available metrics:** `'mse'`, `'psnr'`, `'ssim'`, `'gradient'` (Z-axis), `'gradient_x'`, `'gradient_y'`, `'gradient_z'`

### Example Usage

```python
from pycemrg_image_analysis.utilities.metrics import compare_volumes

# Validate interpolation results
results = compare_volumes(interpolated, ground_truth, metrics=['mse', 'psnr', 'ssim'])
print(f"PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}")

# Test Z-axis gradient preservation
from pycemrg_image_analysis.utilities.metrics import compute_gradient_error
z_error = compute_gradient_error(interpolated, ground_truth, axis=0)
```

### Design Notes
- All inputs must be pre-normalized to `[0, 1]` range
- Arrays must follow `(Z, Y, X)` convention
- Shape mismatches raise `ValueError`
- Returns Python `float`, not NumPy scalars

**Full API:** See `metrics_api.md` for detailed documentation.
