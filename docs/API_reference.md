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
