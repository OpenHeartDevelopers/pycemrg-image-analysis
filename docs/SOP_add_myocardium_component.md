# **SOP: How to Add a New Myocardium Component**

To add a new component (e.g., `"la_myocardium"`) to the `MyocardiumLogic` engine, follow these steps in order:

1.  **Check Utilities:** Does the new component's logic require a new `MaskOperationMode` or a new utility function in `utilities/masks.py`?
    *   **If Yes:**
        *   Add the new function to `masks.py` with unit tests.
        *   Add the new mode to the `MaskOperationMode` Enum.
        *   Register the new mode and function in the `_get_mask_operation_dispatcher` dictionary in `MyocardiumLogic`.
    *   **If No:** Proceed.

2.  **Define the Schematic:** In `scaffolding.py`, create the new internal schematic (e.g., `_LA_MYOCARDIUM_SCHEMATIC`). This dictionary is the "source of truth."
    *   Define the required `labels`.
    *   Define the required `parameters`.
    *   Define the `semantic_map`, including the `APPLICATION_STEPS`.

3.  **Register the Component:** Add the new schematic to the main `_COMPONENT_SCHEMATICS` dictionary in the `ImageAnalysisScaffolder`.

4.  **Write the Integration Test:** In `tests/integration/test_myocardium.py`, create a new test function (e.g., `test_create_la_myocardium`).
    *   Use the scaffolder to generate the new component's configuration: `scaffolder.scaffold_components(tmp_path, ["la_myocardium"])`.
    *   Load the generated configs.
    *   Call the *exact same* generic logic engine: `myo_logic.create_from_semantic_map(...)`.
    *   Assert that the output is correct.

