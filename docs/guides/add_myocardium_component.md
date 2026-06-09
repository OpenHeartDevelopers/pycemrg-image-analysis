# SOP: How to Add a New Myocardium Component

This is the in-repo contribution flow for adding a component (e.g.
`"la_myocardium"`) to the library's built-in schematics. If you only want a
component for your own project, you do **not** need to edit the library — define
the schematic dict in your project and inject it via
`ImageAnalysisScaffolder(extra_schematics=...)`. See
[authoring_a_recipe.md](authoring_a_recipe.md), Step 2.

To add a built-in component, follow these steps in order:

1.  **Check utilities.** Does the new component's logic require a new
    `MaskOperationMode` or a new utility function in `utilities/masks.py`?
    *   **If yes:**
        *   Add the new function to `utilities/masks.py` with unit tests.
        *   Add the new mode to the `MaskOperationMode` enum.
        *   Register the mode -> function mapping in
            `get_mask_operation_dispatcher` in `utilities/dispatchers.py`. This is
            the single registration point; do not wire it ad hoc.
    *   **If no:** proceed.

2.  **Define the schematic.** Add a new entry to `MYOCARDIUM_SCHEMATICS` in
    `src/pycemrg_image_analysis/schematics/myocardium.py` (valves and rings live in
    the sibling `valves.py` / `rings.py`). The dict is the "source of truth":
    *   Define the required `labels`.
    *   Define the required `parameters`.
    *   Define the `semantic_map`, including the ordered `APPLICATION_STEPS`.

3.  **Registration is automatic.** `schematics/__init__.py` merges
    `MYOCARDIUM_SCHEMATICS` (and the valve/ring dicts) into `ALL_SCHEMATICS` via a
    `**` spread, and `ImageAnalysisScaffolder` reads `ALL_SCHEMATICS`. Adding the
    dict entry is all that is needed — there is no separate registration list to
    update.

4.  **Write the integration test.** In `tests/integration/test_myocardium_no_cuts.py`
    (or a sibling test module), add a test function (e.g.
    `test_create_la_myocardium`):
    *   Scaffold the component's config: `scaffolder.scaffold_components(tmp_path, ["la_myocardium"])`.
    *   Load the generated `labels.yaml`, `parameters.json`, and `semantic_maps/<name>.json`.
    *   Call the same generic engine every component uses:
        `myo_logic.create_from_semantic_map(image, label_manager, parameters, semantic_map)`.
    *   Assert the output is correct. Integration tests are skipped (not failed)
        when `PYCEMRG_TEST_DATA_ROOT` is unset.
