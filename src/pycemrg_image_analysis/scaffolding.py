# src/pycemrg_image_analysis/scaffolding.py

import json
import yaml
from pathlib import Path
from typing import Union

from pycemrg.files import ConfigScaffolder
from pycemrg_image_analysis.schematics import ALL_SCHEMATICS


class ImageAnalysisScaffolder(ConfigScaffolder):
    """
    Extends the core scaffolder to create fully-populated configuration files
    for specific image analysis workflow recipes.
    """

    _COMPONENT_SCHEMATICS = ALL_SCHEMATICS

    def scaffold_components(
        self,
        output_dir: Union[str, Path],
        component_names: list[str],
        overwrite: bool = False,
    ) -> None:
        """
        Generates a set of config files from a list of component schematics.

        Args:
            output_dir: The directory where configuration files will be saved.
            component_names: A list of component names (e.g., ["aortic_wall"]).
            overwrite: If True, will overwrite existing files.
        """
        output_dir = Path(output_dir)
        merged_labels = {}
        merged_parameters = {}

        # --- 1. Collect and merge data ---
        for name in component_names:
            schematic = self._COMPONENT_SCHEMATICS.get(name)
            if not schematic:
                raise ValueError(f"Unknown component schematic requested: '{name}'")

            merged_labels.update(schematic["labels"])
            merged_parameters.update(schematic["parameters"])

            # --- 2. Write the unique semantic map for this component ---
            maps_dir = output_dir / "semantic_maps"
            json_map = {k.name: v for k, v in schematic["semantic_map"].items()}
            map_content = json.dumps(json_map, indent=2)
            # REUSE the inherited _write_file method for robustness
            self._write_file(maps_dir / f"{name}.json", map_content, overwrite)

        # --- 3. Write the merged configuration files ---
        # Programmatically generate the full content
        labels_content = yaml.dump({"labels": merged_labels}, sort_keys=False)
        params_content = json.dumps(merged_parameters, indent=2)

        # REUSE the inherited _write_file method
        self._write_file(output_dir / "labels.yaml", labels_content, overwrite)
        self._write_file(output_dir / "parameters.json", params_content, overwrite)

        print(
            f"Scaffolding complete in '{output_dir.resolve()}' for: {', '.join(component_names)}"
        )

    def create_labels_manifest(self, *args, **kwargs):
        """Override the base method to prevent confusion."""
        raise NotImplementedError(
            "Use scaffold_components() to generate a populated labels.yaml. "
            "Generating a template-based manifest is not supported by this scaffolder."
        )