# In logic/myocardium.py

class MyocardiumLogic:
    def create_myocardium_from_rule(
        self, contract: MyocardiumCreationContract
    ) -> sitk.Image:
        """The generic engine for creating any piece of myocardium."""
        # Step 1: Translate names to values using the provided managers
        lm = contract.label_manager
        rule = contract.rule
        source_bp_value = lm.get_value(rule.source_bp_label_name)
        target_myo_value = lm.get_value(rule.target_myo_label_name)
        wall_thickness = contract.parameters[rule.wall_thickness_parameter_name]
        
        # Step 2: Execute the core pattern
        dist_map = filters.distance_map(contract.input_image, source_bp_value)
        new_wall_mask = filters.threshold_filter(dist_map, 0, wall_thickness, binarise=True)
        
        # Step 3: Apply the new wall back using the specified rule
        # ... logic to dispatch to the correct masks.py utility based on
        # rule.application_mode and rule.application_rule_label_names ...

        return modified_image