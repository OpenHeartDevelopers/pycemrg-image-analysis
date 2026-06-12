import pytest

from pycemrg_image_analysis.utilities import schematic_tools as st
from pycemrg_image_analysis.utilities.schematic_tools import LabelStatus


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------
def test_resolve_target_category():
    category, names, schematics = st.resolve_target("valves")
    assert category == "valves"
    assert "mitral_valve" in names
    assert set(names) == set(schematics)


def test_resolve_target_single_component():
    category, names, schematics = st.resolve_target("lv_outflow")
    assert category == "myocardium"
    assert names == ["lv_outflow"]
    assert "lv_outflow" in schematics


def test_resolve_target_unknown_raises():
    with pytest.raises(KeyError):
        st.resolve_target("does_not_exist")


# ---------------------------------------------------------------------------
# summarize_schematic
# ---------------------------------------------------------------------------
def test_summarize_includes_values_param_and_steps():
    _, _, schematics = st.resolve_target("lv_outflow")
    text = st.summarize_schematic("lv_outflow", schematics["lv_outflow"])
    assert "LV_BP_label: 1" in text          # label value shown
    assert "LV_neck_WT: 2.0" in text          # parameter shown
    assert "ADD" in text                       # application step rendered


def test_summarize_renders_replace_only_targets():
    _, _, schematics = st.resolve_target("la_myocardium")
    text = st.summarize_schematic("la_myocardium", schematics["la_myocardium"])
    assert "REPLACE_ONLY — write only over: RA_BP_label" in text


def test_summarize_handles_empty_semantic_map():
    _, _, schematics = st.resolve_target("myo_push_steps")
    text = st.summarize_schematic("myo_push_steps", schematics["myo_push_steps"])
    assert "no semantic map" in text


# ---------------------------------------------------------------------------
# reconcile_labels
# ---------------------------------------------------------------------------
def test_reconcile_classifies_reused_mismatch_new():
    schematic = {"A": 1, "B": 2, "C": 3}
    existing = {"A": 1, "B": 99}  # A matches, B differs, C absent
    by_name = {r.name: r for r in st.reconcile_labels(schematic, existing)}

    assert by_name["A"].status is LabelStatus.REUSED
    assert by_name["A"].value == 1

    assert by_name["B"].status is LabelStatus.VALUE_MISMATCH
    assert by_name["B"].value == 99           # existing value wins
    assert by_name["B"].schematic_default == 2

    assert by_name["C"].status is LabelStatus.NEW
    assert by_name["C"].value == 3            # default free -> kept


def test_reconcile_assigns_free_int_on_collision():
    schematic = {"NewLabel": 5}
    existing = {"Other": 5}                    # 5 is taken
    result = st.reconcile_labels(schematic, existing)[0]
    assert result.status is LabelStatus.NEW
    assert result.value != 5
    assert result.value not in existing.values()


def test_reconcile_flags_possible_conflict():
    schematic = {"LV_BP_label": 1}
    existing = {"LV_bloodpool": 1}
    result = st.reconcile_labels(schematic, existing)[0]
    assert result.status is LabelStatus.NEW
    assert "LV_bloodpool" in result.possible_conflicts


def test_merged_labels_preserves_existing_entries():
    schematic = {"A": 1}
    existing = {"Unrelated": 7}
    reconciled = st.reconcile_labels(schematic, existing)
    merged = st.merged_labels(existing, reconciled)
    assert merged["Unrelated"] == 7           # untouched
    assert merged["A"] == 1                    # added
