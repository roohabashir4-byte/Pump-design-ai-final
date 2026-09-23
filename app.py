"""PumpDesign AI - structured Streamlit design interface.

Step 12 connects the frozen structured inputs to the Step 10 agent and the
Step 11 local RAG store. Numerical design remains owned by deterministic
engineering workflows.
"""
from __future__ import annotations

import json
import csv
import io
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import PumpDesignAgent
from standards.rag_store import RAGStore
from memory.database import MemoryDB
from memory.projects import ProjectMemory
from memory.revisions import create_design_revision
from reports.calculation_report import (
    build_report_payload,
    render_markdown,
    render_html,
    generate_pdf,
)
from engineering.pump_selection import validate_commercial_pump


st.set_page_config(
    page_title="PumpDesign AI",
    page_icon="💧",
    layout="wide",
)

UNITS = {
    "length": ["ft", "m"],
    "volume": ["US gal", "m³", "L"],
    "time": ["min", "h", "s"],
    "flow": ["gpm", "L/s", "m³/s"],
    "pressure": ["psi", "kPa", "bar"],
    "temperature": ["°C", "°F"],
}


def val_unit(
    label: str,
    key: str,
    kind: str,
    default: float | None = None,
) -> dict[str, Any] | None:
    c1, c2 = st.columns([2.2, 1])

    value = c1.number_input(
        label,
        value=default if default is not None else 0.0,
        key=f"v_{key}",
    )

    unit = c2.selectbox(
        "Unit",
        UNITS[kind],
        key=f"u_{key}",
    )

    if value == 0.0:
        return None

    return {
        "value": float(value),
        "unit": unit,
    }


def optional_val_unit(
    label: str,
    key: str,
    kind: str,
) -> dict[str, Any] | None:
    known = st.checkbox(
        f"I know the {label.lower()}",
        key=f"known_{key}",
        value=False,
    )

    if not known:
        st.caption(
            "Leave unknown when the value is not available. "
            "PumpDesign AI will not invent it."
        )
        return None

    return val_unit(label, key, kind)


def project_fields() -> dict[str, Any]:
    c1, c2 = st.columns(2)

    project_name = c1.text_input(
        "Project Name",
        key="project_name",
    )

    location = c2.text_input(
        "Location",
        key="location",
    )

    c3, c4 = st.columns(2)

    jurisdiction = c3.text_input(
        "Jurisdiction / Authority",
        key="jurisdiction",
        placeholder="e.g. Punjab / WASA / AHJ",
    )

    building_type = c4.text_input(
        "Building Type",
        key="building_type",
        placeholder="e.g. Apartment",
    )

    return {
        "project_name": project_name,
        "location": location,
        "jurisdiction": jurisdiction,
        "building_type": building_type,
    }


def demand_fields(prefix: str) -> dict[str, Any]:
    basis = st.radio(
        "Demand basis",
        [
            "Total Design Demand",
            "Calculate from Population",
        ],
        horizontal=True,
        key=f"demand_basis_{prefix}",
    )

    if basis == "Total Design Demand":
        demand = val_unit(
            "Total Design Demand",
            f"{prefix}_total_demand",
            "flow",
        )

        return {
            "demand_basis": "TOTAL_DESIGN_DEMAND",
            "total_design_demand": demand,
        }

    c1, c2 = st.columns(2)

    population = c1.number_input(
        "Population",
        min_value=1.0,
        value=1.0,
        step=1.0,
        key=f"{prefix}_population",
    )

    per_capita = val_unit(
        "Per-capita Demand",
        f"{prefix}_percapita",
        "volume",
    )

    return {
        "demand_basis": "POPULATION_PER_CAPITA",
        "population": population,
        "per_capita_demand": per_capita,
    }


def transfer_ui() -> dict[str, Any]:
    data = project_fields()

    st.subheader("Demand")
    data.update(demand_fields("transfer"))

    st.subheader("Tanks")

    data["ugt_capacity"] = val_unit(
        "UGT Capacity",
        "transfer_ugt_capacity",
        "volume",
    )

    data["oht_capacity"] = val_unit(
        "OHT Capacity",
        "transfer_oht_capacity",
        "volume",
    )

    c1, c2 = st.columns(2)

    with c1:
        data["ugt_design_water_level"] = val_unit(
            "UGT Design Water Level",
            "transfer_ugt_level",
            "length",
        )

    with c2:
        data["oht_design_water_level"] = val_unit(
            "OHT Design Water Level",
            "transfer_oht_level",
            "length",
        )

    st.subheader("Pump Requirement")

    data["required_transfer_time"] = val_unit(
        "Required Transfer Time",
        "transfer_time",
        "time",
    )

    st.subheader("Pipe")

    c1, c2 = st.columns(2)

    with c1:
        data["suction_pipe_length"] = val_unit(
            "Suction Pipe Length",
            "transfer_suction_length",
            "length",
        )

    with c2:
        data["delivery_pipe_length"] = val_unit(
            "Delivery Pipe Length",
            "transfer_delivery_length",
            "length",
        )

    data["pipe_material"] = st.selectbox(
        "Pipe Material",
        [
            "MS / Carbon Steel",
            "PVC-U",
            "HDPE / PE100",
            "PP-R / PPR",
            "Other / Custom",
        ],
        key="transfer_material",
    )

    data["different_scenario"] = st.text_area(
        "Different Scenario (optional)",
        key="transfer_custom",
    )

    return data


def booster_ui() -> dict[str, Any]:
    data = project_fields()

    data["use_case"] = st.selectbox(
        "Use Case",
        [
            "Upper floors of one building",
            "Upper floors of multiple buildings supplied by common OHT",
            "Entire building / pressure zone because OHT gravity pressure is insufficient",
            "Different Scenario",
        ],
        key="booster_use_case",
    )

    st.subheader("Demand")
    data.update(demand_fields("booster"))

    st.subheader("OHT / Source")

    data["oht_design_water_level"] = val_unit(
        "OHT Design Water Level",
        "booster_oht_level",
        "length",
    )

    data["available_inlet_pressure"] = optional_val_unit(
        "Available Inlet Pressure",
        "booster_inlet_pressure",
        "pressure",
    )

    st.subheader("Booster Zone")

    c1, c2 = st.columns(2)

    with c1:
        data["lowest_elevation_served"] = val_unit(
            "Lowest Elevation Served",
            "booster_lowest",
            "length",
        )

    with c2:
        data["highest_elevation_served"] = val_unit(
            "Highest Elevation Served",
            "booster_highest",
            "length",
        )

    data["required_residual_pressure"] = val_unit(
        "Required Residual Pressure",
        "booster_residual",
        "pressure",
    )

    st.subheader("Critical Route")

    data["critical_route_length"] = val_unit(
        "Critical Route Length",
        "booster_route",
        "length",
    )

    data["pipe_material"] = st.selectbox(
        "Pipe Material",
        [
            "MS / Carbon Steel",
            "PVC-U",
            "HDPE / PE100",
            "PP-R / PPR",
            "Other / Custom",
        ],
        key="booster_material",
    )

    data["different_scenario"] = st.text_area(
        "Different Scenario (optional)",
        key="booster_custom",
    )

    return data


def submersible_ui() -> dict[str, Any]:
    data = project_fields()

    data["primary_use"] = st.selectbox(
        "Primary Use",
        [
            "TUBE_WELL_TO_UGT",
            "TUBE_WELL_TO_OHT",
            "SUMP_TO_UGT",
            "SUMP_TO_OHT",
            "DIFFERENT_SCENARIO",
        ],
        key="sub_primary_use",
    )

    st.subheader("Demand")
    data.update(demand_fields("submersible"))

    st.subheader("Source")

    if data["primary_use"].startswith("TUBE_WELL"):
        data["static_water_level"] = optional_val_unit(
            "Static Water Level",
            "sub_static",
            "length",
        )

        data["dynamic_water_level"] = optional_val_unit(
            "Dynamic Water Level",
            "sub_dynamic",
            "length",
        )

        data["well_yield"] = optional_val_unit(
            "Well Yield",
            "sub_yield",
            "flow",
        )

    elif data["primary_use"].startswith("SUMP"):
        data["sump_water_level"] = optional_val_unit(
            "Sump Water Level",
            "sub_sump_level",
            "length",
        )

        data["sump_capacity"] = optional_val_unit(
            "Sump Capacity",
            "sub_sump_capacity",
            "volume",
        )

    st.subheader("Destination")

    data["destination_water_level"] = val_unit(
        "Destination Water Level",
        "sub_destination_level",
        "length",
    )

    st.subheader("Flow")

    flow_basis = st.radio(
        "Flow basis",
        [
            "Calculate from Required Filling Time",
            "Known Design Flow",
        ],
        horizontal=True,
        key="sub_flow_basis",
    )

    if flow_basis == "Known Design Flow":
        data["known_design_flow"] = val_unit(
            "Known Design Flow",
            "sub_known_flow",
            "flow",
        )

    else:
        st.info(
            "For filling-time sizing, the deterministic workflow requires "
            "the destination filling volume. It can come from project/tank "
            "data; it is not invented by the app."
        )

        data["destination_capacity"] = optional_val_unit(
            "Destination Capacity",
            "sub_destination_capacity",
            "volume",
        )

        data["required_filling_time"] = val_unit(
            "Required Filling Time",
            "sub_filling_time",
            "time",
        )

    st.subheader("Pipe")

    data["delivery_pipe_length"] = val_unit(
        "Delivery Pipe Length",
        "sub_delivery_length",
        "length",
    )

    data["pipe_material"] = st.selectbox(
        "Pipe Material",
        [
            "MS / Carbon Steel",
            "PVC-U",
            "HDPE / PE100",
            "PP-R / PPR",
            "Other / Custom",
        ],
        key="sub_material",
    )

    data["different_scenario"] = st.text_area(
        "Different Scenario (optional)",
        key="sub_custom",
    )

    return data


def hot_water_ui() -> dict[str, Any]:
    data = project_fields()

    data["system_arrangement"] = st.selectbox(
        "System Arrangement",
        [
            "Central heater + vertical riser",
            "Central heater + horizontal floor loops",
            "Combined vertical + horizontal distribution",
            "Separate hot-water zones",
            "Different Scenario",
        ],
        key="hw_arrangement",
    )

    st.subheader("Temperature")

    c1, c2 = st.columns(2)

    with c1:
        data["hot_water_supply_temperature"] = val_unit(
            "Hot-Water Supply Temperature",
            "hw_supply_temp",
            "temperature",
        )

    with c2:
        data["required_return_temperature"] = val_unit(
            "Required Return Temperature",
            "hw_return_temp",
            "temperature",
        )

    st.subheader("Piping")

    data["total_hot_water_circulation_pipe_length"] = val_unit(
        "Total Hot-Water Circulation Pipe Length",
        "hw_length",
        "length",
    )

    data["pipe_material"] = st.selectbox(
        "Pipe Material",
        [
            "MS / Carbon Steel",
            "PVC-U",
            "HDPE / PE100",
            "PP-R / PPR",
            "Copper",
            "Other / Custom",
        ],
        key="hw_material",
    )

    data["different_scenario"] = st.text_area(
        "Different Scenario (optional)",
        key="hw_custom",
    )

    return data


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: clean(v)
            for k, v in value.items()
            if v is not None and v != ""
        }

    if isinstance(value, list):
        return [clean(v) for v in value]

    return value


def commercial_pump_ui(
    duty_result: dict[str, Any] | None,
) -> None:
    """Validate a user-supplied manufacturer pump curve; never invent pump data."""

    if not duty_result or not duty_result.get("pump_duty"):
        return

    duty = duty_result["pump_duty"]

    if duty.get("flow") is None or duty.get("head") is None:
        return

    st.divider()
    st.subheader("Commercial Pump Curve Validation")

    st.caption(
        "Enter data exactly from the manufacturer pump curve/datasheet. "
        "PumpDesign AI validates the supplied pump; it does not rank pumps "
        "or invent manufacturer data."
    )

    c1, c2, c3 = st.columns(3)

    manufacturer = c1.text_input(
        "Manufacturer",
        key="pump_mfr",
    )

    model = c2.text_input(
        "Pump Model",
        key="pump_model",
    )

    source = c3.text_input(
        "Curve / Datasheet Source",
        key="pump_curve_source",
        placeholder="Manufacturer datasheet, curve number, URL, etc.",
    )

    c1, c2, c3 = st.columns(3)

    flow_unit = c1.selectbox(
        "Curve Flow Unit",
        ["m³/s", "L/s", "gpm"],
        key="pump_curve_flow_unit",
    )

    head_unit = c2.selectbox(
        "Curve Head Unit",
        ["m", "ft"],
        key="pump_curve_head_unit",
    )

    npshr_unit = c3.selectbox(
        "NPSHr Unit",
        ["m", "ft"],
        key="pump_curve_npshr_unit",
    )

    st.caption(
        "CSV columns: flow, head, and optional efficiency "
        "(fraction, e.g. 0.78) and npshr. At least two curve points are required."
    )

    csv_text = st.text_area(
        "Manufacturer Curve CSV",
        value=(
            "flow,head,efficiency,npshr\n"
            "50,55,0.65,2\n"
            "100,45,0.78,3\n"
            "150,30,0.74,5"
        ),
        height=150,
        key="pump_curve_csv",
    )

    motor_power = st.number_input(
        "Manufacturer Motor Power (optional)",
        min_value=0.0,
        value=0.0,
        key="pump_motor_power",
    )

    motor_unit = st.selectbox(
        "Motor Power Unit",
        ["kW", "hp"],
        key="pump_motor_unit",
    )

    validate = st.button(
        "Validate Manufacturer Pump",
        type="secondary",
        use_container_width=True,
        key="validate_commercial_pump",
    )

    if validate:
        try:
            rows = []

            for row in csv.DictReader(
                io.StringIO(csv_text.strip())
            ):
                if not row.get("flow") or not row.get("head"):
                    continue

                item = {
                    "flow": float(row["flow"]),
                    "head": float(row["head"]),
                }

                if row.get("efficiency") not in (None, ""):
                    item["efficiency"] = float(row["efficiency"])

                if row.get("npshr") not in (None, ""):
                    item["npshr"] = float(row["npshr"])

                rows.append(item)

            manufacturer_data = {
                "manufacturer": manufacturer or None,
                "model": model or None,
                "curve_source": source or None,
                "flow_unit": flow_unit,
                "head_unit": head_unit,
                "npshr_unit": npshr_unit,
                "curve_points": rows,
            }

            if motor_power > 0:
                manufacturer_data["motor_power"] = motor_power
                manufacturer_data["motor_power_unit"] = motor_unit

            validation = validate_commercial_pump(
                duty,
                manufacturer_data,
            )

            st.session_state["commercial_pump_validation"] = validation
            st.session_state["commercial_pump_data"] = manufacturer_data

        except Exception as exc:
            st.error(
                f"Manufacturer curve validation failed: {exc}"
            )
            return

    validation = st.session_state.get(
        "commercial_pump_validation"
    )

    if validation:
        if validation.get("status") == "PASS":
            st.success(
                "Manufacturer pump curve is compatible with the "
                "calculated duty point based on the supplied data."
            )
        else:
            st.warning(
                "Manufacturer pump validation did not pass. "
                "Review the checks and unresolved items before using "
                "the pump for design."
            )

        st.json(validation)


def render_result(result: Any) -> None:
    if isinstance(result, str):
        st.markdown(result)
        return

    if not isinstance(result, dict):
        st.write(result)
        return

    st.subheader("Design Result")

    status = result.get("status", "UNKNOWN")

    if status in {
        "CALCULATED",
        "CALCULATED_WITH_WARNINGS",
    }:
        st.success(
            f"Design status: {status}"
        )

    elif status in {
        "MISSING_INPUT",
        "NOT_CALCULABLE",
    }:
        st.warning(
            f"Design status: {status}"
        )

    else:
        st.info(
            f"Design status: {status}"
        )

    if result.get("message"):
        st.info(result["message"])

    if result.get("calculations"):
        st.subheader("Calculations")

        for item in result["calculations"]:
            with st.expander(
                item.get("name", "Calculation"),
                expanded=False,
            ):
                st.write(
                    {
                        k: v
                        for k, v in item.items()
                        if v is not None
                    }
                )

    if result.get("validation"):
        st.subheader("Validation")

        for item in result["validation"]:
            st.write(item)

    if result.get("pipe_sizing"):
        st.subheader("Pipe Sizing")
        st.json(result["pipe_sizing"])

    st.download_button(
        "Download Result JSON",
        data=json.dumps(
            result,
            indent=2,
            default=str,
        ),
        file_name="pumpdesign_result.json",
        mime="application/json",
    )


@st.cache_resource(show_spinner=False)
def get_memory_db() -> MemoryDB:
    return MemoryDB(
        os.getenv(
            "PUMPDESIGN_MEMORY_DB",
            str(ROOT / "pumpdesign_memory.sqlite3"),
        )
    )


@st.cache_resource(show_spinner=False)
def get_project_memory() -> ProjectMemory:
    return ProjectMemory(get_memory_db())


@st.cache_resource(show_spinner=False)
def get_rag_store() -> RAGStore:
    return RAGStore(top_k=6)


@st.cache_resource(show_spinner=False)
def get_agent(
    _rag_store: RAGStore,
) -> PumpDesignAgent:
    return PumpDesignAgent(
        rag_retriever=_rag_store.retriever
    )


st.title("💧 PumpDesign AI")

st.caption(
    "AI-powered building water-supply pump design assistant"
)

st.markdown(
    "Design reasoning by AI • Engineering criteria from RAG • "
    "Numerical calculations by deterministic tools"
)


with st.sidebar:
    st.header("Design Assistant")

    st.write(
        "Choose a pump application and enter the project facts. "
        "Unknown values are left unknown — the system will not fabricate them."
    )

    st.divider()

    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key detected")
    else:
        st.warning("GROQ_API_KEY not detected")

    st.caption(
        "The AI agent requires a Groq API key. "
        "Deterministic engineering workflows remain testable without it."
    )


if "loaded_design_id" not in st.session_state:
    st.session_state["loaded_design_id"] = None

if "loaded_inputs" not in st.session_state:
    st.session_state["loaded_inputs"] = None

if "last_report_payload" not in st.session_state:
    st.session_state["last_report_payload"] = None


with st.sidebar:
    st.subheader("Saved Projects & Designs")

    db = get_memory_db()

    projects = db.list_projects()

    project_options = {
        "New Project": None
    }

    for p in projects:
        label = (
            f"{p['project_name']} — "
            f"{p['building_type']}"
        )

        project_options[label] = p["project_id"]

    selected_project_label = st.selectbox(
        "Project",
        list(project_options),
        key="saved_project_selector",
    )

    selected_project_id = project_options[
        selected_project_label
    ]

    if selected_project_id:

        designs = db.list_designs(
            selected_project_id
        )

        design_options = {
            "New Design": None
        }

        for d in designs:
            label = (
                f"{d['application']} — "
                f"Rev {d['current_revision']} — "
                f"{d['status']}"
            )

            design_options[label] = d["design_id"]

        selected_design_label = st.selectbox(
            "Saved Design",
            list(design_options),
            key="saved_design_selector",
        )

        selected_design_id = design_options[
            selected_design_label
        ]

        if selected_design_id and st.button(
            "Load Saved Design",
            use_container_width=True,
        ):
            saved = db.get_design(
                selected_design_id
            )

            if saved:
                st.session_state["loaded_design_id"] = (
                    selected_design_id
                )

                st.session_state["loaded_inputs"] = (
                    saved["inputs"]
                )

                st.session_state["loaded_application"] = (
                    saved["application"].upper()
                )

                st.session_state["loaded_result"] = (
                    saved.get("result")
                )

                st.rerun()

    if st.session_state.get("loaded_design_id"):
        st.caption(
            f"Loaded design: "
            f"{st.session_state['loaded_design_id']}"
        )


# ============================================================
# FIXED SAVED-DESIGN LOADING
# ============================================================

def seed_loaded_form_state() -> None:
    """Populate Streamlit widget state from the loaded structured design."""

    loaded = (
        st.session_state.get("loaded_inputs")
        or {}
    )

    if not loaded:
        return

    app = st.session_state.get(
        "loaded_application"
    )

    # --------------------------------------------------------
    # Restore application selector
    # --------------------------------------------------------

    app_labels = {
        "TRANSFER":
            "Transfer Pump — UGT → OHT",

        "BOOSTER":
            "Booster Pump",

        "SUBMERSIBLE":
            "Submersible Pump",

        "HOT_WATER_RECIRCULATION":
            "Hot-Water Recirculation Pump",
    }

    if app in app_labels:
        st.session_state["application"] = (
            app_labels[app]
        )

    # --------------------------------------------------------
    # Restore project fields
    # --------------------------------------------------------

    project_keys = [
        "project_name",
        "location",
        "jurisdiction",
        "building_type",
    ]

    for key in project_keys:

        if (
            key in loaded
            and loaded[key] is not None
        ):
            st.session_state[key] = loaded[key]

    # --------------------------------------------------------
    # Saved database field
    # -> actual Streamlit widget key
    # --------------------------------------------------------

    widget_map = {

        "TRANSFER": {

            "total_design_demand":
                "transfer_total_demand",

            "population":
                "transfer_population",

            "per_capita_demand":
                "transfer_percapita",

            "ugt_capacity":
                "transfer_ugt_capacity",

            "oht_capacity":
                "transfer_oht_capacity",

            "ugt_design_water_level":
                "transfer_ugt_level",

            "oht_design_water_level":
                "transfer_oht_level",

            "required_transfer_time":
                "transfer_time",

            "suction_pipe_length":
                "transfer_suction_length",

            "delivery_pipe_length":
                "transfer_delivery_length",

            "pipe_material":
                "transfer_material",

            "different_scenario":
                "transfer_custom",
        },

        "BOOSTER": {

            "total_design_demand":
                "booster_total_demand",

            "population":
                "booster_population",

            "per_capita_demand":
                "booster_percapita",

            "oht_design_water_level":
                "booster_oht_level",

            "available_inlet_pressure":
                "booster_inlet_pressure",

            "lowest_elevation_served":
                "booster_lowest",

            "highest_elevation_served":
                "booster_highest",

            "required_residual_pressure":
                "booster_residual",

            "critical_route_length":
                "booster_route",

            "pipe_material":
                "booster_material",

            "different_scenario":
                "booster_custom",

            "use_case":
                "booster_use_case",
        },

        "SUBMERSIBLE": {

            "total_design_demand":
                "submersible_total_demand",

            "population":
                "submersible_population",

            "per_capita_demand":
                "submersible_percapita",

            "static_water_level":
                "sub_static",

            "dynamic_water_level":
                "sub_dynamic",

            "well_yield":
                "sub_yield",

            "sump_water_level":
                "sub_sump_level",

            "sump_capacity":
                "sub_sump_capacity",

            "destination_water_level":
                "sub_destination_level",

            "known_design_flow":
                "sub_known_flow",

            "destination_capacity":
                "sub_destination_capacity",

            "required_filling_time":
                "sub_filling_time",

            "delivery_pipe_length":
                "sub_delivery_length",

            "pipe_material":
                "sub_material",

            "different_scenario":
                "sub_custom",

            "primary_use":
                "sub_primary_use",
        },

        "HOT_WATER_RECIRCULATION": {

            "hot_water_supply_temperature":
                "hw_supply_temp",

            "required_return_temperature":
                "hw_return_temp",

            "total_hot_water_circulation_pipe_length":
                "hw_length",

            "pipe_material":
                "hw_material",

            "different_scenario":
                "hw_custom",

            "system_arrangement":
                "hw_arrangement",
        },
    }

    current_map = widget_map.get(
        app,
        {},
    )

    # --------------------------------------------------------
    # Restore mapped values
    # --------------------------------------------------------

    optional_fields = {
        "available_inlet_pressure",
        "static_water_level",
        "dynamic_water_level",
        "well_yield",
        "sump_water_level",
        "sump_capacity",
        "destination_capacity",
    }

    for saved_key, widget_key in current_map.items():

        if saved_key not in loaded:
            continue

        value = loaded[saved_key]

        # Numeric + unit fields
        if (
            isinstance(value, dict)
            and "value" in value
        ):

            numeric_value = value.get(
                "value"
            )

            unit_value = value.get(
                "unit"
            )

            if numeric_value is not None:
                st.session_state[
                    f"v_{widget_key}"
                ] = numeric_value

            if unit_value is not None:
                st.session_state[
                    f"u_{widget_key}"
                ] = unit_value

            # Restore optional-field checkbox
            if saved_key in optional_fields:
                st.session_state[
                    f"known_{widget_key}"
                ] = True

        # Text / selectbox / normal values
        elif value is not None:
            st.session_state[
                widget_key
            ] = value

    # --------------------------------------------------------
    # Restore demand basis
    # --------------------------------------------------------

    demand_basis = loaded.get(
        "demand_basis"
    )

    if app == "TRANSFER":
        prefix = "transfer"

    elif app == "BOOSTER":
        prefix = "booster"

    elif app == "SUBMERSIBLE":
        prefix = "submersible"

    else:
        prefix = None

    if prefix and demand_basis:

        demand_labels = {
            "TOTAL_DESIGN_DEMAND":
                "Total Design Demand",

            "POPULATION_PER_CAPITA":
                "Calculate from Population",
        }

        if demand_basis in demand_labels:
            st.session_state[
                f"demand_basis_{prefix}"
            ] = demand_labels[demand_basis]

    # --------------------------------------------------------
    # Restore submersible flow basis
    # --------------------------------------------------------

    if app == "SUBMERSIBLE":

        if loaded.get(
            "known_design_flow"
        ) is not None:

            st.session_state[
                "sub_flow_basis"
            ] = "Known Design Flow"

        elif (
            loaded.get(
                "required_filling_time"
            ) is not None
            or loaded.get(
                "destination_capacity"
            ) is not None
        ):

            st.session_state[
                "sub_flow_basis"
            ] = (
                "Calculate from Required Filling Time"
            )


seed_loaded_form_state()


application = st.selectbox(
    "Pump Application",
    [
        "Transfer Pump — UGT → OHT",
        "Booster Pump",
        "Submersible Pump",
        "Hot-Water Recirculation Pump",
    ],
    key="application",
)


if application.startswith("Transfer"):

    application_code = "TRANSFER"
    data = transfer_ui()

elif application.startswith("Booster"):

    application_code = "BOOSTER"
    data = booster_ui()

elif application.startswith("Submersible"):

    application_code = "SUBMERSIBLE"
    data = submersible_ui()

else:

    application_code = (
        "HOT_WATER_RECIRCULATION"
    )

    data = hot_water_ui()


data = clean(data)


if st.session_state.get("loaded_inputs"):

    st.info(
        "A saved design is loaded. Current form values "
        "are shown for creating the next revision. "
        "Edit any input, then run the design."
    )

    with st.expander(
        "Loaded design inputs",
        expanded=False,
    ):
        st.json(
            st.session_state["loaded_inputs"]
        )


st.divider()

run = st.button(
    "🚀 Run Pump Design",
    type="primary",
    use_container_width=True,
)


if run:

    if not os.getenv("GROQ_API_KEY"):

        st.error(
            "GROQ_API_KEY is required to run the AI agent. "
            "Add it to your environment or Streamlit secrets and run again."
        )

    else:

        request = (
            f"Design a {application_code} pump system "
            "from the structured project inputs below. "
            "First retrieve the applicable engineering criteria "
            "from RAG, then call exactly the relevant deterministic "
            "workflow. Do not invent missing values. "
            "Explain the deterministic result, warnings, "
            "unresolved inputs, and references. "
            "Do not claim a commercial pump selection unless "
            "manufacturer data exists."
        )

        context = {
            "application": application_code,
            "inputs": data,
        }

        try:

            with st.spinner(
                "PumpDesign AI is reviewing criteria and "
                "running the engineering workflow..."
            ):

                agent = get_agent(
                    get_rag_store()
                )

                answer = agent.run(
                    request,
                    context=context,
                )

            st.session_state[
                "last_answer"
            ] = answer

            st.session_state[
                "last_agent_state"
            ] = agent.state

            # ------------------------------------------------
            # Persist structured project/design data only
            # ------------------------------------------------

            try:

                memory = get_project_memory()

                tool_results = (
                    agent.state.last_tool_results
                    or []
                )

                deterministic = next(
                    (
                        x["result"]
                        for x in reversed(
                            tool_results
                        )
                        if (
                            isinstance(
                                x.get("result"),
                                dict,
                            )
                            and x["result"].get(
                                "calculations"
                            )
                            is not None
                        )
                    ),
                    None,
                )

                if deterministic is None:

                    deterministic = {
                        "status":
                            "NOT_CALCULABLE",

                        "message":
                            "No deterministic calculation "
                            "result was returned by the agent.",
                    }

                result_payload = deterministic

                old_design_id = (
                    st.session_state.get(
                        "loaded_design_id"
                    )
                )

                old_inputs = (
                    st.session_state.get(
                        "loaded_inputs"
                    )
                )

                # --------------------------------------------
                # Existing design -> create revision
                # --------------------------------------------

                if old_design_id and old_inputs:

                    rev = create_design_revision(
                        get_memory_db(),
                        old_design_id,
                        "Engineer-created design revision",
                        old_inputs,
                        data,
                        result_payload,
                    )

                    st.session_state[
                        "revision_number"
                    ] = rev

                    existing_design = (
                        get_memory_db().get_design(
                            old_design_id
                        )
                    )

                    if existing_design:

                        memory.db.save_design(
                            {
                                "design_id":
                                    old_design_id,

                                "project_id":
                                    existing_design[
                                        "project_id"
                                    ],

                                "application":
                                    application_code,

                                "scenario":
                                    data.get(
                                        "different_scenario"
                                    )
                                    or application_code,

                                "inputs":
                                    data,

                                "current_revision":
                                    rev,
                            },
                            result=result_payload,
                            status="AI_COMPLETED",
                        )

                    st.session_state[
                        "loaded_inputs"
                    ] = data

                    st.session_state[
                        "loaded_application"
                    ] = application_code

                    st.success(
                        f"Revision {rev} saved."
                    )

                # --------------------------------------------
                # New design -> create project + design
                # --------------------------------------------

                else:

                    pid, did = (
                        memory.save_project_and_design(
                            data,
                            application_code,
                            result_payload,
                        )
                    )

                    st.session_state[
                        "loaded_design_id"
                    ] = did

                    st.session_state[
                        "loaded_inputs"
                    ] = data

                    st.session_state[
                        "loaded_application"
                    ] = application_code

                    st.success(
                        "Project and design saved for future reuse."
                    )

                # --------------------------------------------
                # Build report
                # --------------------------------------------

                design_id_for_report = (
                    st.session_state.get(
                        "loaded_design_id"
                    )
                    or did
                )

                st.session_state[
                    "last_report_payload"
                ] = build_report_payload(
                    design_id=design_id_for_report,
                    application=application_code,
                    inputs=data,
                    result=result_payload,
                    rag_context=agent.state.rag_context,
                    project={
                        k: data.get(k)
                        for k in [
                            "project_name",
                            "location",
                            "jurisdiction",
                            "building_type",
                        ]
                        if data.get(k) is not None
                    },
                )

            except Exception as memory_exc:

                st.warning(
                    "Design completed, but memory could not be saved: "
                    f"{memory_exc}"
                )

        except Exception as exc:

            st.error(
                f"Design run failed: {exc}"
            )


if st.session_state.get("last_answer"):

    st.divider()

    st.subheader(
        "AI Engineering Explanation"
    )

    st.markdown(
        st.session_state["last_answer"]
    )

    report_payload = (
        st.session_state.get(
            "last_report_payload"
        )
    )

    if report_payload:

        st.subheader(
            "Engineering Calculation Report"
        )

        st.caption(
            "The report presents deterministic calculation "
            "results, formulas, inputs, assumptions, validation "
            "status, and retrieved references. Reports are "
            "generated for download and are not stored as "
            "project memory."
        )

        md_report = render_markdown(
            report_payload
        )

        html_report = render_html(
            report_payload
        )

        pdf_path = (
            ROOT
            / "reports"
            / f"{report_payload['report_id']}.pdf"
        )

        try:

            generate_pdf(
                report_payload,
                pdf_path,
            )

            pdf_bytes = pdf_path.read_bytes()

            st.download_button(
                "Download Engineering Calculation Report (PDF)",
                data=pdf_bytes,
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception as report_exc:

            st.warning(
                "PDF report could not be generated: "
                f"{report_exc}"
            )

        st.download_button(
            "Download Calculation Report (Markdown)",
            data=md_report,
            file_name=(
                f"{report_payload['report_id']}.md"
            ),
            mime="text/markdown",
        )

        st.download_button(
            "Download Calculation Report (HTML)",
            data=html_report,
            file_name=(
                f"{report_payload['report_id']}.html"
            ),
            mime="text/html",
        )

    # --------------------------------------------------------
    # Commercial pump validation
    # --------------------------------------------------------

    last_deterministic = None

    state_for_pump = (
        st.session_state.get(
            "last_agent_state"
        )
    )

    if state_for_pump:

        tool_results_for_pump = (
            state_for_pump.last_tool_results
            or []
        )

        last_deterministic = next(
            (
                x["result"]
                for x in reversed(
                    tool_results_for_pump
                )
                if (
                    isinstance(
                        x.get("result"),
                        dict,
                    )
                    and x["result"].get(
                        "pump_duty"
                    )
                    is not None
                )
            ),
            None,
        )

    commercial_pump_ui(
        last_deterministic
    )

    # --------------------------------------------------------
    # Revision history
    # --------------------------------------------------------

    loaded_id = (
        st.session_state.get(
            "loaded_design_id"
        )
    )

    if loaded_id:

        revisions = (
            get_memory_db().list_revisions(
                loaded_id
            )
        )

        if revisions:

            with st.expander(
                "Revision History"
            ):

                for rev in revisions:

                    st.write(
                        f"Revision {rev['revision_number']} — "
                        f"{rev['reason']} — "
                        f"{rev['created_at']}"
                    )

                    st.json(
                        rev["changed_parameters"]
                    )

    # --------------------------------------------------------
    # RAG context
    # --------------------------------------------------------

    state = (
        st.session_state.get(
            "last_agent_state"
        )
    )

    if state and state.rag_context:

        with st.expander(
            "Engineering Criteria Retrieved from RAG"
        ):

            st.json(
                state.rag_context
            )
