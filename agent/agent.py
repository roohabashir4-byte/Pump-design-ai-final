"""Groq-backed PumpDesign AI agent with RAG and deterministic tool orchestration."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from models.engineering_contract import Criterion
from standards.retrieval import build_criteria_packet
from agent.engineering_tools import build_engineering_router
from agent.prompts import SYSTEM_PROMPT, RAG_TOOL_DESCRIPTION


APPLICATIONS = {
    "transfer": "TRANSFER",
    "transfer_pump": "TRANSFER",
    "booster": "BOOSTER",
    "booster_pump": "BOOSTER",
    "submersible": "SUBMERSIBLE",
    "submersible_pump": "SUBMERSIBLE",
    "hot_water_recirc": "HOT_WATER_RECIRCULATION",
    "hot_water_recirculation": "HOT_WATER_RECIRCULATION",
}


@dataclass
class AgentState:
    rag_context: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_tool_results: list[dict[str, Any]] = field(default_factory=list)


class PumpDesignAgent:
    """Orchestrates LLM reasoning, RAG retrieval and deterministic workflows.

    The LLM handles engineering reasoning and orchestration.

    RAG supplies engineering criteria.

    Deterministic engineering workflows remain the numerical authority.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        rag_retriever: Callable[
            ...,
            list[Criterion | dict[str, Any]]
        ]
        | None = None,
        max_tool_rounds: int = 4,
        client: Any | None = None,
    ):
        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        self.rag_retriever = rag_retriever

        self.max_tool_rounds = max_tool_rounds

        self._state = AgentState()

        self._requested_application: str | None = None
        self._current_inputs: dict[str, Any] = {}

        self._client = client

        self.router = build_engineering_router(
            lambda: self._state.rag_context
        )

    @property
    def state(self) -> AgentState:
        return self._state

    def _get_client(self):

        if self._client is not None:
            return self._client

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required to run the LLM agent."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'groq' package before running the LLM agent."
            ) from exc

        self._client = Groq(
            api_key=self.api_key
        )

        return self._client

    def _rag_context(
        self,
        application: str,
        service: str | None,
        jurisdiction: str | None,
        query: str | None = None,
        material: str | None = None,
    ) -> dict[str, Any]:

        if self.rag_retriever is None:
            raise RuntimeError(
                "No RAG retriever is configured."
            )

        retrieval_query = (
            query
            or application
        )

        if material:
            retrieval_query = (
                f"{retrieval_query} "
                f"{material} "
                "Hazen Williams pipe material"
            )

        raw = self.rag_retriever(
            application=application,
            service=service,
            jurisdiction=jurisdiction,
            query=retrieval_query,
        )

        criteria: list[Criterion] = []

        for item in raw:

            criteria.append(
                item
                if isinstance(
                    item,
                    Criterion,
                )
                else Criterion.model_validate(
                    item
                )
            )

        packet = build_criteria_packet(
            criteria,
            application=application,
            service=service,
            jurisdiction=jurisdiction,
        )

        velocity = [
            c
            for c in packet
            if c.criterion_type.value
            == "VELOCITY"
        ]

        hw = [
            c
            for c in packet
            if (
                c.criterion_type.value
                == "MATERIAL"
                and c.parameter.lower()
                in {
                    "hazen_williams_c",
                    "hazen_williams_c_value",
                }
            )
        ]

        # ----------------------------------------------------
        # Resolve Hazen-Williams C by material
        # ----------------------------------------------------

        if material:

            mat = material.lower()

            aliases = [mat]

            if (
                "steel" in mat
                or mat in {
                    "ms",
                    "carbon steel",
                    "ms/carbon steel",
                }
            ):
                aliases.extend(
                    [
                        "steel",
                        "steel pipe",
                    ]
                )

            if "pvc" in mat:
                aliases.extend(
                    [
                        "pvc",
                        "plastic pipe",
                    ]
                )

            if (
                "hdpe" in mat
                or "pe" in mat
            ):
                aliases.extend(
                    [
                        "pe / polyethylene",
                        "plastic pipe",
                        "plastic",
                    ]
                )

            if "ppr" in mat:
                aliases.extend(
                    [
                        "ppr",
                        "pp-r",
                        "polypropylene",
                        "plastic pipe",
                    ]
                )

            if "copper" in mat:
                aliases.extend(
                    [
                        "copper",
                        "copper tubing",
                    ]
                )

            hw = [
                c
                for c in hw
                if any(
                    alias
                    in c.applicability.lower()
                    for alias in aliases
                )
            ]

        # ----------------------------------------------------
        # Resolve conflicting Hazen-Williams records
        # ----------------------------------------------------

        if len(hw) > 1:

            primary = [
                c
                for c in hw
                if (
                    "PRIMARY"
                    in c.source_status.upper()
                )
            ]

            new_pipe = [
                c
                for c in primary
                if (
                    "new"
                    in c.applicability.lower()
                )
            ]

            if len(new_pipe) == 1:

                hw = new_pipe

            elif len(primary) == 1:

                hw = primary

            else:

                raise ValueError(
                    "Multiple Hazen-Williams C criteria "
                    "matched; source precedence/pipe "
                    "condition must be resolved first."
                )

        # ----------------------------------------------------
        # Resolve velocity criteria
        # ----------------------------------------------------

        velocity_context: dict[str, float] = {}

        for c in velocity:

            key = c.parameter.lower()

            if (
                key
                in {
                    "min_velocity",
                    "minimum_velocity",
                    "velocity_min",
                }
                and c.value is not None
            ):

                if (
                    "min_mps"
                    in velocity_context
                    and velocity_context[
                        "min_mps"
                    ]
                    != c.value
                ):
                    raise ValueError(
                        "Conflicting minimum velocity "
                        "criteria matched."
                    )

                velocity_context[
                    "min_mps"
                ] = c.value

            elif (
                key
                in {
                    "max_velocity",
                    "maximum_velocity",
                    "velocity_max",
                }
                and c.value is not None
            ):

                if (
                    "max_mps"
                    in velocity_context
                    and velocity_context[
                        "max_mps"
                    ]
                    != c.value
                ):
                    raise ValueError(
                        "Conflicting maximum velocity "
                        "criteria matched."
                    )

                velocity_context[
                    "max_mps"
                ] = c.value

        # Send only the engineering facts actually needed by the LLM.
        # The complete RAG store remains available internally; it is never
        # copied into the model context.
        selected = []

        # Hydraulic method / core calculation criteria.
        priority_parameters = {
            "hydraulic_method",
            "hydraulic_method_selection",
            "hydraulic_diameter_input",
            "pipe_sizing_method",
            "minimum_pressure",
            "minimum_available_pressure",
        }

        for c in packet:
            if c in velocity or c in hw:
                selected.append(c)
                continue
            if c.parameter.lower() in priority_parameters:
                selected.append(c)

        # Keep the LLM context deliberately small.
        selected = selected[:6]

        self._state.rag_context = {
            "application": application,
            "service": service,
            "jurisdiction": jurisdiction,
            "criteria": [
                {
                    "criterion_id": c.criterion_id,
                    "parameter": c.parameter,
                    "value": c.value,
                    "unit": c.unit,
                    "applicability": c.applicability,
                    "source_reference_id": c.source_reference_id,
                    "source_section": c.source_section,
                    "source_page": c.source_page,
                }
                for c in selected
            ],
            "references": sorted(
                {c.source_reference_id for c in selected}
            ),
            "velocity_criteria": (
                velocity_context
                if velocity_context
                else None
            ),
            "hazen_williams_c": (
                hw[0].value
                if hw
                else None
            ),
        }

        return self._state.rag_context

    def _rag_tool(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:

        app_raw = str(
            args.get("application")
            or self._requested_application
            or ""
        ).lower().strip()

        application = APPLICATIONS.get(
            app_raw,
            app_raw.upper(),
        )

        if application not in {
            "TRANSFER",
            "BOOSTER",
            "SUBMERSIBLE",
            "HOT_WATER_RECIRCULATION",
        }:
            raise ValueError(
                f"Unsupported pump application: "
                f"{application}"
            )

        service = (
            args.get("service")
            or self._current_inputs.get("service")
        )
        jurisdiction = (
            args.get("jurisdiction")
            or self._current_inputs.get("jurisdiction")
        )
        material = (
            args.get("material")
            or self._current_inputs.get("pipe_material")
        )
        query = args.get("query") or application

        return self._rag_context(
            application,
            service,
            jurisdiction,
            query,
            material,
        )

    def _tool_schemas(
        self,
        *,
        include_rag: bool = True,
        include_workflows: bool = True,
    ) -> list[dict[str, Any]]:

        schemas = []

        if include_rag:
            schemas.append(
            {
                "type": "function",
                "function": {
                    "name":
                        "get_engineering_criteria",

                    "description":
                        RAG_TOOL_DESCRIPTION,

                    "parameters": {
                        "type": "object",

                        "properties": {
                            "application": {
                                "type": "string"
                            },

                            "service": {
                                "type": "string"
                            },

                            "jurisdiction": {
                                "type": "string"
                            },

                            "query": {
                                "type": "string"
                            },

                            "material": {
                                "type": "string"
                            },
                        },

                        "required": [],
                    },
                },
            }
            )

        if include_workflows:
            for name in self.router.available_tools():

                tool = self.router.get(name)

                schemas.append(
                    {
                        "type": "function",

                        "function": {
                            "name": name,

                            "description":
                                tool.description,

                            "parameters": (
                                tool.input_schema
                                or {
                                    "type": "object",
                                    "additionalProperties": True,
                                }
                            ),
                        },
                    }
                )

        return schemas

    def _execute_tool_call(
        self,
        name: str,
        args: dict[str, Any],
    ) -> Any:

        if name == "get_engineering_criteria":

            return self._rag_tool(args)

        if (
            name
            not in self.router.available_tools()
        ):
            raise LookupError(
                f"Unknown agent tool: {name}"
            )

        if not self._state.rag_context:

            raise RuntimeError(
                "Engineering criteria must be retrieved "
                "before running a design workflow."
            )

        from models.engineering_contract import (
            CalculationRequest,
        )

        workflow_inputs = (
            dict(self._current_inputs)
            if self._current_inputs
            else dict(args)
        )

        req = CalculationRequest(
            request_id=str(
                uuid.uuid4()
            ),

            application=name,

            tool_name=name,

            inputs=workflow_inputs,

            purpose=(
                "Agent-orchestrated "
                "pump design calculation"
            ),
        )

        result = self.router.execute(
            req
        )

        return result.model_dump(
            mode="json"
        )

    def run(
        self,
        user_request: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Run exactly one RAG call, one deterministic workflow call, then one final LLM call."""

        client = self._get_client()
        self._state = AgentState()
        self._requested_application = None
        self._current_inputs = {}

        if context:
            self._requested_application = str(
                context.get("application") or ""
            ).strip().upper()
            self._current_inputs = dict(
                context.get("inputs") or {}
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if context:
            messages.append({
                "role": "system",
                "content": (
                    "Structured project context supplied by the user/app. "
                    "Use these inputs exactly; do not invent missing values:\n"
                    + json.dumps(context, default=str)
                ),
            })

        messages.append({"role": "user", "content": user_request})

        # ------------------------------------------------------------
        # STAGE 1: RAG only.
        # ------------------------------------------------------------
        rag_tools = self._tool_schemas(
            include_rag=True,
            include_workflows=False,
        )

        rag_response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            tools=rag_tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_engineering_criteria"},
            },
            temperature=0,
            parallel_tool_calls=False,
        )

        rag_message = rag_response.choices[0].message
        rag_calls = getattr(rag_message, "tool_calls", None) or []
        if not rag_calls:
            raise RuntimeError(
                "The AI did not request engineering criteria from RAG."
            )

        rag_call = rag_calls[0]
        if rag_call.function.name != "get_engineering_criteria":
            raise RuntimeError(
                f"Unexpected RAG tool call: {rag_call.function.name}"
            )
        if len(rag_calls) != 1:
            raise RuntimeError(
                "RAG stage returned multiple tool calls; exactly one is allowed."
            )
        rag_args = json.loads(rag_call.function.arguments or "{}")
        rag_result = self._execute_tool_call(
            rag_call.function.name,
            rag_args,
        )

        messages.append({
            "role": "assistant",
            "content": rag_message.content or "",
            "tool_calls": [{
                "id": rag_call.id,
                "type": "function",
                "function": {
                    "name": rag_call.function.name,
                    "arguments": rag_call.function.arguments,
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": rag_call.id,
            "content": json.dumps(rag_result, default=str),
        })

        # ------------------------------------------------------------
        # STAGE 2: only the workflow relevant to the selected
        # application is exposed. This prevents tool loops and keeps
        # the request small.
        # ------------------------------------------------------------
        workflow_names = {
            "TRANSFER": "run_transfer_design",
            "BOOSTER": "run_booster_design",
            "SUBMERSIBLE": "run_submersible_design",
            "HOT_WATER_RECIRCULATION": "run_hot_water_recirc_design",
        }
        workflow_name = workflow_names.get(
            self._requested_application or ""
        )
        if not workflow_name:
            raise RuntimeError(
                "A valid pump application is required before running the design workflow."
            )

        all_workflow_tools = self._tool_schemas(
            include_rag=False,
            include_workflows=True,
        )
        workflow_tools = [
            tool for tool in all_workflow_tools
            if tool["function"]["name"] == workflow_name
        ]

        workflow_response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            tools=workflow_tools,
            tool_choice={
                "type": "function",
                "function": {"name": workflow_name},
            },
            temperature=0,
            parallel_tool_calls=False,
        )

        workflow_message = workflow_response.choices[0].message
        workflow_calls = getattr(workflow_message, "tool_calls", None) or []
        if not workflow_calls:
            raise RuntimeError(
                f"The AI did not execute the required {workflow_name} workflow."
            )

        workflow_call = workflow_calls[0]
        if workflow_call.function.name != workflow_name:
            raise RuntimeError(
                f"Unexpected workflow tool call: {workflow_call.function.name}"
            )
        if len(workflow_calls) != 1:
            raise RuntimeError(
                "Workflow stage returned multiple tool calls; exactly one is allowed."
            )
        workflow_args = json.loads(
            workflow_call.function.arguments or "{}"
        )
        workflow_result = self._execute_tool_call(
            workflow_call.function.name,
            workflow_args,
        )

        self._state.last_tool_results.append({
            "tool": workflow_call.function.name,
            "result": workflow_result,
        })

        messages.append({
            "role": "assistant",
            "content": workflow_message.content or "",
            "tool_calls": [{
                "id": workflow_call.id,
                "type": "function",
                "function": {
                    "name": workflow_call.function.name,
                    "arguments": workflow_call.function.arguments,
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": workflow_call.id,
            "content": json.dumps(workflow_result, default=str),
        })

        # ------------------------------------------------------------
        # STAGE 3: final explanation. No tools are exposed, so the
        # model cannot start another tool-call round.
        # ------------------------------------------------------------
        final_response = client.chat.completions.create(
            model=self.model,
            messages=messages + [{
                "role": "system",
                "content": (
                    "Provide the final engineering explanation now. "
                    "Use only the deterministic workflow result and retrieved "
                    "criteria above. Clearly state calculated results, missing "
                    "inputs, warnings, unresolved items, and references. "
                    "Do not invent engineering values or manufacturer data."
                ),
            }],
            max_tokens=800,
            tools=[],
            tool_choice="none",
            temperature=0,
            parallel_tool_calls=False,
        )

        final_message = final_response.choices[0].message
        final = final_message.content or (
            "The deterministic engineering workflow completed, but the AI explanation was empty."
        )

        self._state.messages = messages + [{
            "role": "assistant",
            "content": final,
        }]
        return final
