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

        self._state.rag_context = {
            "criteria": [
                c.model_dump(
                    mode="json"
                )
                for c in packet
            ],

            "references": sorted(
                {
                    c.source_reference_id
                    for c in packet
                }
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

        return self._rag_context(
            application,
            args.get("service") or self._current_inputs.get("service"),
            args.get("jurisdiction") or self._current_inputs.get("jurisdiction"),
            args.get("query") or application,
            args.get("material") or self._current_inputs.get("pipe_material"),
        )

    def _tool_schemas(
        self,
        *,
        include_rag: bool = True,
        workflow_name: str | None = None,
    ) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []

        if include_rag:
            schemas.append({
                "type": "function",
                "function": {
                    "name": "get_engineering_criteria",
                    "description": RAG_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "application": {"type": "string"},
                            "service": {"type": "string"},
                            "jurisdiction": {"type": "string"},
                            "query": {"type": "string"},
                            "material": {"type": "string"},
                        },
                        "required": [],
                    },
                },
            })

        if workflow_name:
            tool = self.router.get(workflow_name)
            schemas.append({
                "type": "function",
                "function": {
                    "name": workflow_name,
                    "description": tool.description,
                    "parameters": tool.input_schema or {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
            })

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

        req = CalculationRequest(
            request_id=str(
                uuid.uuid4()
            ),

            application=name,

            tool_name=name,

            inputs=(dict(self._current_inputs) if self._current_inputs else args),

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
        """Run deterministic RAG -> deterministic workflow -> final LLM explanation.

        RAG retrieval and numerical workflow execution are controlled by the
        application. The LLM is used only for the final engineering
        explanation, so tool-calling failures cannot block the calculation.
        """
        client = self._get_client()
        self._state = AgentState()

        context = context or {}
        self._requested_application = str(
            context.get("application") or ""
        ).strip().upper()
        self._current_inputs = dict(
            context.get("inputs") or {}
        )

        workflow_map = {
            "TRANSFER": "run_transfer_design",
            "BOOSTER": "run_booster_design",
            "SUBMERSIBLE": "run_submersible_design",
            "HOT_WATER_RECIRCULATION": "run_hot_water_recirculation_design",
        }
        workflow_name = workflow_map.get(
            self._requested_application
        )
        if not workflow_name:
            raise ValueError(
                f"Unsupported pump application: "
                f"{self._requested_application}"
            )

        # --------------------------------------------------------
        # STAGE 1: RAG is executed directly by Python.
        # The LLM is NOT asked to call the RAG tool.
        # --------------------------------------------------------
        rag_result = self._rag_context(
            self._requested_application,
            self._current_inputs.get("service"),
            self._current_inputs.get("jurisdiction"),
            self._requested_application,
            self._current_inputs.get("pipe_material"),
        )

        # --------------------------------------------------------
        # STAGE 2: deterministic engineering workflow.
        # The LLM is NOT asked to call the calculation tool either.
        # The complete structured inputs remain inside Python.
        # --------------------------------------------------------
        workflow_result = self._execute_tool_call(
            workflow_name,
            {},
        )
        self._state.last_tool_results.append(
            {
                "tool": workflow_name,
                "result": workflow_result,
            }
        )

        # --------------------------------------------------------
        # STAGE 3: one small LLM call for explanation only.
        # Send only the selected criteria and deterministic result.
        # --------------------------------------------------------
        compact_rag = {
            "velocity_criteria": rag_result.get(
                "velocity_criteria"
            ),
            "hazen_williams_c": rag_result.get(
                "hazen_williams_c"
            ),
            "references": rag_result.get(
                "references", []
            ),
            "criteria": rag_result.get(
                "criteria", []
            ),
        }

        final_payload = {
            "application": self._requested_application,
            "user_request": user_request,
            "engineering_criteria": compact_rag,
            "deterministic_design_result": workflow_result,
        }

        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n\n"
                    + "You are now in the final explanation stage. "
                    + "The engineering calculation has already been executed "
                    + "by deterministic Python tools. Do not calculate, "
                    + "change, or invent numerical values. Explain the supplied "
                    + "result, show important formulas/inputs when present, "
                    + "identify missing inputs or warnings, and cite the "
                    + "provided engineering references."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    final_payload,
                    default=str,
                ),
            },
        ]

        self._state.messages = messages

        final_response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=800,
            tools=[],
            tool_choice="none",
            temperature=0,
        )

        final_message = final_response.choices[0].message
        final = final_message.content

        if not final:
            final = (
                "The deterministic engineering calculation completed, "
                "but the AI explanation was empty."
            )

        self._state.messages = messages + [
            {
                "role": "assistant",
                "content": final,
            }
        ]

        return final
