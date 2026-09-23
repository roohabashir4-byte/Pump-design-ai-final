from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from models.engineering_contract import Criterion, CriterionType


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "standards"
INDEX = DATA / "index"


def _norm(s: str) -> str:
    return re.sub(
        r"[^a-z0-9_]+",
        " ",
        str(s).lower()
    ).strip()


def _criterion_type(
    name: str,
    parameter: str,
    topic: str = "",
) -> CriterionType:

    n = f"{name} {parameter} {topic}".lower()

    if (
        "hazen" in n
        or "hazen_williams_c" in n
        or parameter.lower() == "hazen_williams_c"
    ):
        return CriterionType.MATERIAL

    if "velocity" in n:
        return CriterionType.VELOCITY

    if "pressure" in n or "prv" in n:
        return CriterionType.PRESSURE

    if (
        "friction" in n
        or "head loss" in n
        or "headloss" in n
        or "darcy" in n
    ):
        return CriterionType.HEAD_LOSS

    if "demand" in n or "flow" in n:
        return CriterionType.DEMAND

    if (
        "hot" in n
        or "recirculation" in n
        or "temperature" in n
    ):
        return CriterionType.HOT_WATER

    if "npsh" in n:
        return CriterionType.NPSH

    if "method" in n or "hydraulic" in n:
        return CriterionType.HYDRAULIC_METHOD

    if "pump" in n:
        return CriterionType.PUMP

    if "valve" in n or "cutoff" in n:
        return CriterionType.CONTROL

    return CriterionType.OTHER


def _load_json(path: Path, default):
    if not path.exists():
        return default

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


class RAGStore:

    def __init__(
        self,
        *,
        top_k: int = 8,
    ):

        self.top_k = top_k

        self.criteria = self._load_criteria()

        self.chunks = self._load_chunks()

        self.sources = {
            x["source_id"]: x
            for x in _load_json(
                DATA / "sources" / "source_registry.json",
                []
            )
        }

        self.vectorizer = None
        self.matrix = None
        self.metadata = []

        self._load_or_build_index()

    # ============================================================
    # LOAD CRITERIA
    # ============================================================

    def _load_criteria(self):

        rows = _load_json(
            DATA
            / "criteria"
            / "criteria_registry_v2.json",
            []
        )

        out = []

        for r in rows:

            try:

                val = r.get("value")

                rng = (
                    val
                    if (
                        isinstance(val, list)
                        and len(val) == 2
                        and all(
                            isinstance(
                                x,
                                (int, float)
                            )
                            for x in val
                        )
                    )
                    else None
                )

                out.append(
                    Criterion(
                        criterion_id=str(
                            r["criterion_id"]
                        ),

                        criterion_type=_criterion_type(
                            str(
                                r.get(
                                    "name",
                                    ""
                                )
                            ),
                            str(
                                r.get(
                                    "parameter",
                                    r.get(
                                        "name",
                                        ""
                                    )
                                )
                            ),
                            str(
                                r.get(
                                    "topic",
                                    ""
                                )
                            ),
                        ),

                        parameter=str(
                            r.get(
                                "parameter"
                            )
                            or r.get(
                                "name"
                            )
                            or r["criterion_id"]
                        ),

                        value=(
                            val
                            if isinstance(
                                val,
                                (int, float)
                            )
                            else None
                        ),

                        min_value=(
                            rng[0]
                            if rng
                            else r.get(
                                "min_value"
                            )
                        ),

                        max_value=(
                            rng[1]
                            if rng
                            else r.get(
                                "max_value"
                            )
                        ),

                        unit=r.get(
                            "unit"
                        ),

                        application=r.get(
                            "application"
                        ),

                        service=r.get(
                            "service"
                        ),

                        jurisdiction=(
                            r.get(
                                "jurisdiction"
                            )
                            or r.get(
                                "source_location"
                            )
                        ),

                        applicability=str(
                            r.get(
                                "applicability"
                            )
                            or r.get(
                                "scope"
                            )
                            or r.get(
                                "applicability_context"
                            )
                            or "source-defined"
                        ),

                        method=r.get(
                            "method"
                        ),

                        source_reference_id=str(
                            r.get(
                                "source_id"
                            )
                            or "UNSPECIFIED"
                        ),

                        source_status=str(
                            r.get(
                                "source_status"
                            )
                            or r.get(
                                "status"
                            )
                            or "UNVERIFIED"
                        ),

                        edition_year=(
                            str(
                                r.get(
                                    "edition_year"
                                )
                            )
                            if r.get(
                                "edition_year"
                            )
                            else None
                        ),

                        section=(
                            r.get(
                                "section"
                            )
                            or r.get(
                                "reference"
                            )
                        ),

                        page=r.get(
                            "page"
                        ),

                        notes=str(
                            r.get(
                                "notes"
                            )
                            or ""
                        ),

                        system=r.get(
                            "system"
                        ),

                        subsystem=r.get(
                            "subsystem"
                        ),

                        flow_type=r.get(
                            "flow_type"
                        ),

                        topic=r.get(
                            "topic"
                        ),

                        source_location=r.get(
                            "source_location"
                        ),

                        applicability_context=r.get(
                            "applicability_context"
                        ),

                        retrieval_tags=r.get(
                            "retrieval_tags",
                            []
                        ),
                    )
                )

            except Exception:
                continue

        return out

    # ============================================================
    # LOAD CHUNKS
    # ============================================================

    def _load_chunks(self):

        path = (
            DATA
            / "corpus"
            / "chunks_v2.jsonl"
        )

        if not path.exists():
            return []

        return [
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    # ============================================================
    # INDEX
    # ============================================================

    def _load_or_build_index(self):

        pkl = (
            INDEX
            / "tfidf_vectorizer.pkl"
        )

        npz = (
            INDEX
            / "tfidf_matrix.npz"
        )

        meta = (
            INDEX
            / "metadata.jsonl"
        )

        if (
            pkl.exists()
            and npz.exists()
            and meta.exists()
        ):

            from scipy.sparse import load_npz

            self.vectorizer = pickle.loads(
                pkl.read_bytes()
            )

            self.matrix = load_npz(
                npz
            ).tocsr()

            self.metadata = [
                json.loads(line)
                for line in meta.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            return

        self._build_index()

    def _build_index(self):

        from sklearn.feature_extraction.text import (
            TfidfVectorizer
        )

        from scipy.sparse import save_npz

        records = []

        # --------------------------------------------------------
        # Corpus chunks
        # --------------------------------------------------------

        for c in self.chunks:

            text = (
                " ".join(
                    str(
                        c.get(k)
                        or ""
                    )
                    for k in [
                        "heading",
                        "text",
                        "topic",
                        "system",
                        "subsystem",
                        "flow_type",
                        "application",
                        "source_location",
                        "applicability_context",
                    ]
                )
                + " "
                + " ".join(
                    c.get(
                        "retrieval_tags",
                        []
                    )
                )
            )

            records.append(
                {
                    "record_id": c["chunk_id"],
                    "record_type": "chunk",
                    "text": text,
                    "metadata": c,
                }
            )

        # --------------------------------------------------------
        # Engineering criteria
        # --------------------------------------------------------

        for c in self.criteria:

            dumped = c.model_dump(
                mode="json"
            )

            text = (
                " ".join(
                    str(
                        dumped.get(k)
                        or ""
                    )
                    for k in [
                        "criterion_id",
                        "parameter",
                        "application",
                        "jurisdiction",
                        "applicability",
                        "notes",
                        "system",
                        "subsystem",
                        "flow_type",
                        "topic",
                        "source_location",
                    ]
                )
                + " "
                + " ".join(
                    c.retrieval_tags
                )
            )

            records.append(
                {
                    "record_id": c.criterion_id,
                    "record_type": "criterion",
                    "text": text,
                    "metadata": dumped,
                }
            )

        self.metadata = records

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            norm="l2",
        )

        self.matrix = (
            self.vectorizer
            .fit_transform(
                [
                    r["text"]
                    for r in records
                ]
            )
            .tocsr()
        )

        INDEX.mkdir(
            parents=True,
            exist_ok=True
        )

        (
            INDEX
            / "tfidf_vectorizer.pkl"
        ).write_bytes(
            pickle.dumps(
                self.vectorizer
            )
        )

        save_npz(
            INDEX / "tfidf_matrix.npz",
            self.matrix
        )

        (
            INDEX
            / "metadata.jsonl"
        ).write_text(
            "\n".join(
                json.dumps(
                    x,
                    ensure_ascii=False
                )
                for x in records
            )
            + "\n",
            encoding="utf-8",
        )

    # ============================================================
    # TOPIC DETECTION
    # ============================================================

    def _topic_candidates(
        self,
        query: str,
    ):

        q = _norm(query)

        mapping = {

            "PIPE_VELOCITY": [
                "velocity",
                "velocities",
                "speed",
            ],

            "PIPE_FRICTION": [
                "hazen",
                "friction",
                "head loss",
                "headloss",
                "darcy",
            ],

            "PIPE_SIZING": [
                "pipe size",
                "diameter",
                "sizing",
                "size selection",
            ],

            "PRESSURE": [
                "pressure",
                "residual",
                "terminal",
                "prv",
            ],

            "DEMAND_FLOW": [
                "flow",
                "demand",
                "peak",
                "transfer time",
            ],

            "STORAGE": [
                "tank",
                "oht",
                "ugt",
                "storage",
                "reservoir",
            ],

            "PUMP_HEAD": [
                "pump head",
                "tdh",
                "total dynamic head",
            ],

            "PUMP_PROTECTION": [
                "booster",
                "cutoff",
                "relief",
            ],

            "SOURCE_CAPACITY": [
                "well yield",
                "yield",
                "source",
            ],

            "VALVES_AIR_MANAGEMENT": [
                "air valve",
                "check valve",
                "non return",
            ],

            "HOT_WATER_RECIRCULATION": [
                "hot water",
                "recirculation",
                "return temperature",
            ],

            "NPSH": [
                "npsh",
                "cavitation",
            ],

            "PIPE_MATERIAL": [
                "material",
                "steel",
                "hdpe",
                "pvc",
                "ppr",
                "copper",
            ],

            "WATER_QUALITY_PROTECTION": [
                "backflow",
                "potable",
                "cross connection",
            ],

            "GOVERNING_CODE": [
                "code",
                "ahj",
                "ipc",
                "bcp",
            ],
        }

        return {
            topic
            for topic, terms in mapping.items()
            if any(
                term in q
                for term in terms
            )
        }

    # ============================================================
    # APPLICATION CONTEXT
    # ============================================================

    def _application_context(
        self,
        application,
        service,
        query,
        material,
    ):

        a = (
            application
            or ""
        ).upper()

        q = (
            query
            or ""
        ).lower()

        topics = self._topic_candidates(
            q
        )

        # --------------------------------------------------------
        # Transfer
        # --------------------------------------------------------

        if a == "TRANSFER":

            topics |= {
                "PIPE_VELOCITY",
                "PIPE_FRICTION",
                "PIPE_SIZING",
                "PUMP_HEAD",
                "STORAGE",
                "DEMAND_FLOW",
            }

        # --------------------------------------------------------
        # Booster
        # --------------------------------------------------------

        elif a == "BOOSTER":

            topics |= {
                "PIPE_VELOCITY",
                "PIPE_FRICTION",
                "PIPE_SIZING",
                "PRESSURE",
                "PUMP_HEAD",
            }

        # --------------------------------------------------------
        # Submersible
        # --------------------------------------------------------

        elif a == "SUBMERSIBLE":

            topics |= {
                "PIPE_VELOCITY",
                "PIPE_FRICTION",
                "PIPE_SIZING",
                "PUMP_HEAD",
                "SOURCE_CAPACITY",
                "NPSH",
            }

        # --------------------------------------------------------
        # Hot-water recirculation
        # --------------------------------------------------------

        elif a == "HOT_WATER_RECIRCULATION":

            topics |= {
                "HOT_WATER_RECIRCULATION",
                "PIPE_FRICTION",
                "PIPE_SIZING",
                "PUMP_HEAD",
            }

        subsystem = None

        if a == "TRANSFER":
            subsystem = "RISING_MAIN"

        elif a == "BOOSTER":
            subsystem = "BOOSTER_SYSTEM"

        elif a == "HOT_WATER_RECIRCULATION":
            subsystem = "HOT_WATER_LOOP"

        return topics, subsystem

    # ============================================================
    # VECTOR SEARCH
    # ============================================================

    def _search(
        self,
        query,
        top_k,
    ):

        q = self.vectorizer.transform(
            [query]
        )

        scores = (
            self.matrix @ q.T
        ).toarray().ravel()

        order = np.argsort(
            -scores
        )

        return [
            dict(
                self.metadata[i],
                score=float(
                    scores[i]
                ),
            )
            for i in order[
                :top_k
            ]
            if scores[i] > 0
        ]

    # ============================================================
    # MAIN RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        query,
        *,
        application=None,
        service=None,
        jurisdiction=None,
        system="WATER_SUPPLY",
        flow_type="PRESSURIZED",
        topic=None,
        subsystem=None,
        material=None,
        top_k=None,
    ):

        app_topics, app_subsystem = (
            self._application_context(
                application,
                service,
                query,
                material,
            )
        )

        topics = (
            {topic}
            if topic
            else (
                self._topic_candidates(
                    query
                )
                | app_topics
            )
        )

        subsystem = (
            subsystem
            or app_subsystem
        )

        # --------------------------------------------------------
        # Search once per engineering topic.
        # This prevents a generic keyword match from dominating
        # the entire retrieval result.
        # --------------------------------------------------------

        queries = [
            f"{query} {material or ''}"
        ]

        queries += [
            f"{query} {t} {material or ''}"
            for t in sorted(topics)
        ]

        pool = {}

        for qq in queries:

            for r in self._search(
                qq,
                max(
                    12,
                    (top_k or self.top_k) * 2
                ),
            ):

                rid = (
                    r.get(
                        "record_type"
                    ),
                    r.get(
                        "record_id"
                    ),
                )

                if (
                    rid not in pool
                    or r["score"]
                    > pool[rid]["score"]
                ):

                    pool[rid] = r

        # --------------------------------------------------------
        # Metadata-first deterministic criterion retrieval.
        #
        # This is critical:
        # system/topic/flow/subsystem are authoritative filters.
        # Jurisdiction remains metadata and does NOT discard a
        # technically relevant criterion.
        # --------------------------------------------------------

        for c in self.criteria:

            md = c.model_dump(
                mode="json"
            )

            if (
                md.get("system")
                and md.get("system")
                != system
            ):
                continue

            if (
                flow_type
                and md.get("flow_type")
                and md.get("flow_type")
                not in {
                    flow_type,
                    "PRESSURIZED_FORCE_MAIN",
                }
            ):
                continue

            if (
                topics
                and md.get("topic")
                not in topics
            ):
                continue

            score = 0.02

            tags = " ".join(
                str(x)
                for x in c.retrieval_tags
            ).lower()

            name = (
                f"{c.parameter} "
                f"{c.applicability} "
                f"{c.application or ''}"
            ).lower()

            # ----------------------------------------------------
            # Material relevance
            # ----------------------------------------------------

            if material:

                mt = material.lower()

                aliases = []

                if (
                    "hdpe" in mt
                    or "pe100" in mt
                    or re.search(
                        r"\bpe\b",
                        mt,
                    )
                ):
                    aliases = [
                        "hdpe",
                        "pe / polyethylene",
                        "plastic",
                        "plastic pipe",
                    ]

                elif "pvc" in mt:

                    aliases = [
                        "pvc",
                        "plastic",
                        "plastic pipe",
                    ]

                elif (
                    "ppr" in mt
                    or "pp-r" in mt
                ):

                    aliases = [
                        "ppr",
                        "plastic",
                        "plastic pipe",
                    ]

                elif (
                    "steel" in mt
                    or "ms" in mt
                    or "carbon" in mt
                ):

                    aliases = [
                        "steel",
                        "di/ms",
                    ]

                elif "copper" in mt:

                    aliases = [
                        "copper",
                        "copper tubing",
                    ]

                if any(
                    alias in tags
                    or alias in name
                    for alias in aliases
                ):
                    score += 0.50

            # ----------------------------------------------------
            # Subsystem relevance
            # ----------------------------------------------------

            if (
                subsystem
                and c.subsystem
                == subsystem
            ):
                score += 0.25

            rid = (
                "criterion",
                c.criterion_id,
            )

            if (
                rid not in pool
                or score
                > pool[rid].get(
                    "score",
                    0,
                )
            ):

                pool[rid] = {
                    "record_id": c.criterion_id,
                    "record_type": "criterion",
                    "text": "",
                    "metadata": md,
                    "score": score,
                }

        # --------------------------------------------------------
        # Candidate filtering
        # --------------------------------------------------------

        candidates = sorted(
            pool.values(),
            key=lambda x: -x["score"],
        )

        selected = []

        for r in candidates:

            m = r.get(
                "metadata",
                {}
            )

            ms = m.get(
                "system"
            )

            if (
                ms
                and ms != system
            ):
                continue

            mf = m.get(
                "flow_type"
            )

            if (
                flow_type
                and mf
                and mf not in {
                    flow_type,
                    "PRESSURIZED_FORCE_MAIN",
                }
            ):
                continue

            mt = m.get(
                "topic"
            )

            if (
                topics
                and mt not in topics
            ):
                continue

            if (
                subsystem
                and m.get(
                    "subsystem"
                )
                and m.get(
                    "subsystem"
                ) != subsystem
                and m.get(
                    "subsystem"
                ) not in {
                    "DISTRIBUTION_MAIN",
                    None,
                }
            ):
                continue

            bonus = 0.0

            tags = " ".join(
                str(x)
                for x in m.get(
                    "retrieval_tags",
                    [],
                )
            ).lower()

            name = str(
                m.get(
                    "name",
                    "",
                )
            ).lower()

            if material:

                material_tokens = [
                    tok
                    for tok in _norm(
                        material
                    ).split()
                    if len(tok) > 2
                ]

                if any(
                    tok in (
                        tags
                        + " "
                        + name
                    )
                    for tok in material_tokens
                ):
                    bonus += 0.35

            if application:

                if (
                    application.lower()
                    in str(
                        m.get(
                            "application",
                            "",
                        )
                    ).lower()
                ):
                    bonus += 0.10

            # Jurisdiction is only a relevance bonus.
            # It is NOT a hard filter.
            if jurisdiction:

                if (
                    jurisdiction.lower()
                    in str(
                        m.get(
                            "source_location",
                            "",
                        )
                    ).lower()
                ):
                    bonus += 0.05

            r["score"] += bonus

            selected.append(r)

        # ========================================================
        # IMPORTANT TOPIC-COVERAGE FIX
        # ========================================================
        #
        # Before this section, six Hazen-Williams records could
        # occupy all six result slots.
        #
        # Now we reserve at least one result for every requested
        # engineering topic whenever a matching criterion exists.
        #
        # Therefore TRANSFER retrieves PIPE_VELOCITY even when
        # several material-specific C-values have higher scores.
        # ========================================================

        limit = top_k or self.top_k

        candidates = sorted(
            selected,
            key=lambda x: -x["score"],
        )

        selected_final = []

        selected_ids = set()

        # --------------------------------------------------------
        # Reserve one highest-scoring criterion/chunk per topic
        # --------------------------------------------------------

        for requested_topic in sorted(
            topics
        ):

            topic_matches = [
                r
                for r in candidates
                if r.get(
                    "metadata",
                    {}
                ).get(
                    "topic"
                )
                == requested_topic
            ]

            if topic_matches:

                chosen = topic_matches[0]

                rid = (
                    chosen.get(
                        "record_type"
                    ),
                    chosen.get(
                        "record_id"
                    ),
                )

                if rid not in selected_ids:

                    selected_final.append(
                        chosen
                    )

                    selected_ids.add(
                        rid
                    )

        # --------------------------------------------------------
        # Fill remaining slots by score
        # --------------------------------------------------------

        for r in candidates:

            rid = (
                r.get(
                    "record_type"
                ),
                r.get(
                    "record_id"
                ),
            )

            if rid in selected_ids:
                continue

            selected_final.append(
                r
            )

            selected_ids.add(
                rid
            )

            if (
                len(selected_final)
                >= limit
            ):
                break

        selected = selected_final[
            :limit
        ]

        # ========================================================
        # BUILD RESULT
        # ========================================================

        crit = []
        refs = {}

        for r in selected:

            if (
                r["record_type"]
                == "criterion"
            ):

                c = next(
                    (
                        x
                        for x in self.criteria
                        if x.criterion_id
                        == r["record_id"]
                    ),
                    None,
                )

                if c:

                    crit.append(
                        c.model_dump(
                            mode="json"
                        )
                    )

                    refs[
                        c.source_reference_id
                    ] = self.sources.get(
                        c.source_reference_id
                    )

        return {
            "criteria": crit,

            "chunks": [
                r.get(
                    "metadata",
                    {}
                )
                | {
                    "score": r.get(
                        "score"
                    )
                }
                for r in selected
                if r["record_type"]
                == "chunk"
            ],

            "references": [
                x
                for x in refs.values()
                if x
            ],
        }

    # ============================================================
    # RETRIEVER INTERFACE
    # ============================================================

    def retrieve_criteria(
        self,
        *,
        application,
        service=None,
        jurisdiction=None,
        query="engineering criteria",
        topic=None,
        subsystem=None,
        material=None,
    ):

        result = self.retrieve(
            f"{application} "
            f"{service or ''} "
            f"{query} "
            f"{material or ''}",

            application=application,

            service=service,

            jurisdiction=jurisdiction,

            topic=topic,

            subsystem=subsystem,

            material=material,
        )

        return [
            Criterion.model_validate(
                x
            )
            for x in result[
                "criteria"
            ]
        ]

    def retriever(
        self,
        **kwargs
    ):

        return self.retrieve_criteria(
            **kwargs
        )
