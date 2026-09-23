from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from models.engineering_contract import Criterion, CriterionType


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "standards"
INDEX = DATA / "index"


def _norm(s: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(s).lower(),
    ).strip()


def _criterion_type(
    name: str,
    parameter: str,
    value: Any,
) -> CriterionType:

    n = f"{name} {parameter}".lower()

    if "velocity" in n:
        return CriterionType.VELOCITY

    if "pressure" in n or "prv" in n:
        return CriterionType.PRESSURE

    if "hazen" in n or parameter.lower().startswith("hazen"):
        return CriterionType.MATERIAL

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

    if isinstance(value, (int, float)):
        return CriterionType.OTHER

    return CriterionType.OTHER


def _load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_criteria() -> list[Criterion]:

    records: list[dict[str, Any]] = []

    for path in [
        DATA
        / "criteria"
        / "engineering_criteria.json",

        DATA
        / "criteria"
        / "extended_criteria.json",
    ]:

        if path.exists():

            obj = _load_json(path)

            if isinstance(obj, list):
                records.extend(obj)

    out: list[Criterion] = []
    seen: set[str] = set()

    for r in records:

        cid = str(
            r.get(
                "criterion_id",
                "",
            )
        ).strip()

        if not cid or cid in seen:
            continue

        seen.add(cid)

        value = r.get("value")

        numeric_value = (
            value
            if isinstance(
                value,
                (int, float),
            )
            else None
        )

        min_value = None
        max_value = None

        if (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(
                    x,
                    (int, float),
                )
                for x in value
            )
        ):
            min_value, max_value = value

        app = (
            r.get("application")
            or (
                r.get("scope")
                if r.get("scope")
                else "all"
            )
        )

        if isinstance(app, list):

            app = ",".join(
                str(x).lower()
                for x in app
            )

        elif isinstance(app, str):

            app = app.lower()

        source_id = (
            r.get("source_id")
            or (
                r.get("source_ids")
                or [None]
            )[0]
        )

        if not source_id:
            source_id = "UNSPECIFIED"

        note_value = (
            r.get("notes")
            or r.get("rule")
            or r.get("value")
        )

        if not isinstance(
            note_value,
            str,
        ):
            note_value = json.dumps(
                note_value,
                sort_keys=True,
            )

        criterion = Criterion(
            criterion_id=cid,

            criterion_type=_criterion_type(
                str(
                    r.get(
                        "name",
                        "",
                    )
                ),
                str(
                    r.get(
                        "parameter",
                        "",
                    )
                ),
                value,
            ),

            parameter=str(
                r.get("parameter")
                or r.get("name")
                or cid
            ),

            value=numeric_value,

            min_value=min_value,
            max_value=max_value,

            unit=r.get("unit"),

            application=app,

            service=r.get("service"),

            jurisdiction=r.get(
                "jurisdiction"
            ),

            applicability=str(
                r.get("applicability")
                or r.get("scope")
                or "source-defined"
            ),

            method=r.get("method"),

            source_reference_id=source_id,

            source_status=str(
                r.get("source_status")
                or r.get("status")
                or "UNVERIFIED"
            ),

            edition_year=(
                str(
                    r.get(
                        "edition_year"
                    )
                )
                if r.get("edition_year")
                else None
            ),

            section=(
                r.get("section")
                or r.get("reference")
            ),

            page=r.get("page"),

            notes=note_value,
        )

        out.append(criterion)

    # --------------------------------------------------------
    # Hazen-Williams C registry
    # --------------------------------------------------------

    hw_path = (
        DATA
        / "pipe_data"
        / "hazen_williams_c_registry.json"
    )

    if hw_path.exists():

        hw = _load_json(hw_path)

        sources = hw.get(
            "sources",
            {},
        )

        for rec in hw.get(
            "records",
            [],
        ):

            preferred = rec.get(
                "preferred"
            )

            if preferred is None:
                continue

            source_id = rec.get(
                "source",
                "UNKNOWN",
            )

            src = sources.get(
                source_id,
                {},
            )

            cid = (
                f"HW-C-"
                f"{_norm(rec.get('material', '')).replace(' ', '-')}-"
                f"{_norm(rec.get('condition', '')).replace(' ', '-')}"
            )

            if any(
                c.criterion_id == cid
                for c in out
            ):
                continue

            out.append(
                Criterion(
                    criterion_id=cid,

                    criterion_type=CriterionType.MATERIAL,

                    parameter="hazen_williams_c",

                    value=float(
                        preferred
                    ),

                    unit="dimensionless",

                    application="all",

                    applicability=(
                        f"{rec.get('material')} / "
                        f"{rec.get('condition')}"
                    ),

                    source_reference_id=source_id,

                    source_status=str(
                        rec.get(
                            "status",
                            "UNVERIFIED",
                        )
                    ),

                    edition_year=(
                        str(src.get("year"))
                        if src.get("year")
                        else None
                    ),

                    notes=(
                        "Registry value; governing "
                        "project/AHJ criterion takes precedence."
                    ),
                )
            )

    return out


def load_chunks() -> list[dict[str, Any]]:

    path = (
        DATA
        / "corpus"
        / "chunks.jsonl"
    )

    rows = []

    if path.exists():

        for line in path.read_text(
            encoding="utf-8"
        ).splitlines():

            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def load_sources() -> dict[str, dict[str, Any]]:

    path = (
        DATA
        / "sources"
        / "source_registry.json"
    )

    if not path.exists():
        return {}

    return {
        x["source_id"]: x
        for x in _load_json(path)
    }


class RAGStore:
    """Local RAG store.

    Uses FAISS when installed; otherwise a deterministic
    TF-IDF cosine fallback is used for local testing.

    The returned engineering context is deliberately kept
    compact because the LLM is not the numerical authority.
    """

    def __init__(
        self,
        *,
        top_k: int = 6,
    ):

        self.top_k = top_k

        self.criteria = load_criteria()

        self.chunks = load_chunks()

        self.sources = load_sources()

        self.vectorizer = None
        self.matrix = None
        self.faiss_index = None

        self._load_or_build_index()

    def _load_or_build_index(self):

        pkl = (
            INDEX
            / "tfidf_vectorizer.pkl"
        )

        npy = (
            INDEX
            / "tfidf_matrix.npz"
        )

        if pkl.exists() and npy.exists():

            from scipy.sparse import load_npz

            self.vectorizer = pickle.loads(
                pkl.read_bytes()
            )

            self.matrix = load_npz(
                npy
            ).tocsr()

        else:
            self._build_tfidf()

        faiss_path = (
            INDEX
            / "chunks.faiss"
        )

        if faiss_path.exists():

            try:

                import faiss

                self.faiss_index = (
                    faiss.read_index(
                        str(faiss_path)
                    )
                )

            except Exception:

                self.faiss_index = None

    def _build_tfidf(self):

        from sklearn.feature_extraction.text import (
            TfidfVectorizer,
        )

        from scipy.sparse import save_npz

        texts = [
            (
                f"{c.get('heading', '')} "
                f"{c.get('text', '')} "
                f"{' '.join(c.get('tags', []))}"
            )
            for c in self.chunks
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            norm="l2",
        )

        self.matrix = (
            self.vectorizer
            .fit_transform(texts)
            .tocsr()
        )

        INDEX.mkdir(
            parents=True,
            exist_ok=True,
        )

        pkl = (
            INDEX
            / "tfidf_vectorizer.pkl"
        )

        pkl.write_bytes(
            pickle.dumps(
                self.vectorizer
            )
        )

        save_npz(
            INDEX / "tfidf_matrix.npz",
            self.matrix,
        )

    def _search_chunks(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        k = top_k or self.top_k

        if self.faiss_index is not None:

            q = (
                self.vectorizer
                .transform([query])
                .toarray()
                .astype("float32")
            )

            scores, ids = (
                self.faiss_index.search(
                    q,
                    min(
                        k,
                        len(self.chunks),
                    ),
                )
            )

            return [
                {
                    **self.chunks[i],
                    "score": float(
                        scores[0][j]
                    ),
                }
                for j, i in enumerate(
                    ids[0]
                )
                if i >= 0
            ]

        q = self.vectorizer.transform(
            [query]
        )

        scores = (
            self.matrix @ q.T
        ).toarray().ravel()

        order = np.argsort(
            -scores
        )[:k]

        return [
            {
                **self.chunks[i],
                "score": float(
                    scores[i]
                ),
            }
            for i in order
            if scores[i] > 0
        ]

    def _select_mandatory_criteria(
        self,
        criteria: list[Criterion],
        selected: list[Criterion],
        query: str,
        limit: int,
    ) -> list[Criterion]:
        """Keep the LLM context compact while retaining engineering-critical criteria."""

        selected_ids = {
            c.criterion_id
            for c in selected
        }

        query_norm = _norm(query)

        # ----------------------------------------------------
        # 1. Keep one velocity criterion
        # ----------------------------------------------------

        velocity_candidates = [
            c
            for c in criteria
            if c.criterion_type
            == CriterionType.VELOCITY
        ]

        if velocity_candidates:

            velocity_candidates.sort(
                key=lambda c: (
                    -len(
                        set(
                            query_norm.split()
                        )
                        & set(
                            _norm(
                                f"{c.parameter} "
                                f"{c.applicability} "
                                f"{c.notes or ''}"
                            ).split()
                        )
                    ),
                    c.criterion_id,
                )
            )

            best_velocity = (
                velocity_candidates[0]
            )

            if (
                best_velocity.criterion_id
                not in selected_ids
            ):

                if len(selected) >= limit:
                    selected.pop()

                selected.append(
                    best_velocity
                )

                selected_ids.add(
                    best_velocity.criterion_id
                )

        # ----------------------------------------------------
        # 2. Keep one Hazen-Williams C criterion
        # ----------------------------------------------------

        hw_candidates = [
            c
            for c in criteria
            if (
                c.criterion_type
                == CriterionType.MATERIAL
                and c.parameter.lower()
                in {
                    "hazen_williams_c",
                    "hazen_williams_c_value",
                }
            )
        ]

        if hw_candidates:

            def hw_score(
                c: Criterion,
            ) -> tuple[int, int, int, str]:

                text = _norm(
                    f"{c.applicability} "
                    f"{c.notes or ''}"
                )

                overlap = len(
                    set(
                        query_norm.split()
                    )
                    & set(
                        text.split()
                    )
                )

                primary = (
                    1
                    if "PRIMARY"
                    in c.source_status.upper()
                    else 0
                )

                new_pipe = (
                    1
                    if "new"
                    in c.applicability.lower()
                    else 0
                )

                return (
                    overlap,
                    primary,
                    new_pipe,
                    c.criterion_id,
                )

            hw_candidates.sort(
                key=hw_score,
                reverse=True,
            )

            best_hw = (
                hw_candidates[0]
            )

            if (
                best_hw.criterion_id
                not in selected_ids
            ):

                if len(selected) >= limit:
                    selected.pop()

                selected.append(
                    best_hw
                )

        # ----------------------------------------------------
        # 3. Final hard cap
        # ----------------------------------------------------

        return selected[:limit]

    def retrieve(
        self,
        query: str,
        *,
        application: str | None = None,
        jurisdiction: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:

        app = (
            application or ""
        ).lower()

        chunks = self._search_chunks(
            query,
            top_k,
        )

        # ----------------------------------------------------
        # Filter criteria by application
        # ----------------------------------------------------

        criteria = []

        for c in self.criteria:

            capp = (
                c.application
                or "all"
            ).lower()

            capp_parts = {
                x.strip()
                for x in capp.split(",")
            }

            if (
                app
                and "all" not in capp_parts
                and app not in capp_parts
                and not (
                    app in capp
                    or capp in app
                )
            ):
                continue

            criteria.append(c)

        # ----------------------------------------------------
        # Rank criteria by lexical relevance
        # ----------------------------------------------------

        qnorm = _norm(query)

        tokens = set(
            qnorm.split()
        )

        ranked = []

        for c in criteria:

            text = _norm(
                f"{c.criterion_id} "
                f"{c.parameter} "
                f"{c.applicability} "
                f"{c.notes or ''}"
            )

            overlap = len(
                tokens.intersection(
                    text.split()
                )
            )

            ranked.append(
                (
                    overlap,
                    c,
                )
            )

        ranked.sort(
            key=lambda x: (
                -x[0],
                x[1].criterion_id,
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Never send more than the configured compact
        # engineering context.
        # ----------------------------------------------------

        limit = min(
            6,
            max(
                4,
                top_k or self.top_k,
            ),
        )

        selected = [
            c
            for score, c in ranked
            if score > 0
        ][:limit]

        # ----------------------------------------------------
        # If lexical matching is weak, fill the remaining
        # slots with application criteria.
        # ----------------------------------------------------

        if len(selected) < limit:

            selected_ids = {
                c.criterion_id
                for c in selected
            }

            for c in criteria:

                if (
                    c.criterion_id
                    not in selected_ids
                ):

                    selected.append(c)

                    selected_ids.add(
                        c.criterion_id
                    )

                if len(selected) >= limit:
                    break

        # ----------------------------------------------------
        # Always retain the engineering-critical criteria
        # required by the deterministic workflows.
        # ----------------------------------------------------

        selected = (
            self._select_mandatory_criteria(
                criteria,
                selected,
                query,
                limit,
            )
        )

        return {
            "criteria": [
                c.model_dump(
                    mode="json"
                )
                for c in selected
            ],

            "chunks": chunks,

            "references": [
                self.sources[
                    c.source_reference_id
                ]

                for c in selected

                if (
                    c.source_reference_id
                    in self.sources
                )
            ],
        }

    def retrieve_criteria(
        self,
        *,
        application: str,
        service: str | None = None,
        jurisdiction: str | None = None,
        query: str = "engineering criteria",
    ) -> list[Criterion]:

        result = self.retrieve(
            (
                f"{application} "
                f"{service or ''} "
                f"{jurisdiction or ''} "
                f"{query}"
            ),
            application=application,
            jurisdiction=jurisdiction,
        )

        return [
            Criterion.model_validate(x)
            for x in result["criteria"]
        ]

    def retriever(
        self,
        *,
        application: str,
        service: str | None = None,
        jurisdiction: str | None = None,
        query: str = "engineering criteria",
    ) -> list[Criterion]:

        return self.retrieve_criteria(
            application=application,
            service=service,
            jurisdiction=jurisdiction,
            query=query,
        )
