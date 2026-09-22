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


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _criterion_type(name: str, parameter: str, value: Any) -> CriterionType:
    text = f"{name} {parameter}".lower()

    if "velocity" in text:
        return CriterionType.VELOCITY
    if "pressure" in text or "prv" in text:
        return CriterionType.PRESSURE
    if "hazen" in text or parameter.lower().startswith("hazen"):
        return CriterionType.MATERIAL
    if "demand" in text or "flow" in text:
        return CriterionType.DEMAND
    if "hot" in text or "recirculation" in text or "temperature" in text:
        return CriterionType.HOT_WATER
    if "npsh" in text:
        return CriterionType.NPSH
    if "method" in text or "hydraulic" in text:
        return CriterionType.HYDRAULIC_METHOD

    return CriterionType.OTHER


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _material_aliases(material: str) -> set[str]:
    """
    Convert UI/material names into normalized material families.

    This is only an alias map. It does not select a pipe condition
    or invent an engineering value.
    """
    m = _norm(material)

    aliases = {m}

    if (
        "steel" in m
        or m in {
            "ms",
            "carbon steel",
            "ms carbon steel",
            "mild steel",
        }
    ):
        aliases.update(
            {
                "steel",
                "steel pipe",
                "ms carbon steel",
                "carbon steel",
                "mild steel",
            }
        )

    if "pvc" in m:
        aliases.update({"pvc", "pvc u", "plastic pipe", "plastic"})

    if "hdpe" in m or m == "pe 100" or "polyethylene" in m:
        aliases.update(
            {
                "hdpe",
                "pe",
                "pe 100",
                "polyethylene",
                "pe polyethylene",
                "plastic pipe",
                "plastic",
            }
        )

    if "pp r" in m or "ppr" in m or "polypropylene" in m:
        aliases.update(
            {
                "pp r",
                "ppr",
                "polypropylene",
                "plastic pipe",
                "plastic",
            }
        )

    if "copper" in m:
        aliases.update({"copper", "copper tubing"})

    return aliases


def load_criteria() -> list[Criterion]:
    records: list[dict[str, Any]] = []

    # General engineering criteria.
    for path in [
        DATA / "criteria" / "engineering_criteria.json",
        DATA / "criteria" / "extended_criteria.json",
    ]:
        if path.exists():
            obj = _load_json(path)

            if isinstance(obj, list):
                records.extend(obj)

    out: list[Criterion] = []
    seen: set[str] = set()

    for record in records:
        criterion_id = str(record.get("criterion_id", "")).strip()

        if not criterion_id or criterion_id in seen:
            continue

        seen.add(criterion_id)

        value = record.get("value")

        numeric_value = (
            value if isinstance(value, (int, float)) else None
        )

        min_value = None
        max_value = None

        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(x, (int, float)) for x in value)
        ):
            min_value, max_value = value

        application = record.get("application") or (
            record.get("scope")
            if record.get("scope")
            else "all"
        )

        if isinstance(application, list):
            application = ",".join(
                str(x).lower() for x in application
            )
        elif isinstance(application, str):
            application = application.lower()

        source_id = record.get("source_id") or (
            record.get("source_ids") or [None]
        )[0]

        if not source_id:
            source_id = "UNSPECIFIED"

        note_value = (
            record.get("notes")
            or record.get("rule")
            or record.get("value")
        )

        if not isinstance(note_value, str):
            note_value = json.dumps(
                note_value,
                sort_keys=True,
            )

        out.append(
            Criterion(
                criterion_id=criterion_id,
                criterion_type=_criterion_type(
                    str(record.get("name", "")),
                    str(record.get("parameter", "")),
                    value,
                ),
                parameter=str(
                    record.get("parameter")
                    or record.get("name")
                    or criterion_id
                ),
                value=numeric_value,
                min_value=min_value,
                max_value=max_value,
                unit=record.get("unit"),
                application=application,
                service=record.get("service"),
                jurisdiction=record.get("jurisdiction"),
                applicability=str(
                    record.get("applicability")
                    or record.get("scope")
                    or "source-defined"
                ),
                method=record.get("method"),
                source_reference_id=source_id,
                source_status=str(
                    record.get("source_status")
                    or record.get("status")
                    or "UNVERIFIED"
                ),
                edition_year=str(
                    record.get("edition_year") or ""
                )
                or None,
                section=record.get("section")
                or record.get("reference"),
                page=record.get("page"),
                notes=note_value,
            )
        )

    # ------------------------------------------------------------------
    # Hazen-Williams C registry
    # ------------------------------------------------------------------
    #
    # These records are deliberately loaded separately from the normal
    # text/criteria retrieval. Pipe-material engineering data is structured
    # data and must remain available even when the text RAG top-k results
    # do not contain it.
    #
    hw_path = DATA / "pipe_data" / "hazen_williams_c_registry.json"

    if hw_path.exists():
        hw = _load_json(hw_path)

        sources = hw.get("sources", {})

        for record in hw.get("records", []):
            preferred = record.get("preferred")

            if preferred is None:
                continue

            source_id = record.get("source", "UNKNOWN")
            source = sources.get(source_id, {})

            material = str(record.get("material", ""))
            condition = str(record.get("condition", ""))

            criterion_id = (
                f"HW-C-{_norm(material).replace(' ', '-')}"
                f"-{_norm(condition).replace(' ', '-')}"
            )

            if any(
                c.criterion_id == criterion_id
                for c in out
            ):
                continue

            out.append(
                Criterion(
                    criterion_id=criterion_id,
                    criterion_type=CriterionType.MATERIAL,
                    parameter="hazen_williams_c",
                    value=float(preferred),
                    unit="dimensionless",
                    application="all",
                    applicability=f"{material} / {condition}",
                    source_reference_id=source_id,
                    source_status=str(
                        record.get("status", "UNVERIFIED")
                    ),
                    edition_year=(
                        str(source.get("year"))
                        if source.get("year")
                        else None
                    ),
                    notes=(
                        "Registry value; governing project/AHJ "
                        "criterion takes precedence."
                    ),
                )
            )

    return out


def load_chunks() -> list[dict[str, Any]]:
    path = DATA / "corpus" / "chunks.jsonl"

    rows: list[dict[str, Any]] = []

    if path.exists():
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_sources() -> dict[str, dict[str, Any]]:
    path = DATA / "sources" / "source_registry.json"

    if not path.exists():
        return {}

    return {
        item["source_id"]: item
        for item in _load_json(path)
    }


class RAGStore:
    """
    Local PumpDesign AI RAG store.

    Textual engineering knowledge is retrieved through FAISS/TF-IDF.

    Structured engineering registries, such as Hazen-Williams C,
    remain directly accessible and are not dependent on text-RAG top-k.
    """

    def __init__(self, *, top_k: int = 6):
        self.top_k = top_k

        self.criteria = load_criteria()
        self.chunks = load_chunks()
        self.sources = load_sources()

        self.vectorizer = None
        self.matrix = None
        self.faiss_index = None

        self._load_or_build_index()

    def _load_or_build_index(self):
        pkl = INDEX / "tfidf_vectorizer.pkl"
        npy = INDEX / "tfidf_matrix.npz"

        if pkl.exists() and npy.exists():
            from scipy.sparse import load_npz

            self.vectorizer = pickle.loads(
                pkl.read_bytes()
            )
            self.matrix = load_npz(npy).tocsr()
        else:
            self._build_tfidf()

        faiss_path = INDEX / "chunks.faiss"

        if faiss_path.exists():
            try:
                import faiss

                self.faiss_index = faiss.read_index(
                    str(faiss_path)
                )
            except Exception:
                self.faiss_index = None

    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import save_npz

        texts = [
            (
                f"{chunk.get('heading', '')} "
                f"{chunk.get('text', '')} "
                f"{' '.join(chunk.get('tags', []))}"
            )
            for chunk in self.chunks
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            norm="l2",
        )

        self.matrix = self.vectorizer.fit_transform(
            texts
        ).tocsr()

        INDEX.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            INDEX / "tfidf_vectorizer.pkl"
        ).write_bytes(
            pickle.dumps(self.vectorizer)
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

            scores, ids = self.faiss_index.search(
                q,
                min(k, len(self.chunks)),
            )

            return [
                {
                    **self.chunks[i],
                    "score": float(scores[0][j]),
                }
                for j, i in enumerate(ids[0])
                if i >= 0
            ]

        q = self.vectorizer.transform([query])

        scores = (
            self.matrix @ q.T
        ).toarray().ravel()

        order = np.argsort(-scores)[:k]

        return [
            {
                **self.chunks[i],
                "score": float(scores[i]),
            }
            for i in order
            if scores[i] > 0
        ]

    def _pipe_hazen_williams_criteria(
        self,
        material: str | None,
    ) -> list[Criterion]:
        """
        Return the applicable Hazen-Williams C records directly
        from the structured registry.

        Material matching is deterministic.

        For ordinary new steel, the registry contains the primary
        general-reference value C=140.

        This method does not apply public-water-supply criteria
        merely because the pipe material is steel.
        """
        if not material:
            return []

        aliases = _material_aliases(material)

        candidates = [
            criterion
            for criterion in self.criteria
            if (
                criterion.criterion_type == CriterionType.MATERIAL
                and criterion.parameter.lower()
                in {
                    "hazen_williams_c",
                    "hazen_williams_c_value",
                }
            )
        ]

        matched = [
            criterion
            for criterion in candidates
            if any(
                alias in _norm(criterion.applicability)
                for alias in aliases
            )
        ]

        if not matched:
            return []

        # Prefer primary engineering-reference records.
        primary = [
            criterion
            for criterion in matched
            if "PRIMARY" in criterion.source_status.upper()
        ]

        if primary:
            matched = primary

        # Prefer "new" condition where a primary registry contains
        # both new and degraded pipe conditions.
        new_records = [
            criterion
            for criterion in matched
            if "new" in criterion.applicability.lower()
        ]

        if len(new_records) == 1:
            return new_records

        # If there is exactly one primary record, use it.
        if len(matched) == 1:
            return matched

        # More than one applicable condition remains.
        # Do not silently choose between them.
        return matched

    def retrieve(
        self,
        query: str,
        *,
        application: str | None = None,
        jurisdiction: str | None = None,
        material: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve normal RAG context plus deterministic pipe criteria.

        `material` is optional so existing callers remain compatible.
        """
        app = (application or "").lower()

        chunks = self._search_chunks(
            query,
            top_k,
        )

        criteria = []

        for criterion in self.criteria:
            criterion_application = (
                criterion.application or "all"
            ).lower()

            application_parts = {
                item.strip()
                for item in criterion_application.split(",")
            }

            if (
                app
                and "all" not in application_parts
                and app not in application_parts
                and not (
                    app in criterion_application
                    or criterion_application in app
                )
            ):
                continue

            criteria.append(criterion)

        qnorm = _norm(query)
        tokens = set(qnorm.split())

        ranked = []

        for criterion in criteria:
            text = _norm(
                f"{criterion.criterion_id} "
                f"{criterion.parameter} "
                f"{criterion.applicability} "
                f"{criterion.notes or ''}"
            )

            overlap = len(
                tokens.intersection(text.split())
            )

            ranked.append(
                (overlap, criterion)
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1].criterion_id,
            )
        )

        limit = max(
            12,
            top_k or self.top_k,
        )

        selected = [
            criterion
            for score, criterion in ranked
            if score > 0
        ][:limit]

        # Preserve application-specific baseline criteria.
        if len(selected) < min(
            limit,
            len(criteria),
        ):
            selected_ids = {
                criterion.criterion_id
                for criterion in selected
            }

            for criterion in criteria:
                if criterion.criterion_id not in selected_ids:
                    selected.append(criterion)

                if len(selected) >= limit:
                    break

        # --------------------------------------------------------------
        # ALWAYS attach structured Hazen-Williams data when a material
        # is explicitly known.
        # --------------------------------------------------------------
        pipe_hw = self._pipe_hazen_williams_criteria(
            material
        )

        selected_ids = {
            criterion.criterion_id
            for criterion in selected
        }

        for criterion in pipe_hw:
            if criterion.criterion_id not in selected_ids:
                selected.append(criterion)
                selected_ids.add(
                    criterion.criterion_id
                )

        return {
            "criteria": [
                criterion.model_dump(mode="json")
                for criterion in selected
            ],
            "chunks": chunks,
            "references": [
                self.sources[
                    criterion.source_reference_id
                ]
                for criterion in selected
                if criterion.source_reference_id
                in self.sources
            ],
        }

    def retrieve_criteria(
        self,
        *,
        application: str,
        service: str | None = None,
        jurisdiction: str | None = None,
        query: str = "engineering criteria",
        material: str | None = None,
    ) -> list[Criterion]:
        result = self.retrieve(
            f"{application} "
            f"{service or ''} "
            f"{jurisdiction or ''} "
            f"{query}",
            application=application,
            jurisdiction=jurisdiction,
            material=material,
        )

        return [
            Criterion.model_validate(item)
            for item in result["criteria"]
        ]

    def retriever(
        self,
        *,
        application: str,
        service: str | None = None,
        jurisdiction: str | None = None,
        query: str = "engineering criteria",
        material: str | None = None,
    ) -> list[Criterion]:
        return self.retrieve_criteria(
            application=application,
            service=service,
            jurisdiction=jurisdiction,
            query=query,
            material=material,
        )
