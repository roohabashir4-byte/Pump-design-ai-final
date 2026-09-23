# PumpDesign AI RAG Topic Registry v2.0

This package reorganizes the engineering knowledge base around **engineering topics and system classification**, not jurisdiction-first filtering.

## Core rule

Retrieve by:
1. System (WATER_SUPPLY / SEWERAGE / DRAINAGE)
2. Engineering topic (PIPE_VELOCITY, PIPE_FRICTION, PIPE_SIZING, PRESSURE, etc.)
3. Flow type (PRESSURIZED / GRAVITY / FORCE_MAIN)
4. Application/subsystem as additional context

Jurisdiction, source location, edition, section and page remain attached as metadata and are shown with the selected criterion. They are **not the primary retrieval filter**.

## Water-supply protection

Sewerage self-cleansing velocity and gravity-sewer criteria are classified as SEWERAGE and excluded from WATER_SUPPLY retrieval. Public-water-supply criteria from Punjab/PHED are retained as source-specific criteria; they are not automatically treated as building-internal criteria.

## Vector base

`corpus/vector_records_v2.jsonl` contains the searchable records. `index/tfidf_matrix.npz` and `index/tfidf_vectorizer.pkl` store the local vector index, while `index/metadata.jsonl` stores the complete metadata associated with every indexed record.

FAISS can be generated from the same normalized vectors when `faiss-cpu` is available.
