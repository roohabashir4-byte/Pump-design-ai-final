"""Topic-first RAG boundary.

Validation does not discard a source-specific criterion merely because its
source jurisdiction differs from the project location. Applicability is
carried as metadata for engineering reasoning; system/topic mismatches are
excluded at retrieval.
"""
from __future__ import annotations
from models.engineering_contract import Criterion
from standards.criteria import validate_criteria

def build_criteria_packet(criteria:list[Criterion], *, application:str, service:str|None=None, jurisdiction:str|None=None)->list[Criterion]:
    errors=validate_criteria(criteria)
    if errors: raise ValueError("Invalid RAG criteria: "+"; ".join(errors))
    return criteria

def require_criterion(criteria:list[Criterion], *, criterion_type:str, parameter:str)->Criterion:
    matches=[c for c in criteria if c.criterion_type.value==criterion_type and c.parameter.lower()==parameter.lower()]
    if not matches: raise LookupError(f"No RAG criterion found for {criterion_type}/{parameter}.")
    if len(matches)>1: raise LookupError(f"Multiple criteria found for {criterion_type}/{parameter}; applicability/source precedence must be resolved first.")
    return matches[0]
