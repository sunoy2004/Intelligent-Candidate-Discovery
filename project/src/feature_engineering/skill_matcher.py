"""Skill matching with ontology."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml


def _expand_ontology() -> dict[str, set[str]]:
    ont = load_yaml("skill_ontology.yaml")
    alias_map: dict[str, set[str]] = {}
    for skill, aliases in ont.get("edges", {}).items():
        base = skill.lower()
        alias_map.setdefault(base, {base})
        for a in aliases:
            alias_map[base].add(a.lower())
    for canonical, aliases in ont.get("required_aliases", {}).items():
        alias_map.setdefault(canonical.lower(), {canonical.lower()})
        for a in aliases:
            alias_map[canonical.lower()].add(a.lower())
    return alias_map


def _expand_terms(terms: set[str], alias_map: dict[str, set[str]]) -> set[str]:
    expanded: set[str] = set()
    for term in terms:
        expanded.add(term)
        expanded.update(alias_map.get(term, set()))
        for skill, aliases in alias_map.items():
            if term in aliases or skill == term:
                expanded.update(aliases)
                expanded.add(skill)
    return expanded


def _matches(skill: str, expanded: set[str]) -> bool:
    return any(skill == e or e in skill or skill in e for e in expanded)


def score_skills(df: pd.DataFrame, job_features: dict[str, Any], title_scores: np.ndarray) -> np.ndarray:
    alias_map = _expand_ontology()
    required = {s.strip().lower() for s in job_features.get("required_skills", [])}
    preferred = {s.strip().lower() for s in job_features.get("preferred_skills", [])}
    required_expanded = _expand_terms(required, alias_map)
    preferred_expanded = _expand_terms(preferred, alias_map)

    skill_lists = df["skill_names"].apply(json.loads)
    assessment_lists = df["skill_assessment_json"].apply(json.loads)

    scores = np.zeros(len(df), dtype=np.float32)
    req_total = max(len(required_expanded), 1)
    pref_total = max(len(preferred_expanded), 1)

    for idx in range(len(df)):
        skill_names = skill_lists.iloc[idx]
        assessments = assessment_lists.iloc[idx]
        normalized = {s.strip().lower(): s for s in skill_names}

        req_hits = sum(1.0 for ns in normalized if _matches(ns, required_expanded))
        pref_hits = sum(0.5 for ns in normalized if _matches(ns, preferred_expanded))

        trust_vals = [assessments[s] / 100.0 for s in skill_names if s in assessments]
        trust = (sum(trust_vals) / len(trust_vals)) if trust_vals else 0.5

        base = 0.65 * (req_hits / req_total) + 0.35 * (pref_hits / pref_total)
        score = base * (0.7 + 0.3 * trust)

        if title_scores[idx] < 0.35 and df.iloc[idx].get("ai_keyword_count", 0) >= 7:
            score = min(score, 0.25)

        scores[idx] = np.clip(score, 0.0, 1.0)

    return scores
