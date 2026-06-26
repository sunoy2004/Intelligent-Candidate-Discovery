"""Job description analysis."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_pipeline_config, resolve_path, save_json

logger = logging.getLogger(__name__)

REQUIRED_SKILLS = [
    "embeddings", "vector search", "python", "NDCG", "MRR", "MAP",
    "retrieval", "ranking", "Milvus", "Elasticsearch", "FAISS",
    "sentence-transformers", "production", "evaluation",
]
PREFERRED_SKILLS = [
    "LoRA", "QLoRA", "XGBoost", "MLOps", "LLM fine-tuning",
    "learning-to-rank", "distributed systems", "open-source",
]
LOCATIONS = ["Pune", "Noida", "Delhi NCR", "Mumbai", "Hyderabad", "Bangalore", "Gurgaon"]
DISQUALIFIERS = [
    "consulting_only", "langchain_only", "keyword_stuffer",
    "title_chaser", "research_only", "cv_only",
]


def extract_job_features(jd_text: str) -> dict[str, Any]:
    yoe_match = re.search(r"(\d+)\s*[–-]\s*(\d+)\s*years", jd_text, re.I)
    yoe_min = int(yoe_match.group(1)) if yoe_match else 5
    yoe_max = int(yoe_match.group(2)) if yoe_match else 9

    found_required = [s for s in REQUIRED_SKILLS if s.lower() in jd_text.lower()]
    found_preferred = [s for s in PREFERRED_SKILLS if s.lower() in jd_text.lower()]

    extra_skills: list[str] = []
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(jd_text[:100000])
            for ent in doc.ents:
                if ent.label_ in {"ORG", "PRODUCT", "GPE"}:
                    continue
            for chunk in doc.noun_chunks:
                text = chunk.text.strip()
                if 2 < len(text) < 40 and text.lower() not in {"the role", "this role"}:
                    extra_skills.append(text)
        except OSError:
            logger.warning("spaCy model en_core_web_sm not found; skipping NLP extraction")
    except ImportError:
        logger.warning("spaCy not installed; skipping NLP extraction")

    return {
        "required_skills": sorted(set(found_required + REQUIRED_SKILLS[:8])),
        "preferred_skills": sorted(set(found_preferred + PREFERRED_SKILLS)),
        "extra_phrases": sorted(set(extra_skills))[:50],
        "years_experience_min": yoe_min,
        "years_experience_max": yoe_max,
        "seniority": "senior",
        "domain": "AI/IR/recruiting",
        "locations_preferred": LOCATIONS,
        "country_preferred": "India",
        "startup_weight": 0.8,
        "disqualifiers": DISQUALIFIERS,
        "jd_title": "Senior AI Engineer",
        "jd_text": jd_text,
    }


def embed_jd(jd_text: str, model_name: str, cache_path: Path) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=str(resolve_path("models")))
    embedding = model.encode(
        "Represent this sentence for searching relevant passages: " + jd_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    return embedding


def run_job_analysis() -> dict[str, Any]:
    pipeline = load_pipeline_config()
    jd_path = resolve_path(pipeline["paths"]["job_description"])
    jd_text = jd_path.read_text(encoding="utf-8")
    features = extract_job_features(jd_text)
    out_path = resolve_path(pipeline["paths"]["job_features"])
    save_json(out_path, {k: v for k, v in features.items() if k != "jd_text"})
    embed_jd(
        jd_text,
        pipeline["models"]["embedding_model"],
        resolve_path(pipeline["paths"]["jd_embedding"]),
    )
    logger.info("Job features written to %s", out_path)
    return features
