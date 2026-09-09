"""Option B weighting: disease mix + per-disease racial distribution + sex skew.

These tables drive `weighted` generation so the synthetic dataset matches real
epidemiology rather than a uniform 1:1 mix. All numbers are a tunable DRAFT — edit
freely. Rationale and citations live in demographics_disease_research.md.

  * DISEASE_WEIGHTS  : share of the dataset per disease (must sum to 1.0).
  * RACE_WEIGHTS     : per-disease distribution over the 6 generation ethnicities
                       (each disease's dict must sum to 1.0). Reflects who actually
                       gets each condition (e.g. skin cancer ~90% White).
  * MALE_FRACTION    : per-disease probability the subject is male (0.5 = no skew).
"""
from __future__ import annotations

import random

import prompts

# --- Disease mix (sums to 1.0). Skin cancer + H&N kept dominant. --------------
DISEASE_WEIGHTS = {
    "mohs": 0.28,
    "hn_cancer": 0.20,
    "trauma": 0.12,
    "facial_paralysis": 0.10,
    "cleft": 0.08,
    "burns": 0.07,
    "vascular": 0.05,
    "rhinophyma": 0.04,
    "nevus": 0.03,
    "craniofacial": 0.03,
}

# --- Per-disease racial distribution (each sums to 1.0) -----------------------
# Keys MUST match prompts.ETHNICITIES exactly.
RACE_WEIGHTS = {
    # Skin cancer is overwhelmingly White.
    "mohs":             {"White": 0.90, "Hispanic": 0.05, "Black": 0.01, "East Asian": 0.01, "South Asian": 0.01, "Middle Eastern": 0.02},
    # H&N cancer White-leaning, but Black notable (laryngeal).
    "hn_cancer":        {"White": 0.68, "Hispanic": 0.09, "Black": 0.14, "East Asian": 0.04, "South Asian": 0.02, "Middle Eastern": 0.03},
    # Trauma ~population, Black elevated (assault mechanism).
    "trauma":           {"White": 0.50, "Hispanic": 0.22, "Black": 0.18, "East Asian": 0.03, "South Asian": 0.03, "Middle Eastern": 0.04},
    # Bell's palsy ~population, slight Black/Hispanic bump.
    "facial_paralysis": {"White": 0.54, "Hispanic": 0.22, "Black": 0.14, "East Asian": 0.04, "South Asian": 0.03, "Middle Eastern": 0.03},
    # Cleft: White high among major groups, Hispanic also high, Black/Asian lower.
    "cleft":            {"White": 0.50, "Hispanic": 0.28, "Black": 0.06, "East Asian": 0.06, "South Asian": 0.04, "Middle Eastern": 0.06},
    # Burns ~population.
    "burns":            {"White": 0.50, "Hispanic": 0.22, "Black": 0.18, "East Asian": 0.03, "South Asian": 0.03, "Middle Eastern": 0.04},
    # Vascular: infantile hemangioma White-predominant; port-wine ~all.
    "vascular":         {"White": 0.60, "Hispanic": 0.18, "Black": 0.10, "East Asian": 0.05, "South Asian": 0.04, "Middle Eastern": 0.03},
    # Rhinophyma: strongly White (rosacea, fair skin).
    "rhinophyma":       {"White": 0.88, "Hispanic": 0.05, "Black": 0.01, "East Asian": 0.02, "South Asian": 0.02, "Middle Eastern": 0.02},
    # Giant congenital melanocytic nevus ~population.
    "nevus":            {"White": 0.55, "Hispanic": 0.22, "Black": 0.12, "East Asian": 0.05, "South Asian": 0.03, "Middle Eastern": 0.03},
    # Microtia/craniofacial microsomia higher in Hispanic/Asian.
    "craniofacial":     {"White": 0.45, "Hispanic": 0.28, "Black": 0.08, "East Asian": 0.09, "South Asian": 0.05, "Middle Eastern": 0.05},
}

# --- Per-disease male fraction (0.5 = balanced). Set all to 0.5 to disable. ---
MALE_FRACTION = {
    "trauma": 0.65,
    "rhinophyma": 0.80,
    "hn_cancer": 0.70,
    "mohs": 0.55,
}
DEFAULT_MALE_FRACTION = 0.5


def _validate() -> None:
    assert abs(sum(DISEASE_WEIGHTS.values()) - 1.0) < 1e-6, "DISEASE_WEIGHTS must sum to 1.0"
    for d, w in RACE_WEIGHTS.items():
        assert abs(sum(w.values()) - 1.0) < 1e-6, f"RACE_WEIGHTS[{d}] must sum to 1.0"
        assert set(w) == set(prompts.ETHNICITIES), f"RACE_WEIGHTS[{d}] keys must match ETHNICITIES"
    assert set(DISEASE_WEIGHTS) == set(prompts.CATEGORY_KEYS), "DISEASE_WEIGHTS must cover all categories"


def _weighted_choice(rng: random.Random, weights: dict):
    r = rng.random()
    cum = 0.0
    label = None
    for label, p in weights.items():
        cum += p
        if r <= cum:
            return label
    return label  # floating-point fallback


def weighted_spec(rng: random.Random) -> prompts.PromptSpec:
    disease = _weighted_choice(rng, DISEASE_WEIGHTS)
    cat = prompts.CATEGORIES[disease]
    st = rng.choice(cat["subtypes"])
    ethnicity = _weighted_choice(rng, RACE_WEIGHTS[disease])
    sex = "male" if rng.random() < MALE_FRACTION.get(disease, DEFAULT_MALE_FRACTION) else "female"
    age = st.get("age") or rng.choice(cat["ages"])
    return prompts.build_prompt(
        disease, st, age=age, ethnicity=ethnicity, sex=sex,
        view=st.get("view") or rng.choice(prompts.VIEWS),
        background=rng.choice(prompts.CLINICAL_BACKGROUNDS),
    )


def weighted_specs(n: int, rng: random.Random) -> list:
    _validate()
    return [weighted_spec(rng) for _ in range(n)]
