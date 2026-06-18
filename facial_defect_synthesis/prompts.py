"""Prompt templates and prompt builder for facial-defect synthesis.

Prompt anatomy (assembled by build_prompt):

    <PHOTO_STYLE: clinical photo, {view}, clinical {background}, photorealistic>

    The patient is <subject>. <finding sentence(s) for this subtype>

Design rules:
  * The SUBJECT is named exactly once ("The patient is ...") — never repeated.
  * Each subtype's `text` is a self-contained, grammatical *finding sentence*
    that already encodes its state (preop vs healed), so no extra label is glued on.
  * Only two states exist (no fresh post-surgical wounds / sutures):
      - "preop"  : congenital / long-standing defect the patient presents WITH
                   for facial reconstruction (the pre-operative input).
      - "healed" : well-healed, faint, mature scar after a good reconstruction.
  * Every photo is taken inside a medical environment (clinical background visible).

Categories (from the surgeon's email):
    facial_paralysis, mohs, hn_cancer, cleft, trauma
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict

# --- Photographic style anchor (clinical environment) ------------------------
PHOTO_STYLE = (
    "A realistic clinical photograph taken inside a medical facility. {view} of the "
    "patient's face, the face in sharp focus and well-lit. Setting: {background}. "
    "True-to-life skin tone and texture, shot on a DSLR. Neutral expression unless the "
    "finding states otherwise. Photorealistic — a genuine clinical photograph, NOT an "
    "illustration, painting, render or caricature. No text, watermarks or annotations."
)

# Medical clinical environments (kept softly out of focus behind the patient).
CLINICAL_BACKGROUNDS = [
    "a hospital examination room with an exam table, medical cabinets and a wall-mounted "
    "monitor softly blurred behind the patient",
    "a clinic consultation room with a blood-pressure monitor, instrument tray and "
    "anatomical wall charts softly out of focus behind the patient",
    "a plastic-surgery clinic exam room with an adjustable surgical exam light and medical "
    "supplies blurred in the background",
    "a hospital pre-operative holding area with a patient vitals monitor and IV pole softly "
    "out of focus behind the patient",
    "a dermatology clinic room with a medical exam chair and a stainless instrument tray "
    "blurred behind the patient",
]

# --- Diversity axes -----------------------------------------------------------
ADULT_AGES = ["young adult", "middle-aged", "elderly"]
ETHNICITIES = [
    "White", "Black", "East Asian", "South Asian", "Hispanic", "Middle Eastern",
]
SEXES = ["male", "female"]
VIEWS = ["Frontal view", "Three-quarter oblique view", "Lateral profile view"]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def describe_subject(age: str, ethnicity: str, sex: str) -> str:
    if age == "infant":
        return f"a {ethnicity} infant"
    if age == "young child":
        return f"a young {ethnicity} {'boy' if sex == 'male' else 'girl'}"
    if age == "teenager":
        return f"a {ethnicity} teenage {'boy' if sex == 'male' else 'girl'}"
    noun = "man" if sex == "male" else "woman"
    return f"{_article(age)} {age} {ethnicity} {noun}"


# --- Defect categories --------------------------------------------------------
# Each subtype: {state: preop|healed, slug: <kebab id>, text: <finding sentence>,
#                age: <optional forced age>}. `text` must read grammatically right
# after "The patient is <subject>. " — i.e. it describes the face, not the subject.
CATEGORIES = {
    "facial_paralysis": {
        "label": "Facial paralysis / paresis / synkinesis",
        "ages": ADULT_AGES,
        "subtypes": [
            {"state": "preop", "slug": "complete-flaccid",
             "text": "The right side of the face is completely paralysed and slack: the right "
                     "eyebrow droops, the right eye cannot fully close, the right nasolabial fold "
                     "is flattened, and the right corner of the mouth sags. The left side moves "
                     "normally, giving a marked left–right asymmetry. This is long-standing."},
            {"state": "preop", "slug": "smile-asymmetry",
             "text": "The patient is attempting to smile, which exposes a long-standing left-sided "
                     "facial paralysis: only the right corner of the mouth lifts while the left side "
                     "stays still and droops, producing a strongly asymmetric smile."},
            {"state": "preop", "slug": "synkinesis",
             "text": "There is chronic post-paralytic synkinesis on the left: as the patient smiles "
                     "the left eye narrows involuntarily and the left cheek over-contracts, "
                     "tightening the left nasolabial fold."},
            {"state": "preop", "slug": "partial-paresis",
             "text": "There is a subtle long-standing right-sided facial weakness: mild flattening "
                     "of the right nasolabial fold and slightly reduced movement on the right, with "
                     "only mild asymmetry at rest."},
            {"state": "preop", "slug": "congenital-unilateral", "age": "young child",
             "text": "There is a congenital weakness of the right lower lip: when the child "
                     "grimaces the right lower lip fails to move down, so the lower lip pulls "
                     "asymmetrically to one side. The rest of the face is normal."},
            {"state": "healed", "slug": "post-reanimation-symmetry",
             "text": "After facial-reanimation surgery the face is now well balanced and nearly "
                     "symmetric at rest; the only trace of surgery is a faint, well-healed scar in "
                     "front of the ear."},
            {"state": "healed", "slug": "restored-smile",
             "text": "After facial-reanimation surgery the patient has a restored, nearly symmetric "
                     "smile, with only a faint mature scar remaining."},
        ],
    },
    "mohs": {
        "label": "Mohs / facial skin cancer and reconstruction",
        "ages": ["middle-aged", "elderly"],
        "subtypes": [
            {"state": "preop", "slug": "bcc-nasal-ala",
             "text": "On the left side of the nose (the nasal ala) there is a small skin cancer: a "
                     "pearly, rounded nodule with a rolled border, fine surface blood vessels and a "
                     "small central crust, slowly growing over months."},
            {"state": "preop", "slug": "bcc-nasal-tip",
             "text": "On the tip of the nose there is a small pearly basal-cell-carcinoma nodule "
                     "with tiny surface blood vessels, present for several months."},
            {"state": "preop", "slug": "bcc-medial-canthus",
             "text": "At the inner corner of the right eye there is a pearly, slightly depressed "
                     "skin-cancer lesion with a rolled edge."},
            {"state": "preop", "slug": "scc-cheek",
             "text": "On the left cheek there is a squamous cell carcinoma: a firm, raised, scaly, "
                     "reddened nodule with a hardened base."},
            {"state": "preop", "slug": "bcc-forehead-temple",
             "text": "On the right temple there is a basal cell carcinoma: a pearly nodule with a "
                     "rolled border and fine surface vessels."},
            {"state": "healed", "slug": "nasal-recon-faint",
             "text": "On the side of the nose there is a faint, well-healed, mature scar from "
                     "previous skin-cancer surgery; the nose shape is restored and the scar is "
                     "barely visible."},
            {"state": "healed", "slug": "cheek-recon-faint",
             "text": "On the cheek there is a barely visible, well-healed fine-line scar from "
                     "previous skin-cancer surgery."},
            {"state": "healed", "slug": "forehead-recon-faint",
             "text": "On the forehead there is a faint, mature, well-healed scar from previous "
                     "skin-cancer reconstruction."},
        ],
    },
    "hn_cancer": {
        "label": "Head & neck cancer resection and reconstruction",
        "ages": ["middle-aged", "elderly"],
        "subtypes": [
            {"state": "preop", "slug": "parotid-mass",
             "text": "There is a long-standing parotid tumour: a firm, rounded fullness over the "
                     "right angle of the jaw and in front of the right ear, distorting the facial "
                     "contour on that side."},
            {"state": "preop", "slug": "lip-scc",
             "text": "On the lower lip there is a long-standing squamous cell carcinoma: a firm, "
                     "hardened, ulcerated lesion on the red part of the lip."},
            {"state": "preop", "slug": "cheek-tumour",
             "text": "On the cheek there is a long-standing skin cancer: a raised cutaneous tumour "
                     "with a firm, rolled border and a shallow central ulcer."},
            {"state": "preop", "slug": "maxilla-mass-asymmetry",
             "text": "A tumour of the right upper jaw causes fullness of the right cheek and "
                     "midface, producing visible facial asymmetry with the right side fuller than "
                     "the left."},
            {"state": "healed", "slug": "neck-dissection-faint",
             "text": "In a natural crease of the neck there is a faint, well-healed, mature curved "
                     "scar from a previous neck operation; it is barely noticeable."},
            {"state": "healed", "slug": "cheek-freeflap-healed",
             "text": "The cheek has been reconstructed with a tissue flap that healed well: the "
                     "transplanted skin blends in, the contour looks near-normal, and only a faint "
                     "mature scar remains."},
            {"state": "healed", "slug": "jawline-fibula-healed",
             "text": "The jawline was rebuilt after removal of part of the jawbone; the contour is "
                     "restored and only a faint, well-healed scar remains along the jaw."},
            {"state": "healed", "slug": "midface-maxillectomy-healed",
             "text": "After surgery on the upper jaw and its reconstruction the midface has healed "
                     "well, with only a subtle contour change and a faint mature scar."},
        ],
    },
    "cleft": {
        "label": "Cleft lip (congenital) and its repair",
        # Cleft is clinically dominated by infants/children; minors trip the safety
        # filter stochastically, so generate_image retries moderation blocks.
        "ages": ["infant", "young child"],
        "subtypes": [
            {"state": "preop", "slug": "unilateral-complete", "age": "infant",
             "text": "There is an unrepaired complete cleft of the upper lip on the LEFT side only: "
                     "a single clean vertical gap splits the upper lip from the lip margin straight "
                     "up into the left nostril, so the left nostril is widened and flattened. The "
                     "lip on either side of the gap is otherwise normal. Apart from this single "
                     "cleft the infant's face is completely normal, healthy and well-formed."},
            {"state": "preop", "slug": "unilateral-incomplete", "age": "infant",
             "text": "There is an unrepaired incomplete cleft of the upper lip on the left side: a "
                     "partial vertical notch in the upper lip that does NOT reach the nostril, and "
                     "the nostril is intact. Apart from this the infant's face is completely normal "
                     "and healthy."},
            {"state": "preop", "slug": "bilateral-complete", "age": "infant",
             "text": "There is an unrepaired complete cleft of the upper lip on BOTH sides: two "
                     "vertical gaps split the upper lip, leaving a small central block of lip tissue "
                     "that protrudes slightly forward, and both nostrils are widened. Apart from "
                     "this the infant's face is completely normal and healthy."},
            {"state": "preop", "slug": "microform", "age": "young child",
             "text": "There is a very mild (microform) cleft of the upper lip on the left: just a "
                     "faint vertical groove and a small notch in the lip margin, with slight lip "
                     "asymmetry. The rest of the face is completely normal."},
            {"state": "healed", "slug": "unilateral-repair-faint", "age": "young child",
             "text": "The upper lip was repaired after a cleft and healed very well: there is a "
                     "faint vertical scar up the left side of the lip, the lip is continuous, and "
                     "the cupid's-bow shape is restored with only slight nostril asymmetry."},
            {"state": "healed", "slug": "bilateral-repair-faint", "age": "young child",
             "text": "The upper lip was repaired after a bilateral cleft and healed very well: faint "
                     "symmetric scars, a restored cupid's bow, and a continuous near-normal upper "
                     "lip."},
            {"state": "healed", "slug": "adult-old-repair-faint", "age": "young adult",
             "text": "The upper lip carries a faint, mature, barely visible vertical scar from a "
                     "cleft-lip repair done in childhood, with only subtle residual nostril "
                     "asymmetry; otherwise the lip looks normal."},
        ],
    },
    "trauma": {
        "label": "Repair after facial trauma (long-standing scars / deformity)",
        "ages": ["young adult", "middle-aged"],
        "subtypes": [
            {"state": "preop", "slug": "old-cheek-laceration-scar",
             "text": "Across the right cheek there is a long-standing, widened and slightly raised "
                     "scar from an old deep laceration that healed years ago; the patient now wants "
                     "the scar revised."},
            {"state": "preop", "slug": "saddle-nose-deformity",
             "text": "There is a long-standing saddle-nose deformity from an old healed nasal "
                     "fracture: the bridge of the nose is collapsed and depressed in the middle, "
                     "giving a scooped-out profile."},
            {"state": "preop", "slug": "malar-flattening",
             "text": "There is long-standing facial asymmetry from an old, badly healed cheekbone "
                     "fracture on the right: the right cheekbone is flattened and sunken compared "
                     "with the left."},
            {"state": "preop", "slug": "old-brow-scar",
             "text": "Through the right eyebrow there is a long-standing healed scar that leaves a "
                     "small gap (notch) in the eyebrow hair where it crosses."},
            {"state": "preop", "slug": "old-lip-chin-scar",
             "text": "On the lower lip and chin there is a long-standing healed scar from an old "
                     "injury, with mild irregularity of the lip border and chin contour."},
            {"state": "healed", "slug": "laceration-repair-faint",
             "text": "On the cheek there is a well-healed, faint, fine-line scar where a laceration "
                     "was repaired and has matured; it is barely visible."},
            {"state": "healed", "slug": "fracture-recon-faint",
             "text": "After surgery to repair a facial fracture the face has healed well with "
                     "restored symmetry and only a faint, barely visible mature scar."},
        ],
    },
}

CATEGORY_KEYS = list(CATEGORIES.keys())


@dataclass
class PromptSpec:
    category: str
    state: str
    slug: str
    age: str
    ethnicity: str
    sex: str
    view: str
    background: str
    prompt: str

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("prompt")
        return d


def build_prompt(
    category: str,
    subtype: dict,
    *,
    age: str,
    ethnicity: str,
    sex: str,
    view: str = "Frontal view",
    background: str = CLINICAL_BACKGROUNDS[0],
) -> PromptSpec:
    subject = describe_subject(age, ethnicity, sex)
    style = PHOTO_STYLE.format(view=view, background=background)
    prompt = f"{style}\n\nThe patient is {subject}. {subtype['text']}"
    return PromptSpec(
        category=category, state=subtype["state"], slug=subtype["slug"],
        age=age, ethnicity=ethnicity, sex=sex, view=view, background=background, prompt=prompt,
    )


def catalog_specs() -> list:
    """One prompt per (category, subtype) — full detailed coverage of every
    sub-classification. Demographics are rotated deterministically for diversity."""
    specs = []
    i = 0
    for cat_key, cat in CATEGORIES.items():
        for st in cat["subtypes"]:
            age = st.get("age") or cat["ages"][i % len(cat["ages"])]
            specs.append(build_prompt(
                cat_key, st,
                age=age,
                ethnicity=ETHNICITIES[i % len(ETHNICITIES)],
                sex=SEXES[i % len(SEXES)],
                view="Frontal view",
                background=CLINICAL_BACKGROUNDS[i % len(CLINICAL_BACKGROUNDS)],
            ))
            i += 1
    return specs


def random_prompt(category: str, rng: random.Random) -> PromptSpec:
    """A randomised, diverse prompt for one category (for scaled batch generation)."""
    cat = CATEGORIES[category]
    st = rng.choice(cat["subtypes"])
    age = st.get("age") or rng.choice(cat["ages"])
    return build_prompt(
        category, st,
        age=age,
        ethnicity=rng.choice(ETHNICITIES),
        sex=rng.choice(SEXES),
        view=rng.choice(VIEWS),
        background=rng.choice(CLINICAL_BACKGROUNDS),
    )
