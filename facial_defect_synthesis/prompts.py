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
    "A clinical documentation photograph of the patient's face, {view}, the face centred, "
    "in sharp focus and evenly lit. Background: {background}. True-to-life skin tone and "
    "texture, shot on a DSLR. Neutral expression unless the finding states otherwise. "
    "Any lesion, scar, graft or abnormal area must sit continuously within the skin and "
    "share the exact same lighting, focus, grain and colour balance as the rest of the "
    "face — no bright rim, halo, glow, outline, blur ring or cut-and-paste edge around it, "
    "and nothing that looks composited, pasted-on or photoshopped. "
    "Photorealistic — a genuine clinical photograph, NOT an illustration, painting, render "
    "or caricature. No text, watermarks or annotations."
)

# Plain LIGHT-BLUE backdrop only. Doctors asked for a distraction-free plain background
# (no medical equipment); the clinic walls are light blue, so the backdrop is light blue.
CLINICAL_BACKGROUNDS = [
    "a completely plain, smooth, uniform light-blue photographic backdrop (the pale blue "
    "of the clinic wall), with nothing else in the frame — no equipment, furniture, props "
    "or objects of any kind, just the solid light-blue wall behind the patient",
]

# --- Diversity axes -----------------------------------------------------------
ADULT_AGES = ["young adult", "middle-aged", "elderly"]
ETHNICITIES = [
    "White", "Black", "East Asian", "South Asian", "Hispanic", "Middle Eastern",
]
SEXES = ["male", "female"]
# Project requirement: ALL images are frontal. Keep this list frontal-only.
VIEWS = ["Frontal view"]


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
             "text": "The patient has a calm expression at rest. There is a clear long-standing "
                     "FLACCID paralysis of the RIGHT side of the face: the right eyebrow sits lower, "
                     "the right nose-to-mouth crease (nasolabial fold) is flattened and effaced, the "
                     "right corner of the mouth droops, and the right lower eyelid sags and turns "
                     "slightly outward so the right eye sits a little more open than the left. The "
                     "paralysed right side also shows loss of muscle bulk — the right cheek and "
                     "temple look flatter and slightly hollow and wasted next to the fuller, normal "
                     "left side. The deficit is clearly visible and definite, but the face is "
                     "relaxed and at rest — not a grimace, not exaggerated, and the muscles are not "
                     "actively contracting."},
            {"state": "preop", "slug": "smile-asymmetry",
             "text": "The patient has a gentle, slight, closed-mouth smile. Because of a "
                     "long-standing left-sided facial weakness the smile is a little uneven: the "
                     "right corner of the mouth lifts slightly while the left corner lifts less and "
                     "sits lower. The asymmetry is SUBTLE and natural — a soft lopsided smile, not "
                     "a broad grin and not exaggerated."},
            {"state": "preop", "slug": "synkinesis",
             "text": "The patient has a gentle, slight smile. There is mild post-paralytic "
                     "synkinesis on the left: as the patient smiles slightly, the left eye narrows "
                     "just a little more than the right. The effect is SUBTLE and natural — the "
                     "eyes are not squeezed shut and the face is not contorted."},
            {"state": "preop", "slug": "partial-paresis",
             "text": "The patient has a calm, neutral expression. There is a mild right-sided "
                     "facial weakness: the right nose-to-mouth crease (nasolabial fold) is slightly "
                     "flatter than the left, the right corner of the mouth sits a touch lower, and "
                     "the right cheek is slightly less full than the left (mild muscle wasting). "
                     "The asymmetry is subtle."},
            {"state": "preop", "slug": "congenital-unilateral", "age": "young child",
             "text": "The child has a calm, neutral expression. There is a subtle congenital "
                     "weakness of the right lower lip, so the right lower lip is a little less full "
                     "and the lower-lip line is mildly uneven. The effect is subtle and natural, "
                     "not a grimace."},
            {"state": "healed", "slug": "post-reanimation-symmetry",
             "text": "After facial-reanimation surgery the face is much improved and close to "
                     "balanced at rest, but NOT perfectly normal: a mild residual weakness remains "
                     "on the treated side — the nasolabial fold is a little shallower and that cheek "
                     "slightly less full — together with a faint, well-healed scar in front of the "
                     "ear."},
            {"state": "healed", "slug": "restored-smile",
             "text": "After facial-reanimation surgery the patient has a much-improved but still "
                     "slightly uneven smile: the treated side moves noticeably better yet lifts a "
                     "little less than the normal side, leaving some residual asymmetry and only a "
                     "faint mature scar."},
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
             "text": "There is a parotid tumour presenting as a modest, firm, rounded fullness in "
                     "front of the right ear and over the angle of the jaw, causing a slight visible "
                     "asymmetry on that side. The overlying skin is normal and intact. It is a "
                     "moderate lump, not a huge distorting mass."},
            {"state": "preop", "slug": "lip-scc",
             "text": "On the lower lip there is a long-standing squamous cell carcinoma: a firm, "
                     "hardened, ulcerated lesion on the red part of the lip."},
            {"state": "preop", "slug": "cheek-tumour",
             "text": "On the cheek there is a long-standing skin cancer: a raised cutaneous tumour "
                     "with a firm, rolled border and a shallow central ulcer."},
            {"state": "preop", "slug": "maxilla-mass-asymmetry",
             "text": "A tumour of the right upper jaw causes a mild fullness of the right cheek and "
                     "midface, producing a subtle facial asymmetry with the right side slightly "
                     "fuller than the left. The overlying skin is normal and intact — an early "
                     "presentation, not a large distorting mass."},
            {"state": "healed", "slug": "neck-dissection-faint",
             "text": "In a natural crease of the neck there is a faint, well-healed, mature curved "
                     "scar from a previous neck operation; it is barely noticeable."},
            {"state": "healed", "slug": "cheek-freeflap-healed",
             "text": "The cheek has been reconstructed with a free-tissue flap: a visible oval "
                     "'skin paddle' of transferred skin (from the thigh or forearm) fills the area. "
                     "It is a slightly different colour and texture from the surrounding face — "
                     "smoother, hairless and a bit paler, not a perfect colour match — and is "
                     "bordered by a mature scar. The contour is restored but the reconstructed patch "
                     "is recognisable."},
            {"state": "healed", "slug": "jawline-fibula-healed",
             "text": "The jawline was rebuilt with a fibula free flap after removal of part of the "
                     "jawbone: the contour is restored and a mature scar runs along the jaw. Over "
                     "the reconstruction there is a patch of transferred skin (the flap's skin "
                     "paddle) that is smoother and a slightly different colour from the surrounding "
                     "face, not a perfect match."},
            {"state": "healed", "slug": "midface-maxillectomy-healed",
             "text": "After removal of part of the upper jaw and its reconstruction the midface has "
                     "healed: where skin was replaced there is a slightly mismatched patch of "
                     "transferred skin (smoother and a different colour) with a mature scar, "
                     "otherwise only a subtle contour change."},
            {"state": "healed", "slug": "intraoral-resection-normal-face",
             "text": "After removal of a cancer that was contained entirely inside the mouth, the "
                     "external face is completely normal — natural, even skin with NO scar and no "
                     "contour change visible anywhere on the face."},
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
             "text": "There is a modest long-standing saddle-nose deformity from an old healed "
                     "nasal fracture: the middle of the nasal bridge is low and depressed, so on "
                     "this frontal view the bridge looks flattened and slightly widened and the "
                     "nose appears a little short with a mildly upturned tip. The deformity is "
                     "modest and realistic; the rest of the nose and face are normal."},
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
    "burns": {
        "label": "Facial burn scars and contractures",
        "ages": ["young child", "young adult", "middle-aged"],
        "subtypes": [
            {"state": "preop", "slug": "cheek-contracture",
             "text": "On the right cheek and jaw there is long-standing burn scarring: pale, shiny, "
                     "tight scar tissue with loss of normal skin texture, and a contracture that "
                     "pulls down the lower eyelid and the corner of the mouth."},
            {"state": "preop", "slug": "mixed-pigment-scar",
             "text": "Across the forehead and left cheek there are long-standing burn scars with a "
                     "tight, glossy surface and patchy mixed hyper- and hypo-pigmentation."},
            {"state": "healed", "slug": "graft-reconstructed",
             "text": "After burn-scar reconstruction and skin grafting the face has healed: the "
                     "contracture is released and the grafted skin blends reasonably, with only "
                     "mild residual difference in colour and texture."},
        ],
    },
    "vascular": {
        "label": "Vascular anomalies (hemangioma / port-wine stain)",
        "ages": ["infant", "young child", "young adult"],
        "subtypes": [
            {"state": "preop", "slug": "infantile-hemangioma", "age": "infant",
             "text": "On the right cheek there is a single small (about 2 cm), well-defined, bright "
                     "red, raised, lobulated infantile ('strawberry') hemangioma with a smooth "
                     "glossy surface. It is a small localised lesion; the rest of the infant's face "
                     "is completely normal and healthy."},
            {"state": "preop", "slug": "port-wine-stain",
             "text": "On the right cheek there is a small-to-moderate, well-defined flat "
                     "reddish-pink capillary malformation (a port-wine stain), a few centimetres "
                     "across, present since birth. It is localised to one part of the cheek; the "
                     "rest of the facial skin is completely normal."},
            {"state": "healed", "slug": "port-wine-faded",
             "text": "A small port-wine stain on the cheek has faded substantially after laser "
                     "treatment, leaving only a faint, even pink mark."},
        ],
    },
    "rhinophyma": {
        "label": "Rhinophyma and its repair",
        "ages": ["middle-aged", "elderly"],
        "subtypes": [
            {"state": "preop", "slug": "moderate",
             "text": "The skin of the lower nose is thickened and bumpy with enlarged pores and "
                     "mild redness — a moderate rhinophyma that developed slowly over years."},
            {"state": "preop", "slug": "severe-bulbous",
             "text": "The nose, especially the tip and lower third, is markedly thickened, "
                     "enlarged, bulbous, red and irregular with prominent pores — a severe "
                     "long-standing rhinophyma."},
            {"state": "healed", "slug": "post-debulking",
             "text": "After surgical reshaping (debulking) of rhinophyma the nose is improved and "
                     "smaller but not perfectly normal: some residual thickening and mildly "
                     "irregular texture of the lower-nose skin remain, with faint residual "
                     "redness."},
        ],
    },
    "nevus": {
        "label": "Congenital melanocytic nevus and reconstruction",
        "ages": ["young child", "teenager", "young adult"],
        "subtypes": [
            {"state": "preop", "slug": "cheek-patch",
             "text": "There is a congenital melanocytic nevus present since birth: a single, "
                     "well-defined oval dark-brown pigmented patch only a few centimetres across on "
                     "the left cheek, with a slightly raised, faintly textured surface and a few "
                     "coarse dark hairs. The patch is small and localised; the rest of the face is "
                     "normal skin."},
            {"state": "preop", "slug": "near-eyebrow",
             "text": "There is a small, well-circumscribed brown congenital melanocytic nevus, "
                     "about two centimetres across, just above the outer right eyebrow, present "
                     "since birth. It is a small localised spot; the rest of the face is normal."},
            {"state": "healed", "slug": "reconstructed",
             "text": "After removal and reconstruction of a small congenital melanocytic nevus on "
                     "the cheek, the area has healed with near-normal skin colour and only a faint "
                     "mature scar."},
        ],
    },
    "craniofacial": {
        "label": "Craniofacial microsomia / microtia and repair",
        "ages": ["young child", "teenager"],
        "subtypes": [
            {"state": "preop", "slug": "hemifacial-microsomia",
             "text": "The hair is tucked back so both ears are visible. There is hemifacial "
                     "microsomia of the RIGHT side: the right side of the face is clearly smaller "
                     "and less developed than the left — a flatter, smaller right cheek and jaw and "
                     "a smaller, malformed right ear — giving obvious left-right facial asymmetry. "
                     "The left side of the face is normal and fully formed."},
            {"state": "preop", "slug": "microtia",
             "text": "The hair is tucked back so both ears are fully visible for comparison. There "
                     "is microtia of the RIGHT ear: the right ear is a small, malformed, "
                     "peanut-shaped remnant with none of the normal ear folds, clearly much smaller "
                     "than the normal, fully-formed left ear. The face is otherwise normal."},
            {"state": "healed", "slug": "ear-reconstructed",
             "text": "The hair is tucked back so both ears are visible. After ear reconstruction "
                     "for microtia the right ear has a rebuilt shape and size roughly matching the "
                     "left, but it clearly looks reconstructed: built over a cartilage framework, so "
                     "its folds are smoother, stiffer and less defined than a natural ear and the "
                     "overlying skin is tight, with a mature scar. It is a good reconstruction, not "
                     "a perfect natural ear."},
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
                view=st.get("view", "Frontal view"),
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
        view=st.get("view") or rng.choice(VIEWS),
        background=rng.choice(CLINICAL_BACKGROUNDS),
    )
