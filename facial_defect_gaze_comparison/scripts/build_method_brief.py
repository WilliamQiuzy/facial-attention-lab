from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "output" / "pdf" / "independent_cohort_gaze_comparison_brief.pdf"

MAYO_BLUE = colors.HexColor("#0057B8")
DEEP_BLUE = colors.HexColor("#003B71")
SKY = colors.HexColor("#DDEFFB")
PALE = colors.HexColor("#F4F8FB")
TEXT = colors.HexColor("#233746")
MUTED = colors.HexColor("#5F7482")
GREEN = colors.HexColor("#2A7F62")
RED = colors.HexColor("#B63D3D")
GOLD = colors.HexColor("#A86D00")
LINE = colors.HexColor("#C9D7E1")


def _register_fonts() -> tuple[str, str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"),
    ]
    mono_candidates = [
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/System/Library/Fonts/Supplemental/Andale Mono.ttf"),
    ]

    regular = next((path for path in candidates if path.exists()), None)
    bold = next((path for path in bold_candidates if path.exists()), None)
    mono = next((path for path in mono_candidates if path.exists()), None)
    if regular and bold and mono:
        pdfmetrics.registerFont(TTFont("BriefSans", str(regular)))
        pdfmetrics.registerFont(TTFont("BriefSansBold", str(bold)))
        pdfmetrics.registerFont(TTFont("BriefMono", str(mono)))
        return "BriefSans", "BriefSansBold", "BriefMono"
    return "Helvetica", "Helvetica-Bold", "Courier"


REGULAR, BOLD, MONO = _register_fonts()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=BOLD,
            fontSize=24,
            leading=27,
            textColor=DEEP_BLUE,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=REGULAR,
            fontSize=10.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName=BOLD,
            fontSize=14,
            leading=17,
            textColor=DEEP_BLUE,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=9.2,
            leading=12.4,
            textColor=TEXT,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=7.8,
            leading=10.2,
            textColor=MUTED,
        ),
        "formula": ParagraphStyle(
            "Formula",
            parent=base["Code"],
            fontName=MONO,
            fontSize=8.2,
            leading=11,
            textColor=DEEP_BLUE,
            backColor=PALE,
            borderColor=LINE,
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "card_title": ParagraphStyle(
            "CardTitle",
            parent=base["Heading3"],
            fontName=BOLD,
            fontSize=10.2,
            leading=12,
            textColor=DEEP_BLUE,
            spaceAfter=3,
        ),
        "card_body": ParagraphStyle(
            "CardBody",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=7.9,
            leading=10.2,
            textColor=TEXT,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=BOLD,
            fontSize=9,
            leading=12,
            textColor=DEEP_BLUE,
            alignment=TA_CENTER,
        ),
    }


STYLES = _styles()


def _page(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(MAYO_BLUE)
    canvas.rect(0, height - 0.14 * inch, width, 0.14 * inch, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.line(0.62 * inch, 0.47 * inch, width - 0.62 * inch, 0.47 * inch)
    canvas.setFont(REGULAR, 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.62 * inch, 0.27 * inch, "Facial Attention Lab | Research methods brief | Synthetic demonstration")
    canvas.drawRightString(width - 0.62 * inch, 0.27 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _card(number: str, title: str, body: str) -> Table:
    number_style = ParagraphStyle(
        f"Number{number}",
        fontName=BOLD,
        fontSize=16,
        leading=18,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    card = Table(
        [
            [Paragraph(number, number_style), Paragraph(title, STYLES["card_title"])],
            ["", Paragraph(body, STYLES["card_body"])],
        ],
        colWidths=[0.38 * inch, 2.8 * inch],
        rowHeights=[0.34 * inch, 0.56 * inch],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), MAYO_BLUE),
                ("BACKGROUND", (0, 1), (0, 1), SKY),
                ("BACKGROUND", (1, 0), (1, 1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (1, 0), (1, -1), 8),
                ("RIGHTPADDING", (1, 0), (1, -1), 8),
                ("TOPPADDING", (1, 0), (1, -1), 6),
                ("BOTTOMPADDING", (1, 0), (1, -1), 6),
            ]
        )
    )
    return card


def _decision_table() -> Table:
    rows = [
        ["Step", "Question", "Decision output"],
        ["1", "Comparable protocol and cohorts?", "Pass or stop/restrict"],
        ["2", "Technical differences within margins?", "Similar / different / inconclusive"],
        ["3", "Cross-domain map loss acceptable?", "Noninferior to split-half baseline"],
        ["4", "Can source domain be predicted?", "Technical AUC and attention AUC"],
    ]
    table = Table(rows, colWidths=[0.42 * inch, 2.68 * inch, 3.65 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DEEP_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), BOLD),
                ("FONTNAME", (0, 1), (0, -1), BOLD),
                ("FONTNAME", (1, 1), (-1, -1), REGULAR),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEADING", (0, 0), (-1, -1), 10.2),
                ("GRID", (0, 0), (-1, -1), 0.45, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _build_story() -> list[object]:
    p = Paragraph
    story: list[object] = [
        p("Independent Cohort Gaze Comparison", STYLES["title"]),
        p(
            "A concise framework for comparing 500 Webcam/Prolific participants with 500 different professional-camera participants.",
            STYLES["subtitle"],
        ),
        Table(
            [[p("500 Webcam", STYLES["callout"]), p("Different people", STYLES["callout"]), p("500 Professional", STYLES["callout"])]],
            colWidths=[2.15 * inch, 2.15 * inch, 2.15 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), SKY),
                    ("BACKGROUND", (1, 0), (1, 0), PALE),
                    ("BACKGROUND", (2, 0), (2, 0), SKY),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            ),
        ),
        Spacer(1, 0.12 * inch),
        p("What this design can answer", STYLES["section"]),
        p(
            "It estimates whether the complete Webcam workflow-and-cohort distribution is sufficiently similar to the professional workflow-and-cohort distribution for a named group-level endpoint. Equal sample size improves precision, but does not remove differences in recruitment, setting, display, lighting, or participant composition.",
            STYLES["body"],
        ),
        p(
            "It cannot estimate individual device agreement, device interchangeability, or a pure causal device effect because no participant is measured by both workflows.",
            ParagraphStyle("Boundary", parent=STYLES["body"], textColor=RED, fontName=BOLD),
        ),
        Spacer(1, 0.05 * inch),
        p("Four-step analysis", STYLES["section"]),
        Table(
            [
                [
                    _card("1", "Comparability", "Verify identical stimulus, task, exposure, coordinate transform, and QC versions. Review participant balance with SMD."),
                    _card("2", "Technical quality", "Compare participant-level accuracy, precision, data loss, and valid-trial share using Welch intervals and equivalence margins."),
                ],
                [
                    _card("3", "Group attention", "Compare cross-domain heatmaps against each cohort's own repeated split-half stability using histogram intersection."),
                    _card("4", "Domain detectability", "Use repeated cross-validated logistic regression to test technical and attention feature spaces separately."),
                ],
            ],
            colWidths=[3.3 * inch, 3.3 * inch],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            ),
        ),
        Spacer(1, 0.07 * inch),
        p("1. Comparability", STYLES["section"]),
        p(
            "For a prespecified participant characteristic X, measure balance with the standardized mean difference:",
            STYLES["body"],
        ),
        p("SMD = (mean_W - mean_P) / s_pooled", STYLES["formula"]),
        p(
            "A practical review flag is |SMD| >= 0.10. This is not proof of confounding or randomization. Acquisition-context variables such as display size and lighting describe the real workflow and should not be silently adjusted away.",
            STYLES["body"],
        ),
        p("2. Technical quality and equivalence", STYLES["section"]),
        p(
            "Compute one value per participant. Calibration accuracy is mean distance to known targets. Precision is RMS consecutive displacement at a stable target. Data loss is invalid samples divided by expected samples. Valid-trial share is valid trials divided by expected trials.",
            STYLES["body"],
        ),
        p(
            "Delta = mean_W - mean_P<br/>SE = sqrt(s_W^2/n_W + s_P^2/n_P)<br/>CI90 = Delta +/- t(0.95, Welch df) x SE",
            STYLES["formula"],
        ),
        p(
            "With a prespecified acceptable difference M, support equivalence only when the entire 90% interval is inside (-M, +M). A non-significant difference is not evidence of equivalence.",
            STYLES["body"],
        ),
        PageBreak(),
        p("Group attention and domain diagnostics", STYLES["title"]),
        p("The same-domain question is separated into technical acquisition and attention-pattern endpoints.", STYLES["subtitle"]),
        p("3. Group attention maps", STYLES["section"]),
        p(
            "For participant i and map bin b, sum valid dwell duration d over fixations that fall in that bin. Aggregate participants, apply one common smoothing rule, and normalize the map to sum to 1.",
            STYLES["body"],
        ),
        p(
            "H_i(b) = sum_j d_ij x I(g_ij is in bin b)<br/>P_G(b) = Smooth(sum_i H_i(b)) / sum_b Smooth(sum_i H_i(b))",
            STYLES["formula"],
        ),
        p(
            "Map similarity uses histogram intersection. It is the shared normalized density between two maps:",
            STYLES["body"],
        ),
        p("SIM(P,Q) = sum_b min(P(b), Q(b)), with 0 <= SIM <= 1", STYLES["formula"]),
        p(
            "Repeatedly split each cohort into two random halves. Compute Webcam split-half SIM, professional split-half SIM, and cross-domain SIM. The comparison is:",
            STYLES["body"],
        ),
        p(
            "gap = SIM_cross - min(SIM_W_split, SIM_P_split)<br/>Support noninferiority when the lower 90% split-range is above -M_map.",
            STYLES["formula"],
        ),
        p(
            "This uses each workflow's ordinary participant-sampling variability as a realistic benchmark instead of demanding a perfect SIM of 1. For real inference, add a participant bootstrap with replacement for the population cross-domain interval.",
            STYLES["body"],
        ),
        p("4. Domain detectability", STYLES["section"]),
        p(
            "Set Y=1 for Webcam and Y=0 for professional. Fit two separate regularized logistic models: one using technical-quality features and one using stimulus-by-AOI dwell shares. Standardize with training-fold statistics only.",
            STYLES["body"],
        ),
        p(
            "Pr(Y=1|X) = 1 / [1 + exp(-(beta_0 + sum_j beta_j z_j))]<br/>AUC = Pr(score_W > score_P) + 0.5 x Pr(tie)",
            STYLES["formula"],
        ),
        p(
            "Use repeated stratified five-fold cross-validation and report an out-of-fold AUC with a participant-bootstrap interval. AUC 0.50 means chance discrimination; higher values mean greater separability. Low AUC supports low detectability, but never proves equality.",
            STYLES["body"],
        ),
        Spacer(1, 0.04 * inch),
        p("How the decisions fit together", STYLES["section"]),
        _decision_table(),
        Spacer(1, 0.10 * inch),
        KeepTogether(
            [
                Table(
                    [[p("Synthetic example only", ParagraphStyle("ExampleHead", parent=STYLES["card_title"], textColor=colors.white))]],
                    colWidths=[6.75 * inch],
                    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), MAYO_BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]),
                ),
                Table(
                    [[p("Technical AUC 0.995: workflows are technically distinguishable.<br/>Attention AUC 0.552: attention features have low source detectability.<br/>Cross-domain SIM 0.954-0.961 vs within-cohort SIM about 0.963-0.964: group maps are close to split-half repeatability under the illustrative margin.", STYLES["card_body"]) ]],
                    colWidths=[6.75 * inch],
                    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), SKY), ("BOX", (0, 0), (-1, -1), 0.6, MAYO_BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]),
                ),
            ]
        ),
        Spacer(1, 0.10 * inch),
        p(
            "Defensible conclusion: For identical stimuli and a harmonized protocol, Webcam group maps may be sufficiently close to professional-workflow maps for a prespecified group-level endpoint. This does not establish individual interchangeability or a pure device effect.",
            ParagraphStyle("Conclusion", parent=STYLES["body"], fontName=BOLD, textColor=GREEN, borderColor=GREEN, borderWidth=0.7, borderPadding=7),
        ),
        Spacer(1, 0.08 * inch),
        p(
            "Methods basis: Lakens (equivalence); Lopez-Paz and Oquab (classifier two-sample tests); Kummerer et al. (saliency metrics). Synthetic methods only; not a Mayo clinical validation result.",
            STYLES["small"],
        ),
    ]
    return story


def build_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.60 * inch,
        title="Independent Cohort Gaze Comparison",
        author="Facial Attention Lab",
        subject="Concise independent-cohort Webcam versus professional gaze comparison methods",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="brief-frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates([PageTemplate(id="brief", frames=[frame], onPage=_page)])
    document.build(_build_story())
    return output_path


if __name__ == "__main__":
    print(build_pdf())
