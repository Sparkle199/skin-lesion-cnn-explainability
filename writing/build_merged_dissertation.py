"""Merges the abstract, all five chapters, and references into one
dissertation.docx, with front matter: a title page and Statement of
Originality (Handbook Appendices B and C), abstract, acknowledgements, a live
Table of Contents field, live List of Tables / List of Figures fields
(these rely on the Caption paragraph style now applied to every table and
figure caption in build_docx.py), a glossary, then the chapters in order.

Requires: pip install docxcompose (handles the low-level merge, including
image relationship IDs, correctly).
"""

from pathlib import Path
from docx import Document
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docxcompose.composer import Composer

BLACK = RGBColor(0, 0, 0)


def strip_theme_color(run):
    rpr = run._r.get_or_add_rPr()
    color_el = rpr.find(qn("w:color"))
    if color_el is not None:
        for attr in ("w:themeColor", "w:themeTint", "w:themeShade"):
            if color_el.get(qn(attr)) is not None:
                del color_el.attrib[qn(attr)]


def add_heading_black(doc, text, level):
    p = doc.add_heading(level=level)
    run = p.add_run(text)
    run.font.color.rgb = BLACK
    strip_theme_color(run)
    return p

ROOT = Path(__file__).parent
OUT = ROOT / "final project- 30137304.docx"

FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(12)

TITLE = "Deep learning for multi-class skin cancer detection using CNNs and explainable AI: A performance and interpretability analysis"
STUDENT_NAME = "Mercy Olaronke Olawuyi"
STUDENT_NUMBER = "30137304"
SCHEME = "MSc Artificial Intelligence"
YEAR_OF_STUDY = "2025/2026"
FIRST_SUPERVISOR = "Dr. Mabrouka Abuhmida"

CHAPTER_FILES = [
    "chapter-1-introduction.docx",
    "chapter-2-literature-review.docx",
    "chapter-3-methodology.docx",
    "chapter-4-results.docx",
    "chapter-5-discussion-conclusion-future-work.docx",
    "references.docx",
    "appendices.docx",
]

GLOSSARY = [
    ("akiec", "Actinic keratosis and intraepithelial carcinoma, one of HAM10000's seven diagnostic classes, grouped as malignant in the binary task."),
    ("AUC-ROC", "Area under the receiver operating characteristic curve, a threshold-independent measure of classification performance."),
    ("bcc", "Basal cell carcinoma, one of HAM10000's seven diagnostic classes, grouped as malignant in the binary task."),
    ("bkl", "Benign keratosis-like lesions, one of HAM10000's seven diagnostic classes, grouped as benign in the binary task."),
    ("CI (confidence interval)", "A range of values, computed from a sample, expected to contain the true value of an estimate at a stated confidence level, 95 percent throughout this dissertation."),
    ("CNN", "Convolutional neural network, the class of deep learning model compared throughout this dissertation."),
    ("DDI", "Diverse Dermatology Images, a dataset of 656 images spanning the full Fitzpatrick skin type range, used to test fairness across skin tone."),
    ("df", "Dermatofibroma, one of HAM10000's seven diagnostic classes, grouped as benign in the binary task."),
    ("Dice coefficient", "An overlap measure between a predicted region and a ground-truth region, closely related to IoU, used alongside it to score explanation faithfulness."),
    ("FST", "Fitzpatrick skin type, a classification of skin tone used to stratify the binary task's results."),
    ("GDPR", "General Data Protection Regulation, the UK and EU data protection law governing the handling of the datasets used in this project."),
    ("Grad-CAM", "Gradient-weighted class activation mapping, a post-hoc explainability method that produces a heatmap from a convolutional network's gradients."),
    ("HAM10000", "The primary dataset used in this dissertation, 10,015 dermoscopic images across seven diagnostic classes."),
    ("IoU", "Intersection over union, the primary overlap measure used to score explanation faithfulness against ground-truth lesion masks."),
    ("ISIC2018", "The International Skin Imaging Collaboration's 2018 Task 3 challenge, whose held-out test set is used as an independent evaluation set for the seven-class task."),
    ("Kappa (Cohen's kappa)", "A statistical measure of classification agreement that corrects for the agreement expected by chance, treated as the decisive metric wherever class imbalance could make accuracy misleading."),
    ("mel", "Melanoma, one of HAM10000's seven diagnostic classes, grouped as malignant in the binary task."),
    ("nv", "Melanocytic nevi, HAM10000's largest diagnostic class, grouped as benign in the binary task."),
    ("SHAP", "Shapley additive explanations, a post-hoc explainability method, grounded in cooperative game theory, used alongside Grad-CAM in this dissertation."),
    ("vasc", "Vascular lesions, one of HAM10000's seven diagnostic classes, grouped as benign in the binary task."),
    ("XAI", "Explainable artificial intelligence, the general term for methods that make a model's predictions interpretable to a human reader."),
]


def set_default_style(doc):
    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = FONT_SIZE
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), FONT_NAME)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(10)

    for heading_name, size in [("Heading 1", 18), ("Heading 2", 15), ("Heading 3", 13), ("Heading 4", 12.5)]:
        style = doc.styles[heading_name]
        style.font.name = FONT_NAME
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(14)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    for section in doc.sections:
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)


def add_centered_line(doc, text, size=13, bold=False, italic=False, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = BLACK
    strip_theme_color(run)
    return p


def build_title_page(doc):
    for line in ["University of South Wales", "Prifysgol De Cymru", "Faculty of Computing, Engineering and Science"]:
        add_centered_line(doc, line, size=13)
    add_centered_line(doc, "", size=13)
    add_centered_line(doc, "MSc Project", size=14, bold=True)
    add_centered_line(doc, "", size=13)
    add_centered_line(doc, "", size=13)
    add_centered_line(doc, TITLE, size=16, bold=True, space_after=20)
    add_centered_line(doc, "", size=13)
    add_centered_line(doc, STUDENT_NAME, size=13)
    add_centered_line(doc, f"Student Number: {STUDENT_NUMBER}", size=13)
    add_centered_line(doc, "", size=13)
    add_centered_line(doc, "", size=13)
    add_centered_line(doc, f"Supervisor: {FIRST_SUPERVISOR}", size=13)
    add_centered_line(doc, f"Year of Study: {YEAR_OF_STUDY}", size=13)
    add_centered_line(doc, f"Scheme: {SCHEME}", size=13)


def build_declaration_page(doc):
    for line in ["University of South Wales", "Prifysgol De Cymru", "Faculty of Computing, Engineering and Science"]:
        add_centered_line(doc, line, size=13)
    add_centered_line(doc, "", size=13)
    add_centered_line(doc, "STATEMENT OF ORIGINALITY", size=14, bold=True)
    add_centered_line(doc, "", size=13)

    body = doc.add_paragraph()
    body.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    body.paragraph_format.space_after = Pt(24)
    run = body.add_run(
        "This is to certify that, except where specific reference is made, the work described in "
        "this project is the result of the investigation carried out by the student, and that "
        "neither this project nor any part of it has been presented, or is currently being "
        "submitted in candidature for any award other than in part for the MSc award, Faculty of "
        "Computing, Engineering and Science from the University of South Wales."
    )
    run.font.name = FONT_NAME
    run.font.size = FONT_SIZE

    sig = doc.add_paragraph()
    sig.paragraph_format.space_after = Pt(4)
    sig_run = sig.add_run("Signed: " + "_" * 45)
    sig_run.font.name = FONT_NAME
    sig_run.font.size = FONT_SIZE

    name_p = doc.add_paragraph()
    name_p.paragraph_format.space_after = Pt(4)
    name_run = name_p.add_run(f"{STUDENT_NAME} (Student)")
    name_run.font.name = FONT_NAME
    name_run.font.size = FONT_SIZE

    date_p = doc.add_paragraph()
    date_run = date_p.add_run("Date: " + "_" * 45)
    date_run.font.name = FONT_NAME
    date_run.font.size = FONT_SIZE


def add_page_break(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_break_type = None
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)


def add_field(paragraph, field_code, placeholder_text):
    r1 = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    r1._r.append(fld_begin)

    r2 = paragraph.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field_code
    r2._r.append(instr)

    r3 = paragraph.add_run()
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    r3._r.append(fld_sep)

    r4 = paragraph.add_run()
    r4.font.name = FONT_NAME
    r4.font.size = FONT_SIZE
    r4.italic = True
    r4.text = placeholder_text

    r5 = paragraph.add_run()
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    r5._r.append(fld_end)


def insert_page_break_before_first_paragraph(doc):
    """Forces the appended chapter to start on a fresh page regardless of
    how the previous section ended."""
    first_p = doc.paragraphs[0]._p
    new_p = OxmlElement("w:p")
    first_p.addprevious(new_p)
    new_run = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    new_run.append(br)
    new_p.append(new_run)


def build_front_matter():
    doc = Document()
    set_default_style(doc)

    # Title page and Statement of Originality, per Handbook Appendices B and C.
    build_title_page(doc)
    add_page_break(doc)
    build_declaration_page(doc)
    add_page_break(doc)

    # Abstract -- single-spaced, indented on both sides plus a first-line
    # indent, distinct from the 1.5-spaced, non-indented body.
    add_heading_black(doc, "Abstract", 1)
    abstract_text = (ROOT / "abstract.md").read_text(encoding="utf-8").strip()
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.right_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(1.27)
    run = p.add_run(abstract_text)
    run.font.name = FONT_NAME
    run.font.size = FONT_SIZE
    add_page_break(doc)

    # Acknowledgements -- placeholder for the student to fill in.
    add_heading_black(doc, "Acknowledgements", 1)
    ack = doc.add_paragraph()
    ack_run = ack.add_run("[Write your acknowledgements here.]")
    ack_run.font.name = FONT_NAME
    ack_run.font.size = FONT_SIZE
    ack_run.italic = True
    add_page_break(doc)

    # Table of Contents -- live field, driven by the Heading styles already
    # used consistently throughout every chapter. Populated with real page
    # numbers by the Word COM update pass that runs after this script.
    add_heading_black(doc, "Table of Contents", 1)
    toc_p = doc.add_paragraph()
    add_field(toc_p, 'TOC \\o "1-4" \\h \\z \\u', "")
    add_page_break(doc)

    # List of Tables -- live field, driven by the "Table Caption" paragraph
    # style applied to every "Table X.X: ..." caption (build_docx.py). Word's
    # \c switch needs a real SEQ field, which plain styled text does not
    # have, so this uses the style-based \t switch instead.
    add_heading_black(doc, "List of Tables", 1)
    lot_p = doc.add_paragraph()
    add_field(lot_p, 'TOC \\t "Table Caption,1" \\h \\z', "")
    add_page_break(doc)

    # List of Figures -- live field, driven by the "Figure Caption" paragraph
    # style applied to every "Figure X.X: ..." caption (build_docx.py).
    add_heading_black(doc, "List of Figures", 1)
    lof_p = doc.add_paragraph()
    add_field(lof_p, 'TOC \\t "Figure Caption,1" \\h \\z', "")
    add_page_break(doc)

    # Glossary.
    add_heading_black(doc, "Glossary", 1)
    for term, definition in GLOSSARY:
        gp = doc.add_paragraph()
        gp.paragraph_format.left_indent = Cm(1.0)
        gp.paragraph_format.first_line_indent = Cm(-1.0)
        gp.paragraph_format.space_after = Pt(8)
        gp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        term_run = gp.add_run(f"{term}: ")
        term_run.font.name = FONT_NAME
        term_run.font.size = FONT_SIZE
        term_run.bold = True
        def_run = gp.add_run(definition)
        def_run.font.name = FONT_NAME
        def_run.font.size = FONT_SIZE

    return doc


def add_page_numbers(path):
    """Adds a centred page-number field to the footer of every section in
    the final merged file. Done as a separate pass on the saved file,
    rather than on the front-matter doc before merging, because each
    appended chapter's own trailing sectPr can become an internal section
    break during the docxcompose merge -- applying the footer per-section
    afterwards guarantees every resulting section gets one, regardless of
    how many section breaks the merge produced."""
    doc = Document(path)
    for section in doc.sections:
        section.footer.is_linked_to_previous = False
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        add_field(p, "PAGE", "")
        for run in p.runs:
            run.font.name = FONT_NAME
            run.font.size = Pt(10)
            run.italic = False
    doc.save(path)


def build():
    master = build_front_matter()
    composer = Composer(master)

    for filename in CHAPTER_FILES:
        sub_doc = Document(ROOT / filename)
        insert_page_break_before_first_paragraph(sub_doc)
        composer.append(sub_doc)

    composer.save(OUT)
    add_page_numbers(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
