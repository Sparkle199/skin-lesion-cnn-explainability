"""Builds abstract.docx: 11pt Times New Roman, single line spacing (not the
1.5 used elsewhere), full paragraph indented left and right, no heading
numbering, matching standard dissertation abstract convention."""

from pathlib import Path
from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Pt, Cm

ROOT = Path(__file__).parent
TEXT = (ROOT / "abstract.md").read_text(encoding="utf-8").strip()
OUT = ROOT / "abstract.docx"

FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(12)

doc = Document()

normal = doc.styles["Normal"]
normal.font.name = FONT_NAME
normal.font.size = FONT_SIZE

heading = doc.styles["Heading 1"]
heading.font.name = FONT_NAME
heading.font.size = Pt(18)
heading.font.bold = True
heading.paragraph_format.space_after = Pt(14)

for section in doc.sections:
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

doc.add_heading("Abstract", level=1)

p = doc.add_paragraph()
p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
p.paragraph_format.left_indent = Cm(1.27)
p.paragraph_format.right_indent = Cm(1.27)
run = p.add_run(TEXT)
run.font.name = FONT_NAME
run.font.size = FONT_SIZE

word_count = len(TEXT.split())
note = doc.add_paragraph()
note.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
note.paragraph_format.space_before = Pt(10)
note_run = note.add_run(f"({word_count} words)")
note_run.font.name = FONT_NAME
note_run.font.size = Pt(9)
note_run.italic = True

doc.save(OUT)
print(f"Wrote {OUT} ({word_count} words)")
