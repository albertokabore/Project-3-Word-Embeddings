"""Export REPORT.md to matching Word/PDF reports, including tables and plots.

Run from any directory: python scripts/export_report.py
Optional dependencies: pip install -e ".[report]"
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from markdown_it import MarkdownIt
from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    XPreformatted,
)

ROOT = Path(__file__).resolve().parents[1]


def add_docx_inline(paragraph, children):
    bold = italic = False
    for child in children:
        if child.type == "strong_open":
            bold = True
        elif child.type == "strong_close":
            bold = False
        elif child.type == "em_open":
            italic = True
        elif child.type == "em_close":
            italic = False
        elif child.type in {"text", "code_inline", "softbreak", "hardbreak"}:
            run = paragraph.add_run(" " if child.type.endswith("break") else child.content)
            run.bold, run.italic = bold, italic
            if child.type == "code_inline":
                run.font.name = "Consolas"
                run.font.size = Pt(9)


def pdf_inline(children):
    parts = []
    for child in children:
        if child.type == "text":
            parts.append(escape(child.content))
        elif child.type == "code_inline":
            parts.append(f'<font name="Mono" size="8.5">{escape(child.content)}</font>')
        elif child.type in {"strong_open", "strong_close", "em_open", "em_close"}:
            tag = "b" if child.type.startswith("strong") else "i"
            parts.append(f'</{tag}>' if child.type.endswith("close") else f'<{tag}>')
        elif child.type == "link_open":
            parts.append(f'<link href="{escape(child.attrGet("href"), quote=True)}" color="#17547d">')
        elif child.type == "link_close":
            parts.append("</link>")
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    return "".join(parts)


def export_report(docx_name: str = "REPORT.docx"):
    font_dir = Path(get_data_path()) / "fonts" / "ttf"
    for name, filename in [
        ("Body", "DejaVuSans.ttf"), ("BodyBold", "DejaVuSans-Bold.ttf"),
        ("BodyItalic", "DejaVuSans-Oblique.ttf"),
        ("BodyBoldItalic", "DejaVuSans-BoldOblique.ttf"),
        ("Mono", "DejaVuSansMono.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold",
                                  italic="BodyItalic", boldItalic="BodyBoldItalic")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportBody", fontName="Body", fontSize=9.5,
                              leading=14, spaceAfter=7, splitLongWords=True))
    styles.add(ParagraphStyle("ReportTable", parent=styles["ReportBody"],
                              fontSize=8, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle("ReportCode", fontName="Mono", fontSize=8,
                              leading=10, spaceBefore=5, spaceAfter=9,
                              backColor=colors.HexColor("#f2f4f7")))
    for level, size in [(1, 19), (2, 13), (3, 10.5)]:
        styles.add(ParagraphStyle(f"ReportH{level}", fontName="BodyBold", fontSize=size,
                                  leading=size + 4, spaceBefore=12, spaceAfter=7,
                                  keepWithNext=True, textColor=colors.HexColor("#173753")))
    styles.add(ParagraphStyle("Caption", parent=styles["ReportBody"],
                              alignment=TA_CENTER, fontSize=8.5))

    document = Document()
    section = document.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.7)
    section.left_margin = section.right_margin = Inches(0.7)
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(10)
    document.styles["Normal"].paragraph_format.space_after = Pt(7)
    for level in range(1, 4):
        document.styles[f"Heading {level}"].font.color.rgb = RGBColor.from_string("173753")
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run("Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    tokens = MarkdownIt("commonmark").enable("table").parse(
        (ROOT / "REPORT.md").read_text(encoding="utf-8"))
    story = []
    width = letter[0] - 2 * 0.7 * 72
    list_stack = []
    prefix = ""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type in {"bullet_list_open", "ordered_list_open"}:
            list_stack.append(1 if token.type.startswith("ordered") else None)
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            list_stack.pop()
        elif token.type == "list_item_open":
            counter = list_stack[-1]
            prefix = "• " if counter is None else f"{counter}. "
            if counter is not None:
                list_stack[-1] += 1
        elif token.type == "heading_open":
            level = int(token.tag[1])
            inline = tokens[index + 1]
            paragraph = document.add_heading(level=level)
            add_docx_inline(paragraph, inline.children)
            story.append(Paragraph(pdf_inline(inline.children), styles[f"ReportH{level}"]))
            index += 2
        elif token.type == "paragraph_open":
            inline = tokens[index + 1]
            pictures = [c for c in inline.children if c.type == "image"]
            if pictures:
                for picture in pictures:
                    path = ROOT / picture.attrGet("src")
                    document.add_picture(str(path), width=Inches(6.6))
                    document.add_paragraph(picture.content, style="Caption")
                    img = Image(str(path))
                    scale = min(width / img.imageWidth, 285 / img.imageHeight)
                    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
                    story.extend([img, Paragraph(escape(picture.content), styles["Caption"])])
            else:
                if prefix == "• ":
                    paragraph = document.add_paragraph(style="List Bullet")
                else:
                    paragraph = document.add_paragraph(prefix)
                add_docx_inline(paragraph, inline.children)
                story.append(Paragraph(escape(prefix) + pdf_inline(inline.children), styles["ReportBody"]))
            prefix = ""
            index += 2
        elif token.type == "fence":
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(8)
            run = paragraph.add_run(token.content.rstrip())
            run.font.name, run.font.size = "Consolas", Pt(8.5)
            story.append(XPreformatted(escape(token.content.rstrip()), styles["ReportCode"]))
        elif token.type == "table_open":
            rows = []
            row = []
            while tokens[index].type != "table_close":
                item = tokens[index]
                if item.type == "tr_open":
                    row = []
                elif item.type == "inline":
                    row.append(item.children)
                elif item.type == "tr_close":
                    rows.append(row)
                index += 1
            table = document.add_table(rows=0, cols=len(rows[0]))
            table.style = "Light Shading Accent 1"
            for ri, cells in enumerate(rows):
                cells_docx = table.add_row().cells
                for ci, children in enumerate(cells):
                    add_docx_inline(cells_docx[ci].paragraphs[0], children)
                    for run in cells_docx[ci].paragraphs[0].runs:
                        run.font.size = Pt(9)
                        if ri == 0:
                            run.bold = True
                # Keep each table row together in Word.
                tr_properties = table.rows[-1]._tr.get_or_add_trPr()
                tr_properties.append(OxmlElement("w:cantSplit"))
            # Repeat the header if a Word table spans pages.
            table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            document.add_paragraph().paragraph_format.space_after = Pt(0)
            pdf_rows = [[Paragraph(pdf_inline(c), styles["ReportTable"]) for c in r] for r in rows]
            if len(rows[0]) == 2:
                col_widths = [width * .48, width * .52]
            elif len(rows[0]) == 3:
                col_widths = [width * .25, width * .48, width * .27]
            else:
                col_widths = [width / len(rows[0])] * len(rows[0])
            pdf_table = Table(pdf_rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            pdf_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ebf3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#ccd5dd")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(KeepTogether([pdf_table, Spacer(1, 9)]))
        index += 1

    document.core_properties.title = "Project 3: Word Embeddings"
    document.core_properties.subject = "TF-IDF and covariance-based PCA"
    document.core_properties.author = "Albert Kabore"
    document.save(ROOT / docx_name)

    def page_number(canvas, doc):
        canvas.setFont("Body", 8)
        canvas.drawRightString(letter[0] - 0.7 * 72, 25, f"Page {doc.page}")

    pdf = SimpleDocTemplate(str(ROOT / "REPORT.pdf"), pagesize=letter,
                            rightMargin=.7 * 72, leftMargin=.7 * 72,
                            topMargin=.7 * 72, bottomMargin=.7 * 72,
                            title="Project 3: Word Embeddings", author="Albert Kabore")
    pdf.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"Exported {docx_name} and REPORT.pdf from REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx-name", default="REPORT.docx",
                        help="Word output filename; use a new name if the existing file is open")
    export_report(parser.parse_args().docx_name)
