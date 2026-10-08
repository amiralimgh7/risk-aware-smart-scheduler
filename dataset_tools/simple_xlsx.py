"""Dataset serialization and memory-safe spreadsheet helpers.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Mapping, Sequence
from xml.sax.saxutils import escape

INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _clean_text(value: object) -> str:
    """Converts a value to Excel-safe text without presentation rounding."""

    if value is None:
        return ""
    text = str(value)
    return INVALID_XML_RE.sub("", text)


def _column_name(index: int) -> str:
    """Returns Excel column name for a zero-based index."""

    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell_xml(row_index: int, column_index: int, value: object) -> str:
    """Builds one inline-string cell."""

    reference = f"{_column_name(column_index)}{row_index}"
    text = escape(_clean_text(value))
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _worksheet_xml(headers: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    """Builds a minimal valid XLSX worksheet."""

    max_widths = [max(8, min(38, len(str(header)) + 2)) for header in headers]
    row_xml = []
    header_cells = [_cell_xml(1, column_index, header) for column_index, header in enumerate(headers)]
    row_xml.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_number, row in enumerate(rows, start=2):
        cells = []
        for column_index, header in enumerate(headers):
            value = row.get(header, "")
            max_widths[column_index] = max(max_widths[column_index], min(38, len(str(value)) + 2))
            cells.append(_cell_xml(row_number, column_index, value))
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
        for i, width in enumerate(max_widths)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<cols>{cols}</cols><sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def save_rows_as_xlsx(rows: Sequence[Mapping[str, object]], output_path: str | Path, sheet_name: str = "Data") -> None:
    """Writes flat row dictionaries to a minimal XLSX workbook with one sheet.

    The implementation intentionally has no external dependency, so project outputs
    remain reproducible on clean Python environments that only install the listed
    project requirements.
    """

    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    safe_sheet_name = _clean_text(sheet_name)[:31] or "Data"
    worksheet = _worksheet_xml(headers, rows)
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(safe_sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/styles.xml", styles)
