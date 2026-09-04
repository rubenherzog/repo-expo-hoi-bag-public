"""Nature-style Source Data export for figures.

Why this file exists: Nature requires a Source Data file per figure, holding
the exact values plotted in each panel -- not the upstream analysis tables the
panel was derived from. A reviewer must be able to open one workbook and
reproduce every mark in the figure. The sensitivity figures previously shipped
only their heavy upstream CSVs, which contain far more rows than are drawn and
no statement of which rows became which panel.

Contract (one workbook per figure, one sheet per panel):

    <figure_stem>_source_data.xlsx
        [README]      figure, panel, n, test, and a column glossary
        [<panel_id>]  exactly the values drawn in that panel
    <figure_stem>_source_data_<panel_id>.csv   flat mirror of each sheet

The workbook is the paste-ready artifact for the released figure; CSV mirrors
are retained for transparent inspection.

p-values are written as the exact float from disk. Nothing here rounds, floors,
or caps a p-value. Sheet
names are capped at 31 characters by the xlsx format itself, which is a naming
constraint only and never truncates data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Excel hard limit on sheet-name length; not a data constraint.
_SHEET_NAME_MAX = 31
_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")


@dataclass
class Panel:
    """One panel's plotted values plus the provenance a reviewer needs.

    ``panel_id``  short label matching the panel as it appears in the figure.
    ``frame``     exactly the values drawn -- no upstream rows that are not plotted.
    ``description`` what the panel shows in the figure.
    ``columns``   column -> meaning, unit, and estimator where one applies.
    ``test``      the test producing any p-value in the panel, incl. sidedness.
    ``notes``     anything a reader could otherwise get wrong (pool definition,
                  cap, which bundle the numbers came from).
    """

    panel_id: str
    frame: pd.DataFrame
    description: str
    columns: dict[str, str] = field(default_factory=dict)
    test: str = ""
    notes: str = ""

    def sheet_name(self) -> str:
        name = _SHEET_FORBIDDEN.sub("_", str(self.panel_id).strip()) or "panel"
        return name[:_SHEET_NAME_MAX]


def _readme_frame(figure: str, panels: list[Panel], source_paths: list[str]) -> pd.DataFrame:
    """The README sheet: one row per panel, plus one row per documented column.

    Kept as a long table rather than free text so it survives a CSV round-trip
    and can be filtered in Excel.
    """
    rows: list[dict[str, object]] = []
    for panel in panels:
        rows.append(
            {
                "figure": figure,
                "panel": panel.panel_id,
                "sheet": panel.sheet_name(),
                "item": "description",
                "value": panel.description,
            }
        )
        rows.append(
            {
                "figure": figure,
                "panel": panel.panel_id,
                "sheet": panel.sheet_name(),
                "item": "n_rows_plotted",
                "value": int(len(panel.frame)),
            }
        )
        if panel.test:
            rows.append(
                {
                    "figure": figure,
                    "panel": panel.panel_id,
                    "sheet": panel.sheet_name(),
                    "item": "statistical_test",
                    "value": panel.test,
                }
            )
        if panel.notes:
            rows.append(
                {
                    "figure": figure,
                    "panel": panel.panel_id,
                    "sheet": panel.sheet_name(),
                    "item": "notes",
                    "value": panel.notes,
                }
            )
        for column, meaning in panel.columns.items():
            rows.append(
                {
                    "figure": figure,
                    "panel": panel.panel_id,
                    "sheet": panel.sheet_name(),
                    "item": f"column: {column}",
                    "value": meaning,
                }
            )
    for path in source_paths:
        rows.append(
            {
                "figure": figure,
                "panel": "",
                "sheet": "",
                "item": "generated_from",
                "value": path,
            }
        )
    return pd.DataFrame(rows, columns=["figure", "panel", "sheet", "item", "value"])


def write_source_data(
    figure_stem: str,
    panels: list[Panel],
    output_dir: Path | str,
    *,
    source_paths: list[str] | None = None,
) -> list[Path]:
    """Write ``<figure_stem>_source_data.xlsx`` plus per-panel CSV mirrors.

    Files land in a ``source_data/`` subfolder of ``output_dir`` so the figure
    folder holds only the rendered figures: one workbook plus one CSV per panel
    would otherwise bury the pdf/svg/png a reader is looking for.

    Returns every path written, workbook first. Panels with an empty frame are
    skipped rather than written as an empty sheet, so a sheet's presence always
    means the panel was drawn.
    """
    output_dir = Path(output_dir) / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    drawn = [panel for panel in panels if panel.frame is not None and not panel.frame.empty]
    if not drawn:
        return []

    # A truncated sheet name silently drops the part of the panel id that
    # distinguishes level or candidate family, so surface it instead of shipping
    # two sheets a reader cannot tell apart.
    overlong = [p.panel_id for p in drawn if len(_SHEET_FORBIDDEN.sub("_", str(p.panel_id).strip())) > _SHEET_NAME_MAX]
    if overlong:
        raise ValueError(
            f"Source Data panel id(s) for {figure_stem} exceed the {_SHEET_NAME_MAX}-char xlsx "
            f"sheet-name limit and would be truncated: {overlong}. Shorten the panel_id."
        )

    duplicates = {p.sheet_name() for p in drawn if [q.sheet_name() for q in drawn].count(p.sheet_name()) > 1}
    if duplicates:
        raise ValueError(
            f"Source Data panels for {figure_stem} collide on sheet name(s): {sorted(duplicates)}; "
            "give each panel a distinct panel_id."
        )

    written: list[Path] = []
    workbook = output_dir / f"{figure_stem}_source_data.xlsx"
    readme = _readme_frame(figure_stem, drawn, source_paths or [])
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        for panel in drawn:
            panel.frame.to_excel(writer, sheet_name=panel.sheet_name(), index=False)
    written.append(workbook)

    readme_csv = output_dir / f"{figure_stem}_source_data_README.csv"
    expected_csvs = {readme_csv}
    expected_csvs.update(
        output_dir / f"{figure_stem}_source_data_{panel.sheet_name()}.csv"
        for panel in drawn
    )
    # A panel can be removed or renamed between figure revisions. Delete only
    # stale CSV mirrors belonging to this exact figure stem so Source Data never
    # retains sheets from an earlier visual contract.
    for stale in output_dir.glob(f"{figure_stem}_source_data_*.csv"):
        if stale not in expected_csvs:
            stale.unlink()
    readme.to_csv(readme_csv, index=False)
    written.append(readme_csv)
    for panel in drawn:
        csv_path = output_dir / f"{figure_stem}_source_data_{panel.sheet_name()}.csv"
        panel.frame.to_csv(csv_path, index=False)
        written.append(csv_path)

    return written
