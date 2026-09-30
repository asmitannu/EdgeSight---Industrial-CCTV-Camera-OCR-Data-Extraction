"""
Data export module for CSV and Excel output generation.
Saves OCR structured records in standardized formats ready for manual upload
or inspection in Siemens Insights Hub and analytics tools.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataExporter:
    """Handles exporting structured OCR records to CSV and Excel (XLSX) files."""

    COLUMNS = [
        "Image Name",
        "Detected Text",
        "Confidence",
        "Bounding Box",
        "Processing Variant",
        "Timestamp"
    ]

    @classmethod
    def to_dataframe(cls, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts record dictionaries to a pandas DataFrame with standard columns."""
        if not records:
            return pd.DataFrame(columns=cls.COLUMNS)

        df = pd.DataFrame(records)
        # Ensure all required columns exist in proper order
        for col in cls.COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df[cls.COLUMNS]

    @classmethod
    def export_csv(
        cls,
        records: List[Dict[str, Any]],
        output_path: str,
        append: bool = False
    ) -> str:
        """
        Exports records to a CSV file.

        Args:
            records: List of structured dictionary records.
            output_path: Target CSV file path.
            append: If True, appends to existing CSV (without duplicating header).

        Returns:
            str: Absolute path of exported CSV.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = cls.to_dataframe(records)

        write_header = True
        mode = "w"
        if append and os.path.exists(output_path):
            mode = "a"
            write_header = os.path.getsize(output_path) == 0

        df.to_csv(output_path, mode=mode, header=write_header, index=False, encoding="utf-8-sig")
        logger.info(f"Exported {len(df)} records to CSV: {output_path}")
        return os.path.abspath(output_path)

    @classmethod
    def export_excel(
        cls,
        records: List[Dict[str, Any]],
        output_path: str,
        sheet_name: str = "OCR_Results"
    ) -> str:
        """
        Exports records to a formatted Excel (.xlsx) file with styled headers and auto-adjusted columns.

        Args:
            records: List of structured dictionary records.
            output_path: Target Excel file path.
            sheet_name: Worksheet name.

        Returns:
            str: Absolute path of exported Excel file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = cls.to_dataframe(records)

        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Format workbook styling
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]

                # Style header row
                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

                header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
                header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")  # Siemens Navy Blue
                header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

                for cell in worksheet[1]:
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = header_alignment

                # Auto-fit column widths
                for col in worksheet.columns:
                    max_len = 0
                    col_letter = col[0].column_letter
                    for cell in col:
                        val_str = str(cell.value or "")
                        max_len = max(max_len, len(val_str))
                    worksheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 60)

                # Add light border to data cells
                thin_border = Border(
                    left=Side(style="thin", color="D3D3D3"),
                    right=Side(style="thin", color="D3D3D3"),
                    top=Side(style="thin", color="D3D3D3"),
                    bottom=Side(style="thin", color="D3D3D3")
                )
                for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=1, max_col=worksheet.max_column):
                    for cell in row:
                        cell.border = thin_border
                        if cell.column == 3:  # Confidence column
                            cell.number_format = "0.0000"
                            cell.alignment = Alignment(horizontal="right")

            logger.info(f"Exported {len(df)} records to formatted Excel: {output_path}")
        except Exception as e:
            logger.warning(f"openpyxl styling failed ({e}), writing unstyled Excel fallback")
            df.to_excel(output_path, sheet_name=sheet_name, index=False)

        return os.path.abspath(output_path)
