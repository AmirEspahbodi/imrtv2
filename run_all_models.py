import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

import openpyxl
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet




def load_workbook(path: Path) -> Workbook:
    print(f"Loading workbook from {path}")
    return openpyxl.load_workbook(path)


def get_headers(sheet: Worksheet) -> List[str]:
    headers = [cell.value for cell in sheet[1]]
    print(f"Headers found: {headers}")
    return headers  # type: ignore


def process_rows(
    sheet: Worksheet,
    headers: List[str],
    workbook: Workbook,
    path: Path,
) -> None:
    acc_index = headers.index("acc")

    for row in sheet.iter_rows(min_row=2):
        # Map row values to header keys
        row_data: Dict[str, Any] = {header: row[idx].value for idx, header in enumerate(headers)}
        print(f"Row {row[0].row}")

        # Generate and assign a random accuracy
        random_acc = round(random.uniform(0, 1), 4)
        row_data["acc"] = random_acc
        row[acc_index].value = random_acc

        # Log and save immediately on change
        workbook.save(path)
        print(f"Workbook saved after updating row {row[0].row}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Process an Excel file: assign random accuracy per row and save changes immediately."
    )
    parser.add_argument(
        "input", type=Path, help="Path to the Excel file (.xlsx) to process"
    )
    args = parser.parse_args()
    input_path = args.input

    wb = load_workbook(input_path)
    ws = wb.active

    headers = get_headers(ws)
    if "acc" not in headers:
        print("Header 'acc' not found in the Excel sheet.")
        return

    process_rows(ws, headers, wb, input_path)


if __name__ == "__main__":
    main()