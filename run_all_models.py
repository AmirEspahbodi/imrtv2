import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

import openpyxl
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_workbook(path: Path) -> Workbook:
    logging.info(f"Loading workbook from {path}")
    return openpyxl.load_workbook(path)


def get_headers(sheet: Worksheet) -> List[str]:
    headers = [cell.value for cell in sheet[1]]
    logging.debug(f"Headers found: {headers}")
    return headers  # type: ignore


def process_rows(
    sheet: Worksheet, headers: List[str]
) -> List[Dict[str, Any]]:
    processed_data: List[Dict[str, Any]] = []
    acc_index = headers.index("acc")

    for row in sheet.iter_rows(min_row=2):
        # Build a dict of the current row
        row_data: Dict[str, Any] = {header: row[idx].value for idx, header in enumerate(headers)}

        # Assign a random accuracy
        random_acc = round(random.uniform(0, 1), 4)
        row_data["acc"] = random_acc
        # Write it back
        row[acc_index].value = random_acc

        logging.info(f"Row {row[0].row} data: {row_data}")
        processed_data.append(row_data)

    return processed_data


def save_workbook(workbook: Workbook, path: Path) -> None:
    logging.info(f"Saving updated workbook to {path}")
    workbook.save(path)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Process an Excel file, read rows into dicts, assign random accuracy and save results."
    )
    parser.add_argument(
        "input", type=Path, help="Path to the input Excel file (.xlsx)"
    )

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output or args.input

    wb = load_workbook(input_path)
    ws = wb.active

    headers = get_headers(ws)
    if "acc" not in headers:
        logging.error("Header 'acc' not found in the Excel sheet.")
        return

    process_rows(ws, headers)
    save_workbook(wb, output_path)


if __name__ == "__main__":
    main()
