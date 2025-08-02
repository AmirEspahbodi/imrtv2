import argparse
import random
import json
from pathlib import Path
from typing import List, Dict, Any

import openpyxl
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
import subprocess
import sys


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
    metrics = ["acc", "f1", "auc", "precision", "recall"]
    metrics_indexes = {i: headers.index(i) for i in metrics}
    resutl_path = 'resutl_path'
    resutl_path_index = headers.index(resutl_path)
    # TIP
    # sync with hydra config
    save_result = "/content/output/result/"

    for row in sheet.iter_rows(min_row=2):
        # check if this model and configuration isn't runned

        
        # Map row values to header keys
        row_data: Dict[str, Any] = {header: row[idx].value for idx, header in enumerate(headers)}
        model = row_data["model"]
        backbone_trainable_layers = [int(i) for i in row_data['backbone_trainable_layers'].strip().split(" ")] if isinstance(row_data['backbone_trainable_layers'], list) else row_data['backbone_trainable_layers']
        vit1_feature_strame = [int(i) for i in row_data['vit1_feature_strame'].strip().split(" ")] if isinstance(row_data['vit1_feature_strame'], list) else row_data['vit1_feature_strame'] 
        vit2_feature_strame = [int(i) for i in row_data['vit2_feature_strame'].strip().split(" ")] if isinstance(row_data['vit2_feature_strame'], list) else row_data['vit2_feature_strame']
        training_plan = row_data["training_plan"].strip()
        model_id = f"btl{''.join(backbone_trainable_layers)}_v1fs{''.join(vit1_feature_strame)}_v2fs{''.join(vit2_feature_strame)}_tp{training_plan}"

        resutl_path_value = row[resutl_path_index].value
        if resutl_path_value is not None:
            print(f"{row[0].row}==> this model is calculated {model}{model_id} resutl path is {resutl_path_value} <==")
            continue
        try:
            result = subprocess.run(
                f"{sys.executable} main.py network={model} --btl {''.join(backbone_trainable_layers)}  --v1fs {''.join(vit1_feature_strame)} --v2fs {''.join(vit2_feature_strame)} --tp {training_plan}",
                capture_output=True,
                text=True,
                check=True,
            )
            success = (result.returncode == 0)
            print("── STDOUT ─────────────────────────────────────────")
            print(result.stdout or "<no output>")
            print("───────────────────────────────────────────────────")
        except subprocess.CalledProcessError as e:
            print(f"main failed (exit code {e.returncode})\n")
            print("── STDOUT ─────────────────────────────────────────")
            print(e.stdout or "<no output>")
            print("── STDERR ─────────────────────────────────────────")
            print(e.stderr or "<no error output>")
            print("───────────────────────────────────────────────────")
            success = False
            
        if success:
            result_directory = f"{save_result}/{model}/{model_id}"
            
            finall_resutl_path = f"{result_directory}/final_weights_result.json"
            validation_resutl_path = f"{result_directory}/best_validation_weights_result.json"
            finall_resutl = json.loads(finall_resutl_path)
            validation_resutl = json.loads(validation_resutl_path)
            result = None
            if finall_resutl['acc'] < validation_resutl["acc"]:
                result = validation_resutl
            else:
                result = finall_resutl
            for metric, value in result.items():
                row[metrics_indexes[metric]].value = float(value.strip())
            
            row[resutl_path_index] = f"{save_result}/{model}/{model_id}"
            
        else:
            for metric in metrics:
                row[metrics_indexes[metric]].value = "FAILD"
                row[resutl_path_index].value



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