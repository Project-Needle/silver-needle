import pandas as pd
import json
from typing import List, Dict, Optional


def load_and_convert_to_json(
    input_path: str,
    output_path: str,
    max_rows: Optional[int] = None,
    encoding: str = "utf-8"
) -> List[Dict[str, str]]:
    """
    Load CSV or Parquet, convert all values to string, and export to JSON.

    Args:
        input_path: path to csv/parquet file
        output_path: output json file path
        max_rows: only read first N rows (None = all)
        encoding: file encoding for csv/json

    Returns:
        List[Dict[str, str]]
    """

    # =========================
    # 1️⃣ 读取文件
    # =========================
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path, encoding=encoding)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Only support .csv or .parquet")

    # =========================
    # 2️⃣ 截取前 N 行
    # =========================
    if max_rows is not None:
        df = df.head(max_rows)

    # =========================
    # 3️⃣ 全部转 string（处理 NaN）
    # =========================
    df = df.fillna("")  # NaN -> ""
    df = df.astype(str)

    # =========================
    # 4️⃣ 转 List[Dict]
    # =========================
    data: List[Dict[str, str]] = df.to_dict(orient="records")

    # =========================
    # 5️⃣ 写 JSON
    # =========================
    with open(output_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    max_rows = 20
    input_path = "./gpqa/gpqa_main.csv"
    output_path = f"./processed/gpqa-main-first{max_rows}.json"
    load_and_convert_to_json(input_path=input_path, output_path=output_path, max_rows=100)