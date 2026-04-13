"""
Data loading and inspection tools for CCAR backtesting datasets.
Supports CSV, Excel, and Parquet (large files).
"""
import os
import pandas as pd
from smolagents import tool

# ── Module-level registry so datasets persist across agent steps ───────────────
_DATASETS: dict[str, pd.DataFrame] = {}


def _store(name: str, df: pd.DataFrame) -> None:
    _DATASETS[name] = df


def get_df(name: str) -> pd.DataFrame:
    """Internal helper: retrieve a stored dataset by name."""
    if name not in _DATASETS:
        raise KeyError(
            f"Dataset '{name}' not found. "
            f"Available: {list(_DATASETS.keys()) or 'none loaded yet'}. "
            "Call load_dataset or inspect_parquet + aggregate_parquet first."
        )
    return _DATASETS[name]


# ─────────────────────────────────────────────────────────────────────────────

@tool
def inspect_parquet(file_path: str) -> str:
    """Inspect a large Parquet file's schema and basic statistics WITHOUT loading it fully into memory.

    Uses Polars lazy scanning — reads only metadata and a 5-row sample.
    Use this as the first step on any large parquet file to understand columns,
    data types, row count, and sample values before deciding how to aggregate.

    Args:
        file_path: Absolute or relative path to a .parquet file.
    """
    try:
        import polars as pl
    except ImportError:
        return "ERROR: polars not installed. Run: uv pip install polars"

    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    try:
        lf = pl.scan_parquet(file_path)
        schema = lf.collect_schema()
        num_rows = lf.select(pl.len()).collect().item()
        sample_df = lf.head(5).collect()
    except Exception as e:
        return f"ERROR reading parquet: {e}"

    col_lines = [
        f"  [{i:02d}] {name}  ({dtype})"
        for i, (name, dtype) in enumerate(schema.items())
    ]

    return (
        f"Parquet file: {file_path}\n"
        f"Rows: {num_rows:,}  |  Columns: {len(schema)}\n\n"
        f"Schema:\n" + "\n".join(col_lines) +
        f"\n\nFirst 5 rows:\n{sample_df.to_pandas().to_string(index=False)}"
    )


@tool
def load_dataset(file_path: str, dataset_name: str = "main", sheet_name: str = "0") -> str:
    """Load a CSV, Excel, or small Parquet file and register it for later use.

    For LARGE parquet files (millions of rows) use inspect_parquet first,
    then aggregate_parquet to load only the aggregated result into memory.

    The file is stored in memory under `dataset_name` so other tools can
    reference it by that name.

    Args:
        file_path: Absolute or relative path to a .csv, .xlsx, .xls, or .parquet file.
        dataset_name: Short name to identify this dataset (e.g. "round1", "round2"). Defaults to "main".
        sheet_name: For Excel files, the sheet to read. Use a sheet name string or "0" for the first sheet. Defaults to "0".
    """
    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            sn: int | str = int(sheet_name) if sheet_name.isdigit() else sheet_name
            df = pd.read_excel(file_path, sheet_name=sn)
        elif ext == ".parquet":
            import polars as pl
            df = pl.read_parquet(file_path).to_pandas()
        else:
            return f"ERROR: Unsupported file type '{ext}'. Use .csv, .xlsx/.xls, or .parquet."
    except Exception as e:
        return f"ERROR loading file: {e}"

    _store(dataset_name, df)

    # Build a concise summary
    num_rows, num_cols = df.shape
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_null = int(df[col].isna().sum())
        col_info.append(f"  - {col} [{dtype}] | {n_unique} unique | {n_null} nulls")

    col_summary = "\n".join(col_info)
    sample = df.head(3).to_string(index=False)

    return (
        f"Dataset '{dataset_name}' loaded successfully.\n"
        f"Shape: {num_rows:,} rows × {num_cols} columns\n\n"
        f"Columns:\n{col_summary}\n\n"
        f"First 3 rows:\n{sample}"
    )


@tool
def get_dataset_info(dataset_name: str = "main") -> str:
    """Return detailed statistics and column info for a loaded dataset.

    Args:
        dataset_name: Name given when the dataset was loaded. Defaults to "main".
    """
    try:
        df = get_df(dataset_name)
    except KeyError as e:
        return f"ERROR: {e}"

    describe = df.describe(include="all").to_string()
    dtypes = df.dtypes.to_string()
    null_counts = df.isna().sum().to_string()

    return (
        f"=== Dataset: '{dataset_name}' ===\n"
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n"
        f"── Column dtypes ──\n{dtypes}\n\n"
        f"── Null counts ──\n{null_counts}\n\n"
        f"── Descriptive statistics ──\n{describe}"
    )


@tool
def list_loaded_datasets() -> str:
    """List all datasets currently loaded in memory with their shapes.

    Args:
    """
    if not _DATASETS:
        return "No datasets loaded yet. Use load_dataset to load one."
    lines = [f"  - '{name}': {df.shape[0]:,} rows × {df.shape[1]} cols"
             for name, df in _DATASETS.items()]
    return "Loaded datasets:\n" + "\n".join(lines)
