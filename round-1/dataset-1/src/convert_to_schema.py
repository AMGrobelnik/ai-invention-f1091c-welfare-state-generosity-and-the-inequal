#!/usr/bin/env python3
"""Convert merged dataset to exp_sel_data_out.json schema."""
from loguru import logger
from pathlib import Path
import json
import sys
import pandas as pd

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

@logger.catch(reraise=True)
def main():
    # Load current dataset
    src = Path("output/full_dataset.json")
    data = json.loads(src.read_text())
    df = pd.DataFrame(data["examples"])
    
    logger.info(f"Loaded: {len(df)} rows, cols: {df.columns.tolist()}")
    
    # Build examples in exp_sel_data_out schema
    examples = []
    for _, row in df.iterrows():
        # Input: JSON string of features
        inp = {
            "country": row.get("country", ""),
            "year": int(row.get("year", 0)),
            "gini_coeff": float(row.get("gini_coeff", "nan")),
            "social_spending_gdp": float(row.get("social_spending_gdp", "nan")),
            "secondary_edu_enrollment": float(row.get("secondary_edu_enrollment", "nan")),
            "vdem_regime": float(row.get("vdem_regime", "nan")),
            "social_spending_gdp_lag1": float(row.get("social_spending_gdp_lag1", "nan"))
        }
        # Output: vdem_index as string
        out = str(row.get("vdem_index", ""))
        
        ex = {
            "input": json.dumps(inp),
            "output": out,
            "metadata_fold": 0,
            "metadata_task_type": "regression",
            "metadata_n_features": 6
        }
        examples.append(ex)
    
    # Group by dataset
    output = {
        "datasets": [
            {
                "dataset": "owid_welfare_democracy_panel",
                "examples": examples
            }
        ]
    }
    
    # Save
    out_path = Path("full_data_out.json")
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved: {out_path} ({len(examples)} examples)")
    
    # Mini (10% sample)
    mini_examples = pd.DataFrame(examples).sample(frac=0.1, random_state=42).to_dict("records")
    mini_output = {"datasets": [{"dataset": "owid_welfare_democracy_panel", "examples": mini_examples}]}
    mini_path = Path("mini_data_out.json")
    mini_path.write_text(json.dumps(mini_output, indent=2))
    
    # Preview (first 50)
    prev_examples = examples[:50]
    prev_output = {"datasets": [{"dataset": "owid_welfare_democracy_panel", "examples": prev_examples}]}
    prev_path = Path("preview_data_out.json")
    prev_path.write_text(json.dumps(prev_output, indent=2))
    
    logger.info(f"Done: {len(examples)} total examples")

if __name__ == "__main__":
    main()
