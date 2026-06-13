#!/usr/bin/env python3
"""Format supplementary datasets into expected schema.

Expected schema (grouped by dataset):
{
  "datasets": [
    {
      "dataset": "polity_v",
      "examples": [
        {"input": "...", "output": "...", "metadata_*": ...},
        ...
      ]
    },
    ...
  ]
}
"""

from loguru import logger
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


@logger.catch(reraise=True)
def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Loading collected supplementary datasets...")
    
    # ============================================================================
    # Load the 3 collected datasets
    # ============================================================================
    
    # Dataset 1: Polity V (democracy index)
    polity_path = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables/full_garden_democracy_2024-03-07_polity_polity.json")
    logger.info(f"Loading Polity V from {polity_path}")
    polity_data = json.load(open(polity_path))
    polity_df = pd.DataFrame(polity_data)
    polity_df = polity_df[(polity_df['year'] >= 1990) & (polity_df['year'] <= 2022)].copy()
    logger.info(f"Polity V shape: {polity_df.shape}")
    
    # Dataset 2: Oil rents (% of GDP)
    wdi_path = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables/full_meadow_worldbank_wdi_2025-09-08_wdi_wdi.json")
    logger.info(f"Loading WDI (oil rents) from {wdi_path}")
    wdi_data = json.load(open(wdi_path))
    wdi_df = pd.DataFrame(wdi_data)
    wdi_df = wdi_df[(wdi_df['year'] >= 1990) & (wdi_df['year'] <= 2022)].copy()
    logger.info(f"WDI shape: {wdi_df.shape}")
    
    # Extract oil rents column
    oil_col = 'ny_gdp_petr_rt_zs'
    if oil_col not in wdi_df.columns:
        logger.error(f"Oil rents column '{oil_col}' not found")
        sys.exit(1)
    
    # Dataset 3: Gini coefficient (alternative inequality measure)
    gini_path = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables/full_garden_wb_2024-01-17_world_bank_pip_income_consumption_2017_gini.json")
    logger.info(f"Loading Gini from {gini_path}")
    gini_data = json.load(open(gini_path))
    gini_df = pd.DataFrame(gini_data)
    gini_df = gini_df[(gini_df['year'] >= 1990) & (gini_df['year'] <= 2022)].copy()
    logger.info(f"Gini shape: {gini_df.shape}")
    
    # Extract Gini from spell columns
    spell_cols = [col for col in gini_df.columns if 'consumption_spell' in col]
    gini_df['gini_coeff'] = gini_df[spell_cols].apply(
        lambda row: next((val for val in row if pd.notna(val)), np.nan),
        axis=1
    )
    
    # ============================================================================
    # Format each dataset into expected schema
    # ============================================================================
    
    datasets = []
    
    # Dataset 1: Polity V
    logger.info("Formatting Polity V dataset...")
    polity_examples = []
    for _, row in polity_df.iterrows():
        # Input: country-year with regime characteristics
        input_data = {
            'country': row['country'],
            'year': int(row['year']),
            'exec_reccomp_polity': row.get('exec_reccomp_polity'),
            'exec_recopen_polity': row.get('exec_recopen_polity'),
            'exec_constr_polity': row.get('exec_constr_polity'),
            'polpart_reg_polity': row.get('polpart_reg_polity'),
            'polpart_comp_polity': row.get('polpart_comp_polity')
        }
        
        # Output: Polity V democracy score and regime classification
        output_data = {
            'polity_v_democracy': row.get('democracy_polity'),
            'polity_v_regime': row.get('regime_polity')
        }
        
        example = {
            'input': json.dumps(input_data),
            'output': json.dumps(output_data),
            'metadata_dataset': 'polity_v',
            'metadata_source': 'Polity5 Project (OWID: garden/democracy/2024-03-07/polity/polity)',
            'metadata_country': row['country'],
            'metadata_year': int(row['year'])
        }
        polity_examples.append(example)
    
    datasets.append({
        'dataset': 'polity_v',
        'examples': polity_examples
    })
    logger.info(f"Polity V: {len(polity_examples)} examples")
    
    # Dataset 2: Oil rents
    logger.info("Formatting oil rents dataset...")
    oil_examples = []
    for _, row in wdi_df.iterrows():
        if pd.isna(row[oil_col]):
            continue
            
        # Input: country-year
        input_data = {
            'country': row['country'],
            'year': int(row['year'])
        }
        
        # Output: oil rents % GDP and binary exporter indicator
        oil_rents = row[oil_col]
        output_data = {
            'oil_rents_pct_gdp': oil_rents,
            'oil_exporter': 1 if oil_rents > 20 else 0
        }
        
        example = {
            'input': json.dumps(input_data),
            'output': json.dumps(output_data),
            'metadata_dataset': 'oil_rents',
            'metadata_source': 'World Bank WDI (OWID: meadow/worldbank_wdi/2025-09-08/wdi/wdi)',
            'metadata_country': row['country'],
            'metadata_year': int(row['year'])
        }
        oil_examples.append(example)
    
    datasets.append({
        'dataset': 'oil_rents',
        'examples': oil_examples
    })
    logger.info(f"Oil rents: {len(oil_examples)} examples")
    
    # Dataset 3: Gini coefficient
    logger.info("Formatting Gini coefficient dataset...")
    gini_examples = []
    for _, row in gini_df.iterrows():
        if pd.isna(row['gini_coeff']):
            continue
            
        # Input: country-year
        input_data = {
            'country': row['country'],
            'year': int(row['year'])
        }
        
        # Output: Gini coefficient
        output_data = {
            'gini_coeff_alt': row['gini_coeff']
        }
        
        example = {
            'input': json.dumps(input_data),
            'output': json.dumps(output_data),
            'metadata_dataset': 'gini_coeff_alt',
            'metadata_source': 'World Bank PIP (OWID: garden/wb/2024-01-17/world_bank_pip/income_consumption_2017_gini)',
            'metadata_country': row['country'],
            'metadata_year': int(row['year'])
        }
        gini_examples.append(example)
    
    datasets.append({
        'dataset': 'gini_coeff_alt',
        'examples': gini_examples
    })
    logger.info(f"Gini coefficient: {len(gini_examples)} examples")
    
    # ============================================================================
    # Save output in expected schema
    # ============================================================================
    logger.info("Saving output files...")
    
    output_data = {"datasets": datasets}
    
    # Full dataset
    full_path = output_dir / "full_data_out.json"
    full_path.write_text(json.dumps(output_data, indent=2))
    logger.info(f"Saved full_data_out.json: {full_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Count total examples
    total_examples = sum(len(d['examples']) for d in datasets)
    logger.info(f"Total datasets: {len(datasets)}")
    logger.info(f"Total examples: {total_examples}")
    
    # Generate mini version (3 examples per dataset)
    mini_datasets = []
    for dataset in datasets:
        mini_dataset = {
            'dataset': dataset['dataset'],
            'examples': dataset['examples'][:3]
        }
        mini_datasets.append(mini_dataset)
    
    mini_data = {"datasets": mini_datasets}
    mini_path = output_dir / "mini_data_out.json"
    mini_path.write_text(json.dumps(mini_data, indent=2))
    logger.info(f"Saved mini_data_out.json")
    
    # Generate preview version (3 examples per dataset, truncated)
    preview_datasets = []
    for dataset in datasets:
        preview_examples = []
        for example in dataset['examples'][:3]:
            preview_example = {}
            for k, v in example.items():
                if isinstance(v, str) and len(v) > 50:
                    preview_example[k] = v[:50] + "..."
                else:
                    preview_example[k] = v
            preview_examples.append(preview_example)
        
        preview_dataset = {
            'dataset': dataset['dataset'],
            'examples': preview_examples
        }
        preview_datasets.append(preview_dataset)
    
    preview_data = {"datasets": preview_datasets}
    preview_path = output_dir / "preview_data_out.json"
    preview_path.write_text(json.dumps(preview_data, indent=2))
    logger.info(f"Saved preview_data_out.json")
    
    logger.info("Data formatting complete!")


if __name__ == "__main__":
    main()
