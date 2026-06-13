#!/usr/bin/env python3
"""Process and merge supplementary OWID datasets for robustness checks.

Collects 3 supplementary datasets:
1. Polity V democracy index (alternative democracy measure)
2. Palma ratio or Gini coefficient (alternative inequality measure)  
3. Oil rents as % of GDP (to create oil exporter indicator)

Merges with primary dataset from artifact 1.
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
    
    # Create temp directory for dataset files
    temp_dir = Path("temp/tables")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading supplementary datasets...")
    
    # ============================================================================
    # 1. Load Polity V dataset (already downloaded)
    # ============================================================================
    polity_path = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables/full_garden_democracy_2024-03-07_polity_polity.json")
    
    if not polity_path.exists():
        logger.error(f"Polity V dataset not found at {polity_path}")
        logger.info("Please run the OWID download script first to download the Polity V dataset")
        sys.exit(1)
    
    logger.info(f"Loading Polity V data from {polity_path}")
    polity_data = json.load(open(polity_path))
    polity_df = pd.DataFrame(polity_data)
    logger.info(f"Polity V data shape: {polity_df.shape}")
    logger.info(f"Polity V columns: {polity_df.columns.tolist()}")
    
    # Filter to 1990-2022
    polity_df = polity_df[(polity_df['year'] >= 1990) & (polity_df['year'] <= 2022)].copy()
    logger.info(f"Polity V data after filtering to 1990-2022: {polity_df.shape}")
    
    # Extract key Polity V variables
    # democracy_polity = Polity V democracy score (-10 to +10)
    # regime_polity = Political regime classification (0=autocracy, 1=anocracy, 2=democracy)
    polity_subset = polity_df[['country', 'year', 'democracy_polity', 'regime_polity']].copy()
    polity_subset = polity_subset.rename(columns={
        'democracy_polity': 'polity_v_democracy',
        'regime_polity': 'polity_v_regime'
    })
    logger.info(f"Polity V subset shape: {polity_subset.shape}")
    logger.info(f"Polity V democracy range: [{polity_subset['polity_v_democracy'].min()}, {polity_subset['polity_v_democracy'].max()}]")
    logger.info(f"Polity V regime values: {polity_subset['polity_v_regime'].value_counts().to_dict()}")
    
    # ============================================================================
    # 2. Load Oil rents dataset (from WDI - already downloaded)
    # ============================================================================
    wdi_path = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables/full_meadow_worldbank_wdi_2025-09-08_wdi_wdi.json")
    
    if not wdi_path.exists():
        logger.error(f"WDI dataset not found at {wdi_path}")
        logger.info("Please run the OWID download script first to download the WDI dataset")
        sys.exit(1)
    
    logger.info(f"Loading WDI data from {wdi_path}")
    wdi_data = json.load(open(wdi_path))
    wdi_df = pd.DataFrame(wdi_data)
    logger.info(f"WDI data shape: {wdi_df.shape}")
    
    # Extract oil rents column (NY.GDP.PETR.RT.ZS -> ny_gdp_petr_rt_zs)
    oil_col = 'ny_gdp_petr_rt_zs'
    if oil_col not in wdi_df.columns:
        logger.error(f"Oil rents column '{oil_col}' not found in WDI dataset")
        logger.info(f"Available columns: {wdi_df.columns.tolist()[:50]}")
        sys.exit(1)
    
    # Filter to 1990-2022
    wdi_df = wdi_df[(wdi_df['year'] >= 1990) & (wdi_df['year'] <= 2022)].copy()
    
    # Extract oil rents
    oil_subset = wdi_df[['country', 'year', oil_col]].copy()
    oil_subset = oil_subset.rename(columns={oil_col: 'oil_rents_pct_gdp'})
    logger.info(f"Oil rents subset shape: {oil_subset.shape}")
    logger.info(f"Oil rents range: [{oil_subset['oil_rents_pct_gdp'].min()}, {oil_subset['oil_rents_pct_gdp'].max()}]")
    
    # Create oil exporter binary indicator (oil_rents_pct_gdp > 20 = 1, else 0)
    oil_subset['oil_exporter'] = (oil_subset['oil_rents_pct_gdp'] > 20).astype(int)
    logger.info(f"Oil exporter count: {(oil_subset['oil_exporter'] == 1).sum()} country-years")
    
    # ============================================================================
    # 3. Load Palma ratio or Gini coefficient dataset
    # ============================================================================
    # Try Palma ratio first, then fall back to Gini
    palma_path = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables/full_garden_wb_2024-01-17_world_bank_pip_income_consumption_2017_palma_ratio.json")
    gini_path = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables/full_garden_wb_2024-01-17_world_bank_pip_income_consumption_2017_gini.json")
    
    inequality_df = None
    inequality_source = None
    
    # The "spell" columns actually contain the inequality values
    # consumption_spell_* contains consumption-based inequality estimates
    # For each row, find the first non-null spell value
    
    if gini_path.exists():
        logger.info(f"Loading Gini coefficient data from {gini_path}")
        gini_data = json.load(open(gini_path))
        inequality_df = pd.DataFrame(gini_data)
        inequality_source = 'gini'
        logger.info(f"Gini data shape: {inequality_df.shape}")
    elif palma_path.exists():
        logger.info(f"Loading Palma ratio data from {palma_path}")
        palma_data = json.load(open(palma_path))
        inequality_df = pd.DataFrame(palma_data)
        inequality_source = 'palma'
        logger.info(f"Palma ratio data shape: {inequality_df.shape}")
    else:
        logger.error("Neither Gini nor Palma ratio dataset found")
        sys.exit(1)
    
    # Filter to 1990-2022
    inequality_df = inequality_df[(inequality_df['year'] >= 1990) & (inequality_df['year'] <= 2022)].copy()
    
    # Extract inequality value from spell columns
    # Find the first non-null value across consumption_spell_* columns
    spell_cols = [col for col in inequality_df.columns if 'consumption_spell' in col]
    logger.info(f"Inequality spell columns: {spell_cols}")
    
    # Create a single inequality column by taking the first non-null value
    inequality_df['inequality_value'] = inequality_df[spell_cols].apply(
        lambda row: next((val for val in row if pd.notna(val)), np.nan),
        axis=1
    )
    
    # Create subset with country, year, and inequality value
    inequality_subset = inequality_df[['country', 'year', 'inequality_value']].copy()
    
    if inequality_source == 'gini':
        inequality_subset = inequality_subset.rename(columns={'inequality_value': 'gini_coeff_alt'})
        logger.info(f"Gini coefficient range: [{inequality_subset['gini_coeff_alt'].min()}, {inequality_subset['gini_coeff_alt'].max()}]")
    else:
        inequality_subset = inequality_subset.rename(columns={'inequality_value': 'palma_ratio'})
        logger.info(f"Palma ratio range: [{inequality_subset['palma_ratio'].min()}, {inequality_subset['palma_ratio'].max()}]")
    
    logger.info(f"Inequality subset shape: {inequality_subset.shape}")
    
    # ============================================================================
    # 4. Load primary dataset from artifact 1 (if available)
    # ============================================================================
    primary_dataset_path = Path("../gen_art_dataset_1/output/full_dataset.json")
    
    if primary_dataset_path.exists():
        logger.info(f"Loading primary dataset from {primary_dataset_path}")
        primary_data = json.load(open(primary_dataset_path))
        primary_df = pd.DataFrame(primary_data['examples'])
        logger.info(f"Primary dataset shape: {primary_df.shape}")
        logger.info(f"Primary dataset columns: {primary_df.columns.tolist()}")
        
        # Get list of countries in primary dataset
        target_countries = primary_df['country'].unique()
        logger.info(f"Target countries ({len(target_countries)}): {sorted(target_countries)[:10]}...")
        
        # Filter supplementary datasets to target countries
        polity_subset = polity_subset[polity_subset['country'].isin(target_countries)].copy()
        oil_subset = oil_subset[oil_subset['country'].isin(target_countries)].copy()
        inequality_subset = inequality_subset[inequality_subset['country'].isin(target_countries)].copy()
        
        logger.info(f"After filtering to target countries:")
        logger.info(f"  Polity V: {polity_subset.shape}")
        logger.info(f"  Oil rents: {oil_subset.shape}")
        logger.info(f"  Inequality: {inequality_subset.shape}")
        
        # Merge supplementary datasets with primary dataset
        logger.info("Merging supplementary datasets with primary dataset...")
        
        merged_df = primary_df.copy()
        
        # Merge Polity V
        pre_merge = merged_df.shape[0]
        merged_df = merged_df.merge(polity_subset, on=['country', 'year'], how='left')
        logger.info(f"After merging Polity V: {merged_df.shape} (matched {merged_df['polity_v_democracy'].notna().sum()}/{pre_merge} rows)")
        
        # Merge oil rents
        pre_merge = merged_df.shape[0]
        merged_df = merged_df.merge(oil_subset, on=['country', 'year'], how='left')
        logger.info(f"After merging oil rents: {merged_df.shape} (matched {merged_df['oil_rents_pct_gdp'].notna().sum()}/{pre_merge} rows)")
        
        # Merge inequality
        pre_merge = merged_df.shape[0]
        merged_df = merged_df.merge(inequality_subset, on=['country', 'year'], how='left')
        logger.info(f"After merging inequality: {merged_df.shape} (matched {merged_df[inequality_subset.columns[-1]].notna().sum()}/{pre_merge} rows)")
        
    else:
        logger.warning(f"Primary dataset not found at {primary_dataset_path}")
        logger.info("Creating standalone supplementary dataset...")
        
        # Merge all supplementary datasets together
        merged_df = polity_subset.copy()
        merged_df = merged_df.merge(oil_subset, on=['country', 'year'], how='outer')
        merged_df = merged_df.merge(inequality_subset, on=['country', 'year'], how='outer')
        
        logger.info(f"Standalone merged dataset shape: {merged_df.shape}")
    
    # ============================================================================
    # 5. Validate merged dataset
    # ============================================================================
    logger.info("Validating merged dataset...")
    
    # Check for duplicates
    duplicates = merged_df.duplicated(['country', 'year']).sum() if 'country' in merged_df.columns else 0
    logger.info(f"Duplicate country-year pairs: {duplicates}")
    
    # Report missing values
    logger.info("Missing values:")
    for col in merged_df.columns:
        if col not in ['country', 'year']:
            missing_pct = merged_df[col].isna().mean() * 100
            if missing_pct > 0:
                logger.info(f"  {col}: {missing_pct:.1f}% missing")
    
    # Validate data ranges for key variables
    if 'polity_v_democracy' in merged_df.columns:
        logger.info(f"Polity V democracy range: [{merged_df['polity_v_democracy'].min()}, {merged_df['polity_v_democracy'].max()}] (expected: -10 to +10)")
    
    if 'oil_rents_pct_gdp' in merged_df.columns:
        logger.info(f"Oil rents % GDP range: [{merged_df['oil_rents_pct_gdp'].min()}, {merged_df['oil_rents_pct_gdp'].max()}] (expected: 0-100)")
    
    if 'oil_exporter' in merged_df.columns:
        logger.info(f"Oil exporter distribution: {merged_df['oil_exporter'].value_counts().to_dict()}")
    
    # ============================================================================
    # 6. Save output files
    # ============================================================================
    logger.info("Saving output files...")
    
    # Convert to JSON format
    output_data = {
        "examples": merged_df.to_dict(orient='records')
    }
    
    # Add metadata
    output_data["metadata"] = {
        "description": "Supplementary datasets for inequality-democratic resilience analysis",
        "sources": {
            "polity_v": "Polity5 Project, Political Regime Characteristics and Transitions, 1800-2018 (OWID: garden/democracy/2024-03-07/polity/polity)",
            "oil_rents": "World Development Indicators (WDI) - Oil rents (% of GDP) (OWID: meadow/worldbank_wdi/2025-09-08/wdi/wdi)",
            "inequality": f"World Bank Poverty and Inequality Platform (PIP) - {'Gini coefficient' if inequality_source == 'gini' else 'Palma ratio'} (OWID: garden/wb/2024-01-17/world_bank_pip/income_consumption_2017_{inequality_source})"
        },
        "variables": {
            "polity_v_democracy": "Polity V democracy score (-10 to +10)",
            "polity_v_regime": "Polity V regime classification (0=autocracy, 1=anocracy, 2=democracy)",
            "oil_rents_pct_gdp": "Oil rents as percentage of GDP (0-100)",
            "oil_exporter": "Binary indicator for oil exporter (1 if oil_rents_pct_gdp > 20, else 0)",
            "gini_coeff_alt": "Gini coefficient (0-1) - alternative source" if inequality_source == 'gini' else None,
            "palma_ratio": "Palma ratio (ratio of top 10% to bottom 40% income share)" if inequality_source == 'palma' else None
        }
    }
    
    # Remove None values from variables
    output_data["metadata"]["variables"] = {k: v for k, v in output_data["metadata"]["variables"].items() if v is not None}
    
    # Full dataset
    full_path = output_dir / "full_supplementary_data.json"
    full_path.write_text(json.dumps(output_data, indent=2))
    logger.info(f"Saved full dataset: {full_path} ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Mini dataset (3 representative rows)
    mini_df = merged_df.head(3) if len(merged_df) >= 3 else merged_df
    mini_data = {
        "examples": mini_df.to_dict(orient='records'),
        "metadata": output_data["metadata"]
    }
    mini_path = output_dir / "mini_supplementary_data.json"
    mini_path.write_text(json.dumps(mini_data, indent=2))
    logger.info(f"Saved mini dataset: {mini_path}")
    
    # Preview dataset (3 rows with truncated values)
    preview_df = merged_df.head(3) if len(merged_df) >= 3 else merged_df
    # Truncate long string values
    preview_examples = []
    for row in preview_df.to_dict(orient='records'):
        preview_row = {}
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 50:
                preview_row[k] = v[:50] + "..."
            else:
                preview_row[k] = v
        preview_examples.append(preview_row)
    
    preview_data = {
        "examples": preview_examples,
        "metadata": output_data["metadata"]
    }
    preview_path = output_dir / "preview_supplementary_data.json"
    preview_path.write_text(json.dumps(preview_data, indent=2))
    logger.info(f"Saved preview dataset: {preview_path}")
    
    # Create data dictionary
    data_dict_path = output_dir / "data_dictionary.md"
    data_dict_content = f"""# Data Dictionary: Supplementary Datasets for Inequality-Democratic Resilience Analysis

## Dataset Description
Supplementary panel dataset for robustness checks in the inequality-democratic resilience analysis.
Provides alternative measures for democracy (Polity V) and inequality (Gini/Palma), plus oil exporter indicator.

## Variables

| Variable | Description | Range/Values | Source |
|----------|-------------|--------------|--------|
| country | Country name | String | Various |
| year | Year | 1990-2022 | Various |
| polity_v_democracy | Polity V democracy score | -10 to +10 (continuous) | Polity5 Project (OWID) |
| polity_v_regime | Polity V regime classification | 0-2 (ordinal: 0=autocracy, 1=anocracy, 2=democracy) | Polity5 Project (OWID) |
| oil_rents_pct_gdp | Oil rents as % of GDP | 0-100 (continuous) | World Bank WDI (OWID) |
| oil_exporter | Binary oil exporter indicator | 0-1 (binary: 1 if oil_rents_pct_gdp > 20) | Constructed |
{"| gini_coeff_alt | Gini coefficient (alternative source) | 0-1 (continuous) | World Bank PIP (OWID) |" if inequality_source == 'gini' else ""}
{"| palma_ratio | Palma ratio (top 10% / bottom 40% income share) | >0 (continuous) | World Bank PIP (OWID) |" if inequality_source == 'palma' else ""}

## Sources
- Polity V: Polity5 Project, Political Regime Characteristics and Transitions, 1800-2018. OWID: garden/democracy/2024-03-07/polity/polity
- Oil rents: World Development Indicators (WDI), indicator NY.GDP.PETR.RT.ZS. OWID: meadow/worldbank_wdi/2025-09-08/wdi/wdi
- {"Gini coefficient" if inequality_source == 'gini' else "Palma ratio"}: World Bank Poverty and Inequality Platform (PIP). OWID: garden/wb/2024-01-17/world_bank_pip/income_consumption_2017_{inequality_source}

## Sample Size
- Countries: {merged_df['country'].nunique() if 'country' in merged_df.columns else 'N/A'}
- Country-years: {len(merged_df)}

## Missing Values
Reported in validation section above.

## Notes
- Polity V democracy score ranges from -10 (strongly autocratic) to +10 (strongly democratic)
- Oil exporter threshold (>20% GDP from oil rents) follows literature convention
- {"Gini coefficient from World Bank PIP may differ from OECD IDD used in primary dataset" if inequality_source == 'gini' else "Palma ratio is alternatively used inequality measure"}
"""

    data_dict_path.write_text(data_dict_content)
    logger.info(f"Saved data dictionary: {data_dict_path}")
    
    logger.info(f"Supplementary dataset creation complete: {merged_df['country'].nunique() if 'country' in merged_df.columns else 'N/A'} countries, {len(merged_df)} country-year observations")
    
    if 'country' in merged_df.columns:
        logger.info(f"Countries in final dataset: {sorted(merged_df['country'].unique())}")


if __name__ == "__main__":
    main()
