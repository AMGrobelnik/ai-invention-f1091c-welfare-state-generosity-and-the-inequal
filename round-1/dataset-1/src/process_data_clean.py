#!/usr/bin/env python3
"""Process OWID datasets for welfare state-democratic resilience analysis."""
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
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    logger.info("Loading datasets...")
    
    # Load V-Dem data ( electoral democracy index)
    vdem_path = Path("temp/tables/full_garden_democracy_2024-03-07_vdem_vdem_multi_with_regions.json")
    vdem_data = json.loads(vdem_path.read_text())
    vdem_df = pd.DataFrame(vdem_data)
    vdem_df = vdem_df[vdem_df['estimate'] == 'best'].copy()
    vdem_df = vdem_df[(vdem_df['year'] >= 1990) & (vdem_df['year'] <= 2022)].copy()
    vdem_subset = vdem_df[['country', 'year', 'electdem_vdem']].copy()
    vdem_subset = vdem_subset.rename(columns={'electdem_vdem': 'vdem_index'})
    logger.info(f"V-Dem shape: {vdem_subset.shape}")
    
    # Load regime data
    regime_path = Path("temp/tables/full_garden_democracy_2024-03-07_vdem_vdem.json")
    regime_data = json.loads(regime_path.read_text())
    regime_df = pd.DataFrame(regime_data)
    regime_df = regime_df[(regime_df['year'] >= 1990) & (regime_df['year'] <= 2022)].copy()
    if 'regime_redux_row_owid' in regime_df.columns:
        regime_subset = regime_df[['country', 'year', 'regime_redux_row_owid']].copy()
        regime_subset = regime_subset.rename(columns={'regime_redux_row_owid': 'vdem_regime'})
    else:
        regime_subset = regime_df[['country', 'year', 'regime_row_owid']].copy()
        regime_subset = regime_subset.rename(columns={'regime_row_owid': 'vdem_regime'})
    vdem_subset = vdem_subset.merge(regime_subset, on=['country', 'year'], how='left')
    logger.info(f"With regime: {vdem_subset.shape}")
    
    # Identify post-1990 democratizers
    logger.info("Identifying post-1990 democratizers...")
    regime_data = vdem_subset[['country', 'year', 'vdem_regime']].copy()
    democratic_countries = regime_data[regime_data['vdem_regime'] >= 2]['country'].unique()
    post1990_democratizers = []
    for country in democratic_countries:
        cdata = regime_data[regime_data['country'] == country].sort_values('year')
        transition_year = None
        for _, row in cdata.iterrows():
            if row['vdem_regime'] >= 2:
                transition_year = row['year']
                break
        if transition_year and transition_year >= 1990:
            pre = cdata[cdata['year'] < transition_year]
            if len(pre) > 0 and (pre['vdem_regime'] < 2).any():
                post1990_democratizers.append(country)
    logger.info(f"Post-1990 democratizers: {len(post1990_democratizers)}")
    vdem_subset = vdem_subset[vdem_subset['country'].isin(post1990_democratizers)].copy()
    logger.info(f"Filtered to democratizers: {vdem_subset.shape}")
    
    # Load Gini (UNU-WIDER WIID)
    gini_path = Path("temp/tables/full_garden_unu_wider_2024-04-22_world_income_inequality_database_world_income_inequa.json")
    gini_data = json.loads(gini_path.read_text())
    gini_df = pd.DataFrame(gini_data)
    gini_df = gini_df[(gini_df['year'] >= 1990) & (gini_df['year'] <= 2022)].copy()
    gini_subset = gini_df[['country', 'year', 'gini']].copy()
    gini_subset = gini_subset.rename(columns={'gini': 'gini_coeff'})
    gini_subset = gini_subset.dropna(subset=['gini_coeff'])
    logger.info(f"Gini shape: {gini_subset.shape}")
    
    # Load social spending (OECD SOCX)
    socx_path = Path("temp/tables/full_garden_oecd_2025-02-25_social_expenditure_social_expenditure.json")
    socx_data = json.loads(socx_path.read_text())
    socx_df = pd.DataFrame(socx_data)
    socx_df = socx_df[(socx_df['year'] >= 1990) & (socx_df['year'] <= 2022)].copy()
    socx_filtered = socx_df[(socx_df['programme_type_category'] == 'All') & (socx_df['expenditure_source'] == 'Public')].copy()
    socx_subset = socx_filtered[['country', 'year', 'share_gdp']].copy()
    socx_subset = socx_subset.rename(columns={'share_gdp': 'social_spending_gdp'})
    logger.info(f"Social spending shape: {socx_subset.shape}")
    
    # Load education (World Bank EdStats)
    edu_path = Path("temp/tables/full_garden_wb_2024-11-04_edstats_edstats.json")
    edu_data = json.loads(edu_path.read_text())
    edu_df = pd.DataFrame(edu_data)
    edu_df = edu_df[(edu_df['year'] >= 1990) & (edu_df['year'] <= 2022)].copy()
    # Use net enrollment secondary if available
    if 'total_net_enrolment_rate__secondary__both_sexes__pct' in edu_df.columns:
        edu_col = 'total_net_enrolment_rate__secondary__both_sexes__pct'
    else:
        cols = [c for c in edu_df.columns if 'secondary' in c.lower() and ('enrol' in c.lower() or 'net' in c.lower())]
        edu_col = cols[0] if len(cols) > 0 else None
    if edu_col:
        edu_subset = edu_df[['country', 'year', edu_col]].copy()
        edu_subset = edu_subset.rename(columns={edu_col: 'secondary_edu_enrollment'})
        logger.info(f"Education shape: {edu_subset.shape}")
    else:
        edu_subset = pd.DataFrame(columns=['country', 'year', 'secondary_edu_enrollment'])
        logger.warning("No secondary education column found")
    
    # Merge all datasets
    logger.info("Merging datasets...")
    merged = vdem_subset.copy()
    merged = merged.merge(gini_subset, on=['country', 'year'], how='left')
    logger.info(f"After Gini: {merged.shape}")
    merged = merged.merge(socx_subset, on=['country', 'year'], how='left')
    logger.info(f"After social spending: {merged.shape}")
    merged = merged.merge(edu_subset, on=['country', 'year'], how='left')
    logger.info(f"After education: {merged.shape}")
    
    # Create lagged social spending
    merged = merged.sort_values(['country', 'year'])
    merged['social_spending_gdp_lag1'] = merged.groupby('country')['social_spending_gdp'].shift(1)
    
    # Handle missing values - only drop if vdem_index is missing
    merged = merged.dropna(subset=['vdem_index'])
    logger.info(f"After dropping NA in vdem_index: {merged.shape}")
    
    # Simple interpolation for numeric columns
    for col in ['gini_coeff', 'social_spending_gdp', 'secondary_edu_enrollment']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
    
    # Report missing
    logger.info("Missing values:")
    for col in ['vdem_index', 'gini_coeff', 'social_spending_gdp', 'secondary_edu_enrollment', 'social_spending_gdp_lag1']:
        if col in merged.columns:
            pct = merged[col].isna().mean() * 100
            logger.info(f"  {col}: {pct:.1f}% missing")
    
    # Save output
    logger.info("Saving output...")
    output_data = {"examples": merged.to_dict(orient='records')}
    
    (output_dir / "full_dataset.json").write_text(json.dumps(output_data, indent=2))
    logger.info(f"Saved full: {len(merged)} rows")
    
    # Mini (10% sample)
    mini = merged.sample(frac=0.1, random_state=42) if len(merged) > 10 else merged
    mini_data = {"examples": mini.to_dict(orient='records')}
    (output_dir / "mini_dataset.json").write_text(json.dumps(mini_data, indent=2))
    
    # Preview (first 50)
    prev = merged.head(50)
    prev_data = {"examples": prev.to_dict(orient='records')}
    (output_dir / "preview_dataset.json").write_text(json.dumps(prev_data, indent=2))
    
    # Data dictionary
    dd = f"""# Data Dictionary: Welfare State-Democratic Resilience Analysis
## Dataset Description
Panel dataset of post-1990 democratizer countries (1990-2022).
## Variables
| Variable | Description | Range/Values |
|----------|-------------|--------------|
| country | Country name | String |
| year | Year | 1990-2022 |
| vdem_index | V-Dem Electoral Democracy Index | 0-1 |
| gini_coeff | Gini coefficient | 0-1 |
| social_spending_gdp | Public social expenditure % GDP | 0-100 |
| secondary_edu_enrollment | Secondary enrollment rate | 0-100 |
| vdem_regime | V-Dem regime (0-3) | 0=closed,1=elect.autoc.,2=elect.dem.,3=lib.dem. |
| social_spending_gdp_lag1 | Lagged social spending | 0-100 |
## Sources
- V-Dem: OWID garden/democracy/2024-03-07/vdem
- Gini: UNU-WIDER WIID (garden/unu_wider/2024-04-22)
- Social spending: OECD SOCX (garden/oecd/2025-02-25)
- Education: World Bank EdStats (garden/wb/2024-11-04)
## Sample Size
- Countries: {merged['country'].nunique()}
- Country-years: {len(merged)}
"""
    (output_dir / "data_dictionary.md").write_text(dd)
    logger.info(f"Done: {merged['country'].nunique()} countries, {len(merged)} observations")
    logger.info(f"Countries: {sorted(merged['country'].unique())}")

if __name__ == "__main__":
    main()
