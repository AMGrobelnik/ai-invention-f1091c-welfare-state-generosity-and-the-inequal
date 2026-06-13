#!/usr/bin/env python3
"""Process and merge OWID datasets for welfare state-democratic resilience analysis."""

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
    
    logger.info("Loading datasets...")
    
    # Load V-Dem data (has electoral democracy index)
    vdem_path = Path("temp/tables/full_garden_democracy_2024-03-07_vdem_vdem_multi_with_regions.json")
    logger.info(f"Loading V-Dem data from {vdem_path}")
    vdem_data = json.loads(vdem_path.read_text())
    vdem_df = pd.DataFrame(vdem_data)
    logger.info(f"V-Dem data shape: {vdem_df.shape}")
    
    # Filter to best estimate only and 1990-2022 period
    vdem_df = vdem_df[vdem_df['estimate'] == 'best'].copy()
    vdem_df = vdem_df[(vdem_df['year'] >= 1990) & (vdem_df['year'] <= 2022)].copy()
    logger.info(f"V-Dem data after filtering (best estimate, 1990-2022): {vdem_df.shape}")
    
    # Extract electoral democracy index
    vdem_subset = vdem_df[['country', 'year', 'electdem_vdem']].copy()
    vdem_subset = vdem_subset.rename(columns={'electdem_vdem': 'vdem_index'})
    logger.info(f"V-Dem subset shape: {vdem_subset.shape}")
    
    # Load V-Dem regime data (has political regime classification)
    regime_path = Path("temp/tables/full_garden_democracy_2024-03-07_vdem_vdem.json")
    logger.info(f"Loading V-Dem regime data from {regime_path}")
    regime_data = json.loads(regime_path.read_text())
    regime_df = pd.DataFrame(regime_data)
    logger.info(f"V-Dem regime data shape: {regime_df.shape}")
    logger.info(f"Regime columns: {regime_df.columns.tolist()}")
    
    # Filter to 1990-2022
    regime_df = regime_df[(regime_df['year'] >= 1990) & (regime_df['year'] <= 2022)].copy()
    
    # Extract regime column (regime_redux_row_owid: 0=closed autocracy, 1=electoral autocracy, 2=electoral democracy, 3=liberal democracy)
    if 'regime_redux_row_owid' in regime_df.columns:
        regime_subset = regime_df[['country', 'year', 'regime_redux_row_owid']].copy()
        regime_subset = regime_subset.rename(columns={'regime_redux_row_owid': 'vdem_regime'})
    elif 'regime_row_owid' in regime_df.columns:
        regime_subset = regime_df[['country', 'year', 'regime_row_owid']].copy()
        regime_subset = regime_subset.rename(columns={'regime_row_owid': 'vdem_regime'})
    else:
        logger.error(f"Regime column not found. Available columns: {regime_df.columns.tolist()}")
        raise ValueError("Regime column not found in V-Dem data")
    
    logger.info(f"Regime subset shape: {regime_subset.shape}")
    logger.info(f"V-Dem regime values: {regime_subset['vdem_regime'].value_counts().to_dict()}")
    
    # Merge regime into V-Dem subset
    vdem_subset = vdem_subset.merge(regime_subset, on=['country', 'year'], how='left')
    logger.info(f"V-Dem data with regime: {vdem_subset.shape}")
    
    # Load Gini coefficient data (UNU-WIDER WID - comprehensive coverage)
    gini_path = Path("temp/tables/full_garden_unu_wider_2024-04-22_world_income_inequality_database_world_income_inequa.json")
    logger.info(f"Loading Gini data from {gini_path}")
    gini_data = json.loads(gini_path.read_text())
    gini_df = pd.DataFrame(gini_data)
    logger.info(f"Gini data shape: {gini_df.shape}")
    logger.info(f"Gini columns: {gini_df.columns.tolist()}")
    
    # Filter Gini to 1990-2022
    gini_df = gini_df[(gini_df['year'] >= 1990) & (gini_df['year'] <= 2022)].copy()
    # Gini is in 'gini' column
    gini_subset = gini_df[['country', 'year', 'gini']].copy()
    gini_subset = gini_subset.rename(columns={'gini': 'gini_coeff'})
    gini_subset = gini_subset.dropna(subset=['gini_coeff'])
    logger.info(f"Gini subset shape (after dropping NAs): {gini_subset.shape}")
    logger.info(f"Gini countries: {gini_subset['country'].nunique()}")
    
    # Load Social Expenditure data (OECD SOCX)
    socx_path = Path("temp/tables/full_garden_oecd_2025-02-25_social_expenditure_social_expenditure.json")
    logger.info(f"Loading social expenditure data from {socx_path}")
    socx_data = json.loads(socx_path.read_text())
    socx_df = pd.DataFrame(socx_data)
    logger.info(f"Social expenditure data shape: {socx_df.shape}")
    logger.info(f"Social expenditure columns: {socx_df.columns.tolist()}")
    logger.info(f"Social expenditure types: {socx_df['programme_type_category'].unique()}")
    
    # Filter to All programs, Public expenditure, 1990-2022
    socx_filtered = socx_df[
        (socx_df['programme_type_category'] == 'All') & 
        (socx_df['expenditure_source'] == 'Public') &
        (socx_df['year'] >= 1990) & (socx_df['year'] <= 2022)
    ].copy()
    logger.info(f"Social expenditure filtered shape: {socx_filtered.shape}")
    
    socx_subset = socx_filtered[['country', 'year', 'share_gdp']].copy()
    socx_subset = socx_subset.rename(columns={'share_gdp': 'social_spending_gdp'})
    logger.info(f"Social expenditure subset shape: {socx_subset.shape}")
    
    # Load Education data (World Bank EdStats - broader coverage)
    edu_path = Path("temp/tables/full_garden_wb_2024-11-04_edstats_edstats.json")
    logger.info(f"Loading education data from {edu_path}")
    edu_data = json.loads(edu_path.read_text())
    edu_df = pd.DataFrame(edu_data)
    logger.info(f"Education data shape: {edu_df.shape}")
    logger.info(f"Education columns: {edu_df.columns.tolist()}")
    
    # Filter to 1990-2022
    edu_df = edu_df[(edu_df['year'] >= 1990) & (edu_df['year'] <= 2022)].copy()
    
    # Use secondary enrollment - check available columns
    # Look for secondary enrollment columns
    secondary_cols = [c for c in edu_df.columns if 'secondary' in c.lower() and ('enrol' in c.lower() or 'net' in c.lower() or 'gross' in c.lower())]
    logger.info(f"Secondary enrollment columns available: {secondary_cols[:5]}")
    
    # Use total_net_enrolment_rate__secondary if available, otherwise first available
    if 'total_net_enrolment_rate__secondary__both_sexes__pct' in edu_df.columns:
        edu_col = 'total_net_enrolment_rate__secondary__both_sexes__pct'
    elif len(secondary_cols) > 0:
        edu_col = secondary_cols[0]
    else:
        # Fallback to Lee-Lee dataset
        logger.warning("No secondary enrollment in EdStats, using Lee-Lee dataset")
        edu_path = Path("temp/tables/full_garden_education_2023-07-17_education_lee_lee_education_lee_lee.json")
        edu_data = json.loads(edu_path.read_text())
        edu_df = pd.DataFrame(edu_data)
        edu_df = edu_df[(edu_df['year'] >= 1990) & (edu_df['year'] <= 2022)].copy()
        edu_col = 'mf_secondary_enrollment_rates_combined_wb'
    
    edu_subset = edu_df[['country', 'year', edu_col]].copy()
    edu_subset = edu_subset.rename(columns={edu_col: 'secondary_edu_enrollment'})
    logger.info(f"Education subset shape: {edu_subset.shape}")
    
    # Identify post-1990 democratizers
    logger.info("Identifying post-1990 democratizers...")
    
    # A country is a post-1990 democratizer if it transitioned from vdem_regime < 2 to >= 2 between 1990-2022
    # Regime codes: 0=closed autocracy, 1=electoral autocracy, 2=electoral democracy, 3=liberal democracy
    
    # Get regime data for all years
    regime_data = vdem_subset[['country', 'year', 'vdem_regime']].copy()
    
    # Find countries that have at least one year with regime >= 2 (democracy)
    democratic_countries = regime_data[regime_data['vdem_regime'] >= 2]['country'].unique()
    logger.info(f"Countries with democratic regime (>=2) 1990-2022: {len(democratic_countries)}")
    
    # For each democratic country, check if it transitioned from <2 to >=2 after 1990
    post1990_democratizers = []
    for country in democratic_countries:
        country_data = regime_data[regime_data['country'] == country].sort_values('year')
        
        # Check if country had regime < 2 before transitioning to >= 2
        transition_year = None
        for _, row in country_data.iterrows():
            if row['vdem_regime'] >= 2:
                transition_year = row['year']
                break
        
        if transition_year and transition_year >= 1990:
            # Check that before transition, regime was < 2
            pre_transition = country_data[country_data['year'] < transition_year]
            if len(pre_transition) > 0 and (pre_transition['vdem_regime'] < 2).any():
                post1990_democratizers.append(country)
                logger.debug(f"Post-1990 democratizer: {country} (transition year: {transition_year})")
    
    logger.info(f"Post-1990 democratizers identified: {len(post1990_democratizers)}")
    logger.info(f"Post-1990 democratizers: {post1990_democratizers[:10]}...")  # Show first 10
    
    # Filter V-Dem data to post-1990 democratizers
    vdem_subset = vdem_subset[vdem_subset['country'].isin(post1990_democratizers)].copy()
    logger.info(f"V-Dem data after filtering to post-1990 democratizers: {vdem_subset.shape}")
    
    # Merge all datasets by country-year with left joins to preserve all V-Dem observations
    logger.info("Merging datasets...")
    
    # Start with V-Dem as base (post-1990 democratizers, 1990-2022)
    merged_df = vdem_subset.copy()
    logger.info(f"Base V-Dem dataset: {merged_df.shape}")
    
    # Merge Gini (left join to keep all V-Dem observations)
    pre_merge_shape = merged_df.shape
    merged_df = merged_df.merge(gini_subset, on=['country', 'year'], how='left')
    logger.info(f"After merging Gini: {merged_df.shape} (was {pre_merge_shape})")
    logger.info(f"Gini missing values: {merged_df['gini_coeff'].isna().sum()} ({merged_df['gini_coeff'].isna().mean()*100:.1f}%)")
    
    # Merge social expenditure (left join)
    pre_merge_shape = merged_df.shape
    merged_df = merged_df.merge(socx_subset, on=['country', 'year'], how='left')
    logger.info(f"After merging social expenditure: {merged_df.shape} (was {pre_merge_shape})")
    logger.info(f"Social spending missing values: {merged_df['social_spending_gdp'].isna().sum()} ({merged_df['social_spending_gdp'].isna().mean()*100:.1f}%)")
    
    # Merge education (left join)
    pre_merge_shape = merged_df.shape
    merged_df = merged_df.merge(edu_subset, on=['country', 'year'], how='left')
    logger.info(f"After merging education: {merged_df.shape} (was {pre_merge_shape})")
    logger.info(f"Education missing values: {merged_df['secondary_edu_enrollment'].isna().sum()} ({merged_df['secondary_edu_enrollment'].isna().mean()*100:.1f}%)")
    
    # Create lagged social spending variable (t-1)
    logger.info("Creating lagged social spending variable...")
    merged_df = merged_df.sort_values(['country', 'year'])
    merged_df['social_spending_gdp_lag1'] = merged_df.groupby('country')['social_spending_gdp'].shift(1)
    logger.info(f"Lagged social spending missing values: {merged_df['social_spending_gdp_lag1'].isna().sum()} ({merged_df['social_spending_gdp_lag1'].isna().mean()*100:.1f}%)")
    
    # Handle missing values - DON'T do listwise deletion, keep observations with vdem_index
    logger.info("Handling missing values...")
    
    # Only drop rows where dependent variable (vdem_index) is missing
    pre_shape = merged_df.shape
    merged_df = merged_df.dropna(subset=['vdem_index'])
    logger.info(f"After dropping NAs in vdem_index: {merged_df.shape} (was {pre_shape})")
    
    # For other variables, interpolate small gaps within each country
    # Use a simpler approach to avoid pandas groupby apply issues
    numeric_cols = ['gini_coeff', 'social_spending_gdp', 'secondary_edu_enrollment']
    for col in numeric_cols:
        if col in merged_df.columns:
            # Interpolate per country
            merged_df = merged_df.groupby('country')[col].apply(
                lambda x: x.interpolate(limit=2).bfill().ffill()
            ).reset_index(level=0, drop=True)
            # Assign back carefully
            merged_df[col] = merged_df.groupby('country')[col].apply(
                lambda x: x.interpolate(limit=2).bfill().ffill()
            ).values
    
    # Report missing values after interpolation
    logger.info("Missing values after processing:")
    for col in ['vdem_index', 'gini_coeff', 'social_spending_gdp', 'secondary_edu_enrollment', 'social_spending_gdp_lag1']:
        if col in merged_df.columns:
            missing_pct = merged_df[col].isna().mean() * 100
            logger.info(f"  {col}: {missing_pct:.1f}% missing ({merged_df[col].notna().sum()} non-missing)")
    
    # Check for duplicates
    duplicates = merged_df.duplicated(['country', 'year']).sum()
    logger.info(f"Duplicate country-year pairs: {duplicates}")
    if duplicates > 0:
        merged_df = merged_df.drop_duplicates(['country', 'year'], keep='first')
        logger.info(f"Dropped duplicates, new shape: {merged_df.shape}")
    
    # Validate data ranges
    logger.info("Validating data ranges...")
    logger.info(f"V-Dem index range: [{merged_df['vdem_index'].min():.3f}, {merged_df['vdem_index'].max():.3f}] (expected: 0-1)")
    logger.info(f"Gini coefficient range: [{merged_df['gini_coeff'].min():.3f}, {merged_df['gini_coeff'].max():.3f}] (expected: 0-1)")
    logger.info(f"Social spending % GDP range: [{merged_df['social_spending_gdp'].min():.3f}, {merged_df['social_spending_gdp'].max():.3f}] (expected: 0-100)")
    logger.info(f"Secondary edu enrollment range: [{merged_df['secondary_edu_enrollment'].min():.3f}, {merged_df['secondary_edu_enrollment'].max():.3f}]")
    logger.info(f"V-Dem regime values: {merged_df['vdem_regime'].value_counts().to_dict()}")
    
    # Check for duplicates
    duplicates = merged_df.duplicated(['country', 'year']).sum()
    logger.info(f"Duplicate country-year pairs: {duplicates}")
    
    # Save output files
    logger.info("Saving output files...")
    
    # Convert to JSON format
    output_data = {
        "examples": merged_df.to_dict(orient='records')
    }
    
    # Full dataset
    full_path = output_dir / "full_dataset.json"
    full_path.write_text(json.dumps(output_data, indent=2))
    logger.info(f"Saved full dataset: {full_path} ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Mini dataset (10% random sample)
    mini_df = merged_df.sample(frac=0.1, random_state=42)
    mini_data = {
        "examples": mini_df.to_dict(orient='records')
    }
    mini_path = output_dir / "mini_dataset.json"
    mini_path.write_text(json.dumps(mini_data, indent=2))
    logger.info(f"Saved mini dataset: {mini_path} ({mini_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Preview dataset (first 50 rows)
    preview_df = merged_df.head(50)
    preview_data = {
        "examples": preview_df.to_dict(orient='records')
    }
    preview_path = output_dir / "preview_dataset.json"
    preview_path.write_text(json.dumps(preview_data, indent=2))
    logger.info(f"Saved preview dataset: {preview_path} ({preview_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Create data dictionary
    data_dict_path = output_dir / "data_dictionary.md"
    data_dict_content = """# Data Dictionary: Welfare State-Democratic Resilience Analysis

## Dataset Description
Panel dataset of post-1990 democratizer countries (1990-2022) for analyzing the relationship between welfare state spending and democratic resilience.

## Variables

| Variable | Description | Range/Values |
|----------|-------------|--------------|
| country | Country name | String |
| year | Year | 1990-2022 |
| vdem_index | V-Dem Electoral Democracy Index | 0-1 (continuous) |
| gini_coeff | Gini coefficient (income inequality) | 0-1 (continuous) |
| social_spending_gdp | Public social expenditure as % of GDP | 0-100 (continuous) |
| secondary_edu_enrollment | Secondary education enrollment rate (both sexes, %) | 0-100 (continuous) |
| vdem_regime | V-Dem political regime classification | 0-3 (ordinal: 0=closed autocracy, 1=electoral autocracy, 2=electoral democracy, 3=liberal democracy) |
| social_spending_gdp_lag1 | Lagged social spending (t-1 year) | 0-100 (continuous) |

## Sources
- V-Dem Electoral Democracy Index and Regime Classification: Varieties of Democracy (V-Dem) Project, OWID garden/democracy/2024-03-07/vdem/vdem_multi_with_regions
- Gini Coefficient: OECD Income Distribution Database (IDD), OWID garden/oecd/2023-06-06/income_distribution_database
- Social Spending: OECD Social Expenditure Database (SOCX), OWID garden/oecd/2025-02-25/social_expenditure
- Secondary Education Enrollment: Lee-Lee Human Capital Dataset, OWID garden/education/2023-07-17/education_lee_lee

## Sample Size
- Countries: POST-1990 DEMOCRATIZERS COUNT
- Country-years: FINAL ROW COUNT

## Missing Values
Handled via linear interpolation (1-2 year gaps) and listwise deletion.

## Notes
- Post-1990 democratizers defined as countries transitioning from V-Dem regime <2 to >=2 between 1990-2022
- Lagged social spending variable created for exogeneity in regression analysis
- Only "best" V-Dem estimates included
"""
    
    # Update data dictionary with actual counts
    n_countries = merged_df['country'].nunique()
    n_rows = len(merged_df)
    data_dict_content = data_dict_content.replace("POST-1990 DEMOCRATIZERS COUNT", str(n_countries))
    data_dict_content = data_dict_content.replace("FINAL ROW COUNT", str(n_rows))
    
    data_dict_path.write_text(data_dict_content)
    logger.info(f"Saved data dictionary: {data_dict_path}")
    
    logger.info(f"Dataset creation complete: {n_countries} countries, {n_rows} country-year observations")
    logger.info(f"Countries in final dataset: {sorted(merged_df['country'].unique())}")

if __name__ == "__main__":
    main()
