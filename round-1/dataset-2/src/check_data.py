#!/usr/bin/env python3
"""Check data quality of final supplementary dataset."""
import json
import pandas as pd

data = json.load(open('output/full_supplementary_data.json'))
df = pd.DataFrame(data['examples'])

print('Dataset info:')
print(f"  Rows: {len(df)}")
print(f"  Countries: {df['country'].nunique()}")
print(f"  Years: {df['year'].min()}-{df['year'].max()}")
print()
print('Missing values:')
for col in ['polity_v_democracy', 'polity_v_regime', 'oil_rents_pct_gdp', 'oil_exporter', 'gini_coeff_alt']:
    if col in df.columns:
        missing = df[col].isna().sum()
        pct = missing/len(df)*100
        print(f"  {col}: {missing} ({pct:.1f}%)")
print()
print('Data ranges:')
if 'polity_v_democracy' in df.columns:
    print(f"  Polity V democracy: [{df['polity_v_democracy'].min()}, {df['polity_v_democracy'].max()}]")
if 'oil_rents_pct_gdp' in df.columns:
    print(f"  Oil rents % GDP: [{df['oil_rents_pct_gdp'].min()}, {df['oil_rents_pct_gdp'].max()}]")
if 'gini_coeff_alt' in df.columns:
    print(f"  Gini coefficient: [{df['gini_coeff_alt'].min()}, {df['gini_coeff_alt'].max()}]")
print()
print('Oil exporter distribution:')
if 'oil_exporter' in df.columns:
    print(f"  {df['oil_exporter'].value_counts().to_dict()}")
