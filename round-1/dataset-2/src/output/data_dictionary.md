# Data Dictionary: Supplementary Datasets for Inequality-Democratic Resilience Analysis

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
| gini_coeff_alt | Gini coefficient (alternative source) | 0-1 (continuous) | World Bank PIP (OWID) |


## Sources
- Polity V: Polity5 Project, Political Regime Characteristics and Transitions, 1800-2018. OWID: garden/democracy/2024-03-07/polity/polity
- Oil rents: World Development Indicators (WDI), indicator NY.GDP.PETR.RT.ZS. OWID: meadow/worldbank_wdi/2025-09-08/wdi/wdi
- Gini coefficient: World Bank Poverty and Inequality Platform (PIP). OWID: garden/wb/2024-01-17/world_bank_pip/income_consumption_2017_gini

## Sample Size
- Countries: 1
- Country-years: 93

## Missing Values
Reported in validation section above.

## Notes
- Polity V democracy score ranges from -10 (strongly autocratic) to +10 (strongly democratic)
- Oil exporter threshold (>20% GDP from oil rents) follows literature convention
- Gini coefficient from World Bank PIP may differ from OECD IDD used in primary dataset
