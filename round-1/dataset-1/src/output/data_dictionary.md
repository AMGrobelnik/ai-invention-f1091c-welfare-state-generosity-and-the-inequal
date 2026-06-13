# Data Dictionary: Welfare State-Democratic Resilience Analysis
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
- Countries: 55
- Country-years: 5564
