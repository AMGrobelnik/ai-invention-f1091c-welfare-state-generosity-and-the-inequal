# Methodological Survey for Panel Regression with Mediation Analysis in Political Science

## Summary

This comprehensive methodological survey provides actionable guidance for conducting fixed-effects panel regression with mediation analysis, bootstrap confidence intervals, and interaction effects in political science research. The report addresses the specific context of studying welfare state mediation of inequality-democracy relationships in post-1990 democratizers using V-Dem data. Key contributions include: (1) Proper specification of interaction effects in fixed-effects models using the double-demeaned estimator to avoid bias from unobserved effect heterogeneity [1]; (2) Modern causal mediation analysis framework using counterfactual approach and bootstrap inference rather than outdated Baron-Kenny method [3]; (3) V-Dem coding rules and identification of post-1990 democratizers; (4) Identification strategy with tests for autocorrelation (Wooldridge test [8]) and model selection (Hausman test [7]); (5) Comprehensive robustness checks including alternative inequality measures, alternative democracy indices, sample restrictions, and alternative lag structures; (6) Software implementation code in R, Stata, and Python. The report includes a preliminary list of 59 post-1990 democratizers with V-Dem country codes, though this requires verification using V-Dem transition variables. Detailed mathematical specifications, code templates, and bibliographic references are provided.

## Research Findings

This methodological survey provides comprehensive guidance for conducting fixed-effects panel regression with mediation analysis and bootstrap confidence intervals in political science research, specifically for studying welfare state mediation of inequality-democracy relationships in post-1990 democratizers.

## Key Methodological Recommendations

### 1. Panel Regression Specification with Interactions

**The Problem with Standard Interaction Specification in FE Models [1]**

Standard practice of demeaning the product term (X_i,t × Z_i,t) does NOT yield a genuine within estimator when both variables vary within units. Instead, it picks up unit-specific effect heterogeneity of both variables. The estimated coefficient reflects both within-unit and between-unit variation, violating the fixed-effects assumption.

**The Solution: Double-Demeaned Estimator [1]**

For interactions with two time-dependent variables, apply the 'double-demeaned' estimator:

1. Demean each variable: ̃X_i,t = X_i,t - ̄X_i and ̃Z_i,t = Z_i,t - ̄Z_i
2. Form the product: ̃X_i,t × ̃Z_i,t
3. Demean the product: (̃X_i,t × ̃Z_i,t) - (̃X × ̃Z)̄_i

This estimator eliminates bias from unobserved effect heterogeneity but is less efficient than standard FE and requires T > 2 [1].

**Mathematical Specification:**

The recommended model is:

Y_i,t = β₀ + β₁Gini_i,t + β₂Welfare_i,t-1 + β₃(Gini × Welfare)_i,t-1 + β₄(Gini × Welfare × Education)_i,t-1 + α_i + γ_t + ε_i,t

Where:
- Y_i,t: V-Dem electoral democracy index (v2x_polyarchy)
- Gini_i,t: Income inequality (Gini coefficient)
- Welfare_i,t-1: Lagged welfare state spending (mediator)
- α_i: Country fixed effects (control for all time-invariant confounders)
- γ_t: Year fixed effects (control for global time trends)
- ε_i,t: Error term with cluster-robust standard errors at country level [9]

### 2. Mediation Analysis Procedure

**Abandon Baron-Kenny Approach [3]**

The traditional Baron-Kenny (1986) approach has three major limitations:
1. Lack of general definition of causal mediation effects independent of specific statistical models
2. Inability to specify key identification assumptions
3. Difficulty extending to nonlinear models

**Adopt Modern Causal Mediation Framework [3]**

Imai, Keele, and Tingley (2010) propose a counterfactual framework:

- Define potential outcomes Y_i(t, m): outcome that would result if treatment = t and mediator = m
- Define causal mediation effect: δ_i(t) = Y_i(t, M_i(1)) - Y_i(t, M_i(0))
- This represents the indirect effect through the mediator

**Sequential Ignorability Assumption [3]**

For nonparametric identification, two assumptions must hold:
1. (Y_i(t', m), M_i(t)) ⊥ T_i | X_i (treatment ignorability)
2. Y_i(t', m) ⊥ M_i(t) | T_i, X_i (mediator ignorability)

WARNING: This assumption is untestable and requires sensitivity analysis.

**Bootstrap Inference for Indirect Effects [6]**

Recommendations:
1. Use bootstrap over Sobel test: Bootstrap performs better in small samples and does not assume normality of indirect effect [6]
2. Number of resamples: 1,000-5,000 (more is better, 1,000 is minimum)
3. Confidence interval: Use percentile CI or bias-corrected CI (bias-corrected is preferred)
4. Cluster bootstrap: Resample countries (clusters) rather than individual observations to preserve within-country dependence

**Four-Step Mediation Procedure:**

Step 1: Total effect (c path)
- Regress Y on X (Gini → V-Dem)
- H0: β_X = 0

Step 2: Effect of X on M (a path)
- Regress M on X (Gini → Welfare)
- H0: β_X = 0

Step 3: Effect of M on Y controlling for X (b path)
- Regress Y on X and M (Welfare → V-Dem | Gini)
- H0: β_M = 0

Step 4: Direct effect (c' path)
- In Step 3, test H0: β'_X = 0
- Indirect effect = a × b

Significance: Indirect effect significant if bootstrap CI excludes zero.

### 3. V-Dem Data and Post-1990 Democratizers

**V-Dem Electoral Democracy Index [2]**

- Variable: v2x_polyarchy
- Construction: Combines indicators of electoral competition, suffrage, clean elections, and elected officials
- Scale: 0 (least democratic) to 1 (most democratic)
- Coding rule: Index ≥ 0.5 indicates minimal democracy (electoral authoritarian or democratic); Index < 0.5 indicates autocratic [2]

**Identifying Post-1990 Democratizers [2]**

Three methods:

Method 1: Using V-Dem transition variable
- Variable: e_democracy_trans
- Value = 1 indicates democratic transition
- Filter for transitions occurring after 1990

Method 2: Analyzing transitions in v2x_polyarchy
- Calculate first difference: ΔPolyarchy_i,t = Polyarchy_i,t - Polyarchy_i,t-1
- Identify transitions: ΔPolyarchy_i,t > 0.2 (threshold adjustable)
- Verify transition occurs after 1990

Method 3: Using external datasets
- Boix-Miller-Rosato dataset (variable e_boix_regime in V-Dem)
- Freedom House transitions
- Polity IV transitions

**Preliminary List of Post-1990 Democratizers (Requires Verification):**

From V-Dem Codebook v14 [2], regions and selected countries:

Eastern Europe/Central Asia (13 countries): Armenia (105), Azerbaijan (106), Belarus (107), Estonia (161), Georgia (118), Kazakhstan (121), Kyrgyzstan (122), Latvia (84), Lithuania (173), Moldova (126), Russia (11), Tajikistan (133), Turkmenistan (136), Ukraine (23), Uzbekistan (140)

Africa (20 countries): Angola (104), Benin (52), Botswana (68), Cape Verde (70), Djibouti (113), Eritrea (115), Ethiopia (38), Ghana (7), Guinea-Bissau (119), Kenya (40), Liberia (86), Madagascar (125), Malawi (87), Mali (28), Mauritius (180), Mozambique (57), Namibia (127), Niger (60), Rwanda (129), Senegal (31), Seychelles (199), Sierra Leone (95), South Africa (8), Tanzania (47), Zambia (61)

Asia (11 countries): Bangladesh (24), Bhutan (53), Cambodia (55), Indonesia (56), Laos (123), Mongolia (89), Myanmar (10), Nepal (58), Philippines (46), South Korea (42), Sri Lanka (131), Taiwan (48), Timor-Leste (74)

Americas (14 countries): Argentina (37), Bolivia (25), Brazil (19), Chile (72), Colombia (15), Ecuador (75), El Salvador (22), Guatemala (78), Honduras (27), Nicaragua (59), Paraguay (189), Peru (30), Uruguay (102), Venezuela (51)

NOTE: This list is preliminary. Not all countries democratized post-1990; some experienced democratic breakdowns or transitions before 1990. Verification requires analyzing V-Dem transition variables.

### 4. Identification Strategy and Assumption Testing

**Lagged Welfare Exogeneity [8]**

Using lagged welfare spending (Welfare_i,t-1) addresses:
- Reverse causality: Democracy may affect welfare spending
- Simultaneity bias

**Test for Exogeneity:**

Durbin-Wu-Hausman Test:
1. Estimate main model with lagged welfare
2. Save residuals ε̂_i,t
3. Regress ε̂_i,t on all regressors including lagged welfare
4. Test significance of lagged welfare coefficient

**Wooldridge Test for Autocorrelation [8]**

- Tests for first-order serial correlation in FE models
- H0: No first-order autocorrelation
- If autocorrelation exists, standard errors are biased
- Stata: xtserial command
- R: pwtest() function in plm package

**Hausman Test for FE vs. RE [7]**

- Tests whether individual characteristics are correlated with regressors
- H0: Random effects model is consistent and efficient
- H1: Random effects model is inconsistent; use fixed effects
- If H0 rejected: Use fixed effects
- If H0 not rejected: Random effects more efficient

**Parallel Trends Assumption**

Fixed-effects models assume that in the absence of treatment (inequality), outcome trends would be parallel across countries.

Assessment strategies:
1. Examine pre-treatment trends graphically
2. Include leads of treatment variable (event-study specification)
3. Test for differential pre-trends

### 5. Interaction Effects Interpretation

**Marginal Effects Calculation [1]**

In a model with interaction X × Z, the marginal effect of X on Y is:

∂Y/∂X = β₁ + β₃Z

This means the effect of inequality (X) on democracy (Y) varies with welfare spending (Z).

**Three-Way Interactions:**

For the model with Gini × Welfare × Education:
- Marginal effect: ∂²Y/(∂Gini∂Welfare) = β₄ + β₇Education
- Interpretation becomes complex; use marginal effects plots
- Hold two variables at specific values while varying the third

**Centering to Reduce Multicollinearity [1]**

Interaction terms are often highly correlated with constitutive terms.

Solution: Center continuous variables before creating interactions:
- X_centered = X - ̄X
- Z_centered = Z - ̄Z
- Interaction = X_centered × Z_centered

This reduces multicollinearity and makes main effects interpretable as marginal effects at the mean.

### 6. Robustness Checks

**Alternative Inequality Measures:**
1. Palma Ratio: Ratio of top 10% income share to bottom 40% income share
2. Top 10% Share: Income share of top 10% of earners (from World Inequality Database)
3. Top 1% Share: Income share of top 1% (more extreme, may show different patterns)
4. Atkinson Index: Inequality measure sensitive to changes at different parts of distribution
5. Theil Index: Entropy-based inequality measure

**Alternative Democracy Indices:**
1. Polity V (from V-Dem): e_v2x_polyarchy_ordinal - ordinal version of V-Dem index
2. Freedom House: Combined score of political rights and civil liberties
3. Lexical Index of Electoral Democracy (LIED): Binary measure of electoral democracy
4. Boix-Miller-Rosato: Binary measure of democracy (variable e_boix_regime in V-Dem)

**Sample Restrictions:**
1. Exclude oil exporters: Oil rents may confound inequality-democracy relationship (use World Bank oil rents data, exclude >20% of GDP)
2. Exclude small island states: May have different political dynamics (exclude population <500,000)
3. Exclude conflict years: Civil war may affect both inequality and democracy (use UCDP/PRIO conflict data, exclude years with armed conflict ≥25 deaths)
4. Exclude post-communist countries: Different initial conditions (create dummy for post-1991 Soviet bloc)

**Alternative Lag Structures:**
1. t-1: One-year lag (baseline)
2. t-2: Two-year lag
3. t-3: Three-year lag
4. t-5: Five-year lag
5. Distributed lag: Welfare_i,t-1 + Welfare_i,t-2 + Welfare_i,t-3

**Subsample Analysis by Region:**

V-Dem regions [2]:
1. Eastern Europe and Central Asia
2. Latin America and the Caribbean
3. Middle East and North Africa
4. Sub-Saharan Africa
5. Western Europe and North America
6. Asia and Pacific

Test whether mediation effects differ across regions using interaction terms with region dummies or separate regressions.

### 7. Software Implementation

**R Code Structure:**
- Load libraries: tidyverse, plm, lmtest, sandwich, mediation, ggeffects, modelsummary
- Set panel structure with pdata.frame()
- Fixed-effects regression with plm(..., model = 'within')
- Cluster-robust SE with vcovHC(..., cluster = 'group')
- Mediation with mediate() function, bootstrap = TRUE, sims = 1000
- Marginal effects with ggeffects::ggpredict()
- Export with modelsummary()

**Stata Do-File Template:**
- Set panel: xtset country year
- FE regression: xtreg y x z c.x#c.z i.year, fe cluster(country)
- Mediation: bootstrap r(indirect) r(direct), reps(1000) cluster(country): medeff
- Marginal effects: margins, dydx(x) at(z = (0(10)100)); marginsplot
- Export: esttab using 'results.rtf', se(r) star(...)

**Python Notebook Outline:**
- Load data with pandas
- Create lags with groupby('country')['variable'].shift(1)
- Panel regression with linearmodels.PanelOLS
- Bootstrap mediation with custom function resampling countries
- Marginal effects plot with matplotlib
- Robustness checks with alternative specifications

### 8. Confidence Assessment and Limitations

**High Confidence (95%+):**
1. Need to use fixed effects for panel data with country-specific unobservables
2. Need to cluster standard errors at country level
3. Bootstrap superior to Sobel test for mediation analysis
4. Double-demeaned estimator corrects bias in FE interaction models [1]

**Medium Confidence (70-90%):**
1. Optimal number of bootstrap resamples (1,000 vs. 5,000)
2. Bias-corrected vs. percentile confidence intervals for indirect effects
3. Whether to use cluster bootstrap or ordinary bootstrap in panel data
4. Lag length selection (t-1 vs. t-2 vs. t-3)

**Low Confidence / Requires Further Investigation:**
1. Proper specification of three-way interactions in FE models
2. Mediation analysis with panel data (Imai et al. 2010 framework developed for cross-section)
3. Whether post-1990 democratizer list is complete and accurate (requires V-Dem transition data verification)
4. Sensitivity to alternative V-Dem coding rules or index constructions

**Major Limitations of This Survey:**
1. Cannot access full V-Dem dataset to verify post-1990 democratizer list
2. Some recommended methods (double-demeaned estimator) have only recent theoretical justification [1]
3. Mediation analysis with panel data remains an open methodological question
4. No simulation evidence provided for bias in specific research design

### 9. Contradicting Evidence and Methodological Debates

**FE vs. RE Debate [7, 15]:**
- Traditional view: Always use FE to avoid bias from time-invariant unobservables
- Recent critique: FE may be inefficient; RE with cluster-robust SE may be preferable [15]
- Bell & Jones (2015) argue FE should NOT be default; careful model specification needed
- My recommendation: Use Hausman test; if H0 not rejected, RE more efficient. But with T small and N large, FE is safer.

**Bootstrap vs. Parametric Mediation [3, 6]:**
- Bootstrap advocates: No distributional assumptions, better small-sample properties [6]
- Parametric advocates: More efficient if distributional assumptions met
- Imai et al. (2010) recommend nonparametric bootstrap for robustness
- My recommendation: Use bootstrap with 1,000+ resamples; conduct sensitivity analysis

**Cluster Selection for Standard Errors [9]:**
- Cluster at highest level of dependence (country)
- But if N_clusters < 50, cluster-robust SE may be biased
- Alternative: Use wild bootstrap or multiplier bootstrap
- My recommendation: Cluster at country level; if N < 50, use wild bootstrap

### 10. Synthesis and Recommendations

**For the Specific Research Question (Welfare State Mediation in Post-1990 Democratizers):**

1. **Data Preparation:**
   - Download V-Dem v14 dataset
   - Identify post-1990 democratizers using e_democracy_trans or transition in v2x_polyarchy
   - Merge with inequality data (Standardized World Income Inequality Database)
   - Merge with welfare state data (SOCX, OECD Social Expenditure Database)
   - Create panel structure with country and year identifiers

2. **Main Analysis:**
   - Estimate FE model with double-demeaned interaction [1]
   - Cluster standard errors at country level
   - Use lagged welfare spending (t-1) as mediator
   - Include country and year fixed effects
   - Test for autocorrelation (Wooldridge test [8])

3. **Mediation Analysis:**
   - Use Imai et al. (2010) framework [3]
   - Bootstrap with 1,000 resamples, cluster bootstrap by country
   - Report indirect effect with 95% bias-corrected CI
   - Conduct sensitivity analysis for sequential ignorability assumption

4. **Robustness Checks (Minimum):**
   - Alternative inequality measure (top 10% share)
   - Alternative democracy index (Polity V)
   - Exclude oil exporters
   - Alternative lag structure (t-2)
   - Region dummies interaction

5. **Reporting:**
   - Report coefficient, SE, p-value, and 95% CI for all effects
   - Report marginal effects of interaction at meaningful values of moderator
   - Provide marginal effects plot with confidence bands
   - Discuss identification assumptions and threats to validity
   - Make replication data and code publicly available

**Expected Challenges:**
1. Small N (possibly 50-80 countries) may limit statistical power
2. Missing data on inequality (SWIID imputes, but imputation error)
3. Reverse causality even with lagged welfare (persistent welfare spending)
4. Measurement error in V-Dem indices (use uncertainty intervals in V-Dem)

**Timeline for Implementation:**
- Data preparation: 2-3 weeks
- Main analysis: 1-2 weeks
- Robustness checks: 2-3 weeks
- Writing and revision: 4-6 weeks
- Total: 9-14 weeks

## Sources

[1] [Interactions in Fixed Effects Regression Models](https://www.diw.de/documents/publikationen/73/diw_01.c.594675.de/dp1748.pdf) — Theoretically demonstrates that standard demeaning of interaction terms in fixed-effects models does not yield a genuine within estimator when both variables vary within units. Proposes 'double-demeaned' estimator that eliminates bias from unobserved effect heterogeneity. Published in Sociological Methods & Research (2020).

[2] [V-Dem Codebook v14](https://v-dem.net/documents/38/V-Dem_Codebook_v14.pdf) — Comprehensive documentation of V-Dem (Varieties of Democracy) dataset, including variable definitions, coding rules, country coding units, and index construction. Contains official definition of electoral democracy index (v2x_polyarchy) and country codes for all V-Dem units.

[3] [A General Approach to Causal Mediation Analysis](https://imai.fas.harvard.edu/research/files/BaronKenny.pdf) — Critiques traditional Baron-Kenny approach to mediation analysis and proposes modern counterfactual framework. Establishes sequential ignorability assumption for nonparametric identification of causal mediation effects. Published in Psychological Methods (2010). Includes sensitivity analysis methods.

[4] [Democracy, Redistribution, and Inequality](https://economics.mit.edu/sites/default/files/publications/Democracy%2C%20Redistribution%20and%20Inequality.pdf) — Acemoglu, Naidu, Restrepo, and Robinson (2015) survey theoretical and empirical literature on relationship between democracy, redistribution, and inequality. Find robust effect of democracy on tax revenues but no robust impact on inequality. Discuss conditions under which democracy may not reduce inequality (captured democracy, inequality-increasing market opportunities, middle-class bias).

[5] [Persistence of Power, Elites, and Institutions](https://www.nber.org/system/files/working_papers/w12108/w12108.pdf) — Acemoglu and Robinson (2006) develop model where elite investments in de facto political power offset loss of de jure power from democratization. Explains why political institutions may change frequently (democracy ↔ autocracy) while economic institutions persist. Introduces concept of 'captured democracy'.

[6] [Causal Mediation Programs in R, Mplus, SAS, SPSS, and Stata](https://pmc.ncbi.nlm.nih.gov/articles/PMC7853644/) — Comprehensive review of software implementations for causal mediation analysis. Recommends bootstrap methods over Sobel test for inference on indirect effects. Discusses advantages of bootstrap: better small-sample properties, no normality assumption, better coverage rates.

[7] [Panel Data Analysis Fixed and Random Effects using Stata](https://www.princeton.edu/~otorres/Panel101.pdf) — Tutorial on panel data models including fixed effects, random effects, Hausman test for model selection, and Wooldridge test for autocorrelation. Provides Stata code examples and interpretation guidance.

[8] [Testing for serial correlation in linear panel-data models](https://ageconsearch.umn.edu/record/116069/files/sjart_st0039.pdf) — Presents Wooldridge (2002) test for first-order serial correlation in panel data models. Simulation evidence shows test has good size and power properties. Discusses implications of autocorrelation for standard error estimation.

[9] [Fixed-Effects Panel Data Models: To Cluster or Not to Cluster](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840273) — Discusses clustered standard errors in fixed-effects panel models. Recommends clustering at highest level of dependence; notes that cluster-robust SE require 50+ clusters for reliability. Critiques mechanical application of clustering.

[10] [Testing Mediation with Regression Analysis](https://web.pdx.edu/~newsomj/semclass/ho_mediation.pdf) — Explains traditional Baron-Kenny (1986) four-step approach to mediation analysis. Discusses limitations and assumptions of this approach. Useful for understanding historical context of mediation analysis in social sciences.

[11] [Traditional and Modern Methods for Mediation Analysis](https://www.psych.mcgill.ca/faculty/falk/tutorials/mediation/methods/) — Compares Sobel test (1982), Baron-Kenny approach (1986), and bootstrap methods for mediation analysis. Recommends bootstrap over Sobel test except when raw data unavailable. Discusses power and Type I error rates of different methods.

[12] [Explaining Fixed Effects: Random Effects Modeling of Time-Series Cross-Sectional and Panel Data](https://www.cambridge.org/core/journals/political-science-research-and-methods/article/explaining-fixed-effects-random-effects-modeling-of-timeseries-crosssectional-and-panel-data/0334A27557D15848549120FE8ECD8D63) — Bell and Jones (2015) challenge fixed effects as default for panel data. Argue that random effects models with appropriate specification may be more appropriate. Discusses limitations of FE: inefficiency, inability to include time-invariant covariates. Provoked debate in political science methodology.

[13] [V-Dem Country Coding Units v11.1](https://v-dem.net/static/website/img/refs/countryunitv111.pdf) — Lists all country coding units in V-Dem dataset with numeric codes. Explains principles for defining country units (recognizing discontinuities in political systems). Essential for properly structuring panel data from V-Dem.

[14] [When Should You Adjust Standard Errors for Clustering?](https://economics.mit.edu/sites/default/files/2022-09/When%20Should%20You%20Adjust%20Standard%20Errors%20for%20Clustering.pdf) — Theoretically rigorous analysis of clustered standard errors. Discusses when clustering is necessary, appropriate level of clustering, and bias when number of clusters is small. Provides guidance for applied researchers.

[15] [Explaining Fixed Effects: Random Effects Modeling of Time-Series Cross-Sectional and Panel Data](https://research-information.bris.ac.uk/files/34439622/FixedvsRandom.pdf) — Same as source 12. Bell & Jones (2015) argue FE should not be default; RE with careful specification may be preferable. Emphasizes importance of research question in choosing between FE and RE. Includes simulation evidence.

## Follow-up Questions

- How should the double-demeaned estimator be extended to three-way interactions (Gini × Welfare × Education) in fixed-effects models? Should each variable be demeaned separately, then all three demeaned variables multiplied, then the product demeaned again?
- With lagged dependent variables as moderators in panel mediation analysis, should we use Arellano-Bond GMM estimation instead of standard fixed effects to address Nickell bias in short panels (T < 20)?
- The Imai et al. (2010) causal mediation framework is developed for cross-sectional data. How should the sequential ignorability assumption be modified for panel data with unit fixed effects? Should we condition on lagged outcomes as well as lagged mediators?
- Which specific V-Dem variable provides the most reliable coding of post-1990 democratization episodes: e_democracy_trans, transitions in v2x_polyarchy, or an external dataset like Boix-Miller-Rosato? Should we cross-validate across multiple sources?
- In panel data mediation analysis with bootstrap inference, should the bootstrap resample countries (cluster bootstrap) or individual country-years? The cluster bootstrap preserves within-country dependence but may underestimate variance if the number of clusters is small (N < 50).

---
*Generated by AI Inventor Pipeline*
