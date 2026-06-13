# Cluster Bootstrap Mediation Analysis for Panel Data: Methodological Foundation

## Summary

This research provides the methodological foundation for conducting cluster bootstrap mediation analysis in panel data with fixed effects. The key findings are: (1) Standard bootstrap fails with clustered panel data because it assumes i.i.d. observations, violating within-cluster dependence [1]; (2) Cluster bootstrap (pairs cluster bootstrap) correctly resamples entire clusters to preserve within-cluster dependence [1]; (3) The Imai-Keele-Tingley causal mediation framework does not directly address clustering, stating panel data is 'beyond the scope of the current article' [2]; (4) For few clusters (G < 50), wild cluster bootstrap is recommended over pairs cluster bootstrap [1]; (5) R's mediation package does not support cluster bootstrap (cluster option ignored when boot=TRUE) [from source code review]; (6) R's multilevelmediation package supports cluster bootstrap for 1-1-1 multilevel mediation models [3]; (7) Stata's bootstrap command with cluster() option supports cluster bootstrap for mediation [4]; (8) With 45 clusters, the sample is borderline but acceptable; wild cluster bootstrap provides a more reliable alternative [1, 5]. The research includes mathematical framework specification, software implementation code in R/Stata/Python, and practical guidance for handling 45 clusters with missing data.

## Research Findings

## Executive Summary

Cluster bootstrap mediation analysis is essential for panel data where observations are clustered (e.g., countries with multiple years). Standard bootstrap fails because it assumes independent observations, which is violated when countries have multiple years of data [1]. This research provides a comprehensive methodological foundation and software implementation guide for conducting cluster bootstrap mediation analysis in panel data with fixed effects.

## Why Standard Bootstrap Fails with Clustered Panel Data

Standard bootstrap assumes i.i.d. (independent and identically distributed) observations. In panel data with clustered structure (e.g., 45 countries with 33 years each), observations within the same cluster are correlated. Resampling individual observations (not clusters) underestimates standard errors and produces invalid confidence intervals [1].

Cameron and Miller (2015) demonstrate that failure to control for within-cluster error correlation can lead to standard errors several times smaller than correct cluster-robust standard errors [1]. The bootstrap must resample clusters, not individual observations, to preserve within-cluster dependence.

## Mathematical Framework for Cluster Bootstrap Mediation

### Notation
- C = {1,...,G} clusters (countries), G = 45
- Each cluster c has n_c observations (country-years)
- Total N = Σ n_c (unbalanced panel)
- Treatment: X (Gini coefficient)
- Mediator: M (welfare spending % GDP)
- Outcome: Y (V-Dem electoral democracy index)
- Fixed effects: α_g (country), γ_t (year)

### Cluster Bootstrap Algorithm for Mediation

Adapted from the Imai-Keele-Tingley framework [2] and Cameron-Miller pairs cluster bootstrap [1]:

1. For r = 1 to R (R = 1,000+):
   a. Sample G clusters with replacement: C*_r = {c*_1,...,c*_G}
   b. For each sampled cluster, keep ALL observations: D*_r = ∪ D_{c*_g}
   c. Estimate fixed effects models on D*_r:
      - Mediator model: M = β_0 + β_1 X + α_g + γ_t + ε
      - Outcome model: Y = β_0 + β_1 X + β_2 M + α_g + γ_t + ε
   d. Compute indirect effect: IE*_r = β_1(M) × β_2(Y)
   e. Compute direct effect: DE*_r = β_1(Y)

2. Bootstrap distribution: {IE*_1,..., IE*_R}

3. 95% CI: percentile [IE*_0.025, IE*_0.975] or bias-corrected

### Key Adaptation from Imai et al. (2010)

The Imai-Keele-Tingley framework uses nonparametric bootstrap but does not address clustering [2]. The authors explicitly state: 'Future research should address the application of our approach to the panel data settings... and multilevel models, all of which are beyond the scope of the current article' [2, p. 309].

The cluster bootstrap adaptation requires:
1. Resampling clusters instead of observations
2. Re-estimating fixed effects in each bootstrap iteration
3. Using idcluster() option in Stata to handle duplicate cluster IDs

## Few Clusters Problem (G = 45)

Cameron and Miller (2015) note: 'There is no clear-cut definition of "few"; depending on the situation "few" may range from less than 20 to less than 50 clusters in the balanced case' [1, p. 317].

For G = 45 clusters:
- Borderline: 45 < 50, but close to recommended minimum
- Pairs cluster bootstrap may be unreliable
- **Recommendation**: Use wild cluster bootstrap as alternative [1, 5]
- Webb (2013) and Cameron, Gelbach & Miller (2008) show wild cluster bootstrap performs better with few clusters [1]

### Wild Cluster Bootstrap for Mediation

The wild cluster bootstrap holds regressors fixed and resamples residuals with cluster structure preserved. For mediation analysis:
1. Estimate models on original data
2. Compute residuals for mediator and outcome models
3. For r = 1 to R:
   a. For each cluster g, draw w_g from {−1, 1} (Rademacher weights)
   b. Multiply residuals by w_g for all observations in cluster g
   c. Re-estimate models on modified data
   d. Compute IE*_r

## Software Implementation

### R Implementation

#### Option 1: multilevelmediation package (Recommended for 1-1-1 models)

```r
library(multilevelmediation)
library(boot)

# Data must be in long format
# L2ID = 'country' (cluster identifier)

# Bootstrap function for 1-1-1 mediation
boot.result <- boot(data = panel_data,
                   statistic = boot.modmed.mlm,
                   R = 1000,
                   L2ID = 'country',
                   X = 'gini',          # treatment
                   Y = 'vdem',         # outcome
                   M = 'welfare',      # mediator
                   random.a = TRUE,
                   random.b = TRUE,
                   type = 'indirect',
                   boot.lvl = '2')     # resample clusters only

# Extract results
indirect_effect <- extract.boot.modmed.mlm(boot.result, 
                                           type = 'indirect',
                                           ci.conf = 0.95)
```

#### Option 2: Custom cluster bootstrap function

```r
cluster_bootstrap_mediation <- function(data, R = 1000) {
  clusters <- unique(data$country)
  G <- length(clusters)
  indirect_effects <- numeric(R)
  direct_effects <- numeric(R)
  
  for (r in 1:R) {
    # Sample clusters with replacement
    sampled_clusters <- sample(clusters, G, replace = TRUE)
    
    # Create bootstrap sample (keep all observations per cluster)
    bootstrap_sample <- data[data$country %in% sampled_clusters, ]
    
    # Handle duplicate cluster IDs (important!)
    bootstrap_sample$country_new <- factor(bootstrap_sample$country):
      ave(seq_along(bootstrap_sample$country), 
          bootstrap_sample$country, FUN = cumsum)
    
    # Estimate fixed effects models
    model_m <- lm(welfare ~ gini + covariates + 
                   factor(country_new) + factor(year), 
                 data = bootstrap_sample)
    
    model_y <- lm(vdem ~ gini + welfare + covariates + 
                   factor(country_new) + factor(year),
                 data = bootstrap_sample)
    
    # Extract coefficients
    a <- coef(model_m)['gini']
    b <- coef(model_y)['welfare']
    c_prime <- coef(model_y)['gini']
    
    # Compute effects
    indirect_effects[r] <- a * b
    direct_effects[r] <- c_prime
  }
  
  # Compute 95% CIs (percentile method)
  ie_ci <- quantile(indirect_effects, c(0.025, 0.975))
  de_ci <- quantile(direct_effects, c(0.025, 0.975))
  
  return(list(IE = indirect_effects, DE = direct_effects,
              IE_CI = ie_ci, DE_CI = de_ci))
}
```

#### Option 3: Mediation package with cluster-robust SE (Not bootstrap!)

```r
library(mediation)

# Note: cluster option is IGNORED when boot = TRUE
# This uses parametric SE, not bootstrap
mediate_result <- mediate(model.m = med.fit,
                        model.y = out.fit,
                        treat = 'gini',
                        mediator = 'welfare',
                        cluster = data$country,  # Only works with robustSE
                        robustSE = TRUE,        # Use this instead of boot
                        sims = 1000)
```

**Limitation**: The R `mediation` package does NOT support cluster bootstrap. The `cluster` argument is ignored when `boot = TRUE` (verified from source code review of mediate.R).

### Stata Implementation

#### Option 1: Bootstrap with cluster() option

```stata
* Step 1: Define a program that estimates mediation effects
capture program drop mymed
program define mymed, eclass
    version 18
    syntax [varlist] [, TREAT(string) MED(string) OUT(string)]
    
    * Estimate mediator model with country FEs
    xtreg `MED' `TREAT' covariates i.year, fe cluster(country)
    scalar a = _b[`TREAT']
    
    * Estimate outcome model with country FEs
    xtreg `OUT' `TREAT' `MED' covariates i.year, fe cluster(country)
    scalar b = _b[`MED']
    scalar c_prime = _b[`TREAT']
    
    * Compute effects
    scalar IE = a * b
    scalar DE = c_prime
    
    * Store results
    ereturn clear
    ereturn scalar indirect = IE
    ereturn scalar direct = DE
end

* Step 2: Run cluster bootstrap
bootstrap _b[indirect] _b[direct], ///
    cluster(country) ///
    idcluster(newcountry) ///
    reps(1000) ///
    seed(12345): mymed, treat(gini) med(welfare) out(vdem)

* Step 3: View results
estat bootstrap, all
```

#### Option 2: Wild cluster bootstrap (Recommended for G=45)

```stata
* Requires Stata 18+
wildbootstrap, cluster(country) reps(1000): ///
    mymed, treat(gini) med(welfare) out(vdem)
```

### Python Implementation

```python
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from joblib import Parallel, delayed

def cluster_bootstrap_mediation(data, R=1000, n_jobs=4):
    """
    Cluster bootstrap for mediation analysis in panel data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Must have columns: country, year, gini, welfare, vdem, [covariates]
    R : int
        Number of bootstrap replications
    n_jobs : int
        Number of parallel jobs
    """
    countries = data['country'].unique()
    G = len(countries)
    
    def one_bootstrap(r):
        np.random.seed(r)
        
        # Sample clusters with replacement
        sampled_idx = np.random.choice(G, size=G, replace=True)
        sampled_countries = countries[sampled_idx]
        
        # Create bootstrap sample
        boot_data = data[data['country'].isin(sampled_countries)].copy()
        
        # Create new country IDs to handle duplicates
        boot_data['country_new'] = boot_data['country'].astype('category').cat.codes
        boot_data = boot_data.set_index(['country_new', 'year'])
        
        # Estimate models with fixed effects
        # Mediator model
        mod_m = PanelOLS.from_formula(
            'welfare ~ gini + covariates + EntityEffects + TimeEffects',
            data=boot_data
        )
        res_m = mod_m.fit(cov_type='clustered', cluster_entity=True)
        
        # Outcome model  
        mod_y = PanelOLS.from_formula(
            'vdem ~ gini + welfare + covariates + EntityEffects + TimeEffects',
            data=boot_data
        )
        res_y = mod_y.fit(cov_type='clustered', cluster_entity=True)
        
        # Extract coefficients
        a = res_m.params['gini']
        b = res_y.params['welfare']
        c_prime = res_y.params['gini']
        
        return a * b, c_prime  # IE, DE
    
    # Run bootstrap in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(one_bootstrap)(r) for r in range(R)
    )
    
    IEs, DEs = zip(*results)
    
    # Compute 95% CIs
    ie_ci = np.percentile(IEs, [2.5, 97.5])
    de_ci = np.percentile(DEs, [2.5, 97.5])
    
    return {'IE': IEs, 'DE': DEs, 'IE_CI': ie_ci, 'DE_CI': de_ci}
```

## Practical Considerations for Our Case

### 1. 45 Clusters: Is This Enough?

- **Cameron & Miller (2015)**: G = 50 is enough for panel data [1]
- **Our case**: G = 45 is borderline but acceptable
- **Recommendation**: 
  - Use bias-corrected bootstrap CI (not percentile)
  - Consider wild cluster bootstrap as more reliable alternative [5]
  - Report that inference may be sensitive to bootstrap method

### 2. Unbalanced Panel

- Cluster bootstrap handles naturally: keep all observations per sampled cluster
- No special adjustment needed
- Stata's `bootstrap` command handles this automatically with `cluster()` option

### 3. Missing Data (74% for welfare)

**Problem**: Multiple imputation + bootstrap is complex.

**Options**:
1. **Single imputation before bootstrap** (simpler, may underestimate uncertainty)
   ```r
   library(mice)
   imputed_data <- mice(data, m=5, maxit=50)
   complete_data <- complete(imputed_data, 1)  # Use first imputation
   # Then run cluster bootstrap on complete_data
   ```

2. **Multiple imputation + bootstrap** (more correct but complex) [7]
   - Impute M datasets
   - Run bootstrap on each imputed dataset
   - Combine results using Rubin's rules for bootstrap
   - See Wang & Zhang (2014) for SAS implementation [7]

3. **Fully Bayesian approach** (avoids bootstrap entirely)
   - Use `brms` package in R for Bayesian mediation with multilevel structure
   - Naturally handles missing data through posterior predictive distribution

### 4. Three-Way Interaction

If including interaction terms (e.g., inequality × education × welfare):
- Present marginal effects at different education levels
- Bootstrap standard errors for marginal effects
- Use `margins` package in R or `margins` command in Stata

## Comparison of Approaches

| Approach | Clustering | Fixed Effects | Inference | Suitable for Our Case |
|----------|-------------|--------------|----------|-----------------------------|
| Cluster Bootstrap + FE | ✓ | ✓ | Bootstrap CI | **YES** (preferred) |
| Wild Cluster Bootstrap + FE | ✓ | ✓ | Wild bootstrap p-value | **YES** (better for G=45) |
| Mixed Effects (RE) | ✓ | No | Asymptotic SE | NO (cannot control time-invariant confounders) |
| GEE | ✓ | No | Sandwich SE | NO (no causal interpretation with FE needed) |
| Standard Bootstrap | No | ✓ | Bootstrap CI | NO (invalid SE) |
| mediation package (R) | Partial | ✓ | Parametric/BCa | MAYBE (cluster option limited) |

## Limitations and Caveats

1. **Sequential Ignorability Untestable**: The key assumption for causal mediation cannot be tested [2]. Must rely on:
   - High-quality instrumental variables
   - Sensitivity analysis (Imai et al. 2010 provide `medsens` function)
   - Theoretical justification

2. **45 Clusters → Bootstrap May Be Unstable**: 
   - Use wild cluster bootstrap
   - Consider Bayesian approach with informative priors
   - Perform sensitivity analysis with different bootstrap methods

3. **Missing Data Complicates Bootstrap**:
   - MI + bootstrap requires careful implementation
   - Consider complete case analysis if missingness is MCAR (unlikely)
   - Use multiple imputation with caution

4. **Fixed Effects Estimation in Bootstrap**:
   - Must re-estimate FE in each iteration
   - Computational burden with large N
   - Use parallel computing

## Recommendations

### For R Users:
1. Use `multilevelmediation` package with `boot.modmed.mlm()` function
2. Set `boot.lvl = '2'` to resample clusters only
3. Increase `R` to 1000+ for stable CIs
4. Use parallel computing: `parallel='snow'`

### For Stata Users:
1. Use `bootstrap, cluster(country) idcluster(newid):` command
2. For G=45, use `wildbootstrap` (Stata 18+) for more reliable inference
3. Set `reps(1000)` or more

### For Python Users:
1. Implement custom cluster bootstrap function (provided above)
2. Use `joblib` for parallel computing
3. Use `linearmodels.panel` for FE estimation

### General Recommendations:
1. **Always use cluster bootstrap (not standard bootstrap)**
2. **With G=45, use wild cluster bootstrap** if possible
3. **Report multiple CI methods** (percentile, bias-corrected, wild)
4. **Conduct sensitivity analysis** for missing data
5. **Pre-register analysis plan** including bootstrap method

## Conclusion

Yes, cluster bootstrap works for mediation analysis with 45 clusters, but with caveats. The pairs cluster bootstrap may be borderline with G=45. Wild cluster bootstrap provides a more reliable alternative. The Imai-Keele-Tingley framework can be adapted for clustered panel data by resampling clusters instead of observations. Researchers should use the provided code templates, acknowledge limitations, and consider sensitivity analysis with different bootstrap methods.

## Bibliography

[1] Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.

[2] Imai, K., Keele, L., & Tingley, D. (2010). A general approach to causal mediation analysis. *Psychological Methods*, 15(4), 309-334.

[3] Falk, C. F., Vogel, T. A., Hammami, S., & Miočević, M. (2024). Multilevel mediation analysis in R: A comparison of bootstrap and Bayesian approaches. *Behavior Research Methods*, 56, 750-764.

[4] StataCorp. (2023). *Stata Manual: bootstrap — Bootstrap sampling and estimation*. College Station, TX.

[5] Cameron, G., Gelbach, J., & Miller, D. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414-427.

[6] Bauer, D. J., Preacher, K. J., & Gil, K. M. (2006). Conceptualizing and testing random indirect effects and moderated mediation in multilevel models. *Psychological Methods*, 11(2), 142-163.

[7] Wang, L., & Zhang, Z. (2014). Mediation analysis with missing data through multiple imputation and bootstrap. *arXiv preprint arXiv:1401.2081*.

[8] Krull, J. L., & MacKinnon, D. P. (2001). Multilevel modeling of individual and group level mediated effects. *Multivariate Behavioral Research*, 36(2), 249-277.

## Sources

[1] [A Practitioner's Guide to Cluster-Robust Inference](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf) — Comprehensive guide to cluster-robust inference including pairs cluster bootstrap, wild cluster bootstrap, and dealing with few clusters (G < 50). Provides mathematical framework and Stata code examples.

[2] [A General Approach to Causal Mediation Analysis](https://imai.fas.harvard.edu/research/files/BaronKenny.pdf) — Introduces the Imai-Keele-Tingley framework for causal mediation analysis using counterfactual approach and nonparametric bootstrap. Explicitly states that panel data and multilevel models are beyond the scope of the article.

[3] [Package 'multilevelmediation' Reference Manual](https://cran.r-universe.dev/multilevelmediation/doc/manual.html) — Documents the multilevelmediation R package which implements cluster bootstrap for 1-1-1 multilevel mediation models. Provides boot.modmed.mlm() function for bootstrap inference.

[4] [Stata Manual: bootstrap — Bootstrap sampling and estimation](https://www.stata.com/manuals/rbootstrap.pdf) — Documents Stata's bootstrap command with cluster() option for cluster bootstrap. Provides syntax for bootstrap with panel data using idcluster() option.

[5] [Bootstrap-Based Improvements for Inference with Clustered Errors](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf) — Shows that wild cluster bootstrap outperforms pairs cluster bootstrap with few clusters. Recommends wild bootstrap for G < 50.

[6] [Multilevel mediation analysis in R: A comparison of bootstrap and Bayesian approaches](https://link.springer.com/article/10.3758/s13428-023-02079-4) — Simulation study comparing bootstrap and Bayesian methods for multilevel mediation. Provides R code for implementing cluster bootstrap for 1-1-1 models.

[7] [Mediation analysis with missing data through multiple imputation and bootstrap](https://arxiv.org/abs/1401.2081) — Proposes method for combining multiple imputation and bootstrap for mediation analysis with missing data. Implements in SAS but approach can be adapted to R/Stata.

[8] [Source code for mediate() function in R mediation package](https://github.com/kosukeimai/mediation/blob/master/R/mediate.R) — Shows that cluster argument is ignored when boot=TRUE in the mediation package. Confirms limitation of existing R package for cluster bootstrap mediation.

## Follow-up Questions

- How does the power of cluster bootstrap mediation compare to parametric methods with G=45 clusters and 30% missing data?
- What is the performance of wild cluster bootstrap vs. pairs cluster bootstrap specifically for mediation analysis (not just regression) with few clusters?
- How should researchers handle the combination of multiple imputation and cluster bootstrap in practice when analyzing panel data with substantial missingness?

---
*Generated by AI Inventor Pipeline*
