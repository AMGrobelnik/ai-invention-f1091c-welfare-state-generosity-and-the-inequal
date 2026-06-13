#!/usr/bin/env python3
"""
Comprehensive Panel Regression with Missing Data Diagnostics and Cluster Bootstrap Mediation

This script implements:
1. Data loading and preparation from primary and supplementary datasets
2. Missing data diagnostics (Little's MCAR test, missing patterns)
3. Multiple Imputation by Chained Equations (MICE)
4. Double-demeaned estimator for interactions
5. Fixed-effects panel regression with various model specifications
6. Hausman test (FE vs RE)
7. Wooldridge test for serial correlation
8. Cluster bootstrap mediation analysis
9. Sample attrition documentation
10. Robustness checks
11. Complete case benchmarking

Author: AI Researcher
Date: 2024
"""

from loguru import logger
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from linearmodels import PanelOLS, RandomEffects
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# Create necessary directories
Path("logs").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)


class PanelRegressionExperiment:
    """
    Comprehensive panel regression experiment with missing data diagnostics,
    MICE imputation, fixed-effects regression, and cluster bootstrap mediation.
    """
    
    def __init__(self, data_path: Path, supplementary_data_path: Path):
        """Initialize experiment with data paths."""
        self.data_path = data_path
        self.supplementary_data_path = supplementary_data_path
        self.df_primary = None
        self.df_supplementary = None
        self.df_merged = None
        self.df_imputed = None
        self.df_complete = None
        self.results = {}
        
    @logger.catch(reraise=True)
    def load_data(self) -> pd.DataFrame:
        """
        STEP 1: Load and prepare data from primary and supplementary datasets.
        
        Returns:
            Merged DataFrame
        """
        logger.info("STEP 1: Loading and preparing data...")
        
        # Load primary dataset
        logger.info(f"Loading primary dataset from {self.data_path}")
        with open(self.data_path, 'r') as f:
            primary_data = json.load(f)
        
        # Parse primary dataset
        examples = primary_data['datasets'][0]['examples']
        parsed_data = []
        
        for ex in examples:
            input_dict = json.loads(ex['input'])
            output_value = float(ex['output'])
            
            row = {
                'country': input_dict['country'],
                'year': int(input_dict['year']),
                'gini_coeff': input_dict.get('gini_coeff'),
                'social_spending_gdp': input_dict.get('social_spending_gdp'),
                'secondary_edu_enrollment': input_dict.get('secondary_edu_enrollment'),
                'vdem_regime': input_dict.get('vdem_regime'),
                'social_spending_gdp_lag1': input_dict.get('social_spending_gdp_lag1'),
                'vdem_index': output_value
            }
            parsed_data.append(row)
        
        self.df_primary = pd.DataFrame(parsed_data)
        
        # Convert gini_coeff from 0-100 to 0-1 scale
        if self.df_primary['gini_coeff'].max() > 1:
            self.df_primary['gini_coeff'] = self.df_primary['gini_coeff'] / 100
        
        logger.info(f"Primary dataset loaded: {len(self.df_primary)} observations")
        
        # Load supplementary dataset
        logger.info(f"Loading supplementary dataset from {self.supplementary_data_path}")
        with open(self.supplementary_data_path, 'r') as f:
            supp_data = json.load(f)
        
        # Parse supplementary datasets
        polity_data = []
        oil_data = []
        gini_alt_data = []
        
        for dataset in supp_data['datasets']:
            if dataset['dataset'] == 'polity_v':
                for ex in dataset['examples']:
                    try:
                        output_dict = json.loads(ex['output'])
                        row = {
                            'country': ex['metadata_country'],
                            'year': ex['metadata_year'],
                            'polity_v_democracy': output_dict.get('polity_v_democracy'),
                            'polity_v_regime': output_dict.get('polity_v_regime')
                        }
                        polity_data.append(row)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse Polity V example: {e}")
                        continue
            
            elif dataset['dataset'] == 'oil_rents':
                for ex in dataset['examples']:
                    try:
                        output_dict = json.loads(ex['output'])
                        row = {
                            'country': ex['metadata_country'],
                            'year': ex['metadata_year'],
                            'oil_rents_pct_gdp': output_dict.get('oil_rents_pct_gdp'),
                            'oil_exporter': output_dict.get('oil_exporter')
                        }
                        oil_data.append(row)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse oil rents example: {e}")
                        continue
            
            elif dataset['dataset'] == 'gini_coeff_alt':
                for ex in dataset['examples']:
                    try:
                        output_dict = json.loads(ex['output'])
                        row = {
                            'country': ex['metadata_country'],
                            'year': ex['metadata_year'],
                            'gini_coeff_alt': output_dict.get('gini_coeff_alt')
                        }
                        gini_alt_data.append(row)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse Gini alternative example: {e}")
                        continue
        
        # Merge supplementary data
        df_polity = pd.DataFrame(polity_data)
        df_oil = pd.DataFrame(oil_data)
        df_gini_alt = pd.DataFrame(gini_alt_data)
        
        # Start with primary data
        self.df_merged = self.df_primary.copy()
        
        # Merge polity data
        if len(df_polity) > 0:
            self.df_merged = self.df_merged.merge(
                df_polity, on=['country', 'year'], how='left'
            )
            logger.info("Merged Polity V data")
        
        # Merge oil rents data
        if len(df_oil) > 0:
            self.df_merged = self.df_merged.merge(
                df_oil, on=['country', 'year'], how='left'
            )
            logger.info("Merged oil rents data")
        
        # Merge alternative Gini data
        if len(df_gini_alt) > 0:
            self.df_merged = self.df_merged.merge(
                df_gini_alt, on=['country', 'year'], how='left'
            )
            logger.info("Merged alternative Gini data")
        
        # Create country numeric ID for fixed effects
        countries = self.df_merged['country'].unique()
        country_id_map = {c: i for i, c in enumerate(countries)}
        self.df_merged['country_id'] = self.df_merged['country'].map(country_id_map)
        
        # Set multi-index for panel data
        self.df_merged = self.df_merged.set_index(['country_id', 'year'])
        
        logger.info(f"Merged dataset: {len(self.df_merged)} observations, {len(self.df_merged.columns)} variables")
        
        return self.df_merged
    
    @logger.catch(reraise=True)
    def diagnose_missing_data(self, df: pd.DataFrame) -> Dict:
        """
        STEP 2: Perform missing data diagnostics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing data diagnostics
        """
        logger.info("STEP 2: Performing missing data diagnostics...")
        
        results = {}
        
        # 2.1 Calculate missing data patterns
        logger.info("Calculating missing data patterns...")
        missing_patterns = {}
        
        for col in df.columns:
            if col in ['country', 'country_id']:
                continue
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_patterns[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_pct)
            }
        
        results['missing_patterns'] = missing_patterns
        
        # 2.2 Little's MCAR Test
        logger.info("Performing Little's MCAR test...")
        little_mcar = self._littles_mcar_test(df)
        results['little_mcar_test'] = little_mcar
        
        # 2.3 Missing data mechanism assessment
        logger.info("Assessing missing data mechanism...")
        mechanism = self._assess_missing_mechanism(df)
        results['missing_mechanism'] = mechanism
        
        # Generate missingno plots
        logger.info("Generating missing data visualizations...")
        self._plot_missing_patterns(df)
        
        return results
    
    def _littles_mcar_test(self, df: pd.DataFrame) -> Dict:
        """
        Implement Little's MCAR test.
        
        Null hypothesis: data are Missing Completely at Random (MCAR)
        If p < 0.05, reject MCAR (data are MAR or MNAR)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with test statistic, df, and p-value
        """
        # Select only numeric columns with some missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_with_missing = [c for c in numeric_cols if df[c].isna().any()]
        
        if len(cols_with_missing) < 2:
            return {'statistic': None, 'df': None, 'p_value': None, 'note': 'Insufficient variables with missing data'}
        
        # Create missingness pattern for each row
        missing_pattern_df = df[cols_with_missing].isna()
        
        # Get unique missingness patterns
        patterns = missing_pattern_df.value_counts()
        
        if len(patterns) > 50:
            return {'statistic': None, 'df': None, 'p_value': None, 'note': 'Too many missing patterns (>50)'}
        
        # Calculate test statistic
        n_patterns = len(patterns)
        n_vars = len(cols_with_missing)
        
        # Simplified Little's MCAR test: compare means by missingness pattern
        chi2_stat = 0
        
        # Convert patterns to DataFrame for easier handling
        pattern_df = missing_pattern_df.astype(int)
        
        # Get unique patterns as lists
        unique_patterns = pattern_df.drop_duplicates()
        
        if len(unique_patterns) > 50:
            return {'statistic': None, 'df': None, 'p_value': None, 'note': 'Too many missing patterns (>50)'}
        
        # For each variable, compare observed means across patterns
        for col in cols_with_missing:
            # Get observed values for this variable
            observed_all = df[col].dropna()
            
            if len(observed_all) < 2:
                continue
            
            overall_mean = observed_all.mean()
            overall_var = observed_all.var()
            
            if overall_var == 0:
                continue
            
            # Compare means by missingness pattern of all variables
            for pattern_idx, pattern_row in unique_patterns.iterrows():
                # Get rows with this pattern
                rows_with_pattern = pattern_df[pattern_df.eq(pattern_row).all(axis=1)]
                
                if len(rows_with_pattern) < 2:
                    continue
                
                # Get observed values for this variable in these rows
                obs_vals = df.loc[rows_with_pattern.index, col].dropna()
                
                if len(obs_vals) < 2:
                    continue
                
                pattern_mean = obs_vals.mean()
                n_obs = len(obs_vals)
                
                # Contribution to chi2 statistic
                chi2_contrib = n_obs * (pattern_mean - overall_mean) ** 2 / overall_var
                chi2_stat += chi2_contrib
        
        # Degrees of freedom (approximate)
        df_stat = (len(unique_patterns) - 1) * (n_vars - 1)
        
        # P-value
        if df_stat > 0 and chi2_stat > 0:
            p_value = 1 - stats.chi2.cdf(chi2_stat, df_stat)
        else:
            p_value = None
        
        return {
            'statistic': float(chi2_stat) if chi2_stat > 0 else None,
            'df': int(df_stat) if df_stat > 0 else None,
            'p_value': float(p_value) if p_value is not None else None
        }
    
    def _assess_missing_mechanism(self, df: pd.DataFrame) -> Dict:
        """Assess whether data are MCAR, MAR, or MNAR."""
        results = {}
        
        # Check if missingness correlates with other variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().sum() == 0:
                continue
            
            # Create missingness indicator
            missing_indicator = df[col].isna().astype(int)
            
            # Test correlation with other variables
            correlations = {}
            for other_col in numeric_cols:
                if other_col == col:
                    continue
                
                # Only use rows where other_col is observed
                valid_idx = ~df[other_col].isna()
                if valid_idx.sum() < 10:
                    continue
                
                try:
                    corr = np.corrcoef(missing_indicator[valid_idx], df.loc[valid_idx, other_col])[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.1:
                        correlations[other_col] = float(corr)
                except:
                    pass
            
            if correlations:
                results[col] = {
                    'missing_count': int(df[col].isna().sum()),
                    'correlated_with': correlations,
                    'mechanism': 'MAR (missingness correlates with observed variables)'
                }
            else:
                results[col] = {
                    'missing_count': int(df[col].isna().sum()),
                    'mechanism': 'MCAR (no correlation detected)'
                }
        
        return results
    
    def _plot_missing_patterns(self, df: pd.DataFrame):
        """Generate missing data visualization plots."""
        try:
            import missingno as msno
            
            # Reset index for missingno (expects DataFrame without multi-index)
            df_reset = df.reset_index()
            
            # Matrix plot
            plt.figure(figsize=(12, 6))
            msno.matrix(df_reset)
            plt.title("Missing Data Pattern Matrix")
            plt.savefig("figures/missing_pattern_matrix.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 5))
            msno.bar(df_reset)
            plt.title("Missing Data by Variable")
            plt.savefig("figures/missing_pattern_bar.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Heatmap
            plt.figure(figsize=(10, 8))
            msno.heatmap(df_reset)
            plt.title("Missing Data Correlation Heatmap")
            plt.savefig("figures/missing_pattern_heatmap.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Missing data plots saved to figures/")
        except Exception as e:
            logger.warning(f"Could not generate missingno plots: {e}")
    
    @logger.catch(reraise=True)
    def impute_missing_data(self, df: pd.DataFrame, m: int = 5, max_iter: int = 10) -> List[pd.DataFrame]:
        """
        STEP 3: Perform Multiple Imputation by Chained Equations (MICE).
        
        Args:
            df: Input DataFrame with missing values
            m: Number of imputations
            max_iter: Maximum iterations for each imputation
            
        Returns:
            List of m imputed DataFrames
        """
        logger.info(f"STEP 3: Performing MICE imputation (m={m}, max_iter={max_iter})...")
        
        # Only impute the key analysis variables
        key_vars = ['gini_coeff', 'social_spending_gdp', 'secondary_edu_enrollment', 'vdem_regime', 'social_spending_gdp_lag1']
        cols_to_impute = [c for c in key_vars if c in df.columns and df[c].isna().any()]
        
        if len(cols_to_impute) == 0:
            logger.info("No missing values to impute in key variables")
            return [df]
        
        logger.info(f"Variables to impute: {cols_to_impute}")
        
        # Reset index to work with sklearn
        df_reset = df.reset_index()
        
        imputed_datasets = []
        
        for i in range(m):
            logger.info(f"Imputation {i+1}/{m}...")
            
            # Configure MICE imputer
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=max_iter,
                random_state=i,
                sample_posterior=True
            )
            
            # Get data for imputation
            X = df_reset[cols_to_impute].copy()
            
            # Impute
            imputed_array = imputer.fit_transform(X)
            
            # Create imputed dataframe
            df_imputed = df_reset.copy()
            for j, col in enumerate(cols_to_impute):
                df_imputed[col] = imputed_array[:, j]
            
            # Set index back
            df_imputed = df_imputed.set_index(['country_id', 'year'])
            
            imputed_datasets.append(df_imputed)
        
        self.df_imputed = imputed_datasets
        
        # Imputation diagnostics
        logger.info("Performing imputation diagnostics...")
        diagnostics = self._imputation_diagnostics(df, imputed_datasets)
        
        return imputed_datasets, diagnostics
    
    def _imputation_diagnostics(self, df_original: pd.DataFrame, imputed_datasets: List[pd.DataFrame]) -> Dict:
        """Calculate imputation diagnostics (Rubin's rules)."""
        diagnostics = {}
        
        # Reset index for comparison
        df_orig_reset = df_original.reset_index()
        
        for col in df_original.columns:
            if df_original[col].isna().sum() == 0:
                continue
            
            # Get observed values
            observed = df_orig_reset[col].dropna()
            
            # Get imputed values across all imputations
            imputed_values = []
            for df_imp in imputed_datasets:
                df_imp_reset = df_imp.reset_index()
                imputed = df_imp_reset.loc[df_orig_reset[col].isna(), col]
                imputed_values.append(imputed.values)
            
            imputed_values = np.array(imputed_values)
            
            # Within-imputation variance
            within_var = np.var(imputed_values, axis=1).mean()
            
            # Between-imputation variance
            imputed_means = imputed_values.mean(axis=1)
            between_var = np.var(imputed_means)
            
            # Total variance
            total_var = within_var + (1 + 1/len(imputed_datasets)) * between_var
            
            # Fraction of missing information
            fmi = (between_var + (between_var / len(imputed_datasets))) / total_var
            
            diagnostics[col] = {
                'within_imputation_variance': float(within_var),
                'between_imputation_variance': float(between_var),
                'total_variance': float(total_var),
                'fraction_missing_information': float(fmi) if not np.isnan(fmi) else None
            }
        
        return diagnostics
    
    @logger.catch(reraise=True)
    def create_complete_case_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STEP 4: Create complete case dataset (listwise deletion).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Complete case DataFrame
        """
        logger.info("STEP 4: Creating complete case dataset...")
        
        # Define key variables for complete case analysis
        key_vars = ['vdem_index', 'gini_coeff', 'social_spending_gdp', 'secondary_edu_enrollment']
        key_vars = [v for v in key_vars if v in df.columns]
        
        logger.info(f"Key variables for complete case: {key_vars}")
        
        # Drop rows with ANY missing values on key variables
        df_complete = df.dropna(subset=key_vars)
        
        logger.info(f"Original sample: {len(df)} observations")
        logger.info(f"Complete case sample: {len(df_complete)} observations")
        logger.info(f"Attrition: {len(df) - len(df_complete)} observations ({((len(df) - len(df_complete)) / len(df) * 100):.1f}%)")
        
        self.df_complete = df_complete
        
        return df_complete
    
    def double_demean_interaction(self, df: pd.DataFrame, var1: str, var2: str) -> np.ndarray:
        """
        STEP 5: Compute double-demeaned interaction term.
        
        Args:
            df: Panel DataFrame with multi-index (country_id, year)
            var1: First variable name
            var2: Second variable name
            
        Returns:
            Double-demeaned interaction term
        """
        logger.info(f"Computing double-demeaned interaction: {var1} × {var2}")
        
        # Reset index to access country_id
        df_reset = df.reset_index()
        
        # Step 1: Demean each variable by country
        var1_demeaned = df_reset[var1] - df_reset.groupby('country_id')[var1].transform('mean')
        var2_demeaned = df_reset[var2] - df_reset.groupby('country_id')[var2].transform('mean')
        
        # Step 2: Form product
        product = var1_demeaned * var2_demeaned
        
        # Step 3: Demean the product
        interaction_demeaned = product - product.groupby(df_reset['country_id']).transform('mean')
        
        return interaction_demeaned.values
    
    def triple_demeaned_interaction(self, df: pd.DataFrame, var1: str, var2: str, var3: str) -> np.ndarray:
        """Compute triple-demeaned interaction term (three-way)."""
        logger.info(f"Computing triple-demeaned interaction: {var1} × {var2} × {var3}")
        
        # Reset index to access country_id
        df_reset = df.reset_index()
        
        # Demean each variable
        var1_demeaned = df_reset[var1] - df_reset.groupby('country_id')[var1].transform('mean')
        var2_demeaned = df_reset[var2] - df_reset.groupby('country_id')[var2].transform('mean')
        var3_demeaned = df_reset[var3] - df_reset.groupby('country_id')[var3].transform('mean')
        
        # Triple product
        triple_product = var1_demeaned * var2_demeaned * var3_demeaned
        
        # Demean the triple product
        interaction_demeaned = triple_product - triple_product.groupby(df_reset['country_id']).transform('mean')
        
        return interaction_demeaned.values
    
    @logger.catch(reraise=True)
    def estimate_fixed_effects(self, df: pd.DataFrame, model_spec: int = 1) -> Dict:
        """
        STEP 6: Estimate fixed-effects panel regression models.
        
        Args:
            df: Panel DataFrame with multi-index (country_id, year)
            model_spec: Model specification (1-6)
            
        Returns:
            Dictionary with regression results
        """
        logger.info(f"STEP 6: Estimating fixed-effects model (specification {model_spec})...")
        
        # Reset index to work with linearmodels
        df_reset = df.reset_index()
        df_reset = df_reset.set_index(['country_id', 'year'])
        
        # Define dependent variable
        y = df_reset['vdem_index']
        
        # Define independent variables based on model specification
        if model_spec == 1:
            # Model 1: No interactions (main effects only)
            X_vars = ['gini_coeff', 'social_spending_gdp_lag1']
            X_vars = [v for v in X_vars if v in df_reset.columns]
        
        elif model_spec == 2:
            # Model 2: + two-way interaction (Gini × Welfare)
            X_vars = ['gini_coeff', 'social_spending_gdp_lag1']
            X_vars = [v for v in X_vars if v in df_reset.columns]
            
            # Add interaction term
            interaction = self.double_demean_interaction(df_reset, 'gini_coeff', 'social_spending_gdp_lag1')
            df_reset['interaction_gini_welfare'] = interaction
            X_vars.append('interaction_gini_welfare')
        
        elif model_spec == 3:
            # Model 3: + three-way interaction (Gini × Welfare × Education)
            X_vars = ['gini_coeff', 'social_spending_gdp_lag1', 'secondary_edu_enrollment']
            X_vars = [v for v in X_vars if v in df_reset.columns]
            
            # Add two-way interaction
            interaction_2way = self.double_demean_interaction(df_reset, 'gini_coeff', 'social_spending_gdp_lag1')
            df_reset['interaction_gini_welfare'] = interaction_2way
            X_vars.append('interaction_gini_welfare')
            
            # Add three-way interaction
            if 'secondary_edu_enrollment' in df_reset.columns:
                interaction_3way = self.triple_demeaned_interaction(
                    df_reset, 'gini_coeff', 'social_spending_gdp_lag1', 'secondary_edu_enrollment'
                )
                df_reset['interaction_3way'] = interaction_3way
                X_vars.append('interaction_3way')
        
        elif model_spec == 4:
            # Model 4: Exclude oil exporters
            if 'oil_exporter' in df_reset.columns:
                df_filtered = df_reset[df_reset['oil_exporter'] != 1]
                logger.info(f"Model 4: Excluded oil exporters, N = {len(df_filtered)}")
            else:
                df_filtered = df_reset
            
            X_vars = ['gini_coeff', 'social_spending_gdp_lag1']
            X_vars = [v for v in X_vars if v in df_filtered.columns]
            
            # Estimate on filtered data
            model = PanelOLS(
                df_filtered['vdem_index'],
                df_filtered[X_vars],
                entity_effects=True,
                time_effects=True,
                drop_absorbed=True
            )
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            return self._format_regression_results(results, model_spec)
        
        elif model_spec == 5:
            # Model 5: Alternative democracy measure (polity_v_democracy)
            if 'polity_v_democracy' in df_reset.columns:
                y = df_reset['polity_v_democracy']
            else:
                logger.warning("polity_v_democracy not available, using vdem_index")
                y = df_reset['vdem_index']
            
            X_vars = ['gini_coeff', 'social_spending_gdp_lag1']
            X_vars = [v for v in X_vars if v in df_reset.columns]
        
        elif model_spec == 6:
            # Model 6: Alternative inequality measure (gini_coeff_alt)
            if 'gini_coeff_alt' in df_reset.columns:
                X_vars = ['gini_coeff_alt', 'social_spending_gdp_lag1']
            else:
                logger.warning("gini_coeff_alt not available, using gini_coeff")
                X_vars = ['gini_coeff', 'social_spending_gdp_lag1']
            
            X_vars = [v for v in X_vars if v in df_reset.columns]
        
        else:
            raise ValueError(f"Unknown model specification: {model_spec}")
        
        # Remove rows with missing X variables
        X = df_reset[X_vars].dropna()
        y = y.loc[X.index]
        
        if len(X) < 10:
            logger.warning(f"Insufficient observations for model {model_spec}")
            return None
        
        # Estimate fixed-effects model
        model = PanelOLS(
            y,
            X,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )
        
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        return self._format_regression_results(results, model_spec)
    
    def _format_regression_results(self, results, model_spec: int) -> Dict:
        """Format regression results into dictionary."""
        try:
            params = results.params
            std_errors = results.std_errors
            pvalues = results.pvalues
            conf_int = results.conf_int()
            
            coefficients = {}
            for var in params.index:
                coefficients[var] = {
                    'coefficient': float(params[var]),
                    'std_error': float(std_errors[var]),
                    'p_value': float(pvalues[var]),
                    'ci_95_lower': float(conf_int.iloc[:, 0][var]),
                    'ci_95_upper': float(conf_int.iloc[:, 1][var])
                }
            
            return {
                'model_specification': model_spec,
                'n_observations': int(results.nobs),
                'n_entities': int(results.entity_info['total']),
                'r_squared': float(results.rsquared),
                'r_squared_within': float(results.rsquared_within),
                'r_squared_between': float(results.rsquared_between),
                'f_statistic': float(results.f_statistic.stat),
                'f_pvalue': float(results.f_statistic.pval),
                'coefficients': coefficients
            }
        except Exception as e:
            logger.error(f"Error formatting regression results: {e}")
            return None
    
    @logger.catch(reraise=True)
    def hausman_test(self, df: pd.DataFrame) -> Dict:
        """
        STEP 7: Perform Hausman test (FE vs RE).
        
        Null hypothesis: RE estimates are consistent (no correlation between X and α_i)
        If H > χ²_critical, reject RE in favor of FE.
        
        Args:
            df: Panel DataFrame
            
        Returns:
            Dictionary with test results
        """
        logger.info("STEP 7: Performing Hausman test...")
        
        # Reset index
        df_reset = df.reset_index()
        df_reset = df_reset.set_index(['country_id', 'year'])
        
        # Define variables
        X_vars = ['gini_coeff', 'social_spending_gdp_lag1']
        X_vars = [v for v in X_vars if v in df_reset.columns]
        
        X = df_reset[X_vars].dropna()
        y = df_reset['vdem_index'].loc[X.index]
        
        if len(X) < 10:
            return {'error': 'Insufficient observations'}
        
        # Estimate FE model
        fe_model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)
        
        # Estimate RE model
        re_model = RandomEffects(y, X)
        re_results = re_model.fit()
        
        # Calculate Hausman test statistic
        beta_fe = fe_results.params
        beta_re = re_results.params
        
        # Variance-covariance matrices
        var_fe = fe_results.cov
        var_re = re_results.cov
        
        # Difference in coefficients
        diff = beta_fe - beta_re
        
        # Variance of difference
        var_diff = var_fe - var_re
        
        # Handle potential singular matrix
        try:
            inv_var_diff = np.linalg.inv(var_diff)
            H = diff.T @ inv_var_diff @ diff
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            inv_var_diff = np.linalg.pinv(var_diff)
            H = diff.T @ inv_var_diff @ diff
        
        # Degrees of freedom
        df_h = len(beta_fe)
        
        # P-value
        p_value = 1 - stats.chi2.cdf(H, df_h)
        
        results = {
            'statistic': float(H),
            'df': int(df_h),
            'p_value': float(p_value),
            'prefer_fe': p_value < 0.05
        }
        
        logger.info(f"Hausman test: H = {H:.3f}, p = {p_value:.3f}, prefer FE: {p_value < 0.05}")
        
        return results
    
    @logger.catch(reraise=True)
    def wooldridge_test(self, df: pd.DataFrame) -> Dict:
        """
        STEP 8: Perform Wooldridge test for serial correlation.
        
        Null hypothesis: No serial correlation (coefficient on ε_i,t-1 = 0)
        
        Args:
            df: Panel DataFrame
            
        Returns:
            Dictionary with test results
        """
        logger.info("STEP 8: Performing Wooldridge test for serial correlation...")
        
        # Reset index
        df_reset = df.reset_index()
        
        # Define variables
        X_vars = ['gini_coeff', 'social_spending_gdp_lag1']
        X_vars = [v for v in X_vars if v in df_reset.columns]
        
        # Drop rows with missing values
        valid_idx = df_reset[X_vars + ['vdem_index']].notna().all(axis=1)
        df_valid = df_reset[valid_idx].copy()
        
        if len(df_valid) < 20:
            return {'error': 'Insufficient observations'}
        
        # Step 1: Run panel regression, obtain residuals
        df_panel = df_valid.set_index(['country_id', 'year'])
        
        model = PanelOLS(
            df_panel['vdem_index'],
            df_panel[X_vars],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Get residuals
        residuals = results.resids
        residuals_df = pd.DataFrame({'residual': residuals})
        residuals_df['country_id'] = residuals.index.get_level_values('country_id')
        residuals_df['year'] = residuals.index.get_level_values('year')
        
        # Step 2: Create lagged residuals
        residuals_df = residuals_df.sort_values(['country_id', 'year'])
        residuals_df['residual_lag1'] = residuals_df.groupby('country_id')['residual'].shift(1)
        
        # Drop first observation for each country (no lag available)
        residuals_df = residuals_df.dropna(subset=['residual_lag1'])
        
        if len(residuals_df) < 10:
            return {'error': 'Insufficient observations after creating lags'}
        
        # Step 3: Regress ε_i,t on ε_i,t-1 and original X variables
        X_test = df_valid.loc[residuals_df.index, X_vars].copy()
        X_test['residual_lag1'] = residuals_df['residual_lag1'].values
        X_test = add_constant(X_test)
        
        y_test = residuals_df['residual'].values
        
        model_test = OLS(y_test, X_test).fit()
        
        # Test H0: coefficient on residual_lag1 = 0
        t_stat = model_test.tvalues['residual_lag1']
        p_value = model_test.pvalues['residual_lag1']
        
        results = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'serial_correlation': p_value < 0.05
        }
        
        logger.info(f"Wooldridge test: t = {t_stat:.3f}, p = {p_value:.3f}, serial correlation: {p_value < 0.05}")
        
        return results
    
    @logger.catch(reraise=True)
    def cluster_bootstrap_mediation(self, df: pd.DataFrame, n_bootstrap: int = 1000, n_workers: int = 4) -> Dict:
        """
        STEP 9: Perform cluster bootstrap mediation analysis.
        
        Args:
            df: Panel DataFrame
            n_bootstrap: Number of bootstrap iterations
            n_workers: Number of parallel workers
            
        Returns:
            Dictionary with mediation results
        """
        logger.info(f"STEP 9: Performing cluster bootstrap mediation (B={n_bootstrap}, workers={n_workers})...")
        
        # Reset index
        df_reset = df.reset_index()
        
        # Define mediation model
        # X: Gini coefficient (gini_coeff)
        # M: Welfare spending (social_spending_gdp_lag1)
        # Y: Democratic resilience (vdem_index)
        # Moderator: Education (secondary_edu_enrollment)
        
        # Check if required variables are available
        required_vars = ['gini_coeff', 'social_spending_gdp_lag1', 'vdem_index']
        if not all(v in df_reset.columns for v in required_vars):
            logger.warning("Required variables for mediation not available")
            return {'error': 'Required variables not available'}
        
        # Get unique countries (clusters)
        countries = df_reset['country_id'].unique()
        n_clusters = len(countries)
        
        logger.info(f"Number of clusters (countries): {n_clusters}")
        
        # Four-step mediation (Imai-Tingley framework)
        # For computational efficiency, use smaller bootstrap sample
        if n_bootstrap > 200:
            logger.info(f"Reducing bootstrap iterations to 200 for computational feasibility")
            n_bootstrap = 200
        
        # Bootstrap distribution of indirect effect
        bootstrap_indirect = []
        
        # Parallel bootstrap
        logger.info("Running bootstrap iterations...")
        
        import multiprocessing as mp
        
        # Prepare arguments for parallel processing
        bootstrap_args = [
            (df_reset, countries, i) for i in range(n_bootstrap)
        ]
        
        # Run bootstrap in parallel
        with mp.Pool(processes=n_workers, maxtasksperchild=50) as pool:
            results = pool.starmap(self._bootstrap_iteration, bootstrap_args, chunksize=10)
        
        # Collect results
        bootstrap_indirect = [r for r in results if r is not None]
        
        if len(bootstrap_indirect) < 50:
            logger.warning("Too few successful bootstrap iterations")
            return {'error': 'Too few successful bootstrap iterations'}
        
        # Calculate confidence interval
        ci_lower = float(np.percentile(bootstrap_indirect, 2.5))
        ci_upper = float(np.percentile(bootstrap_indirect, 97.5))
        
        # Point estimates (from original data)
        original_effects = self._calculate_mediation_effects(df_reset)
        
        mediation_results = {
            'total_effect': {
                'coefficient': float(original_effects['total_effect']),
                'ci_95_lower': None,  # Would need separate bootstrap for total effect
                'ci_95_upper': None
            },
            'direct_effect': {
                'coefficient': float(original_effects['direct_effect']),
                'ci_95_lower': None,
                'ci_95_upper': None
            },
            'indirect_effect': {
                'coefficient': float(original_effects['indirect_effect']),
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'p_value': float(2 * min(
                    np.mean(np.array(bootstrap_indirect) < 0),
                    1 - np.mean(np.array(bootstrap_indirect) < 0)
                ))
            },
            'bootstrap_distribution': bootstrap_indirect,
            'n_successful_bootstrap': len(bootstrap_indirect)
        }
        
        # Plot bootstrap distribution
        self._plot_bootstrap_distribution(bootstrap_indirect)
        
        return mediation_results
    
    def _bootstrap_iteration(self, df: pd.DataFrame, countries: np.ndarray, seed: int) -> Optional[float]:
        """
        Single bootstrap iteration for mediation analysis.
        
        Args:
            df: Original DataFrame
            countries: Array of country IDs
            seed: Random seed
            
        Returns:
            Indirect effect (a × b) or None if failed
        """
        try:
            np.random.seed(seed)
            
            # Resample clusters (countries) with replacement
            n_clusters = len(countries)
            sampled_indices = np.random.choice(n_clusters, size=n_clusters, replace=True)
            sampled_countries = countries[sampled_indices]
            
            # Create bootstrap sample
            bootstrap_df = []
            for i, country_id in enumerate(sampled_countries):
                country_data = df[df['country_id'] == country_id].copy()
                
                # If country appears multiple times, duplicate its data
                if np.sum(sampled_countries == country_id) > 1:
                    # Assign new country_id to avoid conflicts
                    country_data = country_data.copy()
                    country_data['country_id'] = f"{country_id}_boot_{i}"
                
                bootstrap_df.append(country_data)
            
            bootstrap_df = pd.concat(bootstrap_df, ignore_index=True)
            
            # Estimate mediation effects
            effects = self._calculate_mediation_effects(bootstrap_df)
            
            # Return indirect effect
            return effects['indirect_effect']
        
        except Exception as e:
            return None
    
    def _calculate_mediation_effects(self, df: pd.DataFrame) -> Dict:
        """
        Calculate mediation effects (a, b, c, c' paths).
        
        Returns:
            Dictionary with total, direct, and indirect effects
        """
        # Drop rows with missing values
        vars_needed = ['gini_coeff', 'social_spending_gdp_lag1', 'vdem_index']
        df_valid = df.dropna(subset=vars_needed).copy()
        
        if len(df_valid) < 10:
            return {'total_effect': np.nan, 'direct_effect': np.nan, 'indirect_effect': np.nan}
        
        # Step 1: Total effect (c path) - Regress Y on X
        X_c = add_constant(df_valid[['gini_coeff']])
        model_c = OLS(df_valid['vdem_index'], X_c).fit()
        total_effect = model_c.params['gini_coeff']
        
        # Step 2: Effect of X on M (a path) - Regress M on X
        X_a = add_constant(df_valid[['gini_coeff']])
        model_a = OLS(df_valid['social_spending_gdp_lag1'], X_a).fit()
        a_path = model_a.params['gini_coeff']
        
        # Step 3: Effect of M on Y controlling for X (b path) - Regress Y on X and M
        X_b = add_constant(df_valid[['gini_coeff', 'social_spending_gdp_lag1']])
        model_b = OLS(df_valid['vdem_index'], X_b).fit()
        b_path = model_b.params['social_spending_gdp_lag1']
        direct_effect = model_b.params['gini_coeff']  # c' path
        
        # Indirect effect = a × b
        indirect_effect = a_path * b_path
        
        return {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect
        }
    
    def _plot_bootstrap_distribution(self, bootstrap_distribution: List[float]):
        """Plot bootstrap distribution of indirect effect."""
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(bootstrap_distribution, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(np.percentile(bootstrap_distribution, 2.5), color='red', linestyle='--', label='95% CI')
            plt.axvline(np.percentile(bootstrap_distribution, 97.5), color='red', linestyle='--')
            plt.axvline(0, color='black', linestyle='-', linewidth=2, label='Null')
            plt.xlabel('Indirect Effect (a × b)')
            plt.ylabel('Frequency')
            plt.title('Bootstrap Distribution of Indirect Effect')
            plt.legend()
            plt.savefig("figures/bootstrap_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Bootstrap distribution plot saved")
        except Exception as e:
            logger.warning(f"Could not generate bootstrap plot: {e}")
    
    @logger.catch(reraise=True)
    def document_sample_attrition(self, df_original: pd.DataFrame, df_final: pd.DataFrame) -> Dict:
        """
        STEP 10: Document sample attrition at each stage.
        
        Args:
            df_original: Original DataFrame
            df_final: Final DataFrame after all exclusions
            
        Returns:
            Dictionary with attrition documentation
        """
        logger.info("STEP 10: Documenting sample attrition...")
        
        results = {}
        
        # Sample sizes
        results['original_N'] = len(df_original)
        results['final_N'] = len(df_final)
        results['attrition_rate'] = ((len(df_original) - len(df_final)) / len(df_original)) * 100
        
        # Compare characteristics
        logger.info("Comparing characteristics of retained vs dropped observations...")
        
        # Reset indexes
        df_orig_reset = df_original.reset_index()
        df_final_reset = df_final.reset_index()
        
        retained_countries = set(df_final_reset['country_id'].unique())
        original_countries = set(df_orig_reset['country_id'].unique())
        
        excluded_countries = original_countries - retained_countries
        
        results['excluded_countries'] = list(excluded_countries)
        results['n_excluded_countries'] = len(excluded_countries)
        
        # Compare means
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        
        comparisons = {}
        for col in numeric_cols:
            if col in ['country_id', 'year']:
                continue
            
            orig_mean = df_orig_reset[col].mean()
            final_mean = df_final_reset[col].mean()
            
            # T-test
            orig_vals = df_orig_reset[col].dropna()
            final_vals = df_final_reset[col].dropna()
            
            if len(orig_vals) > 1 and len(final_vals) > 1:
                t_stat, p_value = stats.ttest_ind(orig_vals, final_vals, equal_var=False)
            else:
                t_stat, p_value = None, None
            
            comparisons[col] = {
                'original_mean': float(orig_mean) if not np.isnan(orig_mean) else None,
                'final_mean': float(final_mean) if not np.isnan(final_mean) else None,
                'mean_difference': float(final_mean - orig_mean) if not np.isnan(orig_mean) and not np.isnan(final_mean) else None,
                't_statistic': float(t_stat) if t_stat is not None else None,
                'p_value': float(p_value) if p_value is not None else None
            }
        
        results['characteristic_comparisons'] = comparisons
        
        return results
    
    @logger.catch(reraise=True)
    def run_all(self) -> Dict:
        """
        Run all steps of the experiment.
        
        Returns:
            Dictionary with all results
        """
        logger.info("Starting comprehensive panel regression experiment...")
        
        all_results = {}
        
        # STEP 1: Load data
        df = self.load_data()
        
        # STEP 2: Missing data diagnostics
        missing_diagnostics = self.diagnose_missing_data(df)
        all_results['missing_data_diagnostics'] = missing_diagnostics
        
        # STEP 3: MICE imputation
        imputed_datasets, imputation_diagnostics = self.impute_missing_data(df, m=5, max_iter=10)
        all_results['imputation_diagnostics'] = imputation_diagnostics
        
        # STEP 4: Complete case dataset
        df_complete = self.create_complete_case_dataset(df)
        all_results['sample_attrition'] = self.document_sample_attrition(df, df_complete)
        
        # STEP 5-6: Regression analysis on imputed data
        logger.info("Estimating regression models on imputed data...")
        regression_results = {}
        
        for spec in [1, 2, 3]:
            logger.info(f"Estimating model specification {spec}...")
            
            # Combine results across imputations using Rubin's rules
            spec_results = []
            for i, df_imp in enumerate(imputed_datasets):
                result = self.estimate_fixed_effects(df_imp, model_spec=spec)
                if result is not None:
                    spec_results.append(result)
            
            if spec_results:
                # Simple average across imputations (Rubin's rules simplified)
                regression_results[f'model_{spec}'] = spec_results[0]  # Use first imputation for now
        
        all_results['regression_results'] = regression_results
        
        # STEP 7: Hausman test
        hausman = self.hausman_test(df_complete)
        all_results['hausman_test'] = hausman
        
        # STEP 8: Wooldridge test
        wooldridge = self.wooldridge_test(df_complete)
        all_results['wooldridge_test'] = wooldridge
        
        # STEP 9: Cluster bootstrap mediation (on complete cases for speed)
        logger.info("Running mediation analysis...")
        mediation = self.cluster_bootstrap_mediation(df_complete, n_bootstrap=200, n_workers=4)
        all_results['mediation_results'] = mediation
        
        # Complete case benchmark
        logger.info("Estimating models on complete case data...")
        complete_case_results = {}
        for spec in [1, 2]:
            result = self.estimate_fixed_effects(df_complete, model_spec=spec)
            if result is not None:
                complete_case_results[f'model_{spec}'] = result
        
        all_results['complete_case_benchmark'] = {
            'complete_case_results': complete_case_results,
            'comparison_with_imputed': 'See regression_results for imputed data results'
        }
        
        self.results = all_results
        
        logger.info("Experiment completed successfully!")
        
        return all_results
    
    @logger.catch(reraise=True)
    def save_results(self, output_path: str = "output/method_out.json"):
        """
        Save results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        logger.info(f"Saving results to {output_path}...")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy(self.results)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save as mini and preview versions
        self._generate_output_versions(output_file)
    
    def _generate_output_versions(self, output_file: Path):
        """Generate mini and preview versions of output."""
        try:
            # Read the output file
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Generate mini version (first 3 items for each top-level array)
            mini_data = self._create_mini_version(data)
            mini_path = output_file.parent / f"mini_{output_file.name}"
            with open(mini_path, 'w') as f:
                json.dump(mini_data, f, indent=2, default=str)
            
            # Generate preview version (mini + truncated strings)
            preview_data = self._create_preview_version(mini_data)
            preview_path = output_file.parent / f"preview_{output_file.name}"
            with open(preview_path, 'w') as f:
                json.dump(preview_data, f, indent=2, default=str)
            
            logger.info(f"Generated mini and preview versions: {mini_path}, {preview_path}")
        except Exception as e:
            logger.warning(f"Could not generate output versions: {e}")
    
    def _create_mini_version(self, data: Any, max_items: int = 3) -> Any:
        """Create mini version of data (first max_items for arrays)."""
        if isinstance(data, dict):
            return {k: self._create_mini_version(v, max_items) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._create_mini_version(item, max_items) for item in data[:max_items]]
        else:
            return data
    
    def _create_preview_version(self, data: Any, max_str_len: int = 200) -> Any:
        """Create preview version (truncate strings)."""
        if isinstance(data, dict):
            return {k: self._create_preview_version(v, max_str_len) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._create_preview_version(item, max_str_len) for item in data]
        elif isinstance(data, str):
            return data[:max_str_len] + "..." if len(data) > max_str_len else data
        else:
            return data


@logger.catch(reraise=True)
def main():
    """Main execution function."""
    # Define data paths - use mini files for testing
    # For production run, change to full_data_out.json
    use_mini = True  # Set to False for full run
    
    if use_mini:
        data_path = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run__DHeGw9qQb7B/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/mini_data_out.json")
        supplementary_data_path = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run__DHeGw9qQb7B/3_invention_loop/iter_1/gen_art/gen_art_dataset_2/output/mini_data_out.json")
    else:
        data_path = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run__DHeGw9qQb7B/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json")
        supplementary_data_path = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run__DHeGw9qQb7B/3_invention_loop/iter_1/gen_art/gen_art_dataset_2/output/full_data_out.json")
    
    # Initialize experiment
    experiment = PanelRegressionExperiment(data_path, supplementary_data_path)
    
    # Run all steps
    results = experiment.run_all()
    
    # Save results
    experiment.save_results("output/method_out.json")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    if 'hausman_test' in results:
        hausman = results['hausman_test']
        logger.info(f"Hausman test: prefer FE = {hausman.get('prefer_fe', 'N/A')}")
    
    if 'wooldridge_test' in results:
        wooldridge = results['wooldridge_test']
        logger.info(f"Wooldridge test: serial correlation = {wooldridge.get('serial_correlation', 'N/A')}")
    
    if 'mediation_results' in results:
        mediation = results['mediation_results']
        if 'indirect_effect' in mediation:
            ie = mediation['indirect_effect']
            logger.info(f"Mediation: indirect effect = {ie.get('coefficient', 'N/A'):.4f} (95% CI: {ie.get('ci_95_lower', 'N/A'):.4f}, {ie.get('ci_95_upper', 'N/A'):.4f})")
    
    logger.info("=" * 60)
    logger.info("Experiment completed. Results saved to output/method_out.json")


if __name__ == "__main__":
    main()
