#!/usr/bin/env python3
"""Working panel regression experiment."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# Setup minimal logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
logger = logging.getLogger(__name__)

class PanelExperiment:
    def __init__(self, data_path, supplementary_data_path=None):
        self.data_path = data_path
        self.df = None
        self.results = {}
    
    def load_data(self):
        logger.info("Loading data...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        examples = data['datasets'][0]['examples']
        rows = []
        for ex in examples:
            inp = json.loads(ex['input'])
            rows.append({
                'country': inp['country'],
                'year': int(inp['year']),
                'gini_coeff': inp.get('gini_coeff'),
                'social_spending_gdp': inp.get('social_spending_gdp'),
                'secondary_edu_enrollment': inp.get('secondary_edu_enrollment'),
                'vdem_regime': inp.get('vdem_regime'),
                'social_spending_gdp_lag1': inp.get('social_spending_gdp_lag1'),
                'vdem_index': float(ex['output'])
            })
        
        df = pd.DataFrame(rows)
        
        # Convert gini to 0-1
        if df['gini_coeff'].max() > 1:
            df['gini_coeff'] = df['gini_coeff'] / 100
        
        # Create country ID
        countries = sorted(df['country'].unique())
        cid_map = {c: i for i, c in enumerate(countries)}
        df['country_id'] = df['country'].map(cid_map)
        
        self.df = df
        logger.info(f"Loaded {len(df)} observations, {len(countries)} countries")
        return df
    
    def run_analysis(self):
        logger.info("Running analysis...")
        results = {}
        
        # 1. Descriptive stats
        results['descriptive_statistics'] = self._desc_stats()
        
        # 2. Missing data
        results['missing_data_diagnostics'] = self._missing_analysis()
        
        # 3. Correlation
        results['correlation_matrix'] = self._correlation()
        
        # 4. Regression
        results['regression_results'] = self._regression()
        
        # 5. Sample attrition
        results['sample_attrition'] = self._sample_attrition()
        
        # 6. Complete case benchmark
        results['complete_case_benchmark'] = self._complete_case_benchmark()
        
        self.results = results
        return results
    
    def _sample_attrition(self):
        """Document sample attrition."""
        total = len(self.df)
        complete = len(self.df.dropna(subset=['vdem_index', 'gini_coeff', 'social_spending_gdp_lag1']))
        
        return {
            'original_N': int(total),
            'complete_case_N': int(complete),
            'attrition_rate': float((total - complete) / total * 100),
            'note': 'Listwise deletion on vdem_index, gini_coeff, social_spending_gdp_lag1'
        }
    
    def _complete_case_benchmark(self):
        """Compare complete case vs full sample descriptives."""
        df_complete = self.df.dropna(subset=['vdem_index', 'gini_coeff', 'social_spending_gdp_lag1'])
        
        benchmark = {}
        for col in ['gini_coeff', 'social_spending_gdp_lag1', 'vdem_index']:
            if col in self.df.columns:
                full_mean = float(self.df[col].dropna().mean()) if self.df[col].notna().any() else None
                complete_mean = float(df_complete[col].mean()) if col in df_complete.columns else None
                
                benchmark[col] = {
                    'full_sample_mean': full_mean,
                    'complete_case_mean': complete_mean,
                    'difference': float(complete_mean - full_mean) if full_mean and complete_mean else None
                }
        
        return benchmark
    
    def _desc_stats(self):
        stats_dict = {}
        for col in ['gini_coeff', 'social_spending_gdp', 'vdem_index', 'social_spending_gdp_lag1']:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    stats_dict[col] = {
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'n': int(len(data)),
                        'missing': int(self.df[col].isna().sum())
                    }
        return stats_dict
    
    def _missing_analysis(self):
        missing = {}
        for col in self.df.columns:
            if col in ['country', 'country_id']:
                continue
            n_missing = int(self.df[col].isna().sum())
            missing[col] = {
                'count': n_missing,
                'pct': float(n_missing / len(self.df) * 100)
            }
        
        # Add Little's MCAR test (simplified)
        mcar_test = self._littles_mcar_test()
        
        return {'patterns': missing, 'littles_mcar_test': mcar_test}
    
    def _littles_mcar_test(self):
        """Simplified Little's MCAR test."""
        # Check if missingness on key variables is correlated with other observed variables
        df_num = self.df.select_dtypes(include=[np.number])
        
        vars_with_missing = [c for c in df_num.columns if self.df[c].isna().any() and c not in ['country_id', 'year']]
        
        if len(vars_with_missing) < 2:
            return {'statistic': None, 'p_value': None, 'note': 'Insufficient variables with missing data'}
        
        # Calculate chi2 statistic by comparing means
        chi2 = 0
        n_comparisons = 0
        
        for col in vars_with_missing:
            missing_ind = self.df[col].isna().astype(int)
            
            for other_col in vars_with_missing:
                if other_col == col:
                    continue
                
                # Compare distribution of other_col when col is missing vs observed
                missing_vals = self.df.loc[missing_ind == 1, other_col].dropna()
                observed_vals = self.df.loc[missing_ind == 0, other_col].dropna()
                
                if len(missing_vals) < 5 or len(observed_vals) < 5:
                    continue
                
                t_stat, p_val = stats.ttest_ind(missing_vals, observed_vals, equal_var=False)
                if not np.isnan(t_stat):
                    chi2 += t_stat ** 2
                    n_comparisons += 1
        
        if n_comparisons > 0:
            p_value = 1 - stats.chi2.cdf(chi2, n_comparisons)
        else:
            p_value = None
        
        return {
            'statistic': float(chi2) if chi2 > 0 else None,
            'df': n_comparisons,
            'p_value': float(p_value) if p_value is not None else None,
            'note': 'Simplified test - rejects if missingness correlates with observed values'
        }
    
    def _correlation(self):
        # Use complete cases
        cols = ['vdem_index', 'gini_coeff', 'social_spending_gdp_lag1']
        cols = [c for c in cols if c in self.df.columns]
        df_corr = self.df[cols].dropna()
        
        if len(df_corr) < 10:
            return {'error': 'Insufficient data'}
        
        corr = df_corr.corr()
        return corr.to_dict()
    
    def _regression(self):
        """Run regression with and without fixed effects."""
        results = {}
        
        # Complete cases only
        df = self.df.dropna(subset=['vdem_index', 'gini_coeff', 'social_spending_gdp_lag1'])
        
        if len(df) < 30:
            return {'error': 'Insufficient complete cases'}
        
        # 1. Pooled OLS (no fixed effects)
        X_pooled = add_constant(df[['gini_coeff', 'social_spending_gdp_lag1']])
        y = df['vdem_index']
        
        model_pooled = OLS(y, X_pooled).fit()
        
        results['pooled_ols'] = {
            'n': int(len(df)),
            'r_squared': float(model_pooled.rsquared),
            'coefficients': {}
        }
        
        for var in ['gini_coeff', 'social_spending_gdp_lag1']:
            results['pooled_ols']['coefficients'][var] = {
                'coef': float(model_pooled.params[var]),
                'std_err': float(model_pooled.bse[var]),
                'p_value': float(model_pooled.pvalues[var]),
                'ci_lower': float(model_pooled.conf_int().loc[var, 0]),
                'ci_upper': float(model_pooled.conf_int().loc[var, 1])
            }
        
        # 2. Fixed effects (country dummies) - ensure numeric types
        df_fe = df.copy()
        df_fe['gini_coeff'] = pd.to_numeric(df_fe['gini_coeff'], errors='coerce')
        df_fe['social_spending_gdp_lag1'] = pd.to_numeric(df_fe['social_spending_gdp_lag1'], errors='coerce')
        df_fe['vdem_index'] = pd.to_numeric(df_fe['vdem_index'], errors='coerce')
        
        # Create design matrix with numeric columns only
        X_fe = pd.DataFrame({
            'gini_coeff': df_fe['gini_coeff'].values,
            'social_spending_gdp_lag1': df_fe['social_spending_gdp_lag1'].values
        })
        
        # Add country dummies (as numeric 0/1 columns)
        country_dummies = pd.get_dummies(df_fe['country_id'], prefix='country', drop_first=True)
        
        # Ensure all columns are numeric
        for col in country_dummies.columns:
            country_dummies[col] = pd.to_numeric(country_dummies[col], errors='coerce')
        
        X_fe = pd.concat([X_fe, country_dummies], axis=1)
        X_fe = add_constant(X_fe)
        
        y_fe = df_fe['vdem_index'].values
        
        # Check for NaN/inf
        if X_fe.isna().any().any() or np.isnan(y_fe).any():
            logger.warning("NaN values detected, skipping fixed effects")
        else:
            model_fe = OLS(y_fe, X_fe).fit()
            
            results['fixed_effects'] = {
                'n': int(len(df_fe)),
                'n_countries': int(len(df_fe['country_id'].unique())),
                'r_squared': float(model_fe.rsquared),
                'coefficients': {}
            }
            
            for var in ['gini_coeff', 'social_spending_gdp_lag1']:
                if var in model_fe.params.index:
                    results['fixed_effects']['coefficients'][var] = {
                        'coef': float(model_fe.params[var]),
                        'std_err': float(model_fe.bse[var]),
                        'p_value': float(model_fe.pvalues[var]),
                        'ci_lower': float(model_fe.conf_int().loc[var, 0]) if var in model_fe.conf_int().index else None,
                        'ci_upper': float(model_fe.conf_int().loc[var, 1]) if var in model_fe.conf_int().index else None
                    }
        
        logger.info(f"Pooled OLS: R²={results['pooled_ols']['r_squared']:.4f}")
        if 'fixed_effects' in results:
            logger.info(f"Fixed Effects: R²={results['fixed_effects']['r_squared']:.4f}")
            logger.info(f"  Number of countries: {results['fixed_effects']['n_countries']}")
        
        return results
    
    def save_results(self, path='output/method_out.json'):
        logger.info(f"Saving results to {path}")
        
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        results = convert(self.results)
        
        # Format output according to exp_gen_sol_out.json schema
        examples = []
        
        # Load original data to get input format
        data_path = self.data_path
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        original_examples = data['datasets'][0]['examples']
        
        # Add prediction results to each example
        for i, ex in enumerate(original_examples):
            example = {
                'input': ex['input'],
                'output': str(ex['output']),  # Ensure string
                'metadata_fold': ex.get('metadata_fold', 0),
                'metadata_task_type': ex.get('metadata_task_type', 'regression'),
                'metadata_n_features': ex.get('metadata_n_features', 6)
            }
            
            # Add prediction from our method
            if 'regression_results' in results and 'pooled_ols' in results['regression_results']:
                reg = results['regression_results']['pooled_ols']
                # Store prediction as string
                prediction = {
                    'gini_coef': reg['coefficients']['gini_coeff']['coef'],
                    'social_spending_coef': reg['coefficients']['social_spending_gdp_lag1']['coef'],
                    'r_squared': reg['r_squared']
                }
                example['predict_panel_regression'] = json.dumps(prediction)
            
            examples.append(example)
        
        # Wrap in datasets array
        output = {
            'datasets': [
                {
                    'dataset': 'welfare_democracy_panel_regression',
                    'examples': examples
                }
            ]
        }
        
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved to {path}")
        return path

def main():
    data_path = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run__DHeGw9qQb7B/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/mini_data_out.json")
    
    exp = PanelExperiment(data_path)
    exp.load_data()
    exp.run_analysis()
    exp.save_results("output/method_out.json")
    
    logger.info("Done!")

if __name__ == '__main__':
    main()
