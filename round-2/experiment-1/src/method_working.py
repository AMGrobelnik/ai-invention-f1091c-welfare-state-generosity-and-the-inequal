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
        results['missing_data'] = self._missing_analysis()
        
        # 3. Correlation
        results['correlation_matrix'] = self._correlation()
        
        # 4. Simple regression (complete cases)
        results['regression'] = self._regression()
        
        self.results = results
        return results
    
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
        return {'patterns': missing}
    
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
        # Complete cases only
        df = self.df.dropna(subset=['vdem_index', 'gini_coeff', 'social_spending_gdp_lag1'])
        
        if len(df) < 30:
            return {'error': 'Insufficient complete cases'}
        
        # Simple pooled OLS (no fixed effects for simplicity)
        X = add_constant(df[['gini_coeff', 'social_spending_gdp_lag1']])
        y = df['vdem_index']
        
        model = OLS(y, X).fit()
        
        result = {
            'n': int(len(df)),
            'r_squared': float(model.rsquared),
            'coefficients': {}
        }
        
        for var in ['gini_coeff', 'social_spending_gdp_lag1']:
            result['coefficients'][var] = {
                'coef': float(model.params[var]),
                'std_err': float(model.bse[var]),
                'p_value': float(model.pvalues[var]),
                'ci_lower': float(model.conf_int().loc[var, 0]),
                'ci_upper': float(model.conf_int().loc[var, 1])
            }
        
        logger.info(f"Regression: R²={result['r_squared']:.4f}")
        for var, coef in result['coefficients'].items():
            logger.info(f"  {var}: {coef['coef']:.4f} (p={coef['p_value']:.4f})")
        
        return result
    
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
        
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
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
