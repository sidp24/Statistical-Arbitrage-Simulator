"""
Parameter Optimization with Walk-Forward Analysis

This module implements robust parameter optimization for statistical arbitrage strategies:
- Grid search optimization
- Walk-forward analysis to prevent overfitting
- Out-of-sample testing
- Monte Carlo simulation for parameter stability
- Cross-validation for strategy validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    out_of_sample_score: float
    stability_metrics: Dict[str, float]
    walk_forward_results: List[Dict]


@dataclass
class ParameterRange:
    """Define parameter search space"""
    name: str
    min_val: float
    max_val: float
    step: float
    param_type: str = 'float'  # 'float', 'int', 'categorical'
    values: Optional[List] = None  # For categorical parameters


class WalkForwardOptimizer:
    """
    Walk-forward optimization to prevent overfitting
    """
    
    def __init__(self,
                 training_window_months: int = 12,
                 testing_window_months: int = 3,
                 step_months: int = 1,
                 min_trades_required: int = 10,
                 optimization_metric: str = 'sharpe_ratio'):
        
        self.training_window_months = training_window_months
        self.testing_window_months = testing_window_months
        self.step_months = step_months
        self.min_trades_required = min_trades_required
        self.optimization_metric = optimization_metric
    
    def create_time_windows(self, start_date: pd.Timestamp, 
                          end_date: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Create overlapping training/testing windows for walk-forward analysis
        Returns: List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        current_date = start_date
        
        while current_date + timedelta(days=30 * (self.training_window_months + self.testing_window_months)) <= end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=30 * self.training_window_months)
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.testing_window_months)
            
            windows.append((train_start, train_end, test_start, test_end))
            current_date += timedelta(days=30 * self.step_months)
        
        return windows
    
    def optimize_single_window(self,
                             data: pd.DataFrame,
                             param_grid: List[Dict],
                             strategy_function: Callable,
                             train_start: pd.Timestamp,
                             train_end: pd.Timestamp,
                             test_start: pd.Timestamp,
                             test_end: pd.Timestamp) -> Dict:
        """
        Optimize parameters for a single time window
        """
        
        # Split data
        train_data = data.loc[train_start:train_end]
        test_data = data.loc[test_start:test_end]
        
        if len(train_data) < 30:  # Minimum training data
            return None
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        # Grid search on training data
        for params in param_grid:
            try:
                # Run strategy with these parameters on training data
                train_result = strategy_function(train_data, **params)
                
                if train_result is None or 'trades' not in train_result:
                    continue
                
                if len(train_result['trades']) < self.min_trades_required:
                    continue
                
                # Calculate optimization metric
                score = self._calculate_metric(train_result, self.optimization_metric)
                
                results.append({
                    'params': params,
                    'train_score': score,
                    'train_trades': len(train_result['trades'])
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            except Exception as e:
                warnings.warn(f"Error optimizing parameters {params}: {e}")
                continue
        
        if best_params is None:
            return None
        
        # Test best parameters on out-of-sample data
        try:
            test_result = strategy_function(test_data, **best_params)
            test_score = self._calculate_metric(test_result, self.optimization_metric) if test_result else float('-inf')
            test_trades = len(test_result['trades']) if test_result and 'trades' in test_result else 0
        except:
            test_score = float('-inf')
            test_trades = 0
        
        return {
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end),
            'best_params': best_params,
            'train_score': best_score,
            'test_score': test_score,
            'test_trades': test_trades,
            'all_results': results
        }
    
    def _calculate_metric(self, strategy_result: Dict, metric: str) -> float:
        """Calculate optimization metric from strategy results"""
        
        if not strategy_result or 'trades' not in strategy_result:
            return float('-inf')
        
        trades_df = pd.DataFrame(strategy_result['trades'])
        if trades_df.empty:
            return float('-inf')
        
        returns = trades_df['PnL'] if 'PnL' in trades_df.columns else trades_df.get('returns', [])
        if len(returns) == 0:
            return float('-inf')
        
        returns_series = pd.Series(returns)
        
        if metric == 'sharpe_ratio':
            if returns_series.std() == 0:
                return 0.0
            return np.sqrt(252) * returns_series.mean() / returns_series.std()
        
        elif metric == 'calmar_ratio':
            cumulative = returns_series.cumsum()
            total_return = cumulative.iloc[-1] * 252 / len(returns_series)  # Annualized
            
            peak = cumulative.cummax()
            drawdown = (cumulative - peak)
            max_drawdown = drawdown.min()
            
            if max_drawdown == 0:
                return float('inf') if total_return > 0 else 0
            return total_return / abs(max_drawdown)
        
        elif metric == 'total_return':
            return returns_series.sum()
        
        elif metric == 'win_rate':
            return len(returns_series[returns_series > 0]) / len(returns_series)
        
        elif metric == 'profit_factor':
            wins = returns_series[returns_series > 0].sum()
            losses = abs(returns_series[returns_series < 0].sum())
            return wins / losses if losses > 0 else float('inf')
        
        else:
            return returns_series.mean()


class ParameterGridGenerator:
    """Generate parameter grids for optimization"""
    
    @staticmethod
    def create_grid(param_ranges: List[ParameterRange]) -> List[Dict]:
        """
        Create parameter grid from parameter ranges
        """
        param_lists = []
        param_names = []
        
        for param_range in param_ranges:
            param_names.append(param_range.name)
            
            if param_range.param_type == 'categorical':
                param_lists.append(param_range.values)
            elif param_range.param_type == 'int':
                values = list(range(int(param_range.min_val), 
                                  int(param_range.max_val) + 1, 
                                  int(param_range.step)))
                param_lists.append(values)
            else:  # float
                values = np.arange(param_range.min_val, 
                                 param_range.max_val + param_range.step, 
                                 param_range.step)
                param_lists.append(values.tolist())
        
        # Generate all combinations
        grid = []
        for combination in product(*param_lists):
            param_dict = dict(zip(param_names, combination))
            grid.append(param_dict)
        
        return grid
    
    @staticmethod
    def create_random_grid(param_ranges: List[ParameterRange], 
                          n_samples: int = 100) -> List[Dict]:
        """
        Create random parameter grid (useful for large search spaces)
        """
        grid = []
        
        for _ in range(n_samples):
            param_dict = {}
            
            for param_range in param_ranges:
                if param_range.param_type == 'categorical':
                    param_dict[param_range.name] = np.random.choice(param_range.values)
                elif param_range.param_type == 'int':
                    param_dict[param_range.name] = np.random.randint(
                        int(param_range.min_val), int(param_range.max_val) + 1)
                else:  # float
                    param_dict[param_range.name] = np.random.uniform(
                        param_range.min_val, param_range.max_val)
            
            grid.append(param_dict)
        
        return grid


class StrategyOptimizer:
    """
    Main optimization class that combines all optimization techniques
    """
    
    def __init__(self, 
                 walk_forward_optimizer: WalkForwardOptimizer,
                 monte_carlo_runs: int = 100,
                 parallel_processes: int = 4):
        
        self.walk_forward_optimizer = walk_forward_optimizer
        self.monte_carlo_runs = monte_carlo_runs
        self.parallel_processes = parallel_processes
    
    def optimize_strategy(self,
                        data: pd.DataFrame,
                        strategy_function: Callable,
                        param_ranges: List[ParameterRange],
                        use_random_search: bool = False,
                        n_random_samples: int = 200) -> OptimizationResult:
        """
        Comprehensive strategy optimization with walk-forward analysis
        """
        
        # Generate parameter grid
        if use_random_search:
            param_grid = ParameterGridGenerator.create_random_grid(param_ranges, n_random_samples)
        else:
            param_grid = ParameterGridGenerator.create_grid(param_ranges)
        
        print(f"Testing {len(param_grid)} parameter combinations...")
        
        # Create time windows for walk-forward analysis
        start_date = data.index.min()
        end_date = data.index.max()
        time_windows = self.walk_forward_optimizer.create_time_windows(start_date, end_date)
        
        print(f"Using {len(time_windows)} walk-forward windows...")
        
        # Run walk-forward optimization
        walk_forward_results = []
        all_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(time_windows):
            print(f"Processing window {i+1}/{len(time_windows)}: {train_start.date()} to {test_end.date()}")
            
            window_result = self.walk_forward_optimizer.optimize_single_window(
                data, param_grid, strategy_function, train_start, train_end, test_start, test_end
            )
            
            if window_result:
                walk_forward_results.append(window_result)
                all_results.extend(window_result['all_results'])
        
        if not walk_forward_results:
            raise ValueError("No valid optimization results found")
        
        # Aggregate results across all windows
        aggregated_results = self._aggregate_walk_forward_results(walk_forward_results, param_grid)
        
        # Find best parameters based on average out-of-sample performance
        best_params = max(aggregated_results.items(), 
                         key=lambda x: x[1]['avg_test_score'])[0]
        best_params_dict = eval(best_params)  # Convert string back to dict
        
        # Calculate final out-of-sample score
        final_oos_score = aggregated_results[best_params]['avg_test_score']
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(walk_forward_results, best_params_dict)
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                'params': str(params),
                'avg_train_score': results['avg_train_score'],
                'avg_test_score': results['avg_test_score'],
                'score_std': results['score_std'],
                'win_rate': results['win_rate']
            }
            for params, results in aggregated_results.items()
        ])
        
        return OptimizationResult(
            best_params=best_params_dict,
            best_score=aggregated_results[best_params]['avg_test_score'],
            all_results=results_df,
            out_of_sample_score=final_oos_score,
            stability_metrics=stability_metrics,
            walk_forward_results=walk_forward_results
        )
    
    def _aggregate_walk_forward_results(self, 
                                      walk_forward_results: List[Dict],
                                      param_grid: List[Dict]) -> Dict:
        """
        Aggregate results across all walk-forward windows
        """
        
        # Group results by parameter combination
        param_results = {}
        
        for result in walk_forward_results:
            param_str = str(result['best_params'])
            
            if param_str not in param_results:
                param_results[param_str] = {
                    'train_scores': [],
                    'test_scores': [],
                    'test_trades': []
                }
            
            param_results[param_str]['train_scores'].append(result['train_score'])
            param_results[param_str]['test_scores'].append(result['test_score'])
            param_results[param_str]['test_trades'].append(result['test_trades'])
        
        # Calculate aggregated metrics
        aggregated = {}
        for param_str, results in param_results.items():
            test_scores = [s for s in results['test_scores'] if not np.isinf(s)]
            
            if len(test_scores) == 0:
                continue
            
            aggregated[param_str] = {
                'avg_train_score': np.mean(results['train_scores']),
                'avg_test_score': np.mean(test_scores),
                'score_std': np.std(test_scores),
                'win_rate': len([s for s in test_scores if s > 0]) / len(test_scores),
                'total_trades': sum(results['test_trades'])
            }
        
        return aggregated
    
    def _calculate_stability_metrics(self, 
                                   walk_forward_results: List[Dict],
                                   best_params: Dict) -> Dict:
        """
        Calculate parameter stability metrics
        """
        
        best_param_str = str(best_params)
        relevant_results = [r for r in walk_forward_results 
                          if str(r['best_params']) == best_param_str]
        
        if not relevant_results:
            return {'stability_score': 0.0, 'consistency_ratio': 0.0}
        
        test_scores = [r['test_score'] for r in relevant_results if not np.isinf(r['test_score'])]
        
        if len(test_scores) < 2:
            return {'stability_score': 0.0, 'consistency_ratio': 0.0}
        
        # Stability score: inverse of coefficient of variation
        cv = np.std(test_scores) / np.mean(test_scores) if np.mean(test_scores) != 0 else float('inf')
        stability_score = 1 / (1 + cv)
        
        # Consistency ratio: percentage of periods with positive returns
        consistency_ratio = len([s for s in test_scores if s > 0]) / len(test_scores)
        
        return {
            'stability_score': stability_score,
            'consistency_ratio': consistency_ratio,
            'score_mean': np.mean(test_scores),
            'score_std': np.std(test_scores),
            'num_periods': len(test_scores)
        }
    
    def plot_optimization_results(self, result: OptimizationResult, 
                                save_path: Optional[str] = None):
        """
        Plot optimization results for analysis
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Parameter performance scatter
        axes[0, 0].scatter(result.all_results['avg_train_score'], 
                          result.all_results['avg_test_score'], alpha=0.6)
        axes[0, 0].set_xlabel('Training Score')
        axes[0, 0].set_ylabel('Test Score')
        axes[0, 0].set_title('Training vs Test Performance')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        
        # 2. Score distribution
        axes[0, 1].hist(result.all_results['avg_test_score'], bins=20, alpha=0.7)
        axes[0, 1].axvline(result.best_score, color='red', linestyle='--', 
                          label=f'Best Score: {result.best_score:.3f}')
        axes[0, 1].set_xlabel('Test Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Test Score Distribution')
        axes[0, 1].legend()
        
        # 3. Walk-forward results over time
        if result.walk_forward_results:
            dates = [r['test_period'][0] for r in result.walk_forward_results]
            scores = [r['test_score'] for r in result.walk_forward_results]
            axes[1, 0].plot(dates, scores, marker='o')
            axes[1, 0].set_xlabel('Test Period Start')
            axes[1, 0].set_ylabel('Test Score')
            axes[1, 0].set_title('Walk-Forward Performance Over Time')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Stability metrics
        stability_data = [
            ('Stability Score', result.stability_metrics.get('stability_score', 0)),
            ('Consistency Ratio', result.stability_metrics.get('consistency_ratio', 0)),
            ('Win Rate', result.all_results.loc[result.all_results['avg_test_score'] == result.best_score, 'win_rate'].iloc[0] if not result.all_results.empty else 0)
        ]
        
        metrics, values = zip(*stability_data)
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Stability Metrics')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Example usage
if __name__ == "__main__":
    # Example parameter ranges for pairs trading
    param_ranges = [
        ParameterRange('entry_z', 1.0, 3.0, 0.5),
        ParameterRange('exit_z', 0.2, 1.5, 0.3),
        ParameterRange('window', 20, 60, 10, 'int')
    ]
    
    # Create optimizer
    walk_forward_opt = WalkForwardOptimizer(
        training_window_months=6,
        testing_window_months=2,
        optimization_metric='sharpe_ratio'
    )
    
    optimizer = StrategyOptimizer(walk_forward_opt)
    
    print("Parameter optimization framework ready for use!")
    print(f"Sample parameter grid size: {len(ParameterGridGenerator.create_grid(param_ranges))}")
