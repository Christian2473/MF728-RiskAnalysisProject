import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, kurtosis

class VaRModule:
    """
    A module for calculating Value-at-Risk (VaR) using different methodologies.

    This module implements various VaR calculation approaches:
    - Historical VaR
    - Parametric VaR (Normal and Student's t-distribution)
    """

    @staticmethod
    def calculate_returns(prices):
        """
        Calculate log returns from price series.

        Args:
            prices (array-like): Time series of prices

        Returns:
            array: Log returns
        """
        return np.log(prices[1:] / prices[:-1])

    @staticmethod
    def historical_var(returns, confidence_level=0.95, holding_period=1, current_value=100):
        """
        Calculate VaR using the historical method.

        Args:
            returns (array-like): Historical returns (log returns)
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            holding_period (int): Holding period in days
            current_value (float): Current portfolio value

        Returns:
            float: Value at Risk
        """

        # Historical VaR requires sufficient data points to be statistically reliable.
        if len(returns) < 100:
            print("Warning: Historical VaR is less reliable with fewer than 100 observations")

        # Find the return threshold at the specified percentile for 1-day horizon
        percentile = 100 * (1 - confidence_level)
        var_return_1day = np.percentile(returns, percentile)

        # Scale 1-day VaR to T-day VaR using square root of time rule
        if holding_period > 1:
            var_return = var_return_1day * np.sqrt(holding_period)
        else:
            var_return = var_return_1day

        # Convert return to monetary loss (positive value), from mathematical conversion - P₁/P₀ = e^(var_return)
        var_monetary = current_value * (1 - np.exp(var_return))

        return var_monetary

    @staticmethod
    def parametric_var(returns, confidence_level=0.95, holding_period=1,
                       current_value=100, distribution='normal', df=None):
        """
        Calculate VaR using parametric method with either normal or t-distribution.

        Args:
            returns (array-like): Historical returns (log returns)
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            holding_period (int): Holding period in days
            current_value (float): Current portfolio value
            distribution (str): Either 'normal' or 't' for Student's t-distribution
            df (float): Degrees of freedom for t-distribution (estimated if None)

        Returns:
            float: Value at Risk
        """

        # Estimate parameters
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Scale parameters for holding period
        mu_scaled = mu * holding_period
        sigma_scaled = sigma * np.sqrt(holding_period)

        # Find quantile based on distribution
        if distribution.lower() == 'normal':
            quantile = norm.ppf(1 - confidence_level)
            var_return = mu_scaled + quantile * sigma_scaled
        elif distribution.lower() == 't':
            # Estimate degrees of freedom if not provided
            if df is None:
                # Better estimation of degrees of freedom, excess_kurtosis = 6/(ν-4)  for ν > 4
                excess_kurtosis = kurtosis(returns)
                df = max(4.0, min(30.0, 6.0 / excess_kurtosis + 4.0)) if excess_kurtosis > 0 else 30.0

            # Calculate t-distribution quantile without adjusting sigma
            quantile = t.ppf(1 - confidence_level, df)
            var_return = mu_scaled + quantile * sigma_scaled
        else:
            raise ValueError("Distribution must be 'normal' or 't'")

        # Convert to monetary loss
        var_monetary = current_value * (1 - np.exp(var_return))

        return var_monetary

    @staticmethod
    def bond_price_from_yield(yield_rate, cash_flows, times):
        """
        Calculate bond price from yield rate.

        Args:
            yield_rate (float): Yield to maturity
            cash_flows (array): Bond cash flows
            times (array): Payment times in years

        Returns:
            float: Bond price
        """
        return np.sum(cash_flows / (1 + yield_rate) ** times)

    @staticmethod
    def compare_var_methods(returns, holding_period=10, confidence_levels=[0.95, 0.99],
                            current_value=100):
        """
        Compare different VaR calculation methods.

        Args:
            returns (array): Historical returns
            holding_period (int): Holding period in days
            confidence_levels (list): List of confidence levels to test
            current_value (float): Current portfolio value

        Returns:
            DataFrame: Comparison of VaR methods
        """
        results = []

        # For both confidence levels
        for cl in confidence_levels:
            # Historical VaR
            hist_var = VaRModule.historical_var(
                returns, confidence_level=cl, holding_period=holding_period,
                current_value=current_value)

            # Parametric VaR (Normal)
            param_norm_var = VaRModule.parametric_var(
                returns, confidence_level=cl, holding_period=holding_period,
                current_value=current_value, distribution='normal')

            # Parametric VaR (t-dist)
            param_t_var = VaRModule.parametric_var(
                returns, confidence_level=cl, holding_period=holding_period,
                current_value=current_value, distribution='t')

            # Add to results
            results.append({
                'Confidence Level': f"{cl * 100:.1f}%",
                'Historical VaR': hist_var,
                'Parametric VaR (Normal)': param_norm_var,
                'Parametric VaR (t-dist)': param_t_var
            })

        return pd.DataFrame(results)


def test_var_module(plot=True):
    """
    Test the VaRModule with realistic fixed income data.

    Args:
        plot (bool): Whether to create visualization plots

    Returns:
        tuple: (VaR comparison DataFrame, plots if plot=True)
    """
    print("Testing VaR Module...")

    # Create test bond data
    face_value = 100
    coupon_rate = 0.04
    years_to_maturity = 7
    payments_per_year = 2
    initial_ytm = 0.035

    # Set up cash flows
    num_periods = int(years_to_maturity * payments_per_year)
    coupon_payment = face_value * coupon_rate / payments_per_year

    times = np.arange(1, num_periods + 1) / payments_per_year
    cash_flows = np.ones(num_periods) * coupon_payment
    cash_flows[-1] += face_value  # Add face value at maturity

    # Calculate initial price
    initial_price = VaRModule.bond_price_from_yield(initial_ytm, cash_flows, times)
    print(f"Initial Bond Price: ${initial_price:.2f}")

    # Generate random historical returns for YTM with higher volatility
    np.random.seed(99)
    num_days = 1000
    volatility = 0.005

    # Generate geometric Brownian motion for yields
    daily_changes = np.random.normal(0, volatility, num_days)
    ytm_history = np.zeros(num_days + 1)
    ytm_history[0] = initial_ytm

    for i in range(num_days):
        # Add some mean reversion
        mean_reversion = 0.005 * (0.035 - ytm_history[i])
        ytm_history[i + 1] = ytm_history[i] * np.exp(mean_reversion + daily_changes[i])

    # Calculate price history and returns
    price_history = np.array([VaRModule.bond_price_from_yield(ytm, cash_flows, times)
                              for ytm in ytm_history])
    returns = VaRModule.calculate_returns(price_history)

    # Test plotting functions if requested
    plots = []
    if plot:
        # Plot yield and price history
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(ytm_history * 100)
        ax1.set_title('Synthetic Yield History')
        ax1.set_ylabel('Yield (%)')
        ax1.grid(True)

        ax2.plot(price_history)
        ax2.set_title('Synthetic Bond Price History')
        ax2.set_ylabel('Price ($)')
        ax2.set_xlabel('Days')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('bond_history.png')
        plots.append('bond_history.png')
        plt.close()

        # Plot return distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        ax.hist(returns, bins=50, density=True, alpha=0.6, color='skyblue')

        # Plot normal distribution
        x = np.linspace(min(returns), max(returns), 1000)
        mu, sigma = np.mean(returns), np.std(returns)
        ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'Normal: μ={mu:.6f}, σ={sigma:.6f}')

        # Plot t-distribution
        # Estimate degrees of freedom
        excess_kurt = kurtosis(returns)
        df = max(4.0, min(30.0, 6.0 / excess_kurt + 4.0)) if excess_kurt > 0 else 30.0
        s = sigma
        ax.plot(x, t.pdf(x, df, loc=mu, scale=s), 'g-', linewidth=2,
                label=f't-dist: df={df:.1f}')

        ax.set_title('Return Distribution with Fitted Normal and t-distributions')
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('return_distribution.png')
        plots.append('return_distribution.png')
        plt.close()

    # Test VaR calculations
    # Holding periods to test
    holding_periods = [1, 5, 10, 20]

    # Test different confidence levels
    confidence_levels = [0.95, 0.99]

    all_results = {}
    for hp in holding_periods:
        print(f"\nTesting {hp}-day holding period VaR:")

        # Compare VaR methods
        var_comparison = VaRModule.compare_var_methods(
            returns,
            holding_period=hp,
            confidence_levels=confidence_levels,
            current_value=initial_price
        )

        all_results[hp] = var_comparison
        print(var_comparison.to_string(index=False, float_format=lambda x: f"${x:.2f}"))

    # Calculate VaR as percentage of initial price
    var_pcts = all_results[10].copy()
    for col in var_pcts.columns:
        if col != 'Confidence Level':
            var_pcts[col] = var_pcts[col] / initial_price * 100
            var_pcts = var_pcts.rename(columns={col: f"{col} (%)"})

    print("\nVaR as percentage of portfolio value (10-day holding period):")
    print(var_pcts.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

    # Plot VaR comparison if requested
    if plot:
        # Reshape for better plotting
        plot_data = all_results[10].melt(
            id_vars=['Confidence Level'],
            var_name='Method',
            value_name='VaR'
        )

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get unique confidence levels and methods
        cl_levels = plot_data['Confidence Level'].unique()
        methods = plot_data['Method'].unique()

        # Set width of bars
        bar_width = 0.15
        index = np.arange(len(methods))

        # Colors for different confidence levels
        colors = ['skyblue', 'tomato']

        # Plot bars for each confidence level
        for i, cl in enumerate(cl_levels):
            values = plot_data[plot_data['Confidence Level'] == cl]['VaR'].values
            ax.bar(index + i * bar_width, values, bar_width, label=cl, color=colors[i])

        # Customize plot
        ax.set_xlabel('VaR Method')
        ax.set_ylabel('Value at Risk ($)')
        ax.set_title('Comparison of VaR Methods (10-day Holding Period)')
        ax.set_xticks(index + bar_width * (len(cl_levels) - 1) / 2)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(title='Confidence Level')
        ax.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('var_comparison.png')
        plots.append('var_comparison.png')
        plt.close()

        # Plot VaR by holding period for 95% confidence level
        hp_data = []
        for hp, df in all_results.items():
            row = df[df['Confidence Level'] == '95.0%'].iloc[0].to_dict()
            row['Holding Period'] = hp
            hp_data.append(row)

        hp_df = pd.DataFrame(hp_data)

        # Reshape for plotting
        hp_plot_data = hp_df.melt(
            id_vars=['Holding Period', 'Confidence Level'],
            var_name='Method',
            value_name='VaR'
        )

        # Create line plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot lines for each method
        methods = hp_plot_data['Method'].unique()
        for method in methods:
            method_data = hp_plot_data[hp_plot_data['Method'] == method]
            ax.plot(method_data['Holding Period'], method_data['VaR'],
                    marker='o', label=method)

        # Square root of time rule reference line
        ref_var = hp_plot_data[
            (hp_plot_data['Method'] == 'Historical VaR') &
            (hp_plot_data['Holding Period'] == 1)
            ]['VaR'].values[0]

        holding_periods_arr = np.array(holding_periods)
        ax.plot(holding_periods_arr, ref_var * np.sqrt(holding_periods_arr),
                'k--', label='√t Rule')

        # Customize plot
        ax.set_xlabel('Holding Period (days)')
        ax.set_ylabel('Value at Risk ($)')
        ax.set_title('VaR by Holding Period (95% Confidence Level)')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig('var_by_holding_period.png')
        plots.append('var_by_holding_period.png')
        plt.close()

    # Return results and plots
    return all_results[10], plots if plot else None


if __name__ == "__main__":
    var_comparison, plots = test_var_module()

    print("\nVaR module tests completed.")