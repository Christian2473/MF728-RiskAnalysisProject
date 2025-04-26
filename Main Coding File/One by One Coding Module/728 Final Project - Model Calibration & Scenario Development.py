import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import seaborn as sns

from YieldCurveConstruction import YieldCurve, TreasuryYieldCurve, SpotRateYieldCurve, ForwardRateYieldCurve
from PricingFunction import PricingModule
from VarFunction import VaRModule

class ScenarioAnalysis:
    """
    Module for creating and analyzing interest rate scenarios and their impact on fixed income portfolios.

    This module supports:
    1. Historical scenario analysis (e.g., 2020 COVID crash, 2022 inflation-driven rate hikes)
    2. Hypothetical scenario analysis (parallel shifts, steepeners, flatteners)
    3. Assessment of portfolio performance under different scenarios
    """

    def __init__(self, base_yield_curve, portfolio=None):
        """
        Initialize the scenario analysis module.

        Parameters:
            base_yield_curve (YieldCurve): Base yield curve to use as a starting point
            portfolio (dict or pd.DataFrame, optional): Portfolio of fixed income instruments
        """
        self.base_curve = base_yield_curve
        self.portfolio = portfolio
        self.scenarios = {}
        self.scenario_results = {}

    def add_portfolio(self, portfolio):
        """
        Add or update the portfolio to analyze.

        Parameters:
            portfolio (dict or pd.DataFrame): Portfolio of fixed income instruments
        """
        self.portfolio = portfolio

    def create_historical_scenario(self, scenario_name, shift_type, magnitude,
                                   reference_date=None, description=None):
        """
        Create a historical scenario based on specific market events.

        Parameters:
            scenario_name (str): Name of the scenario
            shift_type (str): Type of shift ('parallel', 'steepener', 'flattener', 'inversion')
            magnitude (float or dict): Magnitude of the shift in basis points
            reference_date (datetime, optional): Reference date for the scenario
            description (str, optional): Description of the scenario

        Returns:
            YieldCurve: The yield curve under the scenario
        """

        if reference_date is None:
            reference_date = self.base_curve.reference_date

        # Create a copy of the base curve's data
        scenario_tenors = np.copy(self.base_curve.tenors)
        scenario_rates = np.copy(self.base_curve.rates)

        # Apply the appropriate shift based on shift_type
        if shift_type == 'parallel':
            # Simple parallel shift of the entire curve
            scenario_rates = scenario_rates + magnitude / 10000

        elif shift_type == 'steepener':
            # Steepening shift (short end down or unchanged, long end up): r₂(t) = r₁(t) + (t/t_max) × m/10000
            # if we're using the dictionary approach for precise control
            if isinstance(magnitude, dict):
                for tenor, shift in magnitude.items():
                    idx = np.abs(scenario_tenors - tenor).argmin()
                    scenario_rates[idx] += shift / 10000
            else:
                # Apply a graduated increase based on tenor
                steepening_factor = scenario_tenors / np.max(scenario_tenors)
                scenario_rates += steepening_factor * (magnitude / 10000)

        elif shift_type == 'flattener':
            # Flattening shift (short end up, long end down or unchanged): r₂(t) = r₁(t) + (1 - t/t_max) × m/10000
            if isinstance(magnitude, dict):
                for tenor, shift in magnitude.items():
                    idx = np.abs(scenario_tenors - tenor).argmin()
                    scenario_rates[idx] += shift / 10000
            else:
                # Apply a graduated decrease based on tenor
                flattening_factor = 1 - (scenario_tenors / np.max(scenario_tenors))
                scenario_rates += flattening_factor * (magnitude / 10000)

        elif shift_type == 'inversion':
            # Inversion shift (short end up significantly, long end down): r₂(t) = r₁(t) + (1 - 2t/t_max) × m/10000
            if isinstance(magnitude, dict):
                for tenor, shift in magnitude.items():
                    idx = np.abs(scenario_tenors - tenor).argmin()
                    scenario_rates[idx] += shift / 10000
            else:
                # Create an inversion effect
                inversion_factor = 1 - (2 * scenario_tenors / np.max(scenario_tenors))
                scenario_rates += inversion_factor * (magnitude / 10000)

        elif shift_type == 'custom':
            # Custom shift provided as a dictionary of tenor: shift pairs
            if not isinstance(magnitude, dict):
                raise ValueError("For custom shift_type, magnitude must be a dictionary of tenor: shift pairs")

            for tenor, shift in magnitude.items():
                idx = np.abs(scenario_tenors - tenor).argmin()
                scenario_rates[idx] += shift / 10000

        else:
            raise ValueError(f"Unknown shift_type: {shift_type}")

        # Create a new yield curve with the shifted rates
        if isinstance(self.base_curve, TreasuryYieldCurve):
            scenario_curve = TreasuryYieldCurve(scenario_tenors, scenario_rates, reference_date)
        elif isinstance(self.base_curve, SpotRateYieldCurve):
            scenario_curve = SpotRateYieldCurve(scenario_tenors, scenario_rates, reference_date=reference_date)
        elif isinstance(self.base_curve, ForwardRateYieldCurve):
            scenario_curve = ForwardRateYieldCurve(scenario_tenors, scenario_rates, reference_date=reference_date)
        else:
            scenario_curve = YieldCurve(scenario_tenors, scenario_rates, reference_date,
                                        curve_type=f"{self.base_curve.curve_type} - {scenario_name}")

        # Store the scenario
        self.scenarios[scenario_name] = {
            'curve': scenario_curve,
            'shift_type': shift_type,
            'magnitude': magnitude,
            'description': description or f"{shift_type.capitalize()} shift of {magnitude} bps"
        }

        return scenario_curve

    def create_covid_crisis_scenario(self, reference_date=None):
        """
        Create a scenario based on the 2020 COVID market crash.

        Parameters:
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve under the COVID crisis scenario
        """

        # 1. Flight to safety - steep drop in short and medium term rates
        # 2. Significant flattening of the curve
        # 3. Some inversion in parts of the curve

        # Define shifts for different tenors in basis points
        covid_shifts = {
            0.25: -125,  # 3-month rate drop
            0.5: -120,  # 6-month rate drop
            1: -100,  # 1-year rate drop
            2: -70,  # 2-year rate drop
            3: -65,  # 3-year rate drop
            5: -55,  # 5-year rate drop
            7: -45,  # 7-year rate drop
            10: -40,  # 10-year rate drop
            20: -35,  # 20-year rate drop
            30: -30  # 30-year rate drop
        }

        description = ("COVID-19 Crisis Scenario (March 2020): Flight to safety causing "
                       "significant drops across the curve, with more pronounced drops in "
                       "short to medium term rates.")

        return self.create_historical_scenario("COVID_Crisis_2020",
                                               "custom",
                                               covid_shifts,
                                               reference_date,
                                               description)

    def create_inflation_hike_scenario(self, reference_date=None):
        """
        Create a scenario based on the 2022 inflation-driven rate hikes.

        Parameters:
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve under the inflation rate hike scenario
        """

        # 1. Significant increases across the entire curve
        # 2. More pronounced increases in short-term rates
        # 3. Some flattening and inversion at times

        # Define shifts for different tenors in basis points
        inflation_shifts = {
            0.25: +300,  # 3-month rate increase
            0.5: +290,  # 6-month rate increase
            1: +260,  # 1-year rate increase
            2: +230,  # 2-year rate increase
            3: +210,  # 3-year rate increase
            5: +190,  # 5-year rate increase
            7: +170,  # 7-year rate increase
            10: +150,  # 10-year rate increase
            20: +125,  # 20-year rate increase
            30: +100  # 30-year rate increase
        }

        description = ("Inflation Rate Hike Scenario (2022): Fed aggressive tightening "
                       "causing significant increases across the curve, with more pronounced "
                       "increases in short-term rates leading to flattening and some inversion.")

        return self.create_historical_scenario("Inflation_Hike_2022",
                                               "custom",
                                               inflation_shifts,
                                               reference_date,
                                               description)

    def create_parallel_shift_scenario(self, bps_shift, scenario_name=None, reference_date=None):
        """
        Create a parallel shift scenario.

        Parameters:
            bps_shift (int): Shift in basis points (can be positive or negative)
            scenario_name (str, optional): Name for the scenario
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the parallel shift
        """

        if scenario_name is None:
            direction = "Up" if bps_shift > 0 else "Down"
            scenario_name = f"Parallel_{direction}_{abs(bps_shift)}bps"

        description = f"Parallel shift of {bps_shift} bps across the entire yield curve."

        return self.create_historical_scenario(scenario_name,
                                               "parallel",
                                               bps_shift,
                                               reference_date,
                                               description)

    def create_steepener_scenario(self, magnitude=100, scenario_name=None, reference_date=None):
        """
        Create a steepener scenario where the long end increases more than the short end.

        Parameters:
            magnitude (int or dict): Maximum shift in basis points or a dictionary of tenor: shift pairs
            scenario_name (str, optional): Name for the scenario
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the steepening shift
        """
        if scenario_name is None:
            scenario_name = f"Bull_Steepener_{magnitude}bps"

        description = (f"Steepening scenario with graduated increases up to {magnitude} bps "
                       f"at the long end of the curve.")

        return self.create_historical_scenario(scenario_name,
                                               "steepener",
                                               magnitude,
                                               reference_date,
                                               description)

    def create_flattener_scenario(self, magnitude=100, scenario_name=None, reference_date=None):
        """
        Create a flattener scenario where the short end increases more than the long end.

        Parameters:
            magnitude (int or dict): Maximum shift in basis points or a dictionary of tenor: shift pairs
            scenario_name (str, optional): Name for the scenario
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the flattening shift
        """
        if scenario_name is None:
            scenario_name = f"Bear_Flattener_{magnitude}bps"

        description = (f"Flattening scenario with graduated increases up to {magnitude} bps "
                       f"at the short end of the curve.")

        return self.create_historical_scenario(scenario_name,
                                               "flattener",
                                               magnitude,
                                               reference_date,
                                               description)

    def create_bull_steepener_scenario(self, short_end_drop=50, reference_date=None):
        """
        Create a bull steepener scenario (short end drops, long end relatively unchanged).

        Parameters:
            short_end_drop (int): Drop in basis points at the short end
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the bull steepener shift
        """

        # Bull steepener: short rates drop more than long rates
        bull_steepener_shifts = {
            0.25: -short_end_drop,
            0.5: -short_end_drop,
            1: -int(short_end_drop * 0.9),
            2: -int(short_end_drop * 0.7),
            3: -int(short_end_drop * 0.5),
            5: -int(short_end_drop * 0.3),
            7: -int(short_end_drop * 0.2),
            10: -int(short_end_drop * 0.1),
            20: 0,
            30: 0
        }

        scenario_name = f"Bull_Steepener_{short_end_drop}bps"
        description = (f"Bull Steepener: Short-end rates drop by {short_end_drop} bps, "
                       f"with diminishing effect toward the long end.")

        return self.create_historical_scenario(scenario_name,
                                               "custom",
                                               bull_steepener_shifts,
                                               reference_date,
                                               description)

    def create_bear_steepener_scenario(self, long_end_rise=50, reference_date=None):
        """
        Create a bear steepener scenario (long end rises, short end relatively unchanged).

        Parameters:
            long_end_rise (int): Rise in basis points at the long end
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the bear steepener shift
        """

        # Bear steepener: long rates rise more than short rates
        bear_steepener_shifts = {
            0.25: 0,
            0.5: 0,
            1: int(long_end_rise * 0.1),
            2: int(long_end_rise * 0.2),
            3: int(long_end_rise * 0.3),
            5: int(long_end_rise * 0.5),
            7: int(long_end_rise * 0.7),
            10: int(long_end_rise * 0.9),
            20: long_end_rise,
            30: long_end_rise
        }

        scenario_name = f"Bear_Steepener_{long_end_rise}bps"
        description = (f"Bear Steepener: Long-end rates rise by {long_end_rise} bps, "
                       f"with diminishing effect toward the short end.")

        return self.create_historical_scenario(scenario_name,
                                               "custom",
                                               bear_steepener_shifts,
                                               reference_date,
                                               description)

    def create_bull_flattener_scenario(self, long_end_drop=50, reference_date=None):
        """
        Create a bull flattener scenario (long end drops more than short end).

        Parameters:
            long_end_drop (int): Drop in basis points at the long end
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the bull flattener shift
        """

        # Bull flattener: long rates drop more than short rates
        bull_flattener_shifts = {
            0.25: 0,
            0.5: 0,
            1: -int(long_end_drop * 0.1),
            2: -int(long_end_drop * 0.2),
            3: -int(long_end_drop * 0.3),
            5: -int(long_end_drop * 0.5),
            7: -int(long_end_drop * 0.7),
            10: -int(long_end_drop * 0.9),
            20: -long_end_drop,
            30: -long_end_drop
        }

        scenario_name = f"Bull_Flattener_{long_end_drop}bps"
        description = (f"Bull Flattener: Long-end rates drop by {long_end_drop} bps, "
                       f"with diminishing effect toward the short end.")

        return self.create_historical_scenario(scenario_name,
                                               "custom",
                                               bull_flattener_shifts,
                                               reference_date,
                                               description)

    def create_bear_flattener_scenario(self, short_end_rise=50, reference_date=None):
        """
        Create a bear flattener scenario (short end rises more than long end).

        Parameters:
            short_end_rise (int): Rise in basis points at the short end
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the bear flattener shift
        """

        # Bear flattener: short rates rise more than long rates
        bear_flattener_shifts = {
            0.25: short_end_rise,
            0.5: short_end_rise,
            1: int(short_end_rise * 0.9),
            2: int(short_end_rise * 0.7),
            3: int(short_end_rise * 0.5),
            5: int(short_end_rise * 0.3),
            7: int(short_end_rise * 0.2),
            10: int(short_end_rise * 0.1),
            20: 0,
            30: 0
        }

        scenario_name = f"Bear_Flattener_{short_end_rise}bps"
        description = (f"Bear Flattener: Short-end rates rise by {short_end_rise} bps, "
                       f"with diminishing effect toward the long end.")

        return self.create_historical_scenario(scenario_name,
                                               "custom",
                                               bear_flattener_shifts,
                                               reference_date,
                                               description)

    def create_inversion_scenario(self, magnitude=50, reference_date=None):
        """
        Create a yield curve inversion scenario.

        Parameters:
            magnitude (int): Maximum inversion in basis points
            reference_date (datetime, optional): Reference date for the scenario

        Returns:
            YieldCurve: The yield curve with the inversion
        """

        # Inversion scenario: short rates rise, mid rates relatively stable, long rates drop
        inversion_shifts = {
            0.25: magnitude,
            0.5: magnitude,
            1: int(magnitude * 0.8),
            2: int(magnitude * 0.5),
            3: int(magnitude * 0.2),
            5: 0,
            7: -int(magnitude * 0.2),
            10: -int(magnitude * 0.4),
            20: -int(magnitude * 0.6),
            30: -int(magnitude * 0.8)
        }

        scenario_name = f"Inversion_{magnitude}bps"
        description = (f"Yield Curve Inversion: Short-end rates rise by {magnitude} bps, "
                       f"while long-end rates fall, creating an inverted curve.")

        return self.create_historical_scenario(scenario_name,
                                               "custom",
                                               inversion_shifts,
                                               reference_date,
                                               description)

    def evaluate_bond_under_scenario(self, cash_flows, times, base_ytm, scenario_name):
        """
        Evaluate a single bond under a specific scenario.

        Parameters:
            cash_flows (array-like): The cash flows of the bond
            times (array-like): The times at which the cash flows occur
            base_ytm (float): The base yield to maturity of the bond
            scenario_name (str): The name of the scenario to evaluate under

        Returns:
            dict: Dictionary of evaluation results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        scenario = self.scenarios[scenario_name]
        scenario_curve = scenario['curve']

        # Calculate base price
        base_price = PricingModule.calculate_price(cash_flows, times, base_ytm)

        # Calculate base duration and convexity
        base_duration = PricingModule.calculate_modified_duration(cash_flows, times, base_ytm)
        base_convexity = PricingModule.calculate_convexity(cash_flows, times, base_ytm)

        # Find the yield under the scenario - simplification approach

        # Estimate the bond's tenor as the duration
        tenor = base_duration

        # Get the base rate for this tenor
        base_rate = self.base_curve.get_rate(tenor)

        # Get the scenario rate for this tenor
        scenario_rate = scenario_curve.get_rate(tenor)

        # Calculate the yield shift
        yield_shift = scenario_rate - base_rate

        # Apply the yield shift to the bond's YTM
        scenario_ytm = base_ytm + yield_shift

        # Calculate scenario price
        scenario_price = PricingModule.calculate_price(cash_flows, times, scenario_ytm)

        # Calculate price change
        price_change = scenario_price - base_price
        percent_change = price_change / base_price * 100

        # Calculate expected price change using different approximation methods
        duration_approx = PricingModule.duration_based_price_change(
            cash_flows, times, base_ytm, yield_shift)

        convexity_approx = PricingModule.duration_convexity_based_price_change(
            cash_flows, times, base_ytm, yield_shift)

        # Calculate approximation errors
        duration_error = duration_approx - price_change
        duration_error_pct = duration_error / price_change * 100 if price_change != 0 else 0

        convexity_error = convexity_approx - price_change
        convexity_error_pct = convexity_error / price_change * 100 if price_change != 0 else 0

        return {
            'scenario_name': scenario_name,
            'base_ytm': base_ytm,
            'scenario_ytm': scenario_ytm,
            'yield_shift': yield_shift,
            'base_price': base_price,
            'scenario_price': scenario_price,
            'price_change': price_change,
            'percent_change': percent_change,
            'base_duration': base_duration,
            'base_convexity': base_convexity,
            'duration_approx': duration_approx,
            'duration_error': duration_error,
            'duration_error_pct': duration_error_pct,
            'convexity_approx': convexity_approx,
            'convexity_error': convexity_error,
            'convexity_error_pct': convexity_error_pct
        }

    def evaluate_portfolio_under_scenario(self, portfolio, scenario_name):
        """
        Evaluate a portfolio of bonds under a specific scenario.

        Parameters:
            portfolio (dict): Dictionary with bond details
            scenario_name (str): The name of the scenario to evaluate under

        Returns:
            dict: Dictionary of evaluation results for the portfolio
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        results = []
        total_base_value = 0
        total_scenario_value = 0

        for bond_id, bond in portfolio.items():
            bond_result = self.evaluate_bond_under_scenario(
                bond['cash_flows'],
                bond['times'],
                bond['ytm'],
                scenario_name
            )

            # Add bond ID and weight to the results
            bond_result['bond_id'] = bond_id
            bond_result['weight'] = bond.get('weight', 1.0)
            bond_result['amount'] = bond.get('amount', 1)

            # Calculate weighted values
            weighted_base_value = bond_result['base_price'] * bond_result['amount'] * bond_result['weight']
            weighted_scenario_value = bond_result['scenario_price'] * bond_result['amount'] * bond_result['weight']

            bond_result['weighted_base_value'] = weighted_base_value
            bond_result['weighted_scenario_value'] = weighted_scenario_value

            total_base_value += weighted_base_value
            total_scenario_value += weighted_scenario_value

            results.append(bond_result)

        # Calculate portfolio-level metrics
        portfolio_change = total_scenario_value - total_base_value
        portfolio_percent_change = portfolio_change / total_base_value * 100 if total_base_value != 0 else 0

        # Calculate portfolio duration and convexity (weighted averages)
        portfolio_duration = sum(r['base_duration'] * r['weighted_base_value'] for r in
                                 results) / total_base_value if total_base_value != 0 else 0
        portfolio_convexity = sum(r['base_convexity'] * r['weighted_base_value'] for r in
                                  results) / total_base_value if total_base_value != 0 else 0

        # Create portfolio summary
        portfolio_summary = {
            'scenario_name': scenario_name,
            'total_base_value': total_base_value,
            'total_scenario_value': total_scenario_value,
            'portfolio_change': portfolio_change,
            'portfolio_percent_change': portfolio_percent_change,
            'portfolio_duration': portfolio_duration,
            'portfolio_convexity': portfolio_convexity,
            'bond_results': results
        }

        self.scenario_results[scenario_name] = portfolio_summary

        return portfolio_summary

    def evaluate_portfolio_under_all_scenarios(self, portfolio=None):
        """
        Evaluate a portfolio under all defined scenarios.

        Parameters:
            portfolio (dict, optional): Portfolio to evaluate (uses self.portfolio if None)

        Returns:
            dict: Dictionary of evaluation results for each scenario
        """
        if portfolio is None:
            portfolio = self.portfolio

        if portfolio is None:
            raise ValueError("No portfolio provided to evaluate")

        results = {}

        for scenario_name in self.scenarios:
            results[scenario_name] = self.evaluate_portfolio_under_scenario(portfolio, scenario_name)

        return results

    def calculate_var_under_scenario(self, portfolio, scenario_name, confidence_level=0.95, holding_period=10):
        """
        Calculate Value at Risk for a portfolio under a specific scenario.

        Parameters:
            portfolio (dict): Dictionary with bond details
            scenario_name (str): The name of the scenario to evaluate under
            confidence_level (float): Confidence level for VaR calculation
            holding_period (int): Holding period in days

        Returns:
            dict: VaR calculation results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        # First evaluate the portfolio under the scenario to get price changes
        evaluation = self.evaluate_portfolio_under_scenario(portfolio, scenario_name)

        # Calculate current portfolio value
        current_value = evaluation['total_base_value']

        # Generate synthetic returns based on the scenario - simplified approach

        # Get the average yield shift from the scenario
        avg_yield_shift = np.mean([bond['yield_shift'] for bond in evaluation['bond_results']])

        # Calculate yield volatility - Assume it scales with the magnitude of the shift - simplified approach
        yield_vol = abs(avg_yield_shift) * 0.2  # 20% of the shift as daily volatility

        # Generate synthetic daily yield changes
        np.random.seed(42)  # For reproducibility
        num_days = 1000
        daily_changes = np.random.normal(avg_yield_shift / 100, yield_vol / 100, num_days)

        # Calculate price changes based on portfolio duration and convexity
        price_changes = []
        for dy in daily_changes:
            # Use the portfolio's duration and convexity to approximate price change
            duration_effect = -evaluation['portfolio_duration'] * current_value * dy
            convexity_effect = 0.5 * evaluation['portfolio_convexity'] * current_value * dy ** 2
            price_change = duration_effect + convexity_effect
            price_changes.append(price_change)

        # Convert to returns
        returns = np.array(price_changes) / current_value

        # Calculate VaR using different methods
        historical_var = VaRModule.historical_var(
            returns, confidence_level, holding_period, current_value)

        normal_var = VaRModule.parametric_var(
            returns, confidence_level, holding_period, current_value, 'normal')

        t_var = VaRModule.parametric_var(
            returns, confidence_level, holding_period, current_value, 't')

        var_results = {
            'scenario_name': scenario_name,
            'confidence_level': confidence_level,
            'holding_period': holding_period,
            'current_value': current_value,
            'historical_var': historical_var,
            'normal_var': normal_var,
            't_var': t_var,
            'var_pct_historical': historical_var / current_value * 100,
            'var_pct_normal': normal_var / current_value * 100,
            'var_pct_t': t_var / current_value * 100
        }

        # Add VaR results to the scenario results
        if scenario_name in self.scenario_results:
            self.scenario_results[scenario_name]['var_results'] = var_results

        return var_results

    def compare_scenario_impacts(self):
        """
        Compare the impact of different scenarios on the portfolio.

        Returns:
            pd.DataFrame: Comparison of scenario impacts
        """
        if not self.scenario_results:
            raise ValueError("No scenario results available to compare")

        results = []

        for scenario_name, result in self.scenario_results.items():
            scenario_data = {
                'Scenario': scenario_name,
                'Description': self.scenarios[scenario_name]['description'],
                'Portfolio Change (%)': result['portfolio_percent_change'],
                'Portfolio Duration': result['portfolio_duration'],
                'Portfolio Convexity': result['portfolio_convexity']
            }

            # Add VaR information if available
            if 'var_results' in result:
                var_results = result['var_results']
                scenario_data.update({
                    'Historical VaR (%)': var_results['var_pct_historical'],
                    'Parametric VaR - Normal (%)': var_results['var_pct_normal'],
                    'Parametric VaR - t (%)': var_results['var_pct_t']
                })

            results.append(scenario_data)

        return pd.DataFrame(results)

    def plot_scenario_curves(self, scenarios=None, ax=None):
        """
        Plot yield curves for multiple scenarios.

        Parameters:
            scenarios (list, optional): List of scenario names to plot (all if None)
            ax (matplotlib.axes.Axes, optional): Axes to plot on

        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 8))

        # Plot base curve
        self.base_curve.plot(ax=ax, label="Base Curve")

        # Plot scenario curves
        scenario_names = scenarios or list(self.scenarios.keys())

        for name in scenario_names:
            if name in self.scenarios:
                scenario = self.scenarios[name]
                scenario['curve'].plot(ax=ax, label=name)

        ax.set_title('Yield Curves Under Different Scenarios')
        ax.legend(loc='best')

        return ax

    def plot_scenario_impacts(self, metric='portfolio_percent_change'):
        """
        Plot the impact of different scenarios on the portfolio using a specified metric.

        Parameters:
            metric (str): Metric to plot ('portfolio_percent_change', 'var_pct_historical', etc.)

        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        if not self.scenario_results:
            raise ValueError("No scenario results available to plot")

        # Prepare data for plotting
        scenarios = []
        values = []

        for scenario_name, result in self.scenario_results.items():
            scenarios.append(scenario_name)

            if metric == 'portfolio_percent_change':
                values.append(result['portfolio_percent_change'])
            elif metric.startswith('var_pct_'):
                if 'var_results' in result and metric in result['var_results']:
                    values.append(result['var_results'][metric])
                else:
                    values.append(0)
            else:
                values.append(result.get(metric, 0))

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(scenarios, values)

        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            if height < 0:
                y_position = height - 0.5
                va = 'top'
            else:
                y_position = height + 0.5
                va = 'bottom'

            ax.text(bar.get_x() + bar.get_width() / 2., y_position,
                    f'{height:.2f}', ha='center', va=va)

        # Add labels and title
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_title(f'Impact of Different Scenarios on Portfolio - {metric_name}')
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig

    def plot_portfolio_heat_map(self, metric='percent_change'):
        """
        Plot a heat map showing the impact of scenarios on individual bonds in the portfolio.

        Parameters:
            metric (str): Metric to plot ('percent_change', 'yield_shift', etc.)

        Returns:
            matplotlib.figure.Figure: The figure containing the heat map
        """
        if not self.scenario_results:
            raise ValueError("No scenario results available to plot")

        # Extract bond IDs
        first_scenario = next(iter(self.scenario_results.values()))
        bond_ids = [bond['bond_id'] for bond in first_scenario['bond_results']]

        # Prepare data for the heat map
        data = []
        for scenario_name, result in self.scenario_results.items():
            row = []
            for bond in result['bond_results']:
                row.append(bond[metric])
            data.append(row)

        data = np.array(data)

        # Create heat map
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data, cmap='RdYlGn_r')

        # Add labels
        ax.set_xticks(np.arange(len(bond_ids)))
        ax.set_yticks(np.arange(len(self.scenario_results)))
        ax.set_xticklabels(bond_ids)
        ax.set_yticklabels(list(self.scenario_results.keys()))

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        cbar.ax.set_ylabel(f"{metric_name}", rotation=-90, va="bottom")

        # Add title
        ax.set_title(f"Impact of Scenarios on Individual Bonds - {metric_name}")

        # Add text annotations showing the values
        for i in range(len(self.scenario_results)):
            for j in range(len(bond_ids)):
                value = data[i, j]
                text_color = "white" if abs(value) > np.max(np.abs(data)) / 2 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

        fig.tight_layout()
        return fig

    def generate_scenario_report(self, output_file=None):
        """
        Generate a comprehensive report of all scenario analyses.

        Parameters:
            output_file (str, optional): Path to save the report (prints to console if None)

        Returns:
            str: Report text
        """
        if not self.scenario_results:
            return "No scenario results available for reporting."

        report = ["SCENARIO ANALYSIS REPORT", "=" * 80, ""]

        # Summary of scenarios
        report.append("SCENARIOS ANALYZED:")
        report.append("-" * 50)
        for name, scenario in self.scenarios.items():
            report.append(f"{name}: {scenario['description']}")
        report.append("")

        # Portfolio characteristics
        first_scenario = next(iter(self.scenario_results.values()))
        report.append("PORTFOLIO CHARACTERISTICS:")
        report.append("-" * 50)
        report.append(f"Total Portfolio Value: ${first_scenario['total_base_value']:.2f}")
        report.append(f"Portfolio Duration: {first_scenario['portfolio_duration']:.2f} years")
        report.append(f"Portfolio Convexity: {first_scenario['portfolio_convexity']:.2f}")
        report.append("")

        # Individual bonds
        report.append("INDIVIDUAL BONDS:")
        report.append("-" * 50)
        for bond in first_scenario['bond_results']:
            report.append(f"Bond ID: {bond['bond_id']}")
            report.append(f"  Weight: {bond['weight']:.2f}")
            report.append(f"  Amount: {bond['amount']}")
            report.append(f"  Base Price: ${bond['base_price']:.2f}")
            report.append(f"  YTM: {bond['base_ytm'] * 100:.2f}%")
            report.append(f"  Duration: {bond['base_duration']:.2f} years")
            report.append(f"  Convexity: {bond['base_convexity']:.2f}")
            report.append("")

        # Scenario impacts
        report.append("SCENARIO IMPACTS:")
        report.append("-" * 50)

        # Create a table for scenario impacts
        impact_table = self.compare_scenario_impacts()
        report.append(impact_table.to_string(index=False))
        report.append("")

        # Detailed scenario results
        report.append("DETAILED SCENARIO RESULTS:")
        report.append("-" * 50)

        for scenario_name, result in self.scenario_results.items():
            report.append(f"Scenario: {scenario_name}")
            report.append(f"Description: {self.scenarios[scenario_name]['description']}")
            report.append(f"Portfolio Base Value: ${result['total_base_value']:.2f}")
            report.append(f"Portfolio Value Under Scenario: ${result['total_scenario_value']:.2f}")
            report.append(f"Change: ${result['portfolio_change']:.2f} ({result['portfolio_percent_change']:.2f}%)")

            # Add VaR results if available
            if 'var_results' in result:
                var = result['var_results']
                report.append(
                    f"Value at Risk ({var['confidence_level'] * 100}% confidence, {var['holding_period']}-day holding period):")
                report.append(f"  Historical VaR: ${var['historical_var']:.2f} ({var['var_pct_historical']:.2f}%)")
                report.append(f"  Parametric VaR (Normal): ${var['normal_var']:.2f} ({var['var_pct_normal']:.2f}%)")
                report.append(f"  Parametric VaR (t-dist): ${var['t_var']:.2f} ({var['var_pct_t']:.2f}%)")

            report.append("")

            # Individual bond results
            report.append("  Individual Bond Results:")
            for bond in result['bond_results']:
                report.append(f"  Bond ID: {bond['bond_id']}")
                report.append(f"    Base YTM: {bond['base_ytm'] * 100:.2f}%")
                report.append(f"    Scenario YTM: {bond['scenario_ytm'] * 100:.2f}%")
                report.append(f"    Yield Shift: {bond['yield_shift'] * 10000:.2f} bps")
                report.append(f"    Price Change: ${bond['price_change']:.2f} ({bond['percent_change']:.2f}%)")
                report.append("")

            report.append("-" * 50)

        # Write to file or print to console
        report_text = "\n".join(report)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

        return report_text


def test_scenario_analysis():
    """
    Test the ScenarioAnalysis module with a sample portfolio and various scenarios.
    """
    print("Testing Scenario Analysis Module...")

    # Sample data - create a base yield curve
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    rates = np.array([0.0210, 0.0215, 0.0220, 0.0230, 0.0240, 0.0250, 0.0260, 0.0270, 0.0275, 0.0280])
    reference_date = datetime(2025, 1, 1)

    # Create Treasury yield curve
    base_curve = TreasuryYieldCurve(tenors, rates, reference_date)

    # Create a sample portfolio of bonds
    portfolio = {
        'Bond1': {
            'cash_flows': create_bond_cash_flows(100, 0.030, 2, 2),  # 2-year, 3% coupon bond
            'times': np.arange(1, 5) / 2,  # Semi-annual payments
            'ytm': 0.023,
            'weight': 0.2,
            'amount': 10000
        },
        'Bond2': {
            'cash_flows': create_bond_cash_flows(100, 0.040, 5, 2),  # 5-year, 4% coupon bond
            'times': np.arange(1, 11) / 2,  # Semi-annual payments
            'ytm': 0.025,
            'weight': 0.3,
            'amount': 15000
        },
        'Bond3': {
            'cash_flows': create_bond_cash_flows(100, 0.025, 10, 2),  # 10-year, 2.5% coupon bond
            'times': np.arange(1, 21) / 2,  # Semi-annual payments
            'ytm': 0.027,
            'weight': 0.5,
            'amount': 20000
        }
    }

    # Initialize scenario analysis
    scenario_analysis = ScenarioAnalysis(base_curve, portfolio)

    # Create various scenarios
    print("\nCreating scenarios...")

    # Historical scenarios
    covid_curve = scenario_analysis.create_covid_crisis_scenario()
    inflation_curve = scenario_analysis.create_inflation_hike_scenario()

    # Hypothetical scenarios
    parallel_up_100 = scenario_analysis.create_parallel_shift_scenario(100)
    parallel_up_200 = scenario_analysis.create_parallel_shift_scenario(200)
    parallel_down_50 = scenario_analysis.create_parallel_shift_scenario(-50)

    bull_steepener = scenario_analysis.create_bull_steepener_scenario(50)
    bear_steepener = scenario_analysis.create_bear_steepener_scenario(50)
    bull_flattener = scenario_analysis.create_bull_flattener_scenario(50)
    bear_flattener = scenario_analysis.create_bear_flattener_scenario(50)

    inversion = scenario_analysis.create_inversion_scenario(50)

    # Plot all scenario curves
    print("\nPlotting scenario curves...")
    ax = scenario_analysis.plot_scenario_curves()
    plt.tight_layout()
    plt.savefig('scenario_curves.png')

    # Evaluate portfolio under all scenarios
    print("\nEvaluating portfolio under all scenarios...")
    scenario_analysis.evaluate_portfolio_under_all_scenarios()

    # Calculate VaR under each scenario
    print("\nCalculating VaR under each scenario...")
    for scenario_name in scenario_analysis.scenarios:
        scenario_analysis.calculate_var_under_scenario(portfolio, scenario_name)

    # Compare scenario impacts
    print("\nComparing scenario impacts...")
    comparison = scenario_analysis.compare_scenario_impacts()
    print(comparison)

    # Plot scenario impacts
    print("\nPlotting scenario impacts...")
    fig = scenario_analysis.plot_scenario_impacts()
    plt.tight_layout()
    plt.savefig('scenario_impacts.png')

    # Plot heat map
    print("\nPlotting portfolio heat map...")
    fig = scenario_analysis.plot_portfolio_heat_map()
    plt.tight_layout()
    plt.savefig('portfolio_heat_map.png')

    # Generate report
    print("\nGenerating scenario report...")
    report = scenario_analysis.generate_scenario_report()
    print("Report excerpt:")
    print(report[:500] + "...")  # Print just the beginning of the report

    print("\nScenario analysis tests completed.")
    return scenario_analysis


def create_bond_cash_flows(face_value, coupon_rate, years, payments_per_year):
    """Helper function to create bond cash flows for testing."""
    num_periods = int(years * payments_per_year)
    coupon_payment = face_value * coupon_rate / payments_per_year

    cash_flows = np.ones(num_periods) * coupon_payment
    cash_flows[-1] += face_value  # Add face value at maturity

    return cash_flows


if __name__ == "__main__":
    scenario_analysis = test_scenario_analysis()