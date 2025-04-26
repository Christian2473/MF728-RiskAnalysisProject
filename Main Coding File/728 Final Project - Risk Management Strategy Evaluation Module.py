import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import copy
import os

from YieldCurveConstruction import YieldCurve, TreasuryYieldCurve, CorporateYieldCurve
from PricingFunction import PricingModule
from VarFunction import VaRModule
from ScenarioAnalysis import ScenarioAnalysis


class RiskManagementModule:
    """
    Module for evaluating and comparing fixed income risk management strategies.

    This module tests:
    - Duration-matching strategies
    - Convexity-based adjustments
    - VaR-based position sizing
    - Diversification across credit qualities
    - Performance during various stress periods
    """

    def __init__(self, base_yield_curve, scenario_analyzer=None):
        """
        Initialize the risk management module.

        Parameters:
            base_yield_curve (YieldCurve): Base yield curve for analysis
            scenario_analyzer (ScenarioAnalysis, optional): Existing scenario analyzer
        """
        self.base_curve = base_yield_curve
        self.scenario_analyzer = scenario_analyzer or ScenarioAnalysis(base_yield_curve)
        self.strategies = {}
        self.strategy_performances = {}
        self.portfolio_allocations = {}

    def add_bond(self, bond_id, cash_flows, times, ytm, face_value=100,
                 credit_rating="IG", duration=None, convexity=None):
        """
        Add a bond to the available bond universe.

        Parameters:
            bond_id (str): Identifier for the bond
            cash_flows (array-like): Cash flows of the bond
            times (array-like): Time points for the cash flows
            ytm (float): Yield to maturity
            face_value (float): Face value of the bond
            credit_rating (str): Credit rating of the bond
            duration (float, optional): Pre-calculated duration
            convexity (float, optional): Pre-calculated convexity

        Returns:
            dict: The added bond information
        """
        # Calculate price
        price = PricingModule.calculate_price(cash_flows, times, ytm)

        # Calculate duration and convexity if not provided
        if duration is None:
            duration = PricingModule.calculate_modified_duration(cash_flows, times, ytm)

        if convexity is None:
            convexity = PricingModule.calculate_convexity(cash_flows, times, ytm)

        bond = {
            'cash_flows': cash_flows,
            'times': times,
            'ytm': ytm,
            'face_value': face_value,
            'price': price,
            'duration': duration,
            'convexity': convexity,
            'credit_rating': credit_rating
        }

        if not hasattr(self, 'bond_universe'):
            self.bond_universe = {}

        self.bond_universe[bond_id] = bond
        return bond

    def add_bonds_from_portfolio(self, portfolio):
        """
        Add multiple bonds from an existing portfolio.

        Parameters:
            portfolio (dict): Dictionary of bonds

        Returns:
            dict: Updated bond universe
        """
        for bond_id, bond_data in portfolio.items():
            if 'cash_flows' in bond_data and 'times' in bond_data and 'ytm' in bond_data:
                # Extract optional parameters if they exist
                face_value = bond_data.get('face_value', 100)
                credit_rating = bond_data.get('credit_rating', "IG")
                duration = bond_data.get('duration', None)
                convexity = bond_data.get('convexity', None)

                self.add_bond(
                    bond_id,
                    bond_data['cash_flows'],
                    bond_data['times'],
                    bond_data['ytm'],
                    face_value,
                    credit_rating,
                    duration,
                    convexity
                )
            else:
                print(f"Warning: Bond {bond_id} is missing required parameters and was not added.")

        return self.bond_universe

    def create_duration_matched_portfolio(self, target_duration, portfolio_size=5,
                                          credit_ratings=None, tolerance=0.1):
        """
        Create a portfolio that matches a target duration.

        Parameters:
            target_duration (float): Target modified duration for the portfolio
            portfolio_size (int): Number of bonds to include
            credit_ratings (list, optional): List of acceptable credit ratings
            tolerance (float): Acceptable deviation from target duration

        Returns:
            dict: Duration-matched portfolio
        """
        if not hasattr(self, 'bond_universe') or not self.bond_universe:
            raise ValueError("Bond universe is empty. Add bonds before creating portfolios.")

        # Filter bonds by credit rating if specified
        available_bonds = self.bond_universe.copy()
        if credit_ratings:
            available_bonds = {
                bond_id: bond for bond_id, bond in available_bonds.items()
                if bond['credit_rating'] in credit_ratings
            }

            if not available_bonds:
                raise ValueError(f"No bonds with credit ratings {credit_ratings} available.")

        # Sort bonds by how close their duration is to the target
        sorted_bonds = sorted(
            available_bonds.items(),
            key=lambda x: abs(x[1]['duration'] - target_duration)
        )

        # Start with the closest bond
        selected_bonds = {}
        total_weight = 0
        portfolio_duration = 0

        # First add the bond closest to target duration
        bond_id, bond = sorted_bonds[0]
        weight = 1.0
        selected_bonds[bond_id] = bond.copy()
        selected_bonds[bond_id]['weight'] = weight
        selected_bonds[bond_id]['amount'] = 1
        total_weight += weight
        portfolio_duration = bond['duration']

        # Add more bonds if portfolio_size > 1
        if portfolio_size > 1:
            # Try to get to target duration by adding bonds with appropriate durations and adjusting weights
            remaining_bonds = sorted_bonds[1:portfolio_size]

            for bond_id, bond in remaining_bonds:
                # Calculate initial weight - give higher weight to bonds that help reach target duration
                if portfolio_duration < target_duration and bond['duration'] > target_duration:
                    # If portfolio duration is below target and this bond is above,
                    # give it higher weight
                    weight = 1.5
                elif portfolio_duration > target_duration and bond['duration'] < target_duration:
                    # If portfolio duration is above target and this bond is below,
                    # give it higher weight
                    weight = 1.5
                else:
                    # Otherwise give standard weight
                    weight = 1.0

                selected_bonds[bond_id] = bond.copy()
                selected_bonds[bond_id]['weight'] = weight
                selected_bonds[bond_id]['amount'] = 1
                total_weight += weight

            # Normalize weights
            for bond_id in selected_bonds:
                selected_bonds[bond_id]['weight'] /= total_weight

            # Calculate portfolio duration with these weights
            portfolio_duration = sum(
                bond['duration'] * bond['weight']
                for bond in selected_bonds.values()
            )

            # If we're still not close enough to target, adjust weights further
            iteration = 0
            max_iterations = 10

            while abs(portfolio_duration - target_duration) > tolerance and iteration < max_iterations:
                # Determine which bonds to adjust
                if portfolio_duration < target_duration:
                    # Need to increase duration
                    # Find highest duration bond and lowest duration bond
                    highest_duration_bond = max(
                        selected_bonds.items(),
                        key=lambda x: x[1]['duration']
                    )
                    lowest_duration_bond = min(
                        selected_bonds.items(),
                        key=lambda x: x[1]['duration']
                    )

                    # Increase weight of highest duration bond
                    highest_id = highest_duration_bond[0]
                    selected_bonds[highest_id]['weight'] *= 1.1

                    # Decrease weight of lowest duration bond
                    lowest_id = lowest_duration_bond[0]
                    selected_bonds[lowest_id]['weight'] *= 0.9
                else:
                    # Need to decrease duration
                    # Find highest duration bond and lowest duration bond
                    highest_duration_bond = max(
                        selected_bonds.items(),
                        key=lambda x: x[1]['duration']
                    )
                    lowest_duration_bond = min(
                        selected_bonds.items(),
                        key=lambda x: x[1]['duration']
                    )

                    # Decrease weight of highest duration bond
                    highest_id = highest_duration_bond[0]
                    selected_bonds[highest_id]['weight'] *= 0.9

                    # Increase weight of lowest duration bond
                    lowest_id = lowest_duration_bond[0]
                    selected_bonds[lowest_id]['weight'] *= 1.1

                # Normalize weights
                total_weight = sum(bond['weight'] for bond in selected_bonds.values())
                for bond_id in selected_bonds:
                    selected_bonds[bond_id]['weight'] /= total_weight

                # Recalculate portfolio duration
                portfolio_duration = sum(
                    bond['duration'] * bond['weight']
                    for bond in selected_bonds.values()
                )

                iteration += 1

        # Store portfolio
        strategy_name = f"duration_matched_{target_duration:.2f}"

        self.strategies[strategy_name] = {
            'name': f"Duration Matched ({target_duration:.2f})",
            'type': 'duration_matched',
            'target_duration': target_duration,
            'achieved_duration': portfolio_duration,
            'portfolio': selected_bonds,
            'performance': {}
        }

        return selected_bonds

    def create_convexity_enhanced_portfolio(self, base_duration, min_convexity=None,
                                            portfolio_size=5, credit_ratings=None):
        """
        Create a portfolio with enhanced convexity while maintaining a target duration.

        Parameters:
            base_duration (float): Target duration for the portfolio
            min_convexity (float, optional): Minimum convexity to target
            portfolio_size (int): Number of bonds to include
            credit_ratings (list, optional): List of acceptable credit ratings

        Returns:
            dict: Convexity-enhanced portfolio
        """
        if not hasattr(self, 'bond_universe') or not self.bond_universe:
            raise ValueError("Bond universe is empty. Add bonds before creating portfolios.")

        # Filter bonds by credit rating if specified
        available_bonds = self.bond_universe.copy()
        if credit_ratings:
            available_bonds = {
                bond_id: bond for bond_id, bond in available_bonds.items()
                if bond['credit_rating'] in credit_ratings
            }

            if not available_bonds:
                raise ValueError(f"No bonds with credit ratings {credit_ratings} available.")

        # First, create a duration-matched portfolio as a base
        base_portfolio = self.create_duration_matched_portfolio(
            base_duration,
            portfolio_size=portfolio_size,
            credit_ratings=credit_ratings
        )

        # Calculate current portfolio convexity
        portfolio_convexity = sum(
            bond['convexity'] * bond['weight']
            for bond in base_portfolio.values()
        )

        # If min_convexity is not specified, target 20% higher than current
        if min_convexity is None:
            min_convexity = portfolio_convexity * 1.2

        # Now enhance convexity while maintaining duration
        # Sort bonds by convexity (highest first)
        sorted_bonds = sorted(
            available_bonds.items(),
            key=lambda x: x[1]['convexity'],
            reverse=True
        )

        # Start with the highest convexity bonds
        selected_bonds = {}
        total_weight = 0

        # Add bonds with high convexity
        for i, (bond_id, bond) in enumerate(sorted_bonds):
            if i >= portfolio_size:
                break

            selected_bonds[bond_id] = bond.copy()

            # Give higher weight to bonds with high convexity but similar duration to target
            weight = 1.0 + (bond['convexity'] / min_convexity) * (
                        1 - abs(bond['duration'] - base_duration) / base_duration)

            selected_bonds[bond_id]['weight'] = weight
            selected_bonds[bond_id]['amount'] = 1
            total_weight += weight

        # Normalize weights
        for bond_id in selected_bonds:
            selected_bonds[bond_id]['weight'] /= total_weight

        # Calculate portfolio duration and convexity with these weights
        portfolio_duration = sum(
            bond['duration'] * bond['weight']
            for bond in selected_bonds.values()
        )

        portfolio_convexity = sum(
            bond['convexity'] * bond['weight']
            for bond in selected_bonds.values()
        )

        # Now adjust weights to get closer to the target duration while maintaining high convexity
        iteration = 0
        max_iterations = 20
        duration_tolerance = 0.1

        while abs(portfolio_duration - base_duration) > duration_tolerance and iteration < max_iterations:
            # Determine which bonds to adjust
            if portfolio_duration < base_duration:
                # Need to increase duration
                # Sort bonds by duration (highest first)
                sorted_by_duration = sorted(
                    selected_bonds.items(),
                    key=lambda x: x[1]['duration'],
                    reverse=True
                )

                # Increase weight of highest duration bonds
                for i, (bond_id, _) in enumerate(sorted_by_duration):
                    if i < portfolio_size // 2:
                        selected_bonds[bond_id]['weight'] *= 1.1
                    else:
                        selected_bonds[bond_id]['weight'] *= 0.95
            else:
                # Need to decrease duration
                # Sort bonds by duration (lowest first)
                sorted_by_duration = sorted(
                    selected_bonds.items(),
                    key=lambda x: x[1]['duration']
                )

                # Increase weight of lowest duration bonds
                for i, (bond_id, _) in enumerate(sorted_by_duration):
                    if i < portfolio_size // 2:
                        selected_bonds[bond_id]['weight'] *= 1.1
                    else:
                        selected_bonds[bond_id]['weight'] *= 0.95

            # Normalize weights
            total_weight = sum(bond['weight'] for bond in selected_bonds.values())
            for bond_id in selected_bonds:
                selected_bonds[bond_id]['weight'] /= total_weight

            # Recalculate portfolio duration and convexity
            portfolio_duration = sum(
                bond['duration'] * bond['weight']
                for bond in selected_bonds.values()
            )

            portfolio_convexity = sum(
                bond['convexity'] * bond['weight']
                for bond in selected_bonds.values()
            )

            iteration += 1

        # Store portfolio
        strategy_name = f"convexity_enhanced_{base_duration:.2f}"

        self.strategies[strategy_name] = {
            'name': f"Convexity Enhanced ({base_duration:.2f})",
            'type': 'convexity_enhanced',
            'target_duration': base_duration,
            'achieved_duration': portfolio_duration,
            'target_convexity': min_convexity,
            'achieved_convexity': portfolio_convexity,
            'portfolio': selected_bonds,
            'performance': {}
        }

        return selected_bonds

    def create_var_optimized_portfolio(self, target_var_pct, confidence_level=0.95,
                                       holding_period=10, portfolio_size=5,
                                       credit_ratings=None, method='normal'):
        """
        Create a portfolio optimized to stay within a target VaR percentage.

        Parameters:
            target_var_pct (float): Target VaR as percentage of portfolio value
            confidence_level (float): Confidence level for VaR
            holding_period (int): Holding period in days
            portfolio_size (int): Number of bonds to include
            credit_ratings (list, optional): List of acceptable credit ratings
            method (str): VaR calculation method ('historical', 'normal', or 't')

        Returns:
            dict: VaR-optimized portfolio
        """
        if not hasattr(self, 'bond_universe') or not self.bond_universe:
            raise ValueError("Bond universe is empty. Add bonds before creating portfolios.")

        # Filter bonds by credit rating if specified
        available_bonds = self.bond_universe.copy()
        if credit_ratings:
            available_bonds = {
                bond_id: bond for bond_id, bond in available_bonds.items()
                if bond['credit_rating'] in credit_ratings
            }

            if not available_bonds:
                raise ValueError(f"No bonds with credit ratings {credit_ratings} available.")

        # Generate synthetic returns for each bond based on its duration and convexity - simplified approach
        # For each bond, generate synthetic returns
        for bond_id, bond in available_bonds.items():
            # Generate 500 random yield changes (normally distributed)
            np.random.seed(42)
            yield_changes = np.random.normal(0, 0.0010, 500)  # 10 bps std dev

            # Calculate price changes based on duration and convexity
            price_changes = []
            for dy in yield_changes:
                # Use duration and convexity to approximate price change
                duration_effect = -bond['duration'] * bond['price'] * dy
                convexity_effect = 0.5 * bond['convexity'] * bond['price'] * dy ** 2
                price_change = duration_effect + convexity_effect
                price_changes.append(price_change)

            # Convert to returns
            returns = np.array(price_changes) / bond['price']

            # Store returns in bond data
            bond['returns'] = returns

            # Calculate individual bond VaR
            if method == 'historical':
                bond['var'] = VaRModule.historical_var(
                    returns, confidence_level, holding_period, bond['price'])
            elif method == 't':
                bond['var'] = VaRModule.parametric_var(
                    returns, confidence_level, holding_period, bond['price'], 't')
            else:  # Default to normal
                bond['var'] = VaRModule.parametric_var(
                    returns, confidence_level, holding_period, bond['price'], 'normal')

            # Calculate VaR as percentage of price
            bond['var_pct'] = bond['var'] / bond['price'] * 100

        # Sort bonds by VaR percentage (lowest first)
        sorted_bonds = sorted(
            available_bonds.items(),
            key=lambda x: x[1]['var_pct']
        )

        # Start with lowest VaR bonds
        selected_bonds = {}
        total_weight = 0
        portfolio_value = 0
        portfolio_var = 0

        # Initially select bonds with lowest VaR
        for i, (bond_id, bond) in enumerate(sorted_bonds):
            if i >= portfolio_size:
                break

            # Calculate weight inversely proportional to VaR percentage
            weight = 1.0 / (bond['var_pct'] + 0.1)  # Add 0.1 to avoid division by zero

            selected_bonds[bond_id] = bond.copy()
            selected_bonds[bond_id]['weight'] = weight
            selected_bonds[bond_id]['amount'] = 1
            total_weight += weight

        # Normalize weights
        for bond_id in selected_bonds:
            selected_bonds[bond_id]['weight'] /= total_weight

        # Calculate portfolio value and VaR
        portfolio_value = sum(
            bond['price'] * bond['weight']
            for bond in selected_bonds.values()
        )

        # We'll use a correlation matrix of 0.5 between all bonds to test.
        # After using real data, we will would estimate this from historical data
        n = len(selected_bonds)
        correlation = 0.5 * np.ones((n, n))
        np.fill_diagonal(correlation, 1.0)

        # Calculate portfolio VaR using the specified method and correlation
        vars = np.array([bond['var'] for bond in selected_bonds.values()])
        weights = np.array([bond['weight'] for bond in selected_bonds.values()])

        # For normal distribution, we can use matrix multiplication to get portfolio VaR
        if method in ['normal', 't']:
            portfolio_var = np.sqrt(weights @ correlation @ (weights * vars ** 2))
        else:
            # For historical method, Calculate combined returns - simplified
            portfolio_returns = np.zeros(500)
            for i, bond_id in enumerate(selected_bonds):
                portfolio_returns += selected_bonds[bond_id]['returns'] * weights[i]

            portfolio_var = VaRModule.historical_var(
                portfolio_returns, confidence_level, holding_period, portfolio_value)

        portfolio_var_pct = portfolio_var / portfolio_value * 100

        # Adjust weights based on target VaR
        # If portfolio VaR exceeds target, increase weights of lower VaR bonds
        # If portfolio VaR is below target, can increase weights of higher VaR bonds

        iteration = 0
        max_iterations = 20
        var_tolerance = 0.2  # Allow 0.2% deviation from target

        while abs(portfolio_var_pct - target_var_pct) > var_tolerance and iteration < max_iterations:
            if portfolio_var_pct > target_var_pct:
                # Need to reduce VaR
                # Sort bonds by VaR (lowest first)
                sorted_by_var = sorted(
                    selected_bonds.items(),
                    key=lambda x: x[1]['var_pct']
                )

                # Increase weight of lowest VaR bonds
                for i, (bond_id, _) in enumerate(sorted_by_var):
                    if i < portfolio_size // 2:
                        selected_bonds[bond_id]['weight'] *= 1.1
                    else:
                        selected_bonds[bond_id]['weight'] *= 0.95
            else:
                # Can increase VaR (may improve returns)
                # Sort bonds by VaR (highest first)
                sorted_by_var = sorted(
                    selected_bonds.items(),
                    key=lambda x: x[1]['var_pct'],
                    reverse=True
                )

                # Increase weight of highest VaR bonds (carefully)
                for i, (bond_id, _) in enumerate(sorted_by_var):
                    if i < portfolio_size // 3:  # Only adjust top third
                        selected_bonds[bond_id]['weight'] *= 1.05  # More conservative adjustment
                    else:
                        selected_bonds[bond_id]['weight'] *= 0.98

            # Normalize weights
            total_weight = sum(bond['weight'] for bond in selected_bonds.values())
            for bond_id in selected_bonds:
                selected_bonds[bond_id]['weight'] /= total_weight

            # Recalculate portfolio value and VaR
            portfolio_value = sum(
                bond['price'] * bond['weight']
                for bond in selected_bonds.values()
            )

            # Update weights array
            weights = np.array([bond['weight'] for bond in selected_bonds.values()])

            # Recalculate portfolio VaR
            if method in ['normal', 't']:
                portfolio_var = np.sqrt(weights @ correlation @ (weights * vars ** 2))
            else:
                # For historical method, recalculate combined returns
                portfolio_returns = np.zeros(500)
                for i, bond_id in enumerate(selected_bonds):
                    portfolio_returns += selected_bonds[bond_id]['returns'] * weights[i]

                portfolio_var = VaRModule.historical_var(
                    portfolio_returns, confidence_level, holding_period, portfolio_value)

            portfolio_var_pct = portfolio_var / portfolio_value * 100
            iteration += 1

        # Store portfolio
        strategy_name = f"var_optimized_{target_var_pct:.2f}"

        self.strategies[strategy_name] = {
            'name': f"VaR Optimized ({target_var_pct:.2f}%)",
            'type': 'var_optimized',
            'target_var_pct': target_var_pct,
            'achieved_var_pct': portfolio_var_pct,
            'confidence_level': confidence_level,
            'holding_period': holding_period,
            'method': method,
            'portfolio': selected_bonds,
            'performance': {}
        }

        return selected_bonds

    def create_credit_diversified_portfolio(self, target_duration=None, portfolio_size=10,
                                            credit_allocations=None):
        """
        Create a portfolio diversified across credit qualities.

        Parameters:
            target_duration (float, optional): Target duration if desired
            portfolio_size (int): Total number of bonds to include
            credit_allocations (dict, optional): Allocation percentages by credit rating
                                               e.g., {'AAA': 0.3, 'AA': 0.3, 'A': 0.2, 'BBB': 0.2}

        Returns:
            dict: Credit-diversified portfolio
        """
        if not hasattr(self, 'bond_universe') or not self.bond_universe:
            raise ValueError("Bond universe is empty. Add bonds before creating portfolios.")

        # Get all available credit ratings
        all_ratings = set(bond['credit_rating'] for bond in self.bond_universe.values())

        # Default to equal allocation if not specified
        if credit_allocations is None:
            credit_allocations = {rating: 1.0 / len(all_ratings) for rating in all_ratings}

        # Validate credit allocations
        if abs(sum(credit_allocations.values()) - 1.0) > 0.001:
            raise ValueError("Credit allocations must sum to 1.0")

        # Group bonds by credit rating
        bonds_by_rating = defaultdict(list)
        for bond_id, bond in self.bond_universe.items():
            bonds_by_rating[bond['credit_rating']].append((bond_id, bond))

        # Make sure we have bonds for each specified credit rating
        for rating in credit_allocations:
            if rating not in bonds_by_rating or not bonds_by_rating[rating]:
                raise ValueError(f"No bonds with credit rating {rating} available")

        # If target_duration is specified, try to match it within each credit rating group
        if target_duration is not None:
            for rating in bonds_by_rating:
                bonds_by_rating[rating].sort(
                    key=lambda x: abs(x[1]['duration'] - target_duration)
                )

        # Select bonds according to allocations and portfolio size
        selected_bonds = {}
        bonds_per_rating = {}

        # Calculate how many bonds to select for each rating
        remaining_bonds = portfolio_size
        for rating, allocation in credit_allocations.items():
            # Calculate bonds for this rating (round down)
            rating_bonds = int(allocation * portfolio_size)
            bonds_per_rating[rating] = min(rating_bonds, len(bonds_by_rating[rating]))
            remaining_bonds -= bonds_per_rating[rating]

        # Allocate any remaining bonds to the highest allocations
        if remaining_bonds > 0:
            sorted_allocations = sorted(
                credit_allocations.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for rating, _ in sorted_allocations:
                if remaining_bonds <= 0:
                    break

                if bonds_per_rating[rating] < len(bonds_by_rating[rating]):
                    bonds_per_rating[rating] += 1
                    remaining_bonds -= 1

        # Select bonds and assign weights
        for rating, count in bonds_per_rating.items():
            rating_allocation = credit_allocations[rating]

            # Select the top 'count' bonds for this rating
            for i in range(count):
                if i < len(bonds_by_rating[rating]):
                    bond_id, bond = bonds_by_rating[rating][i]
                    selected_bonds[bond_id] = bond.copy()

                    # Weight is proportional to rating allocation and number of bonds
                    # Higher allocation with fewer bonds = higher weight per bond
                    weight = rating_allocation / count if count > 0 else 0

                    selected_bonds[bond_id]['weight'] = weight
                    selected_bonds[bond_id]['amount'] = 1

        # Calculate portfolio duration and other metrics
        portfolio_duration = sum(
            bond['duration'] * bond['weight']
            for bond in selected_bonds.values()
        )

        portfolio_convexity = sum(
            bond['convexity'] * bond['weight']
            for bond in selected_bonds.values()
        )

        # Store portfolio
        strategy_name = "credit_diversified"
        if target_duration is not None:
            strategy_name += f"_{target_duration:.2f}"

        self.strategies[strategy_name] = {
            'name': "Credit Diversified" + (f" ({target_duration:.2f})" if target_duration else ""),
            'type': 'credit_diversified',
            'target_duration': target_duration,
            'achieved_duration': portfolio_duration,
            'achieved_convexity': portfolio_convexity,
            'credit_allocations': credit_allocations,
            'portfolio': selected_bonds,
            'performance': {}
        }

        return selected_bonds

    def create_barbell_portfolio(self, target_duration, short_allocation=0.5,
                                 short_max_duration=2, long_min_duration=7,
                                 portfolio_size=6, credit_ratings=None):
        """
        Create a barbell portfolio with concentrations at short and long durations.

        Parameters:
            target_duration (float): Target portfolio duration
            short_allocation (float): Allocation percentage to short-duration bonds
            short_max_duration (float): Maximum duration for short side
            long_min_duration (float): Minimum duration for long side
            portfolio_size (int): Total number of bonds
            credit_ratings (list, optional): Acceptable credit ratings

        Returns:
            dict: Barbell portfolio
        """
        if not hasattr(self, 'bond_universe') or not self.bond_universe:
            raise ValueError("Bond universe is empty. Add bonds before creating portfolios.")

        # Filter bonds by credit rating if specified
        available_bonds = self.bond_universe.copy()
        if credit_ratings:
            available_bonds = {
                bond_id: bond for bond_id, bond in available_bonds.items()
                if bond['credit_rating'] in credit_ratings
            }

            if not available_bonds:
                raise ValueError(f"No bonds with credit ratings {credit_ratings} available.")

        # Split bonds into short and long groups
        short_bonds = {}
        long_bonds = {}

        for bond_id, bond in available_bonds.items():
            if bond['duration'] <= short_max_duration:
                short_bonds[bond_id] = bond
            elif bond['duration'] >= long_min_duration:
                long_bonds[bond_id] = bond

        if not short_bonds:
            raise ValueError(f"No bonds with duration <= {short_max_duration} available.")

        if not long_bonds:
            raise ValueError(f"No bonds with duration >= {long_min_duration} available.")

        # Calculate number of bonds in each group
        short_count = max(1, int(portfolio_size * short_allocation))
        long_count = portfolio_size - short_count

        # Sort bonds by duration
        sorted_short = sorted(
            short_bonds.items(),
            key=lambda x: x[1]['duration'],
            reverse=True  # Start with longer duration short bonds
        )

        sorted_long = sorted(
            long_bonds.items(),
            key=lambda x: x[1]['duration']  # Start with shorter duration long bonds
        )

        # Select bonds for each side
        selected_bonds = {}

        # Short side
        total_short_weight = short_allocation
        for i, (bond_id, bond) in enumerate(sorted_short):
            if i >= short_count:
                break

            selected_bonds[bond_id] = bond.copy()
            selected_bonds[bond_id]['weight'] = total_short_weight / short_count
            selected_bonds[bond_id]['amount'] = 1

        # Long side
        total_long_weight = 1.0 - short_allocation
        for i, (bond_id, bond) in enumerate(sorted_long):
            if i >= long_count:
                break

            selected_bonds[bond_id] = bond.copy()
            selected_bonds[bond_id]['weight'] = total_long_weight / long_count
            selected_bonds[bond_id]['amount'] = 1

        # Calculate portfolio duration and convexity
        portfolio_duration = sum(
            bond['duration'] * bond['weight']
            for bond in selected_bonds.values()
        )

        portfolio_convexity = sum(
            bond['convexity'] * bond['weight']
            for bond in selected_bonds.values()
        )

        # Adjust weights to get closer to target duration
        iteration = 0
        max_iterations = 10
        tolerance = 0.1

        while abs(portfolio_duration - target_duration) > tolerance and iteration < max_iterations:
            # Identify which side to adjust
            if portfolio_duration < target_duration:
                # Need to increase duration - increase long weights
                long_adj = 1.05
                short_adj = 0.95
            else:
                # Need to decrease duration - increase short weights
                long_adj = 0.95
                short_adj = 1.05

            # Adjust weights
            for bond_id, bond in selected_bonds.items():
                if bond['duration'] <= short_max_duration:
                    bond['weight'] *= short_adj
                else:
                    bond['weight'] *= long_adj

            # Normalize weights
            total_weight = sum(bond['weight'] for bond in selected_bonds.values())
            for bond_id in selected_bonds:
                selected_bonds[bond_id]['weight'] /= total_weight

            # Recalculate portfolio duration
            portfolio_duration = sum(
                bond['duration'] * bond['weight']
                for bond in selected_bonds.values()
            )

            iteration += 1

        # Recalculate portfolio convexity with final weights
        portfolio_convexity = sum(
            bond['convexity'] * bond['weight']
            for bond in selected_bonds.values()
        )

        # Store portfolio
        strategy_name = f"barbell_{target_duration:.2f}"

        self.strategies[strategy_name] = {
            'name': f"Barbell Strategy ({target_duration:.2f})",
            'type': 'barbell',
            'target_duration': target_duration,
            'achieved_duration': portfolio_duration,
            'achieved_convexity': portfolio_convexity,
            'short_allocation': short_allocation,
            'short_max_duration': short_max_duration,
            'long_min_duration': long_min_duration,
            'portfolio': selected_bonds,
            'performance': {}
        }

        return selected_bonds

    def create_bullet_portfolio(self, target_duration, duration_window=1.0,
                                portfolio_size=5, credit_ratings=None):
        """
        Create a bullet portfolio concentrated around a target duration.

        Parameters:
            target_duration (float): Target duration
            duration_window (float): Window around target duration to select bonds
            portfolio_size (int): Number of bonds to include
            credit_ratings (list, optional): Acceptable credit ratings

        Returns:
            dict: Bullet portfolio
        """
        if not hasattr(self, 'bond_universe') or not self.bond_universe:
            raise ValueError("Bond universe is empty. Add bonds before creating portfolios.")

        # Filter bonds by credit rating if specified
        available_bonds = self.bond_universe.copy()
        if credit_ratings:
            available_bonds = {
                bond_id: bond for bond_id, bond in available_bonds.items()
                if bond['credit_rating'] in credit_ratings
            }

            if not available_bonds:
                raise ValueError(f"No bonds with credit ratings {credit_ratings} available.")

        # Filter bonds within duration window
        min_duration = target_duration - duration_window / 2
        max_duration = target_duration + duration_window / 2

        filtered_bonds = {
            bond_id: bond for bond_id, bond in available_bonds.items()
            if min_duration <= bond['duration'] <= max_duration
        }

        if not filtered_bonds:
            raise ValueError(f"No bonds with duration between {min_duration} and {max_duration} available.")

        # Sort bonds by how close they are to target duration
        sorted_bonds = sorted(
            filtered_bonds.items(),
            key=lambda x: abs(x[1]['duration'] - target_duration)
        )

        # Select up to portfolio_size bonds
        selected_bonds = {}
        total_weight = 0

        for i, (bond_id, bond) in enumerate(sorted_bonds):
            if i >= portfolio_size:
                break

            # Weight inversely proportional to distance from target duration
            distance = abs(bond['duration'] - target_duration)
            weight = 1.0 / (distance + 0.1)  # Add 0.1 to avoid division by zero

            selected_bonds[bond_id] = bond.copy()
            selected_bonds[bond_id]['weight'] = weight
            selected_bonds[bond_id]['amount'] = 1
            total_weight += weight

        # Normalize weights
        for bond_id in selected_bonds:
            selected_bonds[bond_id]['weight'] /= total_weight

        # Calculate portfolio duration and convexity
        portfolio_duration = sum(
            bond['duration'] * bond['weight']
            for bond in selected_bonds.values()
        )

        portfolio_convexity = sum(
            bond['convexity'] * bond['weight']
            for bond in selected_bonds.values()
        )

        # Store portfolio
        strategy_name = f"bullet_{target_duration:.2f}"

        self.strategies[strategy_name] = {
            'name': f"Bullet Strategy ({target_duration:.2f})",
            'type': 'bullet',
            'target_duration': target_duration,
            'achieved_duration': portfolio_duration,
            'achieved_convexity': portfolio_convexity,
            'duration_window': duration_window,
            'portfolio': selected_bonds,
            'performance': {}
        }

        return selected_bonds

    def prepare_scenario_analysis(self, create_standard_scenarios=True):
        """
        Prepare scenario analysis for evaluating strategies.

        Parameters:
            create_standard_scenarios (bool): Whether to create standard scenarios

        Returns:
            ScenarioAnalysis: Configured scenario analyzer
        """
        if create_standard_scenarios:
            # Create standard scenarios if they don't exist
            if not hasattr(self.scenario_analyzer, 'scenarios') or not self.scenario_analyzer.scenarios:
                # Create historical scenarios
                self.scenario_analyzer.create_covid_crisis_scenario()
                self.scenario_analyzer.create_inflation_hike_scenario()

                # Create hypothetical scenarios
                self.scenario_analyzer.create_parallel_shift_scenario(100)
                self.scenario_analyzer.create_parallel_shift_scenario(200)
                self.scenario_analyzer.create_parallel_shift_scenario(-50)

                self.scenario_analyzer.create_bull_steepener_scenario(50)
                self.scenario_analyzer.create_bear_steepener_scenario(50)
                self.scenario_analyzer.create_bull_flattener_scenario(50)
                self.scenario_analyzer.create_bear_flattener_scenario(50)

                self.scenario_analyzer.create_inversion_scenario(50)

        return self.scenario_analyzer

    def evaluate_strategy_under_scenarios(self, strategy_name, scenarios=None,
                                          include_var=True, confidence_level=0.95,
                                          holding_period=10):
        """
        Evaluate a risk management strategy under various scenarios.

        Parameters:
            strategy_name (str): Name of the strategy to evaluate
            scenarios (list, optional): List of scenario names to evaluate under
            include_var (bool): Whether to calculate VaR for each scenario
            confidence_level (float): Confidence level for VaR calculations
            holding_period (int): Holding period for VaR in days

        Returns:
            dict: Evaluation results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        strategy = self.strategies[strategy_name]
        portfolio = strategy['portfolio']

        # Make sure scenario analyzer is prepared
        self.prepare_scenario_analysis()

        # Default to all scenarios if not specified
        if scenarios is None:
            scenarios = list(self.scenario_analyzer.scenarios.keys())

        results = {}

        for scenario_name in scenarios:
            # Skip scenarios that don't exist
            if scenario_name not in self.scenario_analyzer.scenarios:
                print(f"Warning: Scenario '{scenario_name}' not found. Skipping.")
                continue

            # Evaluate portfolio under scenario
            evaluation = self.scenario_analyzer.evaluate_portfolio_under_scenario(
                portfolio,
                scenario_name
            )

            # Capture key results
            scenario_result = {
                'portfolio_change': evaluation['portfolio_change'],
                'portfolio_percent_change': evaluation['portfolio_percent_change'],
                'portfolio_duration': evaluation['portfolio_duration'],
                'portfolio_convexity': evaluation['portfolio_convexity'],
                'bond_results': evaluation['bond_results']
            }

            # Calculate VaR if requested
            if include_var:
                var_results = self.scenario_analyzer.calculate_var_under_scenario(
                    portfolio,
                    scenario_name,
                    confidence_level,
                    holding_period
                )

                scenario_result['var_results'] = var_results

            results[scenario_name] = scenario_result

        # Store results in strategy
        self.strategies[strategy_name]['performance'] = results

        return results

    def evaluate_all_strategies(self, scenarios=None, include_var=True):
        """
        Evaluate all risk management strategies under various scenarios.

        Parameters:
            scenarios (list, optional): List of scenario names to evaluate under
            include_var (bool): Whether to calculate VaR for each scenario

        Returns:
            dict: Evaluation results for all strategies
        """
        if not self.strategies:
            raise ValueError("No strategies defined to evaluate")

        results = {}

        for strategy_name in self.strategies:
            print(f"Evaluating strategy: {strategy_name}")
            strategy_results = self.evaluate_strategy_under_scenarios(
                strategy_name,
                scenarios,
                include_var
            )

            results[strategy_name] = strategy_results

        return results

    def compare_strategies(self, metrics=None, scenarios=None):
        """
        Compare different risk management strategies across scenarios.

        Parameters:
            metrics (list, optional): List of metrics to compare
            scenarios (list, optional): List of scenarios to include

        Returns:
            pd.DataFrame: Comparison of strategies
        """
        if not self.strategies:
            raise ValueError("No strategies defined to compare")

        # Default metrics
        if metrics is None:
            metrics = ['portfolio_percent_change', 'portfolio_duration', 'portfolio_convexity']

            # Add VaR metrics if available
            first_strategy = next(iter(self.strategies.values()))
            first_performance = first_strategy.get('performance', {})
            if first_performance:
                first_scenario = next(iter(first_performance.values()), {})
                if 'var_results' in first_scenario:
                    metrics.extend(['var_pct_historical', 'var_pct_normal', 'var_pct_t'])

        # Default to all scenarios if not specified
        if scenarios is None:
            all_performances = [s.get('performance', {}) for s in self.strategies.values()]
            all_scenario_names = set()
            for perf in all_performances:
                all_scenario_names.update(perf.keys())
            scenarios = list(all_scenario_names)

        # Create multi-index for rows (strategy, scenario)
        rows = []
        data = []

        for strategy_name, strategy in self.strategies.items():
            strategy_performance = strategy.get('performance', {})

            for scenario_name in scenarios:
                if scenario_name not in strategy_performance:
                    continue

                scenario_result = strategy_performance[scenario_name]
                row_data = []

                for metric in metrics:
                    if metric in scenario_result:
                        row_data.append(scenario_result[metric])
                    elif metric.startswith('var_pct_') and 'var_results' in scenario_result:
                        var_results = scenario_result['var_results']
                        row_data.append(var_results.get(metric, None))
                    else:
                        row_data.append(None)

                rows.append((strategy_name, scenario_name))
                data.append(row_data)

        # Create DataFrame
        multi_index = pd.MultiIndex.from_tuples(rows, names=['Strategy', 'Scenario'])
        df = pd.DataFrame(data, index=multi_index, columns=metrics)

        return df

    def plot_strategy_performance(self, metric='portfolio_percent_change',
                                  scenarios=None, strategies=None,
                                  figsize=(12, 8)):
        """
        Plot performance of strategies across scenarios for a specific metric.

        Parameters:
            metric (str): Metric to plot
            scenarios (list, optional): List of scenarios to include
            strategies (list, optional): List of strategies to include
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        if not self.strategies:
            raise ValueError("No strategies defined to plot")

        # Get comparison data
        comparison = self.compare_strategies(metrics=[metric], scenarios=scenarios)

        # Filter strategies if specified
        if strategies:
            comparison = comparison.loc[strategies]

        # Reset index for easier plotting
        plot_data = comparison.reset_index()

        # Pivot data for grouped bar chart
        pivot_data = plot_data.pivot(index='Scenario', columns='Strategy', values=metric)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        pivot_data.plot(kind='bar', ax=ax)

        # Add labels and title
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        ax.set_ylabel(f'{metric_name}')
        ax.set_title(f'Strategy Performance Comparison - {metric_name}')
        ax.legend(title='Strategy', loc='best')

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)

        plt.tight_layout()
        return fig

    def plot_strategy_heatmap(self, metric='portfolio_percent_change',
                              scenarios=None, strategies=None,
                              figsize=(14, 10), cmap='RdYlGn'):
        """
        Plot a heatmap of strategy performance across scenarios.

        Parameters:
            metric (str): Metric to plot
            scenarios (list, optional): List of scenarios to include
            strategies (list, optional): List of strategies to include
            figsize (tuple): Figure size
            cmap (str): Colormap to use

        Returns:
            matplotlib.figure.Figure: The figure containing the heatmap
        """
        if not self.strategies:
            raise ValueError("No strategies defined to plot")

        # Get comparison data
        comparison = self.compare_strategies(metrics=[metric], scenarios=scenarios)

        # Filter strategies if specified
        if strategies:
            comparison = comparison.loc[strategies]

        # Pivot for heatmap
        pivot_data = comparison.unstack(level=0)

        # Drop the metric level for cleaner visualization
        pivot_data.columns = pivot_data.columns.droplevel(0)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap=cmap, ax=ax,
                    linewidths=0.5, cbar_kws={'label': metric})

        # Add title
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        ax.set_title(f'Strategy Performance Heatmap - {metric_name}')

        plt.tight_layout()
        return fig

    def create_performance_summary(self, output_file=None):
        """
        Create a comprehensive performance summary of all strategies.

        Parameters:
            output_file (str, optional): Path to save the summary

        Returns:
            str: Summary text
        """
        if not self.strategies:
            return "No strategies available for summary."

        # Calculate key metrics across all scenarios

        summary = ["RISK MANAGEMENT STRATEGY PERFORMANCE SUMMARY", "=" * 80, ""]

        # Strategy details
        summary.append("STRATEGIES ANALYZED:")
        summary.append("-" * 50)

        for name, strategy in self.strategies.items():
            summary.append(f"Strategy: {strategy['name']} ({name})")
            summary.append(f"  Type: {strategy['type']}")

            # Add strategy-specific details
            if 'target_duration' in strategy:
                summary.append(f"  Target Duration: {strategy['target_duration']:.2f}")
                summary.append(f"  Achieved Duration: {strategy['achieved_duration']:.2f}")

            if 'target_convexity' in strategy:
                summary.append(f"  Target Convexity: {strategy['target_convexity']:.4f}")
                summary.append(f"  Achieved Convexity: {strategy['achieved_convexity']:.4f}")

            if 'target_var_pct' in strategy:
                summary.append(f"  Target VaR: {strategy['target_var_pct']:.2f}%")
                summary.append(f"  Achieved VaR: {strategy['achieved_var_pct']:.2f}%")

            if 'credit_allocations' in strategy:
                allocations = [f"{rating}: {alloc * 100:.1f}%"
                               for rating, alloc in strategy['credit_allocations'].items()]
                summary.append(f"  Credit Allocations: {', '.join(allocations)}")

            summary.append("")

        # Performance comparison
        summary.append("PERFORMANCE COMPARISON:")
        summary.append("-" * 50)

        # Calculate average performance across scenarios for each strategy
        strategy_avg_performance = {}

        for name, strategy in self.strategies.items():
            if 'performance' in strategy and strategy['performance']:
                # Calculate average portfolio change
                pct_changes = [
                    scenario['portfolio_percent_change']
                    for scenario in strategy['performance'].values()
                ]
                avg_pct_change = sum(pct_changes) / len(pct_changes)

                # Calculate worst-case scenario
                worst_scenario = min(
                    strategy['performance'].items(),
                    key=lambda x: x[1]['portfolio_percent_change']
                )

                # Calculate best-case scenario
                best_scenario = max(
                    strategy['performance'].items(),
                    key=lambda x: x[1]['portfolio_percent_change']
                )

                # Calculate average VaR if available
                var_metrics = {}
                var_methods = ['historical', 'normal', 't']

                for method in var_methods:
                    var_key = f'var_pct_{method}'
                    var_values = []

                    for scenario in strategy['performance'].values():
                        if 'var_results' in scenario and var_key in scenario['var_results']:
                            var_values.append(scenario['var_results'][var_key])

                    if var_values:
                        var_metrics[var_key] = sum(var_values) / len(var_values)

                strategy_avg_performance[name] = {
                    'avg_pct_change': avg_pct_change,
                    'worst_scenario': worst_scenario,
                    'best_scenario': best_scenario,
                    'var_metrics': var_metrics
                }

        # Create average performance table
        summary.append("Average Performance Across All Scenarios:")
        summary.append(f"{'Strategy':<25} {'Avg % Change':<15} {'Worst-Case':<25} {'Best-Case':<25}")
        summary.append("-" * 90)

        for name, perf in strategy_avg_performance.items():
            strategy_name = self.strategies[name]['name']
            avg_change = f"{perf['avg_pct_change']:.2f}%"
            worst = f"{perf['worst_scenario'][0]} ({perf['worst_scenario'][1]['portfolio_percent_change']:.2f}%)"
            best = f"{perf['best_scenario'][0]} ({perf['best_scenario'][1]['portfolio_percent_change']:.2f}%)"

            summary.append(f"{strategy_name:<25} {avg_change:<15} {worst:<25} {best:<25}")

        summary.append("")

        # VaR comparison if available
        has_var = any(
            'var_metrics' in perf and perf['var_metrics']
            for perf in strategy_avg_performance.values()
        )

        if has_var:
            summary.append("Average Value at Risk (% of Portfolio):")
            summary.append(f"{'Strategy':<25} {'Historical VaR':<15} {'Normal VaR':<15} {'t-dist VaR':<15}")
            summary.append("-" * 70)

            for name, perf in strategy_avg_performance.items():
                strategy_name = self.strategies[name]['name']
                var_metrics = perf.get('var_metrics', {})

                hist_var = f"{var_metrics.get('var_pct_historical', 'N/A'):.2f}%" if 'var_pct_historical' in var_metrics else 'N/A'
                norm_var = f"{var_metrics.get('var_pct_normal', 'N/A'):.2f}%" if 'var_pct_normal' in var_metrics else 'N/A'
                t_var = f"{var_metrics.get('var_pct_t', 'N/A'):.2f}%" if 'var_pct_t' in var_metrics else 'N/A'

                summary.append(f"{strategy_name:<25} {hist_var:<15} {norm_var:<15} {t_var:<15}")

            summary.append("")

        # Scenario-specific performance
        summary.append("SCENARIO-SPECIFIC PERFORMANCE:")
        summary.append("-" * 50)

        # Get all scenarios
        all_scenarios = set()
        for strategy in self.strategies.values():
            if 'performance' in strategy:
                all_scenarios.update(strategy['performance'].keys())

        # For each scenario, compare strategy performance
        for scenario in sorted(all_scenarios):
            summary.append(f"Scenario: {scenario}")
            summary.append(f"{'Strategy':<25} {'% Change':<15} {'Duration':<10} {'Convexity':<10}")
            summary.append("-" * 60)

            for name, strategy in self.strategies.items():
                if 'performance' in strategy and scenario in strategy['performance']:
                    result = strategy['performance'][scenario]
                    strategy_name = strategy['name']

                    pct_change = f"{result['portfolio_percent_change']:.2f}%"
                    duration = f"{result['portfolio_duration']:.2f}"
                    convexity = f"{result['portfolio_convexity']:.2f}"

                    summary.append(f"{strategy_name:<25} {pct_change:<15} {duration:<10} {convexity:<10}")

            summary.append("")

        # Key findings and recommendations
        summary.append("KEY FINDINGS AND RECOMMENDATIONS:")
        summary.append("-" * 50)

        # Calculate strategy rankings
        if strategy_avg_performance:
            # Rank by average percent change
            ranked_by_change = sorted(
                strategy_avg_performance.items(),
                key=lambda x: x[1]['avg_pct_change'],
                reverse=True  # Highest first
            )

            # Find the most resilient strategy (best worst-case)
            most_resilient = max(
                strategy_avg_performance.items(),
                key=lambda x: x[1]['worst_scenario'][1]['portfolio_percent_change']
            )

            # Find the strategy with lowest VaR (if available)
            lowest_var_strategy = None
            if has_var:
                strategies_with_hist_var = [
                    (name, perf) for name, perf in strategy_avg_performance.items()
                    if 'var_metrics' in perf and 'var_pct_historical' in perf['var_metrics']
                ]

                if strategies_with_hist_var:
                    lowest_var_strategy = min(
                        strategies_with_hist_var,
                        key=lambda x: x[1]['var_metrics']['var_pct_historical']
                    )

            # Add findings
            summary.append("1. Overall Performance Ranking:")
            for i, (name, _) in enumerate(ranked_by_change):
                strategy_name = self.strategies[name]['name']
                summary.append(f"   {i + 1}. {strategy_name}")

            summary.append("")
            summary.append(f"2. Most Resilient Strategy: {self.strategies[most_resilient[0]]['name']}")
            summary.append(
                f"   Worst-case scenario: {most_resilient[1]['worst_scenario'][0]} ({most_resilient[1]['worst_scenario'][1]['portfolio_percent_change']:.2f}%)")

            if lowest_var_strategy:
                summary.append("")
                summary.append(f"3. Lowest Risk Strategy (by VaR): {self.strategies[lowest_var_strategy[0]]['name']}")
                summary.append(f"   Historical VaR: {lowest_var_strategy[1]['var_metrics']['var_pct_historical']:.2f}%")

            summary.append("")
            summary.append("Recommendations:")

            # Add recommendations based on findings
            if ranked_by_change:
                best_strategy = ranked_by_change[0]
                summary.append(
                    f"- The {self.strategies[best_strategy[0]]['name']} strategy showed the best overall performance")
                summary.append(
                    f"  across all scenarios with an average return of {best_strategy[1]['avg_pct_change']:.2f}%.")

            if most_resilient:
                summary.append(
                    f"- For investors concerned with downside protection, the {self.strategies[most_resilient[0]]['name']}")
                summary.append(f"  strategy offers the best resilience in stressed market scenarios.")

            if lowest_var_strategy:
                summary.append(
                    f"- Risk-averse investors should consider the {self.strategies[lowest_var_strategy[0]]['name']}")
                summary.append(f"  strategy for its lower Value-at-Risk profile.")

            # Strategy-specific insights
            duration_strategies = [s for s in self.strategies.values() if s['type'] in ['duration_matched', 'bullet']]
            convexity_strategies = [s for s in self.strategies.values() if s['type'] == 'convexity_enhanced']
            var_strategies = [s for s in self.strategies.values() if s['type'] == 'var_optimized']
            diversified_strategies = [s for s in self.strategies.values() if s['type'] == 'credit_diversified']
            barbell_strategies = [s for s in self.strategies.values() if s['type'] == 'barbell']

            if duration_strategies and convexity_strategies:
                summary.append("")
                summary.append("- Duration matching alone is insufficient during periods of high volatility.")
                summary.append("  Enhancing convexity provides additional protection against large yield changes.")

            if var_strategies:
                summary.append("")
                summary.append("- VaR-optimized portfolios can effectively balance risk and return,")
                summary.append("  particularly in moderately stressed market conditions.")

            if diversified_strategies:
                summary.append("")
                summary.append("- Credit diversification demonstrates its value during credit stress scenarios,")
                summary.append("  though it may underperform in purely rate-driven scenarios.")

            if barbell_strategies:
                summary.append("")
                summary.append("- The barbell strategy's performance illustrates its usefulness for investors")
                summary.append("  expecting significant yield curve shape changes.")

            # General recommendations
            summary.append("")
            summary.append("General Framework Recommendations:")
            summary.append("- Combine multiple risk metrics rather than relying on a single measure")
            summary.append("- Regularly stress test portfolios using both historical and hypothetical scenarios")
            summary.append("- Adjust risk management approach based on the current economic environment and outlook")
            summary.append("- For most investors, a balanced approach using duration management,")
            summary.append("  convexity enhancement, and proper diversification will provide the best results")

        # Write to file or return as string
        summary_text = "\n".join(summary)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(summary_text)

        return summary_text


def create_bond_cash_flows(face_value, coupon_rate, years, payments_per_year=2):
    """Helper function to create bond cash flows for testing."""
    num_periods = int(years * payments_per_year)
    coupon_payment = face_value * coupon_rate / payments_per_year

    cash_flows = np.ones(num_periods) * coupon_payment
    cash_flows[-1] += face_value  # Add face value at maturity

    times = np.arange(1, num_periods + 1) / payments_per_year

    return cash_flows, times


def test_risk_management_module():
    """
    Test the RiskManagementModule with a sample bond universe and evaluation.
    """
    print("Testing Risk Management Strategy Evaluation Module...")

    # Create a sample yield curve
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    rates = np.array([0.0210, 0.0215, 0.0220, 0.0230, 0.0240, 0.0250, 0.0260, 0.0270, 0.0275, 0.0280])
    reference_date = datetime(2025, 1, 1)

    # Create Treasury yield curve
    base_curve = TreasuryYieldCurve(tenors, rates, reference_date)

    # Initialize risk management module
    risk_mgmt = RiskManagementModule(base_curve)

    # Create a varied bond universe with different durations, convexities, and credit ratings
    print("\nCreating bond universe...")

    # Short-term bonds
    for i in range(3):
        coupon = 0.020 + i * 0.003
        years = 1 + i * 0.5
        cash_flows, times = create_bond_cash_flows(100, coupon, years)
        ytm = 0.021 + i * 0.002

        risk_mgmt.add_bond(
            f"Short{i + 1}",
            cash_flows,
            times,
            ytm,
            credit_rating="AAA"
        )

    # Medium-term bonds with varied credit quality
    for i in range(4):
        coupon = 0.025 + i * 0.003
        years = 3 + i
        cash_flows, times = create_bond_cash_flows(100, coupon, years)
        ytm = 0.024 + i * 0.002

        # Vary credit ratings
        if i == 0:
            rating = "AAA"
        elif i == 1:
            rating = "AA"
        elif i == 2:
            rating = "A"
        else:
            rating = "BBB"

        risk_mgmt.add_bond(
            f"Med{i + 1}",
            cash_flows,
            times,
            ytm,
            credit_rating=rating
        )

    # Long-term bonds
    for i in range(3):
        coupon = 0.030 + i * 0.003
        years = 10 + i * 5
        cash_flows, times = create_bond_cash_flows(100, coupon, years)
        ytm = 0.027 + i * 0.002

        # Vary credit ratings
        if i == 0:
            rating = "AA"
        elif i == 1:
            rating = "A"
        else:
            rating = "BBB"

        risk_mgmt.add_bond(
            f"Long{i + 1}",
            cash_flows,
            times,
            ytm,
            credit_rating=rating
        )

    # Create specialized bonds for testing barbell and others
    cash_flows, times = create_bond_cash_flows(100, 0.035, 15)
    risk_mgmt.add_bond("Long4", cash_flows, times, 0.033, credit_rating="A")

    cash_flows, times = create_bond_cash_flows(100, 0.04, 20)
    risk_mgmt.add_bond("Long5", cash_flows, times, 0.034, credit_rating="BBB")

    cash_flows, times = create_bond_cash_flows(100, 0.015, 0.5)
    risk_mgmt.add_bond("VeryShort", cash_flows, times, 0.020, credit_rating="AAA")

    # Display bond universe summary
    print("Bond Universe Summary:")
    print(f"{'Bond ID':<12} {'YTM':<8} {'Duration':<10} {'Convexity':<10} {'Rating':<6}")
    print("-" * 50)

    for bond_id, bond in risk_mgmt.bond_universe.items():
        print(
            f"{bond_id:<12} {bond['ytm'] * 100:.2f}% {bond['duration']:.2f} {bond['convexity']:.2f} {bond['credit_rating']:<6}")

    print("\nCreating and evaluating risk management strategies...")

    # Create various portfolios with different strategies
    duration_matched_5 = risk_mgmt.create_duration_matched_portfolio(5.0)
    convexity_enhanced_5 = risk_mgmt.create_convexity_enhanced_portfolio(5.0)
    var_optimized_3 = risk_mgmt.create_var_optimized_portfolio(3.0)
    credit_diversified = risk_mgmt.create_credit_diversified_portfolio(
        target_duration=5.0,
        credit_allocations={'AAA': 0.25, 'AA': 0.25, 'A': 0.25, 'BBB': 0.25}
    )
    barbell = risk_mgmt.create_barbell_portfolio(5.0)
    bullet = risk_mgmt.create_bullet_portfolio(5.0)

    # Prepare scenario analysis
    risk_mgmt.prepare_scenario_analysis()

    # Evaluate all strategies
    risk_mgmt.evaluate_all_strategies()

    # Create comparison
    print("\nStrategy comparison:")
    comparison = risk_mgmt.compare_strategies()
    print(comparison)

    # Create visualizations
    print("\nCreating visualizations...")
    # Performance plot
    performance_plot = risk_mgmt.plot_strategy_performance()
    plt.tight_layout()
    plt.savefig('strategy_performance_comparison.png')
    plt.close()

    # Heatmap
    heatmap = risk_mgmt.plot_strategy_heatmap()
    plt.tight_layout()
    plt.savefig('strategy_heatmap.png')
    plt.close()

    # Create summary report
    print("\nGenerating strategy performance summary...")
    summary = risk_mgmt.create_performance_summary("risk_management_summary.txt")
    print("\nSummary report excerpt:")
    print(summary[:1000] + "...")  # Print just the start of the report

    print("\nRisk management strategy evaluation tests completed.")
    return risk_mgmt


if __name__ == "__main__":
    risk_mgmt = test_risk_management_module()