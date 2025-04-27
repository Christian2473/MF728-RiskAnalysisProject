import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import interpolate
from scipy.stats import norm, t, kurtosis
from scipy.optimize import minimize, brentq, fsolve

###############################################################################
#                          YIELD CURVE CONSTRUCTION                           #
###############################################################################

class YieldCurve:
    """
    Class for all yield curve types.

    This class provides common functionality for all yield curve implementations
    and serves as a foundation for specialized yield curve types.
    """

    def __init__(self, tenors, rates, reference_date=None, curve_type="Generic"):
        """
        Initialize a yield curve.

        Parameters:
            tenors (array-like): Time points in years
            rates (array-like): Interest rates corresponding to tenors (in decimal form)
            reference_date (datetime, optional): Reference date for the curve
            curve_type (str): Type of the yield curve
        """
        self.tenors = np.array(tenors)
        self.rates = np.array(rates)
        self.reference_date = reference_date or datetime.today()
        self.curve_type = curve_type
        self._validate_inputs()
        self._interpolator = None
        self._build_interpolator()

    def _validate_inputs(self):
        """Validate input parameters."""
        if len(self.tenors) != len(self.rates):
            raise ValueError("Length of tenors and rates must be the same")
        if not np.all(np.diff(self.tenors) > 0):
            raise ValueError("Tenors must be strictly increasing")

    def _build_interpolator(self):
        """Build the interpolation function."""
        # Estimate rates at any tenor point in between and this is easy to use
        # S(x) = a_i + b_i(x-x_i) + c_i(x-x_i)² + d_i(x-x_i)³, tenors is x and rate is y
        self._interpolator = interpolate.CubicSpline(self.tenors, self.rates)

    def get_rate(self, tenor):
        """
        Get interpolated rate for a specific tenor.

        Parameters:
            tenor (float or array-like): Tenor(s) in years

        Returns:
            float or array: Interpolated rate(s)
        """

        # If no prefer interpolator exist to use then create default one to use
        if self._interpolator is None:
            self._build_interpolator()

        # Converts the input to a NumPy array, even if it's a single value to handle scalar and array inputs
        tenor_array = np.atleast_1d(tenor)

        # Verifies all requested tenors fall within the range of available data
        if np.any(tenor_array < self.tenors[0]) or np.any(tenor_array > self.tenors[-1]):
            raise ValueError(f"Tenor out of bounds. Valid range: [{self.tenors[0]}, {self.tenors[-1]}]")

        # Calls the cubic spline interpolator with the requested tenor(s)
        rates = self._interpolator(tenor_array)

        # Returns a single value if the input was a single value, otherwise, returns an array of rates
        if np.isscalar(tenor):
            return rates[0]

        return rates

    def plot(self, min_tenor=None, max_tenor=None, points=100, ax=None, label=None):
        """
        Plot the yield curve.

        Parameters:
            min_tenor (float, optional): Minimum tenor to plot
            max_tenor (float, optional): Maximum tenor to plot
            points (int): Number of points to plot
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            label (str, optional): Label for the plot

        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        # if no existing plotting area, create one
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        min_tenor = min_tenor or self.tenors[0]
        max_tenor = max_tenor or self.tenors[-1]

        plot_tenors = np.linspace(min_tenor, max_tenor, points)
        plot_rates = self.get_rate(plot_tenors)

        # Plot the curve
        ax.plot(plot_tenors, plot_rates * 100, label=label or self.curve_type)

        # Plot original data points
        ax.scatter(self.tenors, self.rates * 100, marker='o')

        ax.set_xlabel('Tenor (years)')
        ax.set_ylabel('Rate (%)')
        ax.set_title(f'{self.curve_type} Yield Curve - {self.reference_date.strftime("%Y-%m-%d")}')
        ax.grid(True)
        ax.legend()

        return ax


class TreasuryYieldCurve(YieldCurve):
    """
    Treasury Bond yield curve implementation.

    This class represents the yield curve for Treasury bonds and provides
    methods for analyzing and visualizing Treasury yields.
    """

    def __init__(self, tenors, rates, reference_date=None):
        """
        Initialize a Treasury yield curve.

        Parameters:
            tenors (array-like): Time points in years
            rates (array-like): Treasury yields corresponding to tenors (in decimal form)
            reference_date (datetime, optional): Reference date for the curve
        """

        # Inherits all functionality from the base YieldCurve class
        # Call the parent's __init__ method
        super().__init__(tenors, rates, reference_date, "Treasury")

    def calculate_term_premium(self, short_term=1, long_term=10):
        """
        Calculate the term premium between two points on the curve.

        Parameters:
            short_term (float): Short-term tenor in years
            long_term (float): Long-term tenor in years

        Returns:
            float: Term premium in basis points
        """
        if short_term >= long_term:
            raise ValueError("short_term must be less than long_term")

        short_rate = self.get_rate(short_term)
        long_rate = self.get_rate(long_term)

        # Term Premium (in basis points) = (Long-Term Rate - Short-Term Rate) × 10,000
        # Example: term_premium = (0.0425 - 0.0350) * 10000 = 75 basis points for 1 year: 3.50% and 10 year: 4.25%
        return (long_rate - short_rate) * 10000

    def is_inverted(self, short_term=2, long_term=10):
        """
        Check if the yield curve is inverted between two points.

        Parameters:
            short_term (float): Short-term tenor in years
            long_term (float): Long-term tenor in years

        Returns:
            bool: True if the curve is inverted
        """
        return self.calculate_term_premium(short_term, long_term) < 0


class CorporateYieldCurve(YieldCurve):
    """
    Corporate Bond yield curve implementation.

    This class represents the yield curve for Corporate bonds and provides
    methods for analyzing corporate bond spreads over Treasuries.
    """

    def __init__(self, tenors, rates, treasury_curve=None, credit_rating="IG", reference_date=None):
        """
        Initialize a Corporate yield curve.

        Parameters:
            tenors (array-like): Time points in years
            rates (array-like): Corporate yields corresponding to tenors (in decimal form)
            treasury_curve (TreasuryYieldCurve, optional): Treasury curve for spread calculations
            credit_rating (str): Credit rating of the corporate curve
            reference_date (datetime, optional): Reference date for the curve
        """
        self.credit_rating = credit_rating
        self.treasury_curve = treasury_curve
        curve_type = f"Corporate {credit_rating}"
        super().__init__(tenors, rates, reference_date, curve_type)

    def calculate_spread(self, tenor, treasury_curve=None):
        """
        Calculate spread over Treasury for a specific tenor.

        Parameters:
            tenor (float): Tenor in years
            treasury_curve (TreasuryYieldCurve, optional): Treasury curve to use

        Returns:
            float: Spread in basis points
        """
        tc = treasury_curve or self.treasury_curve

        if tc is None:
            raise ValueError("Treasury curve must be provided")

        corporate_rate = self.get_rate(tenor)
        treasury_rate = tc.get_rate(tenor)

        return (corporate_rate - treasury_rate) * 10000

    def get_spread_curve(self, treasury_curve=None):
        """
        Get the full spread curve over Treasury.

        Parameters:
            treasury_curve (TreasuryYieldCurve, optional): Treasury curve to use

        Returns:
            tuple: (tenors, spreads) where spreads are in basis points
        """
        tc = treasury_curve or self.treasury_curve

        if tc is None:
            raise ValueError("Treasury curve must be provided")

        # Use common tenors between both curves
        # Example,  [0.25, 0.5, 1, 2, 5, 10] and [0.5, 1, 2, 5, 10, 30], then [0.5, 1, 2, 5, 10]
        common_tenors = []
        for tenor in self.tenors:
            if tenor >= tc.tenors[0] and tenor <= tc.tenors[-1]:
                common_tenors.append(tenor)

        common_tenors = np.array(common_tenors)
        spreads = np.array([self.calculate_spread(t, tc) for t in common_tenors])

        return common_tenors, spreads

    def plot_spread(self, treasury_curve=None, ax=None):
        """
        Plot the spread curve over Treasury.

        Parameters:
            treasury_curve (TreasuryYieldCurve, optional): Treasury curve to use
            ax (matplotlib.axes.Axes, optional): Axes to plot on

        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        tenors, spreads = self.get_spread_curve(treasury_curve)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(tenors, spreads)
        ax.scatter(tenors, spreads, marker='o')

        ax.set_xlabel('Tenor (years)')
        ax.set_ylabel('Spread (bps)')
        ax.set_title(f'{self.credit_rating} Corporate Spread Curve - {self.reference_date.strftime("%Y-%m-%d")}')
        ax.grid(True)

        return ax


class SpotRateYieldCurve(YieldCurve):
    """
    Spot Rate yield curve implementation.

    This class represents the zero-coupon (spot) yield curve and provides
    methods for bootstrapping spot rates from par yields.
    """

    def __init__(self, tenors, rates=None, par_curve=None, reference_date=None):
        """
        Initialize a Spot Rate yield curve.

        Parameters:
            tenors (array-like): Time points in years
            rates (array-like, optional): Spot rates corresponding to tenors (in decimal form)
            par_curve (YieldCurve, optional): Par yield curve to bootstrap from
            reference_date (datetime, optional): Reference date for the curve
        """
        self.par_curve = par_curve

        if rates is None and par_curve is None:
            raise ValueError("Either rates or par_curve must be provided")

        if rates is None:
            # Bootstrap spot rates from par curve
            rates = self.bootstrap_spot_rates(tenors, par_curve)

        super().__init__(tenors, rates, reference_date, "Spot Rate")

    @staticmethod
    def bootstrap_spot_rates(tenors, par_curve, frequency=2):
        """
        Bootstrap spot rates from par yields.

        Parameters:
            tenors (array-like): Time points in years
            par_curve (YieldCurve): Par yield curve
            frequency (int): Coupon frequency per year

        Returns:
            array: Bootstrapped spot rates
        """

        tenors = np.array(tenors)
        n = len(tenors)
        spot_rates = np.zeros(n)

        # For the first tenor, spot rate equals par rate because a bond with just one payment is effectively a zero-coupon bond
        spot_rates[0] = par_curve.get_rate(tenors[0])

        # Calculate discount factors for the first tenor
        discount_factors = np.array([np.exp(-spot_rates[0] * tenors[0])])

        # Bootstrap remaining tenors
        for i in range(1, n):

            # Gets the current par yield and tenor to process
            par_rate = par_curve.get_rate(tenors[i])
            tenor = tenors[i]

            # Calculate how many coupon payments occur and time between payments
            periods = int(tenor * frequency)
            dt = tenor / periods # frequency = 2 then this is 0.5! Not the same as frequency, for discounting purpose

            # Create coupon payment times by np.arange(start, stop, step), create arrays with evenly spaced values
            # Example: (0.25, 2 + 0.25/2, 0.25), then [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            payment_times = np.arange(dt, tenor + dt / 2, dt)

            # Identifies which payment times already have spot rates calculated from previous iterations
            known_times_mask = payment_times < tenors[i - 1] + dt / 2
            known_times = payment_times[known_times_mask]

            # Interpolate spot rates for known times
            if len(known_times) > 0:
                # Perform linear interpolation to estimate values between data points
                known_spot_rates = np.interp(known_times, tenors[:i], spot_rates[:i])
                # Discount factor b/w them all
                known_discount_factors = np.exp(-known_spot_rates * known_times)
            else:
                # No known times? initialize as an empty array, we can use it later to avoid error if tenors are not evenly spaced
                known_discount_factors = np.array([])

            # Calculate value of coupon payments for known times
            if len(known_discount_factors) > 0:
                coupon_value = par_rate / frequency * known_discount_factors.sum()
            else:
                # No known coupon payments before the current tenor, still work for rest of the code!
                coupon_value = 0

            # Solve for the final discount factor
            final_payment = 1 + par_rate / frequency
            final_df = (1 - coupon_value) / final_payment

            # Calculate the spot rate for the current tenor
            # Discount Factor = e^(-r(t)×t) -> -r(t)×t = ln(Discount Factor)
            spot_rates[i] = -np.log(final_df) / tenor

            # Update discount factors array
            discount_factors = np.append(discount_factors, final_df)

        return spot_rates

    def get_discount_factor(self, tenor):
        """
        Get discount factor for a specific tenor.

        Parameters:
            tenor (float or array-like): Tenor(s) in years

        Returns:
            float or array: Discount factor(s)
        """
        rate = self.get_rate(tenor)

        # Handle scalar and array inputs
        if np.isscalar(tenor):
            return np.exp(-rate * tenor)

        return np.exp(-rate * np.array(tenor))

    def get_discount_factors(self):
        """
        Get all discount factors for the curve tenors.

        Returns:
            tuple: (tenors, discount_factors)
        """
        discount_factors = self.get_discount_factor(self.tenors)
        return self.tenors, discount_factors

    def plot_discount_factors(self, ax=None):
        """
        Plot the discount factors curve.

        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on

        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        tenors, discount_factors = self.get_discount_factors()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(tenors, discount_factors)
        ax.scatter(tenors, discount_factors, marker='o')

        ax.set_xlabel('Tenor (years)')
        ax.set_ylabel('Discount Factor')
        ax.set_title(f'Discount Factors - {self.reference_date.strftime("%Y-%m-%d")}')
        ax.grid(True)

        return ax


class ForwardRateYieldCurve(YieldCurve):
    """
    Forward Rate yield curve implementation.

    This class represents the forward rate yield curve and provides
    methods for calculating forward rates from spot rates.
    """

    def __init__(self, tenors, rates=None, spot_curve=None, reference_date=None):
        """
        Initialize a Forward Rate yield curve.

        Parameters:
            tenors (array-like): Time points in years
            rates (array-like, optional): Forward rates corresponding to tenors (in decimal form)
            spot_curve (SpotRateYieldCurve, optional): Spot rate curve to derive from
            reference_date (datetime, optional): Reference date for the curve
        """
        self.spot_curve = spot_curve

        if rates is None and spot_curve is None:
            raise ValueError("Either rates or spot_curve must be provided")

        if rates is None:
            # Derive forward rates from spot curve
            rates = self.derive_forward_rates(tenors, spot_curve)

        super().__init__(tenors, rates, reference_date, "Forward Rate")

    @staticmethod
    def derive_forward_rates(tenors, spot_curve):
        """
        Derive instantaneous forward rates from spot rates.

        Parameters:
            tenors (array-like): Time points in years
            spot_curve (SpotRateYieldCurve): Spot rate curve

        Returns:
            array: Forward rates
        """
        # Convert input to numpy array for efficient operations
        tenors = np.array(tenors)

        # Get spot rates at each tenor
        spot_rates = np.array([spot_curve.get_rate(t) for t in tenors])

        # P(0,t): e^(-r(t)·t) = e^(-∫₀ᵗf(s)ds) -> r(t)·t = ∫₀ᵗf(s)ds -> r(t) + t·dr(t)/dt = f(t)
        # Discount Factor = e^(-r(t)×t) -> -r(t)×t = ln(Discount Factor)
        # -> r(t) = (1/t) × ∫₀ᵗ f(s) ds -> r(t) × t = ∫₀ᵗ f(s) ds
        # -> d/dt [r(t) × t] = f(t)

        # Build a cubic spline interpolator for r(t)*t
        rt_interpolator = interpolate.CubicSpline(tenors, spot_rates * tenors)

        # Creates a new function that represents the derivative of the cubic spline, and evaluates this derivative function at each point in the tenors array
        forward_rates = rt_interpolator.derivative()(tenors)

        return forward_rates

    def get_forward_rate(self, start_tenor, end_tenor):
        """
        Get the forward rate between two tenors.

        Parameters:
            start_tenor (float): Start tenor in years
            end_tenor (float): End tenor in years

        Returns:
            float: Forward rate
        """
        if start_tenor >= end_tenor:
            raise ValueError("start_tenor must be less than end_tenor")

        # No spot curve is available?
        # Step 1: Finding the midpoint between the start and end tenors
        # Step 2: Using the instantaneous forward rate at that midpoint
        # Approximating the forward rate between two points by using the instantaneous forward
        if self.spot_curve is None:
            avg_tenor = (start_tenor + end_tenor) / 2
            return self.get_rate(avg_tenor)

        # Calculate from spot rates
        spot_rate_start = self.spot_curve.get_rate(start_tenor)
        spot_rate_end = self.spot_curve.get_rate(end_tenor)

        # Calculate implied forward rate
        # e^(r2 × t2) = e^(r1 × t1) × e^(f × (t2 − t1)) -> r2 × t2 = r1 × t1 + f × (t2 − t1)
        # -> f = (r2 × t2 − r1 × t1) / (t2 − t1)
        forward_rate = ((spot_rate_end * end_tenor) - (spot_rate_start * start_tenor)) / (end_tenor - start_tenor)

        return forward_rate

    def plot_forward_curves(self, start_tenors=(1, 2, 5), maturity=10, points=100, ax=None):
        """
        Plot forward rate curves for different starting tenors.

        Parameters:
            start_tenors (tuple): Starting tenors in years
            maturity (float): Maximum maturity in years
            points (int): Number of points to plot
            ax (matplotlib.axes.Axes, optional): Axes to plot on

        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        for start in start_tenors:
            if start >= maturity:
                continue

            end_tenors = np.linspace(start + 0.1, maturity, points)
            forward_rates = [self.get_forward_rate(start, end) for end in end_tenors]

            ax.plot(end_tenors, np.array(forward_rates) * 100,
                    label=f'{start}y forward')

        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Forward Rate (%)')
        ax.set_title(f'Forward Rate Curves - {self.reference_date.strftime("%Y-%m-%d")}')
        ax.grid(True)
        ax.legend()

        return ax


###############################################################################
#                               PRICING MODULE                                #
###############################################################################

class PricingModule:
    """
    A module for pricing fixed income securities using duration and convexity matching.

    This module implements pricing functions based on duration and convexity
    to provide better approximations of bond price changes when interest rates change.
    """

    @staticmethod
    def calculate_price(cash_flows, times, ytm):
        """
        Calculate the price of a bond given its cash flows and yield to maturity.

        Args:
            cash_flows (list or array): The cash flows of the bond
            times (list or array): The times at which the cash flows occur
            ytm (float): The yield to maturity (as a decimal)

        Returns:
            float: The price of the bond
        """
        return np.sum(cash_flows / (1 + ytm) ** times)

    @staticmethod
    def calculate_duration(cash_flows, times, ytm):
        """
        Calculate the Macaulay duration of a bond.

        Args:
            cash_flows (list or array): The cash flows of the bond
            times (list or array): The times at which the cash flows occur
            ytm (float): The yield to maturity (as a decimal)

        Returns:
            float: The Macaulay duration of the bond
        """
        price = PricingModule.calculate_price(cash_flows, times, ytm)
        weighted_sum = np.sum((times * cash_flows) / (1 + ytm) ** times)
        return weighted_sum / price

    @staticmethod
    def calculate_modified_duration(cash_flows, times, ytm):
        """
        Calculate the modified duration of a bond.

        Args:
            cash_flows (list or array): The cash flows of the bond
            times (list or array): The times at which the cash flows occur
            ytm (float): The yield to maturity (as a decimal)

        Returns:
            float: The modified duration of the bond
        """
        mac_duration = PricingModule.calculate_duration(cash_flows, times, ytm)
        return mac_duration / (1 + ytm)

    @staticmethod
    def calculate_convexity(cash_flows, times, ytm):
        """
        Calculate the convexity of a bond.

        Args:
            cash_flows (list or array): The cash flows of the bond
            times (list or array): The times at which the cash flows occur
            ytm (float): The yield to maturity (as a decimal)

        Returns:
            float: The convexity of the bond
        """
        price = PricingModule.calculate_price(cash_flows, times, ytm)
        weighted_sum = np.sum((times * (times + 1) * cash_flows) / (1 + ytm) ** times)
        return weighted_sum / (price * (1 + ytm) ** 2)

    @staticmethod
    def duration_based_price_change(cash_flows, times, ytm, yield_change):
        """
        Calculate the price change of a bond based on duration approximation.

        Args:
            cash_flows (list or array): The cash flows of the bond
            times (list or array): The times at which the cash flows occur
            ytm (float): The yield to maturity (as a decimal)
            yield_change (float): The change in yield (as a decimal)

        Returns:
            float: The estimated price change
        """
        price = PricingModule.calculate_price(cash_flows, times, ytm)
        mod_duration = PricingModule.calculate_modified_duration(cash_flows, times, ytm)

        # First-order approximation (duration only)
        price_change = -mod_duration * price * yield_change
        return price_change

    @staticmethod
    def duration_convexity_based_price_change(cash_flows, times, ytm, yield_change):
        """
        Calculate the price change of a bond based on duration and convexity approximation.

        Args:
            cash_flows (list or array): The cash flows of the bond
            times (list or array): The times at which the cash flows occur
            ytm (float): The yield to maturity (as a decimal)
            yield_change (float): The change in yield (as a decimal)

        Returns:
            float: The estimated price change
        """
        price = PricingModule.calculate_price(cash_flows, times, ytm)
        mod_duration = PricingModule.calculate_modified_duration(cash_flows, times, ytm)
        convexity = PricingModule.calculate_convexity(cash_flows, times, ytm)

        # Second-order approximation (duration and convexity)
        price_change = -mod_duration * price * yield_change + 0.5 * convexity * price * yield_change ** 2
        return price_change

    @staticmethod
    def duration_matching_price(target_cash_flows, target_times, target_ytm,
                                portfolio_cash_flows, portfolio_times, portfolio_ytm,
                                yield_min=0.001, yield_max=0.40):
        """
        Price a target bond by matching its duration to a portfolio's duration.

        This function adjusts the price of the target bond so that its duration
        matches the duration of the portfolio, which is useful for immunization strategies.

        Args:
            target_cash_flows (list or array): The cash flows of the target bond
            target_times (list or array): The times at which the target cash flows occur
            target_ytm (float): The yield to maturity of the target bond (as a decimal)
            portfolio_cash_flows (list or array): The cash flows of the portfolio
            portfolio_times (list or array): The times at which the portfolio cash flows occur
            portfolio_ytm (float): The yield to maturity of the portfolio (as a decimal)
            yield_min (float): Minimum yield to consider (default 0.1%)
            yield_max (float): Maximum yield to consider (default 40%)

        Returns:
            tuple: (adjusted_price, adjusted_ytm, duration_match_quality, warning)
        """
        # Calculate the duration of the portfolio
        portfolio_duration = PricingModule.calculate_duration(
            portfolio_cash_flows, portfolio_times, portfolio_ytm)

        # Important: Not all durations are achievable for a given bond with fixed cash flow structure !!!
        # Calculate min and max possible durations for the target bond
        min_duration = PricingModule.calculate_duration(target_cash_flows, target_times, yield_max)
        max_duration = PricingModule.calculate_duration(target_cash_flows, target_times, yield_min)

        # Check if the portfolio duration is within the achievable range
        warning = ""
        if portfolio_duration < min_duration:
            warning = f"WARNING: Portfolio duration ({portfolio_duration:.4f}) is below the minimum achievable duration ({min_duration:.4f}) for the target bond even at maximum yield. Consider using a different bond."
        elif portfolio_duration > max_duration:
            warning = f"WARNING: Portfolio duration ({portfolio_duration:.4f}) is above the maximum achievable duration ({max_duration:.4f}) for the target bond even at minimum yield. Consider using a different bond."

        # Define a function to find the yield that gives the target duration
        def duration_difference(ytm):
            duration = PricingModule.calculate_duration(target_cash_flows, target_times, ytm)
            return duration - portfolio_duration

        # Find the yield that matches the duration by using root-finding algorithm
        try:
            # Try to find a root within reasonable bounds
            adjusted_ytm = brentq(duration_difference, yield_min, yield_max)
        except ValueError:
            # If no root found, use optimization to get closest match
            result = minimize(lambda ytm: abs(duration_difference(ytm[0])), [target_ytm],
                              bounds=[(yield_min, yield_max)])
            adjusted_ytm = result.x[0]

            # Check if we hit the boundaries, which means the algorithm wanted to go beyond these constraints but couldn't
            if abs(adjusted_ytm - yield_min) < 1e-6:
                if not warning:
                    warning = f"WARNING: Optimization hit minimum yield boundary ({yield_min}). Perfect duration match not possible."
            elif abs(adjusted_ytm - yield_max) < 1e-6:
                if not warning:
                    warning = f"WARNING: Optimization hit maximum yield boundary ({yield_max}). Perfect duration match not possible."

        # Calculate the price at this yield
        adjusted_price = PricingModule.calculate_price(target_cash_flows, target_times, adjusted_ytm)

        # Calculate actual duration at this yield to check match quality
        actual_duration = PricingModule.calculate_duration(target_cash_flows, target_times, adjusted_ytm)
        duration_match_quality = abs(actual_duration - portfolio_duration) / portfolio_duration

        return adjusted_price, adjusted_ytm, duration_match_quality, warning

    @staticmethod
    def convexity_matching_price(target_cash_flows, target_times, target_ytm,
                                 portfolio_cash_flows, portfolio_times, portfolio_ytm,
                                 weight_duration=0.7, weight_convexity=0.3,
                                 yield_min=0.001, yield_max=0.40):
        """
        Price a target bond by matching both its duration and convexity to a portfolio.

        This function adjusts the price and potentially a spread of the target bond
        so that its duration and convexity match those of the portfolio.

        Args:
            target_cash_flows (list or array): The cash flows of the target bond
            target_times (list or array): The times at which the target cash flows occur
            target_ytm (float): The yield to maturity of the target bond (as a decimal)
            portfolio_cash_flows (list or array): The cash flows of the portfolio
            portfolio_times (list or array): The times at which the portfolio cash flows occur
            portfolio_ytm (float): The yield to maturity of the portfolio (as a decimal)
            weight_duration (float): Weight for duration matching (default 0.7)
            weight_convexity (float): Weight for convexity matching (default 0.3)
            yield_min (float): Minimum yield to consider (default 0.1%)
            yield_max (float): Maximum yield to consider (default 40%)

        Returns:
            tuple: (adjusted_price, adjusted_ytm, match_quality, warning)
        """
        # Calculate the duration and convexity of the portfolio
        portfolio_duration = PricingModule.calculate_duration(
            portfolio_cash_flows, portfolio_times, portfolio_ytm)
        portfolio_convexity = PricingModule.calculate_convexity(
            portfolio_cash_flows, portfolio_times, portfolio_ytm)

        # Calculate min and max possible durations and convexities for the target bond
        min_duration = PricingModule.calculate_duration(target_cash_flows, target_times, yield_max)
        max_duration = PricingModule.calculate_duration(target_cash_flows, target_times, yield_min)
        min_convexity = PricingModule.calculate_convexity(target_cash_flows, target_times, yield_max)
        max_convexity = PricingModule.calculate_convexity(target_cash_flows, target_times, yield_min)

        # Check if the portfolio duration and convexity are within achievable ranges
        warning = ""
        if portfolio_duration < min_duration:
            warning = f"WARNING: Portfolio duration ({portfolio_duration:.4f}) is below the minimum achievable duration ({min_duration:.4f}) for the target bond."
        elif portfolio_duration > max_duration:
            warning = f"WARNING: Portfolio duration ({portfolio_duration:.4f}) is above the maximum achievable duration ({max_duration:.4f}) for the target bond."

        if portfolio_convexity < min_convexity:
            if warning:
                warning += " "
            warning += f"Portfolio convexity ({portfolio_convexity:.4f}) is below the minimum achievable convexity ({min_convexity:.4f}) for the target bond."
        elif portfolio_convexity > max_convexity:
            if warning:
                warning += " "
            warning += f"Portfolio convexity ({portfolio_convexity:.4f}) is above the maximum achievable convexity ({max_convexity:.4f}) for the target bond."

        if warning:
            warning += " Consider using a different bond with characteristics closer to the portfolio."

        # Define objective function to minimize (weighted difference in duration and convexity)
        def objective(ytm):
            duration = PricingModule.calculate_duration(target_cash_flows, target_times, ytm[0])
            convexity = PricingModule.calculate_convexity(target_cash_flows, target_times, ytm[0])

            # Normalized differences
            duration_diff = abs(duration - portfolio_duration) / portfolio_duration
            convexity_diff = abs(convexity - portfolio_convexity) / portfolio_convexity

            return weight_duration * duration_diff + weight_convexity * convexity_diff

        # Find the yield that minimizes the weighted differences
        result = minimize(objective, [target_ytm], bounds=[(yield_min, yield_max)])
        adjusted_ytm = result.x[0]

        # Check if we hit the boundaries
        if abs(adjusted_ytm - yield_min) < 1e-6:
            if not warning:
                warning = f"WARNING: Optimization hit minimum yield boundary ({yield_min}). Perfect match not possible."
        elif abs(adjusted_ytm - yield_max) < 1e-6:
            if not warning:
                warning = f"WARNING: Optimization hit maximum yield boundary ({yield_max}). Perfect match not possible."

        # Calculate the price at this yield
        adjusted_price = PricingModule.calculate_price(target_cash_flows, target_times, adjusted_ytm)

        # Calculate actual duration and convexity at this yield to check match quality
        actual_duration = PricingModule.calculate_duration(target_cash_flows, target_times, adjusted_ytm)
        actual_convexity = PricingModule.calculate_convexity(target_cash_flows, target_times, adjusted_ytm)

        duration_match = abs(actual_duration - portfolio_duration) / portfolio_duration
        convexity_match = abs(actual_convexity - portfolio_convexity) / portfolio_convexity
        match_quality = weight_duration * duration_match + weight_convexity * convexity_match

        return adjusted_price, adjusted_ytm, match_quality, warning


###############################################################################
#                                 VAR MODULE                                  #
###############################################################################

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


###############################################################################
#                           SCENARIO ANALYSIS MODULE                          #
###############################################################################

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


###############################################################################
#                        RISK MANAGEMENT MODULE                               #
###############################################################################
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