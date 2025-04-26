import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime, timedelta


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
        # Exmaple,  [0.25, 0.5, 1, 2, 5, 10] and [0.5, 1, 2, 5, 10, 30], then [0.5, 1, 2, 5, 10]
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

def test_yield_curves():

    # Sample data
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    treasury_rates = np.array([0.0516, 0.0518, 0.0520, 0.0510, 0.0490, 0.0480, 0.0470, 0.0455, 0.0445, 0.0440])
    corp_rates = treasury_rates + np.array([0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065])
    reference_date = datetime(2025, 4, 24)

    print("Creating yield curves...")

    # Create Treasury yield curve
    treasury_curve = TreasuryYieldCurve(tenors, treasury_rates, reference_date)

    # Create Corporate yield curve
    corporate_curve = CorporateYieldCurve(tenors, corp_rates, treasury_curve, "A", reference_date)

    # Create Spot rate curve from Treasury curve
    spot_curve = SpotRateYieldCurve(tenors, par_curve=treasury_curve, reference_date=reference_date)

    # Create Forward rate curve from Spot curve
    forward_curve = ForwardRateYieldCurve(tenors, spot_curve=spot_curve, reference_date=reference_date)

    # Test curve properties
    print("\nTesting Treasury curve properties...")
    print(f"2-10 Term Premium: {treasury_curve.calculate_term_premium(2, 10):.2f} bps")
    print(f"Is curve inverted (2-10)? {treasury_curve.is_inverted(2, 10)}")

    print("\nTesting Corporate curve properties...")
    print(f"10-year Corporate-Treasury Spread: {corporate_curve.calculate_spread(10):.2f} bps")

    print("\nTesting Spot curve properties...")
    tenors, discount_factors = spot_curve.get_discount_factors()
    print("Discount Factors:")
    for t, df in zip(tenors, discount_factors):
        print(f"  {t:4.2f}y: {df:.6f}")

    print("\nTesting Forward curve properties...")
    for start in [1, 2, 5]:
        end = start + 5
        forward = forward_curve.get_forward_rate(start, end)
        print(f"Forward rate {start}y to {end}y: {forward * 100:.4f}%")

    # Create plots
    print("\nGenerating plots...")

    # Create a 2x2 grid for the main plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Treasury and Corporate curves
    ax1 = axes[0, 0]
    treasury_curve.plot(ax=ax1, label="Treasury")
    corporate_curve.plot(ax=ax1, label="Corporate A")

    # Plot 2: Corporate-Treasury spread
    ax2 = axes[0, 1]
    corporate_curve.plot_spread(ax=ax2)

    # Plot 3: Spot rates
    ax3 = axes[1, 0]
    spot_curve.plot(ax=ax3, label="Spot Rates")

    # Plot 4: Discount factors
    ax4 = axes[1, 1]
    spot_curve.plot_discount_factors(ax=ax4)

    # Adjust layout
    plt.tight_layout()

    # Create a separate figure for forward curves
    plt.figure(figsize=(10, 6))
    forward_curve.plot_forward_curves()

    plt.show()

    print("\nYield curve tests completed.")
    return treasury_curve, corporate_curve, spot_curve, forward_curve


# Run test
if __name__ == "__main__":
    test_yield_curves()