import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq


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


# Example usage and testing of the PricingModule
def test_pricing_module():
    """Test the PricingModule with examples to verify functionality."""

    # Set up a bond example
    face_value = 100
    coupon_rate = 0.05
    years_to_maturity = 10
    payments_per_year = 2
    ytm = 0.06

    # Create cash flows and time points
    num_periods = int(years_to_maturity * payments_per_year)
    coupon_payment = face_value * coupon_rate / payments_per_year

    times = np.arange(1, num_periods + 1) / payments_per_year
    cash_flows = np.ones(num_periods) * coupon_payment
    cash_flows[-1] += face_value

    # Calculate initial price and risk metrics
    price = PricingModule.calculate_price(cash_flows, times, ytm)
    duration = PricingModule.calculate_duration(cash_flows, times, ytm)
    mod_duration = PricingModule.calculate_modified_duration(cash_flows, times, ytm)
    convexity = PricingModule.calculate_convexity(cash_flows, times, ytm)

    print("Initial Bond Characteristics:")
    print(f"Price: ${price:.2f}")
    print(f"Macaulay Duration: {duration:.4f} years")
    print(f"Modified Duration: {mod_duration:.4f}")
    print(f"Convexity: {convexity:.4f}")
    print("-" * 50)

    # Test price change approximations for different yield changes
    yield_changes = [0.001, 0.005, 0.01, -0.001, -0.005, -0.01]

    results = []
    for dy in yield_changes:
        # Actual price at new yield
        new_price = PricingModule.calculate_price(cash_flows, times, ytm + dy)
        actual_change = new_price - price

        # Duration-based approximation
        duration_approx = PricingModule.duration_based_price_change(cash_flows, times, ytm, dy)

        # Duration + Convexity approximation
        duration_convexity_approx = PricingModule.duration_convexity_based_price_change(
            cash_flows, times, ytm, dy)

        results.append({
            'Yield Change (bps)': dy * 10000,
            'Actual Price': new_price,
            'Actual Change': actual_change,
            'Duration Approx': duration_approx,
            'Duration Error': duration_approx - actual_change,
            'Duration+Convexity Approx': duration_convexity_approx,
            'Duration+Convexity Error': duration_convexity_approx - actual_change
        })

    # Display results as a table
    results_df = pd.DataFrame(results)
    print("Price Change Approximation Comparison:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 50)

    # Test duration matching with a more similar portfolio to avoid boundary issues
    # Create a portfolio with characteristics closer to the target bond
    portfolio_years = 8
    portfolio_coupon = 0.045

    portfolio_num_periods = int(portfolio_years * payments_per_year)
    portfolio_coupon_payment = face_value * portfolio_coupon / payments_per_year

    portfolio_times = np.arange(1, portfolio_num_periods + 1) / payments_per_year
    portfolio_cash_flows = np.ones(portfolio_num_periods) * portfolio_coupon_payment
    portfolio_cash_flows[-1] += face_value

    portfolio_ytm = 0.055

    # Match duration
    adjusted_price, adjusted_ytm, duration_match_quality, warning = PricingModule.duration_matching_price(
        cash_flows, times, ytm,
        portfolio_cash_flows, portfolio_times, portfolio_ytm
    )

    portfolio_duration = PricingModule.calculate_duration(
        portfolio_cash_flows, portfolio_times, portfolio_ytm)

    target_duration = PricingModule.calculate_duration(
        cash_flows, times, adjusted_ytm)

    print("Duration Matching Results:")
    print(f"Portfolio Duration: {portfolio_duration:.4f} years")
    print(f"Target Bond Duration After Adjustment: {target_duration:.4f} years")
    print(f"Adjusted YTM: {adjusted_ytm:.4%}")
    print(f"Adjusted Price: ${adjusted_price:.2f}")
    print(f"Duration Match Quality (lower is better): {duration_match_quality:.8f}")
    if warning:
        print(f"Warning: {warning}")
    print("-" * 50)

    # Test convexity matching
    portfolio_convexity = PricingModule.calculate_convexity(
        portfolio_cash_flows, portfolio_times, portfolio_ytm)

    adjusted_price, adjusted_ytm, match_quality, warning = PricingModule.convexity_matching_price(
        cash_flows, times, ytm,
        portfolio_cash_flows, portfolio_times, portfolio_ytm
    )

    target_duration_after = PricingModule.calculate_duration(
        cash_flows, times, adjusted_ytm)
    target_convexity_after = PricingModule.calculate_convexity(
        cash_flows, times, adjusted_ytm)

    print("Duration and Convexity Matching Results:")
    print(f"Portfolio Duration: {portfolio_duration:.4f} years")
    print(f"Portfolio Convexity: {portfolio_convexity:.4f}")
    print(f"Target Bond Duration After Adjustment: {target_duration_after:.4f} years")
    print(f"Target Bond Convexity After Adjustment: {target_convexity_after:.4f}")
    print(f"Adjusted YTM: {adjusted_ytm:.4%}")
    print(f"Adjusted Price: ${adjusted_price:.2f}")
    print(f"Overall Match Quality (lower is better): {match_quality:.8f}")
    if warning:
        print(f"Warning: {warning}")

    # Demonstrate error improvement with duration+convexity vs duration-only
    print("-" * 50)
    print("Comparing Duration-only vs. Duration+Convexity Approximation Errors:")
    duration_errors = [abs(row['Duration Error']) for row in results]
    convexity_errors = [abs(row['Duration+Convexity Error']) for row in results]

    avg_duration_error = np.mean(duration_errors)
    avg_convexity_error = np.mean(convexity_errors)

    print(f"Average Absolute Error (Duration Only): ${avg_duration_error:.4f}")
    print(f"Average Absolute Error (Duration+Convexity): ${avg_convexity_error:.4f}")
    print(f"Error Reduction: {(1 - avg_convexity_error / avg_duration_error) * 100:.2f}%")

    # Plot yield change vs approximation error
    plt.figure(figsize=(10, 6))
    plt.plot([r['Yield Change (bps)'] for r in results],
             [abs(r['Duration Error']) for r in results],
             'o-', label='Duration Only')
    plt.plot([r['Yield Change (bps)'] for r in results],
             [abs(r['Duration+Convexity Error']) for r in results],
             's-', label='Duration + Convexity')
    plt.xlabel('Yield Change (bps)')
    plt.ylabel('Absolute Error ($)')
    plt.title('Approximation Error vs. Yield Change')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('approximation_error_comparison.png')
    plt.close()

    # Return the plot
    return 'approximation_error_comparison.png'


if __name__ == "__main__":
    test_pricing_module()