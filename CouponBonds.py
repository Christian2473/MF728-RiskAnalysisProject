# %%
import numpy as np
import pandas as pd
import scipy as sci
from collections.abc import Callable
from datetime import datetime, timedelta

class CouponBond:
    def __init__(self, ytm, maturity_years, face_value = 100, frequency=2, issue_date=datetime.today()):
        """
        Initializes a Coupon Bond object.

        Assumes treasuries are issued at par and are compounded semi-annually

        Parameters:
            face_value (float): The bond's face value.
            ytm (float): Current market yield to maturity of the bond.
            maturity_years (float): Years until maturity.
            frequency (int): Coupon payment frequency and compounding frequency per year(default is 2 for semi-annual).
            issue_date (datetime): The bond's issue date.
        """
        self.face_value = face_value
        self.ytm = ytm/100 #yields are quoted in percentages on Bloomberg
        self.maturity_years = maturity_years
        self.frequency = frequency
        self.issue_date = issue_date
        self.maturity_date = self.issue_date + timedelta(days=maturity_years * 365)
        self.periods = int(self.maturity_years * self.frequency)
        self.cash_flow_dates = pd.Series([issue_date + timedelta(weeks=26*i) for i in range(1, self.periods + 1)])
    
    @property
    def cashflows(self)-> np.array[float]:
        """
        Calculates the cash flows of the bond.

        Returns:
            np.array: Cash flows array for each period until maturity.
        """
        periods = self.periods
        if periods == 1:
            return np.array([self.face_value])

        coupon_payment = self.face_value * self.ytm / self.frequency #since we assume treasuries are issued at par, the coupon payment = the yield to maturity
        cash_flows = np.full(periods, coupon_payment)
        cash_flows[-1] += self.face_value  # Adding face value at maturity
        return cash_flows
    
    @property
    def discount_factors(self)->np.array:
        """ returns the discount factors as a numpy array"""

        discount_factors_list = [(1+self.ytm/self.frequency)**(-i) for i in range(1,self.periods + 1)]

        return np.array(discount_factors_list).T

    @property
    def price(self)->float:
        """
        Calculates the bond's price. Since we assume Treasuries are issued at par, the price should be 100

        Returns:
            float: Price of a bond rounded to 2 decimal places.
            example: 98.25 -> $98.25 / 100 face
        """
        cash_flows = self.cashflows

        discounts = self.discount_factors
        price = discounts@cash_flows

        return price[0]

    @property
    def modified_duration(self)->float:
        """
        Calculates the bond's modified duration.

        Returns:
            float: Modified duration.
        """

        y = self.ytm

        der_discount = np.array([-i/self.frequency *(1+ y/self.frequency)**(-i-1) for i in range(1,self.periods + 1)]).T
        fprime = der_discount @ self.cashflows 

        return - fprime / self.price
    
    @property
    def macaulay_duration(self)->float:
        """
        Calculates the Macaulay duration of the bond.

        Returns:
            float: Macaulay duration.
        """

        return  (1+self.ytm/self.frequency) * self.modified_duration

    @property
    def convexity(self)->float:
        """
        Calculates the convexity of the bond.

        Returns:
            float: Convexity.
        """

        y = self.ytm

        
        der_2_discount = np.array([ i*(i+1) /(self.frequency)**2 *(1+self.ytm/self.frequency)**(-i-2) for i in range(1,self.periods + 1)]).T
        f_2_prime = der_2_discount @ self.cashflows
        
        return f_2_prime/self.price

    def value_at_risk(self, confidence_level=0.95)->float:
        """
        Estimates the value at risk (VaR) of the bond based on duration and convexity.

        Parameters:
            confidence_level (float): The confidence level for VaR calculation.

        Returns:
            float: Estimated VaR for the bond.
        """
        modified_duration = self.modified_duration
        convexity = self.convexity
        yield_shock = np.percentile(np.random.normal(0, 0.01, 10000), 1 - confidence_level)
        price_change = -self.price * (modified_duration * yield_shock + 0.5 * convexity * yield_shock**2)
        return price_change
    
    def times_till_coupon(self, current_date)->np.array[float]:
        """Creating a numpy array to get the values of time till coupon payments given a new time
            Returns time till maturity in number of days till coupon/360
        """
        time_till_next_coupon = self.cash_flow_dates.apply( lambda x : (x - current_date))
        time_till_next_coupon = time_till_next_coupon.apply(lambda x: x.days)
        time_till_next_coupon = time_till_next_coupon.apply(lambda x: max(x, 0))
        return np.array(time_till_next_coupon/360)
    
    def interpolated_yield_curve(self, S: pd.Series, T:float)->float:
        """Returning of yield curve interpolation method given a yield curve (pandas series) and time of maturity T."""
        icurve = sci.interpolate.CubicSpline(x= S.index, y = S)

        return icurve(T)
    
    def spot_rates(self, S: pd.Series)->np.array[float]:
        """Bootstrapping spot rates from a yield curve interpolation"""

        # Computing yields from the interpolated yield curve
        tenors = np.arange(.5, 30 + .5, .5)

        yields = self.interpolated_yield_curve(S, tenors)/100

        # Bootstrapping spot rates
        s_rates = np.array([yields[0]])
        discount_factors = (1+s_rates/2)**(-1)

        for i in range(1, len(tenors)):
            y = yields[i]

            first_part = 1 - y/2*discount_factors.sum()
            second_part = (1+y/2)**(-1)

            discount_factors = np.append(discount_factors, first_part * second_part)

            if first_part <= 0:
                raise ValueError(f"Negative or zero discount factor at tenor {tenors[i]}. Check yield inputs.")
            
            s_rates = np.append(s_rates, 2*discount_factors[i]**(-1/(2*tenors[i])) - 2)
        
        return s_rates
    
    def interpolated_spot_curve(self, S:pd.Series, T:float)->pd.Series:
        """Interpolating the Spot Curve given a pandas series containing yields, and a new time T"""
        s_rates = self.spot_rates(S= S)
        tenors = np.arange(.5 , 30+.5, .5)
        icurve = sci.interpolate.CubicSpline(x= tenors, y = s_rates)

        return icurve(T)

    
    def new_price(self, S:pd.DataFrame)->float:
        """Calculating the daily price of a bond given a new yield curve.
        S - S.index : datetime objects or pandas timeseries/timestamps
        Returns a pandas series with price data and indices of the dates
        """

        #initializing an empty series for the for loop
        price_array = np.array([])



        #calculating the price per row
        for i in range(len(S.index)):
            #getting the row for the dataframe
            row = S.iloc[i,:]

            # Calculating Tenors for coupon payments
            ## Remember tenors are (number of days till next coupon/360)
            tenors = self.times_till_coupon(current_date=row.name)
            tenors = tenors[tenors>0]


            ## checking if tenors is empty. if it is, the bond has matured, and is equal to the face value
            if len(tenors) == 0:
                price_array = np.append(price_array, self.face_value)
                continue

            #getting remaining coupons
            coupons = self.cashflows[-len(tenors):]

            # getting spot rates
            ## If there is an error in finding the discount rates, we use the spot curve of the previous day.
            j = i
            s_rates = 0
            while True:
                try:
                    s_rates = self.interpolated_spot_curve(S= S.iloc[j,:], T= tenors)
                except ValueError:
                    j-= 1
                else:
                    break
                if abs(j - i) > 30:
                    j= i
                    break
            
            ## If no spot rate can be found, we go in the reverse direction
            while True:
                try:
                    s_rates = self.interpolated_spot_curve(S= S.iloc[j,:], T= tenors)
                except ValueError:
                    j+=1
                except IndexError:
                    print("Interpolated curves cannot support discount rates")
                    break
                else:
                    break
            
            #setting empty discount factors array
            discount_factors = np.array([])

            if len(s_rates)!= len(coupons):
                print(s_rates)

            #calculating the discount factors
            for k in range(len(tenors)):
                disc_factor = (1+s_rates[k]/2)**(-2*tenors[k])
                discount_factors = np.append(discount_factors, disc_factor)

            #dotting the coupon vector with the discount factors vector to get the new price
            price_array = np.append(price_array, coupons @ discount_factors)

        price_series = pd.Series(price_array, index = S.index)
        return price_series


class CouponBond_DF(CouponBond):
    """Creating a class that takes in a Dataframe, and choice of CouponBond that inherits from the CouponBond class
    
    Paramters:
        maturity_years : float or int
        df: pandas dataframe containing par yield curves
        face_value: int. assumes 100 face
        frequency: float. compounding frequency. assume semi-annual compounding
    
    """
    def __init__(self, maturity_years, df: pd.DataFrame, face_value_ = 100, 
                 frequency_=2):
        
        # collecting data from the dataframe
        ytm_ = df.iloc[0,:][maturity_years]
        i_date = df.index[0]
        self.par_curve = df
        super().__init__(ytm_, maturity_years, face_value_, frequency_, i_date)
    
    def new_price(self):
        """
        Calculate the daily price of the bond using the DataFrame passed during initialization.
        
        Uses the `new_price` method from the parent class but automatically
        uses `self.df` as the DataFrame input.
        
        Returns:
            pd.Series: A Series of bond prices indexed by dates from the DataFrame.
        """
        return super().new_price(self.par_curve)
    
    def open_position(self):
        return self.price

    def close_position(self):
        """A method to get the full closing position of the bond.
            Accounts for the coupon payment during the period.
            TAKES ABOUT A MINUTE TO RUN since it calls Treasury.new_price()
        """
        position = self.new_price().iloc[-1,:][self.maturity_years]
        if self.maturity_years != .5:
            position += self.cashflows[0]
        
        return position 
    
    def price_plus_coupon(self):
        """Calculates price + coupon payment. 
        
        Note: this is only specified for the analysis period. Not for a general coupon bond
        """        

        prices = self.new_price()

        if self.maturity_years == .5:
            return prices

        coupon_date = self.cash_flow_dates[0]
        coupon_payment = self.cashflows[0]

        # a boolean array of when we hold the coupon payment
        date_bool = (prices.index >= coupon_date).astype(int)

        coupon_series = date_bool *coupon_payment

        return coupon_series+ prices 