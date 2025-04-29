import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class CIR_base:
    """
    CIR model class that takes in a series of yields.
    """

    def __init__(self, rates: pd.Series):
        """
        Initialize the CIR model with a series of yields.
        
        Parameters
        ----------
        rates : pd.DataFrame
            A DataFrame containing the yield data.
        """
        self.rates = rates

    def calibrate(self):
        """Calibrate CIR model parameters using OLS regression."""
        self.rates, rates = self.rates/100, self.rates/100

        rt = rates[1:]

        #setting up the parameters for OLS regression
        delta_rt = rates.diff().dropna()
        sqrt_rt = np.sqrt(rt)
        sqrt_delta_t = np.sqrt(pd.Series(rates.index.to_series().diff().dropna().dt.days.values, index=delta_rt.index) / 360 * 12)  # Ensure matching lengths

        #setting up the OLS regression
        y = delta_rt/(sqrt_rt * sqrt_delta_t)
        Y = np.array(y).reshape(-1, 1)

        x1 = sqrt_delta_t/sqrt_rt
        x2 = sqrt_rt*sqrt_delta_t

        X = np.array([x1, x2]).T

        # Fit the linear regression model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, Y)

        # Extract the coefficients
        a, b = model.coef_[0]

        residuals = Y - model.predict(X)

        # Calculate the parameters
        kappa = -b
        theta = a/kappa
        
        self.model = model
        self.theta = theta
        self.kappa = kappa
        self.sigma = residuals.std()

    def simulate(self, N:int = 1, starting_rate:float = None) ->np.ndarray:
        """
        Simulate the CIR process using calibrated parameters.
        
        Parameters
        ----------
        N : int
            Number of simulations to run.
        starting_rate : float
            Starting interest rate for the simulation. If None, uses the first rate in the series.

        Returns
        -------
        np.ndarray
            Simulated interest rates. Each row corresponds to a simulation, and each column corresponds to a time step.
        """
        # Calibrate the CIR process
        self.calibrate()

        if starting_rate is None:
            starting_rate = self.rates.iloc[0]

        # Getting CIR parameters
        theta = self.theta
        kappa = self.kappa
        sigma = self.sigma
        delta_t = 1/360

        rates = np.zeros((N, len(self.rates)))

        for i in range(N):
            # Simulate the process
            # Use the CIR model parameters to generate future rates


            Z = np.random.normal(0, 1, size=(len(self.rates)))

            rates[i, :] = np.zeros(len(self.rates))
            rates[i, 0] = starting_rate

            for t in range(1, len(self.rates)):
                rates[i, t] = rates[i,t-1] + kappa * (theta - rates[i,t-1]) * delta_t + sigma * np.sqrt(delta_t) * np.sqrt(rates[i,t-1]) * Z[t-1]

        return rates
    
class CIR(CIR_base):
    """
    CIR model class that takes in a series of yields.
    """

    def __init__(self, rates: pd.DataFrame):
        """
        Initialize the CIR model with a series of yields.
        
        Parameters
        ----------
        rates : pd.DataFrame
            A DataFrame containing the yield data.
        """

        self.rates = rates/100

        
        self.dataframe = rates.apply(lambda x : CIR_base(x), axis=0)

    def calibrate(self):
        self.dataframe.apply(lambda x : x.calibrate())
    
    def simulate(self, starting_rates:pd.Series|None = None) ->np.ndarray:
        """
        Simulate the CIR process using calibrated parameters.
        
        Parameters
        ----------
        N : int
            Number of simulations to run.
        starting_rate : float
            Starting interest rate for the simulation. If None, uses the first rate in the series.

        Returns
        -------
        np.ndarray
            Simulated interest rates. Each row corresponds to a simulation, and each column corresponds to a time step.
        """
        # Calibrate the CIR process
        if not hasattr(self, 'dataframe'):
            self.calibrate()

        if starting_rates is None:
            simulated_rates = self.dataframe.apply(lambda x : pd.Series(x.simulate()[0])).T
            starting_date = self.rates.index[0]
        else:
            if len(starting_rates) != len(self.dataframe):
                raise ValueError("Starting rates must have the same length as the dataframe.")
            
            starting_date = starting_rates.name
            starting_rates = starting_rates/100

            index = [starting_date + pd.Timedelta(days=i) for i in range(len(self.rates))]

            simulated_rates = pd.DataFrame(columns=self.dataframe.index, index = index)
            
            for i, column in enumerate(starting_rates.index):

                starting_rate = starting_rates[column]
                model = self.dataframe[column]

                # simulated_rates[column] = model.simulate(starting_rate=starting_rate)
                simulated_array = model.simulate(starting_rate=starting_rate)[0]

                simulated_rates[column] = simulated_array
        
        return simulated_rates