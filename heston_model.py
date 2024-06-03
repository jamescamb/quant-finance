import QuantLib as ql

evaluation_date = ql.Date(30, 5, 2024)
ql.Settings.instance().evaluationDate = evaluation_date

expiry_date = ql.Date(20, 9, 2024)
strike_price = 190
option_type = ql.Option.Call

spot_price = 191.62
dividend_rate = 0.0053
risk_free_rate = 0.05
volatility = 0.2361

dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(
        evaluation_date, 
        dividend_rate, 
        ql.Actual360()
    )
)
risk_free_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(
        evaluation_date, 
        risk_free_rate, 
        ql.Actual360()
    )
)

v0 = volatility * volatility
kappa = 2.0
theta = volatility * volatility
sigma = 0.1
rho = 0.0

heston_process = ql.HestonProcess(
    risk_free_ts, 
    dividend_ts, 
    ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    ), 
    v0, 
    kappa, 
    theta, 
    sigma, 
    rho
)
heston_model = ql.HestonModel(heston_process)