import data
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress


def option_one():
    print('Statsmodels solution')
    x = data.syntetic.x1.values
    x = sm.add_constant(x, prepend=True)
    y = data.syntetic.y.values
    mod = sm.OLS(y,x)
    res = mod.fit()
    print(res.summary())

def option_two():
    print('Sklearn solution')
    x = data.syntetic.x1.values.reshape((-1, 1))
    y = data.syntetic.y.values
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

def option_three():
    print('Scipy solution')
    x = data.syntetic.x1.values
    y = data.syntetic.y.values
    result = linregress(x, y)
    print(result.slope, result.stderr)
    print(result.intercept, result.intercept_stderr)

option_one()
print('')
option_two()
print('')
option_three()