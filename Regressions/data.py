import numpy as np
import pandas as pd

syntetic = pd.DataFrame({
    'x1': np.random.uniform(0,10,1000)})
syntetic['y'] = 7 + syntetic.x1 * 0.3 + np.random.normal(0,2,1000)