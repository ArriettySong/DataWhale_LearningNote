from sklearn import datasets
from pygam import LinearGAM
import pandas as pd
boston = datasets.load_boston();
X=boston.data
y=boston.target
features=boston.feature_names
boston_data = pd.DataFrame(X,columns=features)
gam = LinearGAM().fit(boston_data[boston.feature_names], y)
X_test_res = gam.predict(X)
print(X_test_res[:5])


