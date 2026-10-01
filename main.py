import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r"placement.csv")

x = df.iloc[:, 0:1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

print(x_test)
print(y_test)

prediction = lr.predict(x_test.iloc[2].values.reshape(1, 1))
print(prediction)

m = lr.coef_
b = lr.intercept_

prediction_for_3 = m * 3 + b
print(prediction_for_3)

plt.scatter(df['cgpa'], df['package'])
plt.plot(x_train, lr.predict(x_train), color='red')

plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')
plt.show()
