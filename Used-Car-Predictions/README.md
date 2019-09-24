# Predicting Used Car Prices in Chicago

This project uses Linear Regression to predict the price of a used car in Chicago. All code found in "Used Car Predictor.ipynb".

## Methodology

### Data Collection

Webscraped 1,000 postings from Autotrader.com. Used Beautiful Soup and multithreaded requests. 

### Data Cleaning

Removed duplicates and outliers (car price > $53,000 was removed). Also converted data to correct data types. 

### Feature Engineering

Look at correlations and pairplots. Derived new features (dummy variables on make of car, Certified x Luxury, Age x Miles). 

### Modeling

Run linear regression models, lasso regression, and ridge regression. See which model did the best and which features were relevant. 

## Insights

* Some luxury cars (Porsche, BMW, Lexus, Mercedes-Benz) positively impacted the price. 
* Some luxury cars (Cadillac, Infiniti, Lincoln) negatively impacted the price. A reason may be unpopularity.
* Four-wheel drive terrains positively impacted the price. Reason is that they are popular during the Chicago snow.
* High MPG Highway negatively impacts the price. A reason is that high MPG Highway are found in many affordable cars.
* Mileage of the car has little to no impact on the price. A surprising finding!

## Conclusions 

Was able to build a linear regression model with an R^2 of 0.87 and an RMSE of $3,465.88. So whatever price my model predicts, it will be off by $3,465.88. 





