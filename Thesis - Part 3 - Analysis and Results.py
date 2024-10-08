#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install statsmodel


# In[2]:


conda install -c conda-forge statsmodels


# In[ ]:





# In[1]:


import statsmodels.api as sm

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


# In[2]:


import seaborn as sns  
import statsmodels.api as sm


# In[ ]:





# In[ ]:





# In[3]:


# Load our DataFrame 
Offshore_Wind = pd.read_csv('/Users/will/Desktop/Thesis Data/Final ENG Data/CSV_UK_OSW_Final.csv')


# In[4]:


Offshore_Wind.head()


# In[5]:


num_rows_1 = Offshore_Wind.shape[0]

print("Number of rows:", num_rows_1)


# In[6]:


Offshore_Wind.dtypes


# In[7]:


# Create a new DataFrame
Offshore_Wind_new_1 = Offshore_Wind.copy()

# Remove rows with 'N/A' or NaN values from the new DataFrame using dropna
columns_to_check = ['Distance from Shore (km)', 'Area', 'Capacity_MW_x', 'mcz_proximity']
Offshore_Wind_new_1 = Offshore_Wind_new_1.dropna(subset=columns_to_check, how='any', inplace=False)

# Convert selected columns to float in the new DataFrame
columns_to_convert = ['Distance from Shore (km)', 'Area', 'Capacity_MW_x']
for column in columns_to_convert:
    Offshore_Wind_new_1[column] = Offshore_Wind_new_1[column].apply(lambda x: x.replace('.', '', 1) if isinstance(x, str) else x)
    Offshore_Wind_new_1[column] = Offshore_Wind_new_1[column].astype(float)


# In[8]:


Offshore_Wind_new_1.dtypes


# In[ ]:





# In[9]:


num_rows_1_test = Offshore_Wind_new_1.shape[0]

print("Number of rows:", num_rows_1_test)


# In[ ]:





# In[ ]:





# In[10]:





# In[ ]:





# In[11]:


Offshore_Wind_new_1.dtypes


# In[ ]:


km_values = Offshore_Wind_new_1["Distance from Shore (km)"].tolist()
print(km_values)


# In[12]:


highest_value = Offshore_Wind_new_1["Distance from Shore (km)"].max()
print(highest_value)


# In[15]:


km_values = Offshore_Wind_new_1["Distance from Shore (km)"].tolist()
print(km_values)


# In[19]:


# Remove rows where the "Name" column has the value '3702.0' - removing a significant outlier
value_to_remove = 3702.0
Offshore_Wind_new_2 = Offshore_Wind_new_1[Offshore_Wind_new_1["Distance from Shore (km)"] != value_to_remove]

print(Offshore_Wind_new_2)


# In[ ]:





# In[20]:


km_values_2 = Offshore_Wind_new_2["Distance from Shore (km)"].tolist()
print(km_values_2)


# In[ ]:





# In[22]:


num_rows_2_test = Offshore_Wind_new_2.shape[0]

print("Number of rows:", num_rows_2_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# having done final data preparation - now discussion of the results

# First Linear regression between Wind Speed and the proximity to MPZ


# In[39]:


X_Wind_speed = Offshore_Wind_new_2['Wind Speed at 150M']  # Independent variable
y_linear_regression = Offshore_Wind_new_2['mcz_proximity']  # Dependent variable


# In[ ]:





# In[ ]:





# In[40]:




# Fit our linear regression model
X_Wind_speed = sm.add_constant(X_Wind_speed)  
model_windspeed = sm.OLS(y_linear_regression, X_Wind_speed).fit()


# In[41]:


print(model_windspeed.summary())


# In[54]:


# coefficients <3
intercept, slope = model_windspeed.params['const'], model_windspeed.params['Wind Speed at 150M']

# Create scatterplot with regression line
plt.scatter(Offshore_Wind_new_2['Wind Speed at 150M'], Offshore_Wind_new_2['mcz_proximity'], label='Data')
plt.plot(Offshore_Wind_new_2['Wind Speed at 150M'], intercept + slope * Offshore_Wind_new_2['Wind Speed at 150M'], color='red', label='Regression Line')
plt.xlabel('Wind Speed at 150M')
plt.ylabel('mcz_proximity')
plt.legend()
plt.title('Linear Regression: Wind Speed vs. Proximity to Marine Protected Zones')
plt.show()


# In[ ]:





# In[43]:


# Linear regression between Capacity and the proximity to MPZ


# In[45]:


X_Capacity = Offshore_Wind_new_2['Capacity_MW_x']  


# In[46]:


# Fit the linear regression model
X_Capacity = sm.add_constant(X_Capacity)  
model_capacity = sm.OLS(y_linear_regression, X_Capacity).fit()


# In[47]:


print(model_capacity.summary())


# In[49]:


#  Get our regression coefficients
intercept, slope = model_capacity.params['const'], model_capacity.params['Capacity_MW_x']

# Create scatterplot with regression line
plt.scatter(Offshore_Wind_new_2['Capacity_MW_x'], Offshore_Wind_new_2['mcz_proximity'], label='Data')
plt.plot(Offshore_Wind_new_2['Capacity_MW_x'], intercept + slope * Offshore_Wind_new_2['Capacity_MW_x'], color='red', label='Regression Line')
plt.xlabel('Capacity_MW_x')
plt.ylabel('mcz_proximity')
plt.legend()
plt.title('Linear Regression: Capacity MW vs. Proximity to Marine Protected Zones')
plt.show()


# In[ ]:





# In[ ]:


# Linear regression between Distance to Shore and the proximity to MPZ


# In[50]:


X_Distance = Offshore_Wind_new_2['Distance from Shore (km)']  # Independent variable


# In[51]:


# Fit the linear regression model
X_Distance = sm.add_constant(X_Distance)
model_distance = sm.OLS(y_linear_regression, X_Distance).fit()


# In[52]:


print(model_distance.summary())


# In[53]:


#  Get our regression coefficients
intercept, slope = model_distance.params['const'], model_distance.params['Distance from Shore (km)']

# Create scatterplot with regression line
plt.scatter(Offshore_Wind_new_2['Distance from Shore (km)'], Offshore_Wind_new_2['mcz_proximity'], label='Data')
plt.plot(Offshore_Wind_new_2['Distance from Shore (km)'], intercept + slope * Offshore_Wind_new_2['Distance from Shore (km)'], color='red', label='Regression Line')
plt.xlabel('Distance from Shore (km)')
plt.ylabel('mcz_proximity')
plt.legend()
plt.title('Linear Regression: Distance From Shore KM vs. Proximity to Marine Protected Zones')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# MULTIPLE LINEAR REGRESSION: 


# In[88]:


# Define the predictor variables (independent variables)
Independent_VAR = ['Wind Speed at 150M', 'Distance from Shore (km)', 'Capacity_MW_x']


# In[89]:


# Add a constant term to the predictor variables
X_ML = sm.add_constant(Offshore_Wind_new_2[Independent_VAR])


# In[90]:


# Define the dependent variable
y_ML = Offshore_Wind_new_2['mcz_proximity'] 


# In[91]:


test_model_final = sm.OLS(y_ML, X_ML).fit()


# In[92]:


print(test_model_final.summary())


# In[61]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[93]:


# Calculate VIF for each independent variable
vif = pd.DataFrame()
vif["Variable"] = X_ML.columns
vif["VIF"] = [variance_inflation_factor(X_ML.values, i) for i in range(X.shape[1])]

# Display the VIF values
print(vif)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Polynomial Regression - 


# In[63]:


from sklearn.preprocessing import PolynomialFeatures  # Import PolynomialFeatures


# In[64]:


# Polynomial regression with multiple predictors 
X_P = Offshore_Wind_new_2[['Wind Speed at 150M', 'Distance from Shore (km)', 'Capacity_MW_x']]
X_P = sm.add_constant(X_P)  # Add a constant
y_P = Offshore_Wind_new_2['mcz_proximity']

# Define the degree of the polynomial
degree = 2  

# Create polynomial features
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X_P)

# Fit the multiple polynomial regression model
model = sm.OLS(y_P, X_poly).fit()

print(model.summary())


# In[65]:


# Calculate VIF for each independent variable
vif_2 = pd.DataFrame()
vif_2["Variable"] = X_P.columns
vif_2["VIF"] = [variance_inflation_factor(X_P.values, i) for i in range(X_P.shape[1])]

# Display the VIF values
print(vif)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




