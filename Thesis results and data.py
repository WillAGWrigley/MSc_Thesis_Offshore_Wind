#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install statsmodel


# In[2]:


conda install -c conda-forge statsmodels


# In[ ]:





# In[69]:


import statsmodels.api as sm

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


# In[70]:


import seaborn as sns  # Import the seaborn library
import statsmodels.api as sm


# In[ ]:





# In[ ]:





# In[76]:


# Load your DataFrame (replace 'data.csv' with your actual file)
Offshore_Wind = pd.read_csv('/Users/will/Desktop/Thesis Data/Final ENG Data/CSV_UK_OSW_Final.csv')


# In[77]:


Offshore_Wind.head()


# In[78]:


num_rows_1 = Offshore_Wind.shape[0]

print("Number of rows:", num_rows_1)


# In[86]:


Offshore_Wind.dtypes


# In[87]:


# Create a new DataFrame before dropna
Offshore_Wind_new_1 = Offshore_Wind.copy()

# Remove rows with 'N/A' or NaN values from the new DataFrame
columns_to_check = ['Distance from Shore (km)', 'Area', 'Capacity_MW_x', 'mcz_proximity']
Offshore_Wind_new_1 = Offshore_Wind_new_1.dropna(subset=columns_to_check, how='any', inplace=False)

# Convert selected columns to float in the new DataFrame
columns_to_convert = ['Distance from Shore (km)', 'Area', 'Capacity_MW_x']
for column in columns_to_convert:
    Offshore_Wind_new_1[column] = Offshore_Wind_new_1[column].apply(lambda x: x.replace('.', '', 1) if isinstance(x, str) else x)
    Offshore_Wind_new_1[column] = Offshore_Wind_new_1[column].astype(float)


# In[88]:


Offshore_Wind_new_1.dtypes


# In[ ]:





# In[89]:


num_rows_1_test = Offshore_Wind_new_1.shape[0]

print("Number of rows:", num_rows_1_test)


# In[ ]:





# In[ ]:





# In[31]:


# dropping NaN values from the dataframe 
#potentially obsolete
#columns_to_check = ['Wind Speed at 150M', 'Distance from Shore (km)', 'mcz_proximity']  # Replace with the column names you want to check

#OSW_Clean_MR_1 = Offshore_Wind.dropna(subset=columns_to_check).copy()


# In[36]:





# In[90]:


Offshore_Wind_new_1.dtypes


# In[39]:





# In[ ]:





# In[ ]:





# In[91]:


# Define the predictor variables (independent variables)
Independent_VAR = ['Wind Speed at 150M', 'Distance from Shore (km)', 'Capacity_MW_x']


# In[92]:


# Add a constant term to the predictor variables
X = sm.add_constant(Offshore_Wind_new_1[Independent_VAR])


# In[93]:


# Define the dependent variable
y = Offshore_Wind_new_1['mcz_proximity'] 


# In[94]:


test_model = sm.OLS(y, X).fit()


# In[95]:


print(test_model.summary())


# In[ ]:





# In[97]:


# Set up the scatter plots with histograms
g = sns.PairGrid(
    Offshore_Wind_new_1,
    y_vars=['mcz_proximity'],
    x_vars=['Wind Speed at 150M', 'Distance from Shore (km)', 'Capacity_MW_x']
)

# Map regression plots to the scatter plot axes with line of best fit (blue line)
g.map(sns.regplot, scatter_kws={'color': 'blue', 'label': 'Data'})

# Map histograms to the diagonal axes
g.map_diag(plt.hist, color='gray', bins=15)

plt.show()


# In[99]:


from sklearn.preprocessing import PolynomialFeatures  # Import PolynomialFeatures


# In[100]:


# Polynomial regression with multiple predictors (X1, X2, X3)
X_P = Offshore_Wind_new_1[['Wind Speed at 150M', 'Distance from Shore (km)', 'Capacity_MW_x']]
X_P = sm.add_constant(X_P)  # Add a constant (intercept) term
y_P = Offshore_Wind_new_1['mcz_proximity']

# Define the degree of the polynomial
degree = 2  # Adjust the degree as needed

# Create polynomial features
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X_P)

# Fit the multiple polynomial regression model
model = sm.OLS(y_P, X_poly).fit()

# Get the regression summary
print(model.summary())


# In[ ]:





# In[102]:


# Generate scatter plots with regression lines
variables_to_plot = ['Wind Speed at 150M', 'Distance from Shore (km)', 'Capacity_MW_x']
for variable in variables_to_plot:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Offshore_Wind_new_1[variable], y=Offshore_Wind_new_1['mcz_proximity'], label='Data points')
    sns.regplot(x=Offshore_Wind_new_1[variable], y=model.fittedvalues, scatter=False, label='Regression line', color='red')
    plt.xlabel(variable)
    plt.ylabel('Y')
    plt.title(f'Multiple Polynomial Regression with {variable}')
    plt.legend()
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


sm.tools.web.webdoc(func=None, stable=None)


# In[48]:


# New Data-- trying normal linear regression:

X_1 = OSW_Clean_MR_1['Wind Speed at 150M']
y_1 = OSW_Clean_MR_1['mcz_proximity']

# Add a constant term to the independent variable
X_with_const_1 = sm.add_constant(X_1)


# In[50]:


# Create a linear regression model
model_1 = sm.OLS(y_1, X_with_const_1).fit()


# In[51]:


print(model_1.summary())


# In[57]:


# Get the regression line parameters
intercept_1 = model_1.params[0]
slope_1 = model_1.params[1]

# Scatter plot with regression line
plt.scatter(X_1, y_1, label='Data points')
plt.plot(X_1, slope_1 * X_1 + intercept_1, color='red', label='Regression line')
plt.xlabel('Wind_Speed')
plt.ylabel('Proximity to Marine Protected Zones')
plt.title('Regression - Scatter MPZ Proximity vs Wind Speed')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




