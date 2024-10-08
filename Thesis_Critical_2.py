#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Important Numpy, Pandas, Geopandas & Matplotlib
#
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# In[247]:


# England, Ireland and Wales Offshore Wind farm locations - this is from Crown Estate Open Data Portal
# https://opendata-thecrownestate.opendata.arcgis.com/

DF_Crown_Estate_CSV = pd.read_csv('/Users/will/Desktop/Thesis Data/Wind_Site_Agreements_(England%2C_Wales_%26_NI)%2C_The_Crown_Estate.csv')

print(DF_Crown_Estate_CSV)


# In[248]:


# Using the Head function to view data - toggle on/off the hastag


DF_Crown_Estate_CSV.head()


# In[245]:


# England, Wales Ireland -- offshore windfarm location shapefiles -- Crown Estate Open Data Portal:
# https://opendata-thecrownestate.opendata.arcgis.com/
# Using Geopandas library in order to process the shapefile data

Offshore_shapefile_locations = gpd.read_file('/Users/will/Desktop/Thesis Data/Wind_Site_Agreements_(England%2C_Wales_%26_NI)%2C_The_Crown_Estate/WindSiteAgreements_EnglandWalesAndNI_TheCrownEstate.shp')


# In[246]:


Offshore_shapefile_locations.head()


# In[137]:


# Using the plot functionality to view the Offshore wind farm location shapefiles
# Even from the initial viewing, visually you can see that the North Sea is the predomoninant location for North Sea
# offshore wind farm

Offshore_shapefile_locations.plot()
plt.show()


# In[ ]:





# In[249]:


# Import CSV file with Windspeed, Proximity to shore, Area, Capacity data for England, Ireland and Wales
# https://globalwindatlas.info/en
EWI_Wind_Speed_Data = pd.read_csv('/Users/will/Desktop/Thesis Data/_Final Wind Speed Data - England, Wales & Irish Sea - Sheet1.csv')

print(EWI_Wind_Speed_Data)


# In[250]:


EWI_Wind_Speed_Data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[251]:


# Scotland Offshore Wind -- CSV

# Need to import this


# In[252]:


SCT_OFFSHORE_LOCATIONS_CSV = pd.read_csv('/Users/will/Desktop/Thesis Data/Offshore_Wind_(Crown_Estate_Scotland).csv')

print(SCT_OFFSHORE_LOCATIONS_CSV)


# In[253]:


# Scotland Offshore Wind Shapefile -- problem with this data is that is has information regarding substations and other infrastructure thus need to clean data


# In[254]:


Scotland_offshore_shapefile = gpd.read_file('/Users/will/Desktop/Thesis Data/Offshore_Wind_(Crown_Estate_Scotland)/Offshore_Wind__Crown_Estate_Scotland_.shp')

print(Scotland_offshore_shapefile)


# In[255]:


Scotland_offshore_shapefile.head()


# In[256]:


Scotland_offshore_shapefile.plot()
plt.show()


# In[257]:


# Scottish Windspeed CSV Data


# In[258]:


SCT_WINDSPEED = pd.read_csv('/Users/will/Desktop/Thesis Data/Wind Speed Data Scotland - Sheet1.csv')

print(SCT_WINDSPEED)


# In[ ]:





# In[262]:


# Data Cleaning

# Removing unneeded columns, renaming columns & then merging data into one dataframe in preparation for analysis


# In[263]:


DF_Crown_Estate_CSV.head()


# In[264]:


SCT_OFFSHORE_LOCATIONS_CSV.head()


# In[265]:


# renaming columns in the scottish dataframe to align with those in the England, Wales and Scotland dataframe

SCT_OFFSHORE_LOCATIONS_CSV.rename(columns={'Lease_Description': 'Name_Prop'}, inplace=True)
SCT_OFFSHORE_LOCATIONS_CSV.rename(columns={'Tenant_Name': 'Name_Ten'}, inplace=True)
SCT_OFFSHORE_LOCATIONS_CSV.rename(columns={'Lease_Type_Description': 'Lease_Stat'}, inplace=True)
SCT_OFFSHORE_LOCATIONS_CSV.rename(columns={'Project_Phase': 'Inf_Status'}, inplace=True)




SCT_OFFSHORE_LOCATIONS_CSV.head()


# In[266]:


# Removing columns using the drop() method

columns_to_remove_1 = ['Property_Classification']
SCT_OFFSHORE_LOCATIONS_CSV.drop(columns=columns_to_remove_1, inplace=True)


SCT_OFFSHORE_LOCATIONS_CSV.head()


# In[267]:


DF_Crown_Estate_CSV.head()


# In[268]:


# Removing columns using the drop() method

columns_to_remove_2 = ['Wind_Round', 'km2', 'ODP_Hyperlink']
DF_Crown_Estate_CSV.drop(columns=columns_to_remove_2, inplace=True)


# In[270]:


DF_Crown_Estate_CSV.head()


# In[271]:


SCT_OFFSHORE_LOCATIONS_CSV.head()


# In[275]:


num_rows_1 = DF_Crown_Estate_CSV.shape[0]

print("Number of rows:", num_rows_1)


# In[276]:


num_rows_2 = SCT_OFFSHORE_LOCATIONS_CSV.shape[0]

print ("Number of rows:", num_rows_2)


# In[277]:


# Using the concat function in order to merge the dataframes 

UK_OFFSHORE_COMB = pd.concat([DF_Crown_Estate_CSV, SCT_OFFSHORE_LOCATIONS_CSV], ignore_index=True)

# Print the combined DataFrame
print(UK_OFFSHORE_COMB)


# In[278]:


# to create a universal ID that covers both Scotland & the England, Wales Ireland file the index is used

UK_OFFSHORE_COMB['New_Object_ID'] = UK_OFFSHORE_COMB.index + 1

# Print the combined DataFrame
print(UK_OFFSHORE_COMB)


# In[279]:


columns_to_remove_3 = ['OBJECTID']
UK_OFFSHORE_COMB.drop(columns=columns_to_remove_3, inplace=True)


# In[280]:


UK_OFFSHORE_COMB.head()


# In[281]:


# Now with Shapefiles -- :) 


# In[282]:


Offshore_shapefile_locations.head()


# In[283]:


Scotland_offshore_shapefile.head()


# In[ ]:





# In[284]:


columns_to_remove_4 = ['Lease_Desc', 'Tenant_Nam', 'Lease_Type', 'Property_C', 'Project_Ph', 'Capacity_M']
Scotland_offshore_shapefile.drop(columns=columns_to_remove_4, inplace=True)

Scotland_offshore_shapefile.head()


# In[ ]:





# In[285]:


columns_to_remove_5 = ['Name_Prop', 'Name_Ten', 'Wind_Round', 'Lease_Stat', 'Inf_Status', 'km2', 'ODP_Hyperl']
Offshore_shapefile_locations.drop(columns=columns_to_remove_5, inplace=True)

Offshore_shapefile_locations.head()


# In[287]:


# by using the concat function in pandas, the shp file is transformed into a df - later this will be returned to shapefiles

UK_OFFSHORE_SHP_COMB = pd.concat([Offshore_shapefile_locations, Scotland_offshore_shapefile], ignore_index=True)

# Print the combined DataFrame
print(UK_OFFSHORE_SHP_COMB)


# In[288]:


UK_OFFSHORE_SHP_COMB['New_Object_ID'] = UK_OFFSHORE_SHP_COMB.index + 1

# Print the combined DataFrame
print(UK_OFFSHORE_SHP_COMB)


# In[184]:


# Now with Windspeed -- :) 

# 


# In[289]:


SCT_WINDSPEED.head()


# In[290]:


EWI_Wind_Speed_Data.head()


# In[291]:


# Removing uneeded collumns

columns_to_remove_6 = ['OBJECTID']
SCT_WINDSPEED.drop(columns=columns_to_remove_6, inplace=True)

SCT_WINDSPEED.head()


# In[292]:


# renaming columns to be suitable for the merge 

SCT_WINDSPEED.rename(columns={'Lease_Description': 'Name_Prop'}, inplace=True)

SCT_WINDSPEED.head()


# In[293]:


# renaming misspelt collumn name

SCT_WINDSPEED.rename(columns={'Commisioning': 'Commissioning'}, inplace=True)

SCT_WINDSPEED.head()


# In[294]:


WIND_SPEED_COMB = pd.concat([EWI_Wind_Speed_Data, SCT_WINDSPEED], ignore_index=True)

# Print the combined DataFrame
print(WIND_SPEED_COMB)


# In[295]:


WIND_SPEED_COMB.head()


# In[296]:


WIND_SPEED_COMB['New_Object_ID'] = WIND_SPEED_COMB.index + 1

# Print the combined DataFrame
print(WIND_SPEED_COMB)


# In[ ]:





# In[299]:


WIND_SPEED_COMB['Capacity_MW'].fillna(UK_OFFSHORE_COMB['Capacity_MW'], inplace=True)
UK_OFFSHORE_COMB['Capacity_MW'].fillna(WIND_SPEED_COMB['Capacity_MW'], inplace=True)


# In[300]:


print(WIND_SPEED_COMB)


# In[301]:


print(UK_OFFSHORE_COMB)


# In[ ]:





# In[ ]:


# This next step will involve merging the 


# In[ ]:


# WIND_SPEED_COMB


# In[ ]:


# UK_OFFSHORE_SHP_COMB


# In[297]:


# UK_OFFSHORE_COMB


# In[346]:


# Merge the DataFrames based on the 'ID' column
Merged_DF = pd.merge(UK_OFFSHORE_COMB, UK_OFFSHORE_SHP_COMB, on='New_Object_ID', how='inner')  

Final_Merged_DF = pd.merge(Merged_DF, WIND_SPEED_COMB, on='New_Object_ID', how='inner')

print(Final_Merged_DF)


# In[347]:


Final_Merged_DF.head()


# In[ ]:





# In[348]:


columns_to_remove_7 = ['OBJECTID', 'SHAPE_Leng', 'SHAPE_Area_y', 'Name_Prop_y', 'Capacity_MW_y']
Final_Merged_DF.drop(columns=columns_to_remove_7, inplace=True)

Final_Merged_DF.head()


# In[349]:


print(Final_Merged_DF)


# In[350]:


# Rearrange the order of columns
new_order_col = ['New_Object_ID', 'Name_Prop_x', 'Name_Ten', 'Lease_Stat', 'Inf_Status', 'Wind Speed at 100M', 'Wind Speed at 150M', 'Region', 'Commissioning', 'Foundation Type', 'Sea Name', 'Area', 'Capacity_MW_x', 'Distance from Shore (km)', 'Grid Connection Point', 'SHAPE_Length', 'SHAPE_Area_x', 'geometry']
Final_Merged_DF = Final_Merged_DF[new_order_col]

print(Final_Merged_DF)


# In[351]:


# Remove rows with NaN values in column windspeed column because they are values from non windfarms in Scotland Data
Offshore_Clean_DF = Final_Merged_DF.dropna(subset=['Wind Speed at 100M'])

print(Offshore_Clean_DF)


# In[ ]:





# In[352]:


Offshore_Clean_DF_2 = Offshore_Clean_DF.dropna(subset=['Foundation Type'])

print(Offshore_Clean_DF_2)


# In[353]:


# Replace all occurrences of differing terminology 
Offshore_Clean_DF_2['Inf_Status'] = Offshore_Clean_DF_2['Inf_Status'].replace('Active/In Operation', 'Opertational')

print(Offshore_Clean_DF_2)


# In[354]:


Offshore_Clean_DF_2['Inf_Status'] = Offshore_Clean_DF_2['Inf_Status'].replace('Opertational', 'Operational')

print(Offshore_Clean_DF_2)


# In[355]:


Offshore_Clean_DF_2['Inf_Status'] = Offshore_Clean_DF_2['Inf_Status'].replace('Pre-planning Application', 'Pre Planning')

print(Offshore_Clean_DF_2)


# In[ ]:





# In[ ]:





# In[356]:


# This data is the MPA data - not as high priority as being able to get functionality from the more granular data


# In[357]:


MPA_Shapes = gpd.read_file('/Users/will/Desktop/Thesis Data/c20230705-OffshoreMPAs-WGS84/c20230705_OffshoreMPAs_WGS84.shp')


# In[358]:


MPA_Shapes


# In[359]:


MPA_Shapes.plot()
plt.show()


# In[360]:


# Convert the DataFrame to a GeoDataFrame
gdf_Offshore_Clean = gpd.GeoDataFrame(Offshore_Clean_DF_2, geometry='geometry')

print(gdf_Offshore_Clean)


# In[361]:


# Calculate the nearest distance for each wind farm from MPA 
wind_farms_with_mpa_proximity_2 = gdf_Offshore_Clean.copy()  # Create a copy to avoid overwriting of the 
# Offshore wind dataframe


# In[362]:


# resampling both GeoDataFrames to an appropriate projected CRS (e.g., UTM)
utm_crs = 'EPSG:32633'  # UTM Zone 33N -- an appropriate CRS for our study
wind_farms_with_mpa_proximity_2 = wind_farms_with_mpa_proximity_2.to_crs(utm_crs)
MPA_Shapes = MPA_Shapes.to_crs(utm_crs)


# In[363]:


# Ensure both GeoDataFrames have the same CRS (print the same if they are)
if wind_farms_with_mpa_proximity_2.crs != MPA_Shapes.crs:
    wind_farms_with_mpa_proximity_2 = wind_farms_with_mpa_proximity_2.to_crs(MPA_Shapes.crs)
print("The GeoDataFrames have the same crs")


# In[364]:


# Calculate the nearest distance for each wind farm
wind_farms_with_mpa_proximity_2['mpa_proximity'] = wind_farms_with_mpa_proximity_2.geometry.apply(
    lambda geom: MPA_Shapes.geometry.distance(geom).min()
)


# In[365]:


# Save the wind farms with distances to a new shapefile
# wind_farms_with_mpa_proximity.to_file('/Users/will/Desktop/Thesis Data/c20230705-OffshoreMPAs-WGS84/Windfarms_with_MPA_Proximity.shp')


# In[366]:


MPA_Proximity_2 = wind_farms_with_mpa_proximity_2.copy()  # Create a copy to avoid overwriting of the 
# Offshore wind dataframe


# In[367]:


MPA_Proximity_2


# In[368]:


pd.set_option('display.max_rows', None)

# Display the DataFrame
print(MPA_Proximity_2)


# In[ ]:





# In[369]:


MCZ_DF = gpd.read_file('/Users/will/Desktop/Thesis Data/NE_MarineConservationZonesEngland_SHP_Full/data/Marine_Conservation_Zones_England.shp')


# In[370]:


MCZ_DF.plot()
plt.show()


# In[371]:


MCZ_Scotland_DF = gpd.read_file('/Users/will/Desktop/Thesis Data/MPA_SCOTLAND_SHP_27700/MPA_SCOTLAND.shp')


# In[372]:


MCZ_Scotland_DF.plot()
plt.show()


# In[373]:


# Merge the two shapefiles
UK_TOTAL_MCZ = gpd.GeoDataFrame(pd.concat([MCZ_DF, MCZ_Scotland_DF], ignore_index=True))

# Save the merged shapefile
UK_TOTAL_MCZ.to_file('/Users/will/Desktop/Thesis Data/merged_shapefile.shp')


# In[374]:


UK_TOTAL_MCZ.plot()
plt.show()


# In[ ]:





# In[375]:


# # Calculate the nearest distance for each wind farm from MPA 
Offshore_MPA_MCZ_Proximity = MPA_Proximity_2.copy()  # Create a copy to avoid overwriting of the 
# Offshore wind dataframe


# In[376]:


# resampling both GeoDataFrames to an appropriate projected CRS (e.g., UTM)
two_utm_crs = 'EPSG:32633'  # UTM Zone 33N -- an appropriate CRS for our study
Offshore_MPA_MCZ_Proximity = Offshore_MPA_MCZ_Proximity.to_crs(two_utm_crs)
UK_TOTAL_MCZ = UK_TOTAL_MCZ.to_crs(two_utm_crs)


# In[377]:


# Ensure both GeoDataFrames have the same CRS (print the same if they are)
if Offshore_MPA_MCZ_Proximity.crs != UK_TOTAL_MCZ.crs:
    Offshore_MPA_MCZ_Proximity = Offshore_MPA_MCZ_Proximity.to_crs(UK_TOTAL_MCZ.crs)
print("The GeoDataFrames have the same crs")


# In[378]:


# Calculate the nearest distance for each wind farm
Offshore_MPA_MCZ_Proximity['mcz_proximity'] = Offshore_MPA_MCZ_Proximity.geometry.apply(
    lambda geom: UK_TOTAL_MCZ.geometry.distance(geom).min()
)


# In[379]:


pd.set_option('display.max_rows', None)

# Display the DataFrame
print(Offshore_MPA_MCZ_Proximity)


# In[ ]:





# In[ ]:





# In[380]:


FINAL_OFFSHORE_gdf = gpd.GeoDataFrame(Offshore_MPA_MCZ_Proximity, geometry='geometry')


# In[381]:


Offshore_MPA_MCZ_Proximity.to_csv('CSV_UK_OSW_Final.csv', index=False) 

print("DataFrame exported to CSV")


# In[382]:


FINAL_OFFSHORE_gdf.to_file('SHP_UK_OSW_Final.shp')  

print("DataFrame exported to shp")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




