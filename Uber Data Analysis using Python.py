#!/usr/bin/env python
# coding: utf-8

# # Uber Data Analysis using Python

# ### Import the Essential Packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# ### Import Dataset
# 

# In[3]:


apr = pd.read_csv("uber-raw-data-apr14.csv")
may = pd.read_csv("uber-raw-data-may14.csv")
jun = pd.read_csv("uber-raw-data-jun14.csv")
jul = pd.read_csv("uber-raw-data-jul14.csv")
aug = pd.read_csv("uber-raw-data-aug14.csv")
sep = pd.read_csv("uber-raw-data-sep14.csv")


# ### Merge Dataset

# In[4]:


df = pd.concat([apr, may, jun, jul, aug, sep], ignore_index = True)

df


# ### Check Missing Values

# In[5]:


df.isnull().values.any()


# ### Data Structure

# In[6]:


df.info()


# ### Change the Format/Type of Date/Time to datetime

# In[7]:


df["Date/Time"] = pd.to_datetime(df['Date/Time'])

df.dtypes


# ### Create Time Objects 

# In[8]:


df["Time"] = df["Date/Time"].dt.time
df["Hour"] = df["Date/Time"].dt.hour
df["Date"] = df["Date/Time"].dt.date
df["Year"] = df["Date/Time"].dt.year
df["Month"] = df["Date/Time"].dt.month_name() #df["Month"] = df["Date/Time"].dt.month
df["Day"] = df["Date/Time"].dt.day
df["Wday"] = df['Date/Time'].dt.day_name() # df["Wday"] = df["Date/Time"].dt.weekday+1
df


# ### Change the Order of Factors

# In[9]:


df["Wday"] = df["Wday"].astype("category")
df["Wday"] = df["Wday"].cat.reorder_categories(["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])

df["Month"] = df["Month"].astype("category")
df["Month"] = df["Month"].cat.reorder_categories(["April","May","June","July","August","September"])

df.dtypes


# ### Example of Calendar Package

# In[10]:


import calendar
list(calendar.day_name)


# In[11]:


list(calendar.day_abbr)


# In[12]:


calendar.day_abbr[0]


# ### Number of Trips in a Day

# In[13]:


df.Hour.value_counts()


# In[14]:


df.groupby(["Hour"], as_index = False)["Base"].count()


# In[15]:


plt.figure(figsize = (12,8))
# sns.barplot(x = "Hour", y = "Base", data = df.groupby(["Hour"], as_index = False)["Base"].count(), ci = None)
sns.countplot(data = df, x = "Hour")
plt.xlabel('Hour',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips in a Day", y = 1.00, x = 0.5, size = 16)
plt.show()


# In the resulting visualizations, we can understand how the number of passengers fares throughout the day. We observe that the number of trips are higher in the evening around 5:00 and 6:00 PM.

# ### Number of Trips during Every Day of the Month

# In[16]:


df.Day.value_counts()


# In[17]:


df.groupby(["Day"], as_index = False)["Base"].count()


# In[18]:


plt.figure(figsize = (12,8))
#sns.barplot(x = "Day", y = "Base", data = df.groupby(["Day"], as_index = False)["Base"].count(), ci = None)
sns.countplot(data = df, x = "Day")
plt.xlabel('Every Day of Month',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips during Every Day of the Month", y = 1.00, x = 0.5, size = 16)
plt.show()


# We observe from the resulting visualization that 30th of the month had the highest trips in the year.

# ### Number of Trips on Every Day of the Week

# In[19]:


df.Wday.value_counts()


# In[20]:


plt.figure(figsize = (12,8))
#sns.barplot(x = "Wday", y = "Base", data = df.groupby(["Wday"], as_index = False)["Base"].count(), ci = None)
sns.countplot(data = df, x = "Wday")
plt.xlabel('Day of Week',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips on Every Day of the Week", y = 1.00, x = 0.5, size = 16)
plt.show()


# In the output visualization, we observe that most trips were made on Thursday and Friday during the week. Potentially, people like to have meals in the restaurant on the weekend.

# ### Number of Trips on Each Month

# In[21]:


df.Month.value_counts()


# In[22]:


df.groupby(["Month"], as_index = False)["Base"].count()


# In[23]:


plt.figure(figsize = (12,8))
#sns.barplot(x = "Month", y = "Base", data = df.groupby(["Month"], as_index = False)["Base"].count(), ci = None)
sns.countplot(data = df, x = "Month")
plt.xlabel('Month',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips in Each Month", y = 1.00, x = 0.5, size = 16)
plt.ticklabel_format(axis = "y", style = "plain")
plt.show()


# In the output visualization, we observe that most trips were made during the month of September. 

# ### Number of Trips on Day of Week during Months

# In[24]:


week_month = df.groupby(["Month","Wday"], as_index = False)["Base"].count()

week_month


# In[25]:


#sns.catplot(x = "Month", y = "Base", hue = "Wday", data = week_month, kind = "bar", height = 8, aspect = 1.5)
plt.figure(figsize = (12,8))
sns.countplot(x = "Month", data = df, hue = "Wday")
plt.xlabel('Month',size = 16)
plt.ylabel('Total',size = 16)
plt.title("Trips on Day of Week during Months", y = 1.00, x = 0.5, size = 16)
plt.show()


# In[26]:


week_month = week_month.pivot_table(index = "Month", columns = "Wday", values = "Base")
week_month


# In[27]:


# plot the pivoted dataframe

week_month.plot.bar(stacked = True, figsize = (12, 8))   # Pandas Plot Function
plt.legend(title = 'Day of Week', bbox_to_anchor = (1.05, 1), loc = 'upper left')
plt.ticklabel_format(axis = "y", style = "plain")
plt.xlabel("Month", size = 16)
plt.ylabel("Trips", size = 16)
plt.title("Trips on Day of Week during Months", y = 1.00, x = 0.5, size = 16)
plt.xticks(rotation = 30, horizontalalignment = "center")
plt.show()


# ### Number of Trips for Each Base

# In[28]:


df.groupby(["Base"], as_index = False)["Year"].count()


# In[29]:


plt.figure(figsize = (12,8))
#sns.barplot(x = "Base", y = "Year", data = df.groupby(["Base"], as_index = False)["Year"].count(), ci = None)
sns.countplot(x = "Base", data = df)
plt.xlabel('Base',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips for Each Base", y = 1.00, x = 0.5, size = 16)
plt.ticklabel_format(axis = "y", style = "plain")
plt.show()


# In this visualization, we plot the number of trips that have been taken by the passengers from each of the bases. There are five bases in all out of which, we observe that B02617 had the highest number of trips.

# ### Number of Trips for Each Base during Every Month

# In[30]:


df.groupby(["Month","Base"], as_index = False)["Year"].count()


# In[31]:


#sns.catplot(x = "Base", y = "Year", hue = "Month", data = df.groupby(["Month","Base"], as_index = False)["Year"].count(), kind = "bar", height = 8, aspect = 1.5)
plt.figure(figsize = (12,8))
sns.countplot(x = "Base", data = df, hue = "Month")
plt.xlabel('Base',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips for Each Base during Every Month", y = 1.00, x = 0.5, size = 16)
plt.show()


# In[32]:


plt.figure(figsize = (12,8))
sns.countplot(x = "Month", data = df, hue = "Base")
plt.xlabel('Month',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips for Each Base during Every Month", y = 1.00, x = 0.5, size = 16)
plt.show()


# ### Number of Trips for Each Base during Day of Week Every Month

# In[35]:


df.groupby(["Wday","Base"], as_index = False)["Year"].count()


# In[33]:


#sns.catplot(x = "Base", y = "Year", hue = "Wday", data = df.groupby(["Wday","Base"], as_index = False)["Year"].count(), kind = "bar", height = 8, aspect = 1.5)
plt.figure(figsize = (12,8))
sns.countplot(x = "Base", data = df, hue = "Wday")
plt.xlabel('Base',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips for Each Base during Day of Week Every Month", y = 1.00, x = 0.5, size = 16)
plt.show()


# In[34]:


plt.figure(figsize = (12,8))
sns.countplot(x = "Wday", data = df, hue = "Base")
plt.xlabel('Day of Week',size = 16)
plt.ylabel('Trips',size = 16)
plt.title("Trips for Each Base during Day of Week Every Month", y = 1.00, x = 0.5, size = 16)
plt.show()


# Furthermore, B02617 base had the highest number of trips in the month September. Thursday observed highest trips in the three bases – B02598, B02617, B02682.

# ## Heatmap

# In[36]:


# The matplotlib basemap toolkit is a library for plotting 2D data on maps in Python

from mpl_toolkits.basemap import Basemap
from matplotlib import cm # Colormap


# ### Heatmap by Hour and Day

# In[37]:


hour_day = df.groupby(["Hour","Day"], as_index = False)["Base"].count().pivot_table(index = "Hour", columns = "Day", values = "Base")

hour_day 


# In[38]:


plt.figure(figsize = (12,8))
sns.heatmap(hour_day, cmap = cm.YlGnBu, linewidth = .5) # Using the Seaborn Heatmap Function 
plt.xlabel('Day',size = 16)
plt.ylabel('Hour',size = 16)
plt.title("Trips by Hour and Day", y = 1.00, x = 0.5, size = 16)
plt.show()


# We see that the number of trips in increasing throughout the day, with a peak demand in the evening between 16:00 and 18:00.
# 
# It corresponds to the time where employees finish their work and go home.

# ### Heatmap by Hour and Weekday

# In[39]:


hour_wday = df.groupby(["Hour","Wday"], as_index = False)["Base"].count().pivot_table(index = "Hour", columns = "Wday", values = "Base")

hour_wday 


# In[40]:


plt.figure(figsize = (12,8))
sns.heatmap(hour_wday, cmap = cm.YlGnBu, linewidth = .5) 
plt.xlabel('Day of Week',size = 16)
plt.ylabel('Hour',size = 16)
plt.title("Trips by Hour and Day of Week", y = 1.00, x = 0.5, size = 16)
plt.show()


# We can see that on working days (From Monday to Friday) the number of trips is higher from 16:00 to 21:00. It shows even better what we said from the first heatmap.
# 
# On Friday the number of trips remains high until 23:00 and continues on early Saturday. It corresponds to the time where people come out from work, then go out for dinner or drink before the weekend.
# 
# We can notice the same pattern on Saturday, people tend to go out at night, the number of trips remains on high until early Sunday.

# ### Heatmap by Day and Month

# In[41]:


day_month = df.groupby(["Day","Month"], as_index = False)["Base"].count().pivot_table(index = "Day", columns = "Month", values = "Base")

day_month


# In[42]:


plt.figure(figsize = (12,8))
sns.heatmap(day_month, cmap = cm.YlGnBu, linewidth = .5) 
plt.xlabel('Month',size = 16)
plt.ylabel('Day',size = 16)
plt.title("Trips by Day and Month", y = 1.00, x = 0.5, size = 16)
plt.show()


# We observe that the number of trips increases each month, we can say that from April to September 2014, Uber was in a continuous improvement process.
# 
# We can notice from the visualization a dark spot, it corresponds to the 30 April. The number of trips that day was extreme compared to the rest of the month.
# 
# Unfortunatly we have not been able to find any factual information to explain the pulse. A successful marketing strategy can be assumed to be in place that days. So as the analysis go on we consider that day an outliner.

# ### Heatmap by Month and Day of Week

# In[43]:


wday_month = df.groupby(["Wday","Month"], as_index = False)["Base"].count().pivot_table(index = "Month", columns = "Wday", values = "Base")

wday_month


# In[44]:


plt.figure(figsize = (12,8))
sns.heatmap(wday_month, cmap = cm.YlGnBu, linewidth = .5) 
plt.xlabel('Day of Week',size = 16)
plt.ylabel('Month',size = 16)
plt.title("Trips by Month and Day of Week", y = 1.00, x = 0.5, size = 16)
plt.show()


# ## Spatial Visualization

# ### Scatter visualization
# Reduce the need in computational power by dropping the duplicates in Latitude and Longitude

# In[45]:


# Setting up the limits

top, bottom, left, right = 41, 40.55, -74.3, -73.6


# In[46]:


df_reduced = df.drop_duplicates(['Lat','Lon'])


# In[47]:


plt.figure(figsize = (16, 12))
plt.ylim(top = top, bottom = bottom)
plt.xlim(left = left, right = right)

# Extracting the Longitude and Latitude of each pickup in our reduced dataset
plt.plot(df_reduced['Lon'], df_reduced['Lat'], '.', ms = 0.8, alpha = 0.5) 

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('New York Uber Pickups from April to September 2014')
plt.show()


# ### Heatmap Visualization
# This visualization is more demanding in computational power, since we can't use the reduce dataset if we want to get the number of pickups in the heatmap. We will use Basemap to create the spacial heatmap.

# In[48]:


#Extracting the Longitude and Latitude of each pickup in our dataset

Longitudes = df['Lon'].values
Latitudes  = df['Lat'].values

plt.figure(figsize = (18, 14))
plt.title('New York Uber Pickups from April to September 2014')

# https://matplotlib.org/basemap/api/basemap_api.html
map = Basemap(projection = 'merc', urcrnrlat = top, llcrnrlat = bottom, llcrnrlon = left, urcrnrlon = right)
x, y = map(Longitudes, Latitudes)
map.hexbin(x, y, gridsize = 1000, bins = 'log', cmap = cm.YlOrRd)
map.colorbar(location = 'right', format = '%.1f', label = 'Number of Pickups')


# From our spacial visualization we observe that:
# - Most of Uber's trips in New York are made from Midtown to Lower Manhattan.
# - Followed by Upper Manhattan and the Heights of Brooklyn.
# - Lastly Jersey City and the rest of Brooklyn.
# 
# We see some brighter spots in our heatmap, corresponding to :
# - LaGuardia Airport in East Elmhurst.
# - John F. Kennedy International Airport.
# - Newark Liberty International Airport.
# 
# We know that many airports have specific requirements about where customers can be picked up by vehicles on the Uber platform. We can assume that these three airports have them, since they represent a big part of uber's business in new york.
# 

# In[ ]:




