#!/usr/bin/env python
# coding: utf-8

# In[16]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# In[17]:


df = pd.read_csv('Canada_Immigration.csv')
df.head()


# In[18]:


#set country as index
df.set_index('Country', inplace=True)
df.head()


# In[19]:


#year
year=list(map(str, range(1980, 2014)))


# In[20]:


#data for Denmark, Norway & Sweden
df_t=df.loc[['Denmark', 'Norway', 'Sweden'], year].T
df_t.head()


# In[21]:


#histogram
df_t.plot(kind='hist', figsize=(10,6))
plt.title('Histogram of Immigration from Denmark, Norway, Sweden from 1980-2013')
plt.ylabel('Number of years')
plt.xlabel('Number of Immigrants')
plt.show()


# In[22]:


#get the xtick values
count, bin_edges=np.histogram(df_t, 15)


# In[23]:


#Create unstacked histogram
df_t.plot(kind='hist', figsize=(10,6), alpha = 0.6, xticks=bin_edges, color=['coral', 'darkslateblue', 'mediumseagreen'])
plt.title('Histogram of Immigrants from Denmark, Norway, Sweden from 1980-2013')
plt.xlabel('Number of Immigrants')
plt.ylabel('Number of Years')
plt.show()


# In[24]:


#colors available in matplotlib
import matplotlib
for name,hex in matplotlib.colors.cnames.items():
    print(name,hex)


# In[25]:


count, bin_edges = np.histogram(df_t, 15)
xmin=bin_edges[0]-10
xmax=bin_edges[-1]+10


# In[26]:


df_t.plot(kind='hist', figsize=(10,6),bins =15, xticks=bin_edges, color=['gold','green','navy'], stacked=True)
plt.title('Histogram of Immigrants to Canada from Denmark, Norway, Sweden from 1980-2013')
plt.xlabel('Number of Immigrants')
plt.ylabel('Number of Years')
plt.xlim(xmin, xmax)
plt.show()


# In[27]:


#icelandic immigrants to Canada from 1980-2013
df_ice = df.loc['Iceland',year].T
df_ice.head()


# In[47]:


df_ice.plot(kind='bar', figsize=(10,6), color = 'red', rot=90)
plt.title('Icelandic immigrants to Canada from 1980-2013')
plt.xlabel('Year')
plt.ylabel('Number of immigrants')

#annotation
plt.annotate('',
            xy=(32,70),
            xytext=(28,20),
            xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='green', lw=2)
            )
plt.annotate('2008 - 2011 Financial Crisis',
            xy=(27,30),
            rotation = 72.5,
            va='bottom',
            ha='left')
plt.show()
        


# In[58]:


#top 15 countries contributing immigrants to Canada
df.sort_values(['Total'], axis=0, ascending=True, inplace=True)
df_top15=df.tail(15)
df_top15['Total']


# In[63]:


#horizontal bar graph
df_top15['Total'].plot(kind = 'barh', figsize=(12,12), color='navy')
plt.xlabel('Number of immigrants')
plt.title('Top 15 countries contributing to the immigration to Canada from 1980-2013')

#annotate value label to each country
for index, value in enumerate(df_top15['Total']):
    label=format(int(value), ",")
plt.annotate(label, xy=(value-47000,index-0.10), color='green')
plt.show()


# In[66]:


#to visualize immigration to Canada by continent
df_continent=df.groupby('Continent', axis=0).sum()
df_continent.head()
df_continent['Total']


# In[81]:


color_list=['gold', 'green', 'red', 'orange', 'grey','navy']
explode_list=[0.1,0,0,0,0.1,0.1]


# In[136]:


df_continent['Total'].plot(kind='pie',figsize=(8,8), autopct='%1.1f%%', pctdistance=1.0, startangle=90, shadow=True, label=None, colors=color_list, explode = explode_list)
plt.title('Immigration to Canada by continent[1980-2013]')
plt.axis('equal')
plt.legend(labels=df_continent.index, loc='upper left')
plt.show()


# In[89]:


color_list=['gold', 'green', 'red', 'orange', 'grey','navy']


# In[100]:


df_continent['Total'].plot(kind='pie', 
                           figsize=(10,8), 
                           autopct='%1.1f%%',
                           startangle=90,
                           shadow=True,
                          label=None,
                          pctdistance=1.0,
                            explode=explode_list,
                           colors=color_list
                          )
plt.title('Immigration to Canada by continent(1980-2013)',y=1.0, fontsize=15)
plt.axis('equal')
plt.legend(labels=df_continent.index, loc='upper left', fontsize=7)
plt.show()


# In[104]:


#data for japanese immigrants to Canada[1980-2013]
df_japan=df.loc[['Japan'],year].T
df_japan.head()


# In[106]:


#plotting
df_japan.plot(kind='box', figsize=(8,6))
plt.title('Box plot of Japanese immigrants to Canada (1980-2013)')
plt.ylabel('Number of immigrants')
plt.show()


# In[108]:


df_japan.describe()


# In[110]:


#data for China and India immigrants to Canada(1980-2023)
df_ci=df.loc[['China', 'India'], year].T
df_ci.head()


# In[111]:


#plotting box plot
df_ci.plot(kind='box', fig=(8,8))
plt.title('Box plot from indians and chinese immigrants to Canada (1980-2023)')
plt.ylabel('Number of Immigrants')
plt.show()


# In[112]:


df_ci.describe()


# In[113]:


df_ci.plot(kind = 'box', figsize=(8,6), color='blue', vert=False)
plt.title('Box plot of Immigrants from China and India(1980-2013)')
plt.xlabel('Number of Immigrants')
plt.show()


# In[114]:


df_japan.plot(kind='box', figsize=(8,6), color='navy', vert=False)
plt.title('Box plot for immigrants from Japan to canada(1980-2013)')
plt.xlabel('Number of Immigrants')
plt.show()


# In[122]:


#subplot1: Box plot
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

df_ci.plot(kind='box', figsize=(10,8), color='maroon', vert=False, ax=ax0)
#add to subplot1
ax0.set_title('Box plot for China and Indian immigrants tO Canada(1980-2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

#subplot2:line plot
df_ci.plot(kind='line', figsize=(20,6), ax=ax1)
#add to subplot2
ax1.set_title('Line plot of immigrants from China and India (1980-2013)')
ax1.set_ylabel('Number of immigrants')
ax1.set_xlabel('Years')

plt.show()


# create a box plot to visualize the distribution of the top 15 countries (based on total immigration) grouped by decades 1980s, 1990s, and 2000s.
# 

# In[127]:


df_top15=df.sort_values(['Total'], ascending = False, axis=0)
df_top15['Total'].head(15)


# In[128]:


#create a list of all the years
year_80s=list(map(str, range(1980, 1990)))
year_90s=list(map(str, range(1990, 2000)))
year_00s=list(map(str, range(2000, 2010)))


# In[129]:


#slice the original dataframe to create a series
df_80s=df_top15.loc[:,year_80s].sum(axis=1)
df_90s=df_top15.loc[:,year_90s].sum(axis=1)
df_00s=df_top15.loc[:,year_00s].sum(axis=1)


# In[132]:


#merge the 3 series into dataframe called new_df
new_df=pd.DataFrame({'1980s':df_80s, '1990s':df_90s, '2000s':df_00s})
new_df.head()


# In[133]:


new_df.describe()


# In[ ]:


#checking the entries that follow above the outliers threshold

