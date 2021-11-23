#!/usr/bin/env python
# coding: utf-8

# # <center> Predict NBA player wage with multiple regression model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wooldridge as woo
from simple_colors import *


# # Data Description and Motivation

# In[2]:


# import data
df = woo.dataWoo('nbasal')
df_missing = woo.dataWoo('nbasal')


# ##### Motivation:
# 
# Our goal is to study what would influence NBA players' salaries. The outcome would be helpful for players to improve themselves and for teams to determine how much they should pay based on players' performance.

# ##### Data Description:
# 
# We found this data from Wooldridge Dataset. This data was collected by Christopher Torrente, who did for school team project while he was in MSU.
# This is the salary data and the career statistics from The Complete Handbook of Pro Basketball, 1995, edited by Zander Hollander. New York: Signet. The demographic information (marital status, number of children, and so on) was obtained from the teams’ 1994-1995 media guides.
# 

# ##### Variable Explaination:
# 
# 
# marr: =1 if married
# 
# wage: annual salary, thousands $
# 
# exper: years as professional player
# 
# age: age in years
# 
# coll: years played in college
# 
# games: average games per year
# 
# minutes: average minutes per year
# 
# guard: =1 if guard
# 
# forward: =1 if forward
# 
# center: =1 if center
# 
# points: points per game
# 
# rebounds: rebounds per game
# 
# assists: assists per game
# 
# draft: draft number
# 
# allstar: =1 if ever all star
# 
# avgmin: minutes per game
# 
# lwage: log(wage)
# 
# black: =1 if black
# 
# children: =1 if has children
# 
# expersq: exper^2
# 
# agesq: age^2
# 
# marrblck: marr*black

# ##### Variable Exploration

# In[3]:


# explore the data
print(df.info())
print(df.describe())
print(df.isnull().sum())


# We find there are two response variables, wage and lwage, therefore we drop lwage for now.
df.drop('lwage',axis = 1,inplace = True)
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent],axis = 1,keys = ['total','percent'])
print(missing_data)


# #### From the above analysis, we can see that the total number of variables in this dataset is 21. We have 269 observations, where 12% observations miss the value of draft.

# In[4]:


for i in df.columns:
    print('The unique number of comlumn ' + str(i),'is',len(df[i].unique()))


# In order to use Boruta Algorithm and Mallows Cp, we will need to handle missing values first. We removed obeservations who does not have values for draft. We will explain the reasoning in Question 2.

# In[5]:


# drop observations where draft is NA
df.dropna(axis = 0,inplace = True)


# # Q1: Variable Selection

# ## (a) Using the Boruta Algorithm identify the top 10 predictors

# In[6]:


# perform boruta to select potnetial predictor
from BorutaShap import BorutaShap
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor


# In[7]:


x = df.loc[:,(df.columns != 'wage')]
y = df['wage']


# In[10]:


Feature_Selector = BorutaShap(importance_measure='shap', classification=False)
Feature_Selector.fit(X=x, y=y, n_trials=100, random_state=0)
Feature_Selector.plot(which_features='all')


# Based on the result from Boruta Algorithm, we will choose the 8 confirmed important variables and 2 tentative varaibles:
# 
# #####  draft, avgmin, points,center, rebounds,minutes,  expersq, exper, age, agesq (from high to low according to Z-score)

# ## (b) Using Mallows Cp identify the top 10 predictors

# In[87]:


# list of all variables
potential_list = ['center', 'exper', 'rebounds', 'draft', 'points', 'avgmin', 'expersq', 'minutes','marr', 'forward', 'guard', 'allstar', 'coll', 'children', 'assists', 'games', 'black', 'marrblck','age', 'agesq']

# create combinations of 10 variables
import itertools
potential_list = ['center', 'exper', 'rebounds', 'draft', 'points', 'avgmin', 'expersq', 'minutes','marr', 'forward', 'guard', 'allstar', 'coll', 'children', 'assists', 'games', 'black', 'marrblck','age', 'agesq']
variable_list = []
for subset in itertools.combinations(potential_list,10):
    variable_list.append(subset)

    
for i in variable_list[:20]:
    print(i)
    
print("Total number of possible combinations is", len(variable_list))


# In[56]:


# Mallows Cp
from RegscorePy import *
y = df['wage']
x_original = df.loc[:,potential_list]
full_model = sm.OLS(y,sm.add_constant(x_original))
full_model_fit = full_model.fit()
y_pred = np.array(full_model_fit.fittedvalues)
mallow_select = pd.DataFrame()
for i in variable_list:
    x = df.loc[:,i]
    Mallow_CP_mod = sm.OLS(y,sm.add_constant(x))
    Mallow_CP_fit = Mallow_CP_mod.fit()
    y_sub= np.array(Mallow_CP_fit.fittedvalues)
    k = 21
    p = 11
    cp = mallow.mallow(y,y_pred,y_sub,k,p)
    mallow_select = mallow_select.append({'combination' :str(i), 'mallow' :cp},ignore_index = True)


# In[60]:


# calculate distance 
mallow_select["distance"] = np.abs(11-mallow_select.mallow)


# In[61]:


mallow_select


# In[98]:


mallow_select.sort_values(["distance", "mallow"], axis = 0)


# In[100]:


mallow_select.iloc[70820].combination


# In[120]:


mallow_select.iloc[70946].combination


# In[121]:


mallow_select.iloc[168495].combination


# In[15]:


print(red('From above mallow cp result ：',['bold']))
print('We achieve three combinations with same mallow-cp values. They have 8 identical predictors. \nWe suspect that there is perfect multicollinearity between the rest three variables :',blue('guard, forward, center',['bold']))


# In[8]:


df_test = df.copy()
df_test['sum_up'] = df_test['center']+df_test['forward']+df_test['guard']
df_test.sum_up.value_counts()


# In[16]:


print('The made up columns proved our hypothesis, so we only need two variables from guard, forward, and center')


# Based on the result from Mallows Cp, we will choose the following varaibles:
# 
# ##### draft, points, avgmin, forward, center, coll, children, games, age, agesq

# ### (c) Based on your findings from parts (a) and (b) above, select your preferred choice of predictors

# In[6]:


list_brt = ['draft','avgmin','points', 'center','rebounds', 'minutes','expersq', 'exper', 'age','agesq',]
list_cp = ['center', 'draft', 'points', 'avgmin', 'forward', 'coll', 'children', 'games', 'age', 'agesq']
variable_intersect = np.intersect1d(np.array(list_brt), np.array(list_cp))
variable_union = np.union1d(np.array(list_brt), np.array(list_cp))
print(blue('Intersection',['bold']))
print(variable_intersect,'with length of ',len(variable_intersect))
print(blue('Union',['bold']))
print(variable_union,'with length of ',len(variable_union))


# In[7]:


# avgmin = (minutes/games)
df2 = df.copy()
df2['minperg'] = df2['minutes']/df2["games"]
df2[['minperg', "avgmin"]]


# #### Based on the result from (a) and (b), we will choose the following varaibles:

# In[8]:


print(blue('age, avgmin, draft, points, coll, forward, center, rebounds',['bold']))


# #### Reasons:

# >1. We conduct two predictor selection process, we will keep the intersection of predictor selected by two process
# >2. After that, we will delete any high order term of existing predictor as we will conduct linearity check afterwards
# >3. Then we deal with multicollinearty issues that can be easily found: minutes/games = avgmin, guard + center + forward = 1

# # Q2: Descriptive Analysis

# (a) Descriptive analysis of the variables

# In[9]:


variables = df[['age','avgmin','center','draft','points','coll','forward','rebounds','wage']]
print(variables.describe())
print(variables.info())


# We used pairplot to interpret and determine the correlation between different predictors.

# In[10]:


sns.pairplot(data = variables,vars = ['age','avgmin','center','draft','points','coll','forward','rebounds'],diag_kind = 'hist',corner = True)


# In[11]:


# boxplot: 
var = ['age','avgmin','center','draft','points','coll','forward','rebounds','wage']
fig,axes = plt.subplots(3,3,figsize = (40,30))
axes = axes.ravel()
for i in range(9):
    if len(variables[var[i]].unique()) <=5:
        sns.boxplot(ax = axes[i],x = var[i],y = 'wage',data = variables)
    else:
        sns.boxplot(ax = axes[i],x = var[i],data = variables)
    axes[i].set_title(str(var[i]))


# In[12]:


# QQ plot
var = ['age','avgmin','center','draft','points','coll','forward','rebounds','wage']
fig,axes = plt.subplots(3,3,figsize = (40,30))
axes = axes.ravel()
for i in range(9):
    if len(variables[var[i]].unique()) <=5:
        continue
    else:
        sm.qqplot(ax = axes[i],data = variables[var[i]],line = 'r')
    axes[i].set_title(str(var[i]))


# In[15]:


# correlation plot
sub_df = variables[['wage','age','avgmin','center','draft','points','coll','forward','rebounds']]
corr = sub_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
colormap = sns.diverging_palette(220, 10, as_cmap = True)
dropvals = np.zeros_like(corr)
dropvals[np.triu_indices_from(dropvals)] = True
sns.heatmap(corr, cmap = colormap, linewidths = .5, annot = True, fmt = ".2f", mask = dropvals)
plt.show()


# (b) Density plots and fitted distributions.

# In[16]:


# plot histogram and corresponding kde
fig,axes = plt.subplots(3,3,figsize = (15,15))
axes = axes.ravel()
for i in range(9):
    if len(variables[var[i]].unique()) == 2:
        sns.histplot(data=variables, x=var[i],kde = False,ax = axes[i])
    else:
        sns.histplot(data=variables, x=var[i],kde = True,stat = 'density',ax = axes[i])


# In[6]:


# We explore more about response variable and use cullen-frey graph to see the distribution of wage
from fitter import Fitter
f = Fitter(df.wage)
f.fit()
f.summary()


# We may use transformations on 'wage'.

# (c) Identify non-linearities within the variables

# We run a simple linear regression to see if there are any non-linearities:

# In[17]:


# first perform ols regression
y = variables['wage']
x = variables[['age','avgmin','center','draft','points','coll','forward','rebounds']]
reg_r = smf.ols(formula = 'wage ~ age + avgmin + center + draft + points + coll + forward + rebounds',data = variables)
result_r = reg_r.fit()


# In[18]:


result_r.summary()


# In[19]:


# plot predict y V.S residual plot
fig, ax = plt.subplots(1,2,figsize=(8, 6))
fig.tight_layout(pad=6.0)
sns.regplot(x=result_r.fittedvalues, y=variables['wage'], lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
ax[0].set(xlabel='Predicted', ylabel='Observed')

sns.regplot(x=result_r.fittedvalues, y=result_r.resid, lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
ax[1].set(xlabel='Predicted', ylabel='Residuals')


# In[20]:


import statsmodels.regression.linear_model as rg
import statsmodels.stats.diagnostic as dg

test =  dg.linear_reset(result_r, power=2,  test_type='fitted', use_f = True)

print(blue("Ramsey-RESET:",['bold']))
print(test)


# According to P-value which is 0.03047, higher order of some of the variables may be added to the model.

# In[21]:


# ccpr plot
from statsmodels.graphics.regressionplots import add_lowess
fig, axs = plt.subplots(1,6, figsize=(15, 6), facecolor='w', edgecolor='k',sharey = True)
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
predictor_category = ['age','avgmin','draft','points','coll','rebounds']
for i in range(6):
    sm.graphics.plot_ccpr(result_r, predictor_category[i],ax = axs[i])
    axs[i].set_title(str(predictor_category[i]))
    add_lowess(axs[i],frac = 0.5)


# Based on the ccpr plot, we found that draft can be changed to quadratic form. 
# 
# Age and rebounds may also need transformations, but they do not have a strong trend. It becomes unnecessary after we apply log transformation on wage.
# 
# If we do not transform the variables and simply add those non-linear variables, we will break the linearity assumption of the multiple regression model. Our parameter estimates would be biased, and our model would have poor performance.

# (d) Outliers

# In[22]:


# use influence plot to identify outlier
fig,ax = plt.subplots(figsize = (12,8))
fig = sm.graphics.influence_plot(result_r,alpha = 0.05,ax = ax,criterion = 'cooks')
fig,ax = plt.subplots(figsize = (12,8))
fig = sm.graphics.influence_plot(result_r,alpha = 0.05,ax = ax,criterion = 'DFFITS')


# In[51]:


variables.loc[102,:]


# In[52]:


variables.loc[213,:]


# In[53]:


variables.loc[165,:]


# Our model's goal is to predict NBA players' salary by players' performance such as points per game, rebounds per game and average minutes played per game. In this case, every influencial point in the influence plots above represents an active player in that season. We selected some outlier points and search for the players corresponding to these points in order to see if its existence is reasonable.
# 
# Point 213 represents a player named David Robinson.During that season, he was the most valuable player, scoring champion with highest points per game, also, he has the highest win share, which means he was the most capable player to lead the team to victory that season. In response, David Robinson's salary is higher than any other players in the league.
# 
# Point 102 represents Sedale Threatt, the 139th draft pick in 1983,he was just an average substitute player until 1990. He was traded to LA Lakers to play as a backup guard, but Laker's starting point guard abruptly retired due to HIV. Therefore, he was assigned to be the starting point guard and did so well that the team offered him a lucrative contract in return.
# 
# Point 165 represents Tree Rollins, one of the oldest player at that season and meanwhile had the lowest salary of 150 thousand dollars. It's not uncommon for players to be paid the lowest wage, and more than a tenth of the players that season were paid as much as he was.
# 
# These players might be outliers in the graph, but in reality, their situation is not uncommon and are even fairly representative. David Robinson, the MVP player in that season, represents the best players in the NBA. Sedale Threatt, who was a below-average player until he took his chance, represents every regular player who has a dream and fights for it. Tree Rollins represents a player at the end of his career with a veteran's minimum salary. 
# 
# In conclusion, we cannot simply take any of them off based on the influential plot.

# ### Reference: 
# ### https://www.basketball-reference.com/leagues/NBA_1994_per_game.html#per_game_stats::pts_per_g
# ### https://www.basketball-reference.com/players/r/robinda01.html
# ### https://en.wikipedia.org/wiki/Sedale_Threatt
# ### https://www.basketball-reference.com/players/r/rollitr01.html

# (e) NAs

# ##### We only have NAs in "draft".

# In[23]:


# first we observe the index of missing values
miss = df_missing[df_missing['draft'].isnull()]
miss_index = miss['draft'].index.tolist()
miss_index


# In[24]:


# add new a column is_null_draft where 1 means null, 0 means not null
df_test_corr = df_missing.copy()
df_test_corr['is_null_draft'] = 1
df_test_corr['is_null_draft'].where(df_test_corr['draft'].isnull(),0,inplace = True)


# In[25]:


df_test_corr['is_null_draft'].value_counts()


# ##### We run a logit regression to see if any variables could affect the probability of having a missing 'draft'.

# In[26]:


y = df_test_corr['is_null_draft']
x = df_test_corr.drop(['wage','draft','is_null_draft','lwage'],axis = 1)
result = sm.Logit(y, sm.add_constant(x)).fit()
print(result.summary())


# ##### No variable has significant effect. Missing values are not likely to be caused by certain characteristics of the data.

# In[27]:


# we will then draw the correlation plot to see if draft is important
corr = df.corr()
fig,ax = plt.subplots(figsize = (10,8))
mask = np.triu(df.corr())
sns.heatmap(corr,linewidths = 0.5,mask = mask,square = True)


# In[28]:


corr['wage'].sort_values(ascending = False)[1:]


# From above correlation, we found out the variable 'draft' is likely to have a significance effect on wage, which could cause really large impact if we remove the variable 'draft'.
# ##### Thus, we cannot drop the column of 'draft'.

# As shown above, the variable 'draft' has 29 missing values, but we could not tell if the missing values are due to failure in draft or no observations on draft. What's more, the Draft is a ordinal variable, so we couldn't simply replace the missing values with either mean, median, or mode. Therefore, we decided to drop the 29 observations with missing values.

# #### In addition, we want to use KNN to see approximately missing value of draft will belong to which group
# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

# ``` python
# combine with cross validation to build a knn model to predict the missing value
# https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12
# from sklearn.model_selection import KFold
# kf5 = KFold(n_splits = 5, shuffle = False)
# kf3 = KFold(n_splits = 3,shuffle = False)
# ```

# In[3]:


df = woo.dataWoo('nbasal')


# we need to first perform PCA to reduce dimension

# In[4]:


# first pick the data that don't contain null value for draft stored as train_df, then split between train_set and train_label by exluding draft set
from sklearn.preprocessing import StandardScaler
train_df = df[~df['draft'].isnull()]
train_set = train_df.iloc[:,df.columns != 'draft']
train_label = train_df.loc[:,df.columns == 'draft']
# then create prediction df, which only contains null value for draft stored as pred_df, then split between prediction_set 
pred_df = df[df['draft'].isnull()]
pred_set = pred_df.loc[:,df.columns != 'draft']


# In[5]:


# before perform PCA, we need to centerdized our data, so that sum(column i) = 0
train_set_norm = StandardScaler().fit_transform(train_set)
pred_set_norm = StandardScaler().fit_transform(pred_set)


# In[6]:


# we can visualize the centerdized data first for train_set_norm
feature = ['feature'+str(i) for i in range(train_set_norm.shape[1])]
standardized_train_set = pd.DataFrame(train_set_norm,columns = feature,index = train_set.index)
standardized_train_set.head()


# In[7]:


feature = ['feature'+str(i) for i in range(pred_set_norm.shape[1])]
standardized_pred_set = pd.DataFrame(pred_set_norm,columns = feature,index = pred_set.index)
standardized_pred_set.head()


# In[8]:


# next, perform PCA to reduce irrelevant dimension and get pca_df
from sklearn.decomposition import PCA
pca = PCA(n_components = 12) # we choose to keep 10 most important feature
features = ['feature'+str(i) for i in range(12)]
pca_train_set = pd.DataFrame(pca.fit_transform(train_set_norm),columns = features,index = train_set.index)
pca_train_set.head()


# In[9]:


# do the same thing for prediction set
pca_pred_set= pd.DataFrame(pca.fit_transform(pred_set_norm),index = pred_set.index)
pca_pred_set.head()


# ``` python
# From now, we have two dataset that can be used in the next step: cross validation and train true model. 
#     1.pca_train_set train_label
#     2.pca_pred_set
# ```

# In[11]:


from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn_result = cross_validate(knn,pca_train_set,train_label,cv = 10,scoring = ('neg_mean_absolute_error',
                                                                             'neg_mean_absolute_percentage_error',
                                                                             'neg_mean_squared_error', 
                                                                             'neg_root_mean_squared_error'))


# In[12]:


# use regression to predict
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg_result = cross_validate(reg,pca_train_set,train_label,cv = 10,scoring = ('neg_mean_absolute_error',
                                                                             'neg_mean_absolute_percentage_error',
                                                                             'neg_mean_squared_error', 
                                                                             'neg_root_mean_squared_error'))


# In[13]:


reg_score = pd.DataFrame()
for i in sorted(reg_result.keys()):
    reg_score = reg_score.append({'method': i,'score':(np.mean(reg_result[i])*-1)},ignore_index = True)
knn_score = pd.DataFrame()
for i in sorted(knn_result.keys()):
    knn_score = knn_score.append({'method': i,'score':(np.mean(knn_result[i])*-1)},ignore_index = True)


# In[15]:


print(blue('prediction error from regression model'))
print(reg_score)
print(blue('prediction error from knn model'))
print(knn_score)


# From above prediction model, the prediction error are too large. So instead of trying to impute missing value, we we will throw observations with NAs away.

# # Q3: Model Building

# In[7]:


final = df[['age','avgmin','draft','points','coll','forward','center','rebounds','wage']]
final


# #### We have 3 models which involved quadratic term, log, and linear functions. 

# ## Model 1

# From the above observations in our baseline model, we find higher order effects of draft may be appropriate. So we decided to add ${{draft}^2}$ in the linear regression model, the outcome is shown below.

# In[30]:


# model1:

y = final['wage']
x = final[['age','avgmin','draft','points','coll','forward','center','rebounds']]
x["draft**2"] = x["draft"]**2
x = sm.add_constant(x)
model1 = sm.OLS(y,x)
result1 = model1.fit()
result1.summary()


# Comparing to the simple linear regression model, adding a higher order variable slightly imporved our model, the ${R^2}$ increased from 0.560 to 0.596, which means the quadratic regression model can better explain the changes in wage rather than the linear regression model. The value of AIC and BIC are relatively lower than before, which also means the model has been improved.
# 

# In[31]:


# Evaluate transformations of variables
from statsmodels.graphics.regressionplots import add_lowess
fig, axs = plt.subplots(1,7, figsize=(15, 6), facecolor='w', edgecolor='k',sharey = True)
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
predictor_category = ['age','avgmin','draft','points','coll','rebounds','draft**2']
for i in range(7):
    sm.graphics.plot_ccpr(result1, predictor_category[i],ax = axs[i])
    axs[i].set_title(str(predictor_category[i]))
    add_lowess(axs[i],frac = 0.5)


# In[32]:


# Test for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF dataframe
pd.Series([variance_inflation_factor(x.values, i)
 for i in range(x.shape[1])],
 index=x.columns)


# The outcome of variance inflation factor indicated that there is a multicollinearity relationship between minutes played per game, draft, ${draft^2}$ and points per game. Theoretically, dropping these variables would make our model better. 
# 
# However, we decided not to drop any of them because draft, avgmin and points are important references for people to evaluate a player in reality, also, draft and ${draft^2}$ would naturally have some kind of correlation.

# In[33]:


# Test for heteroskedasticity
import statsmodels.stats.api as sms
name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(result1.resid, result1.model.exog)
print(blue("BP Results:",['bold']))
print(list(zip(name, test)))


# From above BP test, we see that p value is small, we reject the null hypothesis: variance = constant, therefore, this model shows heteroskedasticity.

# In[34]:


# Test for model misspecification
import statsmodels.regression.linear_model as rg
import statsmodels.stats.diagnostic as dg

test =  dg.linear_reset(result1, power=2,  test_type='fitted', use_f = True)

print(blue("Ramsey-RESET:",['bold']))
print(test)


# From above Ramsey RESET test, for 5% level of confidence, we see that p value is small that we reject the null hypothesis and conclude that this model is not good enough. We should consider improving this model by including quadratic terms or interactions.

# In[35]:


# Cook’s distance Plot, Residuals Plot, QQ-Plot

fig, ax = plt.subplots(1,2,figsize=(12, 6))
sns.regplot(x=result1.fittedvalues, y=final['wage'], lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
ax[0].set(xlabel='Predicted', ylabel='Observed')

sns.regplot(x=result1.fittedvalues, y=result1.resid, lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
ax[1].set(xlabel='Predicted', ylabel='Residuals')


# From above, we can see that residuals are not random distributed and the variance is not constant. 

# In[36]:


import scipy as sp
figA, axA = plt.subplots(figsize=(6,4))
_, (__, ___, r) = sp.stats.probplot(result1.resid, plot = axA, fit=True)


# From above, we can see that large proportion of the residuals of model 1 follow the red line, yet also have some outliers on the top right of the plot.

# In[37]:


cooks = result1.get_influence().cooks_distance[0]
plt.title("Cook's Distance Plot")
plt.ylabel("Cook's Distance")
plt.xlabel("index")
plt.scatter(range(len(cooks)), cooks)
plt.grid()


# From above, we can see one outlier, we have discuessed outliers in Q2 and we decided to keep them because remove those are representative points. By including those points, our model shows more accuracy.

# In[38]:


# bootstrap
# resample with replacement each row
boot_age = []
boot_avgmin = []
boot_center = []
boot_draft = []
boot_points = []
boot_coll = []
boot_forward = []
boot_rebounds = []
boot_draft2 = []
boot_interc = []
boot_adjR2 = []
n_boots = 1000
n_points = df.shape[0]
plt.figure()
for _ in range(n_boots):
 # sample the rows, same size, with replacement
    sample_df = final.sample(n=n_points, replace=True)
 # fit a linear regression
    y = sample_df['wage']
    x = sample_df[['age','avgmin','draft','points','coll','forward','center','rebounds']]
    x["draft**2"] = x["draft"]**2
    x = sm.add_constant(x)
    ols_model_temp = sm.OLS(y,x)
    results_temp = ols_model_temp.fit()

 
 # append coefficients
    boot_interc.append(results_temp.params[0])
    boot_age.append(results_temp.params[1])
    boot_avgmin.append(results_temp.params[2])
    boot_draft.append(results_temp.params[3])
    boot_points.append(results_temp.params[4])
    boot_coll.append(results_temp.params[5])
    boot_forward.append(results_temp.params[6])
    boot_center.append(results_temp.params[7])
    boot_rebounds.append(results_temp.params[8])
    boot_draft2.append(results_temp.params[9])
    boot_adjR2.append(results_temp.rsquared_adj)
    


# In[39]:


def empirical_ci(sample_median, boot_medians, confidence):
    boot_medians = np.array(boot_medians)
    differences = boot_medians - sample_median
    alpha = (1-confidence)/2
    # calculate the lower percentile confidence
    lower = np.percentile(differences, 100*alpha)
    # calculate the upper percentile of confidence
    upper = np.percentile(differences, (1-alpha)*100)
    # Return results
    return(lower+sample_median, upper+sample_median)


# In[43]:


results = [boot_interc,boot_age,boot_avgmin,boot_draft,boot_points,boot_coll,boot_forward,boot_center,boot_rebounds,boot_draft2]
name = ['const','age','avgmin','draft','points','coll','forward','center','rebounds','draft^2']
fig, axes = plt.subplots(5,2, figsize=(15, 30))
axes = axes.ravel()
for i in range(10):
    
    sns.histplot(results[i], alpha = 0.5, stat="density", ax = axes[i])
    title = 'Bootstrap Estimates: ' + name[i]
    axes[i].set_title(title, fontsize=16)
    axes[i].axvline(x=result1.params[i], color='red', linestyle='--')
    low, up = empirical_ci(result1.params[i], results[i], .95)
    axes[i].axvline(low, color = "lime", label='Lower Empirical CI')
    axes[i].axvline(up, color = "lime", label='Upper Empirical CI')
    axes[i].legend(loc='upper right')
    print('We can expect the coefficient of variable '+ name[i] +' to fall within',str(low),
       'and',str(up),'for 95% of the time.')


# In[45]:


# testing set performance and cross-validation performance

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score


y = final['wage']
x = final[['age','avgmin','draft','points','coll','forward','center','rebounds']]
x["draft**2"] = x["draft"]**2

# Perform an OLS fit using all the data
regr = LinearRegression()
model = regr.fit(x,y)
regr.coef_
regr.intercept_

# Split the data into train  (70%)/test(30%) samples:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train the model:
regr = LinearRegression()
regr.fit(x_train, y_train)

# Make predictions based on the test sample
y_pred = regr.predict(x_test)

# Evaluate Performance

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Perform a 5-fold CV
# Use MSE as the scoring function (there are other options as shown here:
# https://scikit-learn.org/stable/modules/model_evaluation.html

regr = linear_model.LinearRegression()
scores = cross_val_score(regr, x, y, cv=5, scoring='neg_root_mean_squared_error')
print('5-Fold CV RMSE Scores:', scores)


# We can see the error in this model is really large.

# ## Model 2

# In[47]:


# model2:

y = np.log(final['wage'])
x = final[['age','avgmin','draft','points','coll','forward','center','rebounds']]
x = sm.add_constant(x)
model2 = sm.OLS(y,x)
result2 = model2.fit()
result2.summary()


# Comparing to model 1, althought the log-linear regression model has a lower $R^2$ value of 0.513, the outcome of AIC and BIC has significantly decreased. AIC has decreased from 3797 to 424.2, and BIC for model 2 is 455.5 comparing to 3932 in model 1.

# In[48]:


# Evaluate transformations of variables
from statsmodels.graphics.regressionplots import add_lowess
fig, axs = plt.subplots(1,6, figsize=(15, 6), facecolor='w', edgecolor='k',sharey = True)
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
predictor_category = ['age','avgmin','draft','points','coll','rebounds']
for i in range(6):
    sm.graphics.plot_ccpr(result2, predictor_category[i],ax = axs[i])
    axs[i].set_title(str(predictor_category[i]))
    add_lowess(axs[i],frac = 0.5)


# From above plots, we can see that plots show more linearity than in model 1, which agree with us on the decision of making dependent variable log. However, we can still add a quadratic term for draft.

# In[49]:


# Test for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF dataframe
pd.Series([variance_inflation_factor(x.values, i)
 for i in range(x.shape[1])],
 index=x.columns)


# Comparing to model 1, the multicollinearity issue has been improved. As we discussed in model 1, we decided not to delete avgmin or points since they are important parameters to evaluate wages.

# In[50]:


# Test for heteroskedasticity
name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(result2.resid, result2.model.exog)
print(blue("BP Results:",['bold']))
print(list(zip(name, test)))


# From above BP test, we see that p value is small that we reject the null hypothesis: variance = constant. Therefore, this model shows heteroskedasticity.

# In[51]:


# Test for model misspecification
test =  dg.linear_reset(result2, power=2,  test_type='fitted', use_f = True)

print(blue("Ramsey-RESET:",['bold']))
print(test)


# From above Ramsey RESET test, although the p value is larger than p value of model 1 for 5% level of confidence, we see that p value is small than 0.05. We reject the null hypothesis and conclude that this model is not good enough. 

# In[52]:


# Cook’s distance Plot, Residuals Plot, QQ-Plot

fig, ax = plt.subplots(1,2,figsize=(12, 6))
sns.regplot(x=result2.fittedvalues, y=np.log(final['wage']), lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
ax[0].set(xlabel='Predicted', ylabel='Observed')

sns.regplot(x=result2.fittedvalues, y=result2.resid, lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
ax[1].set(xlabel='Predicted', ylabel='Residuals')


# From above, we can see that residuals are not random distributed and the variance is not constant.

# In[53]:


import scipy as sp
figA, axA = plt.subplots(figsize=(6,4))
_, (__, ___, r) = sp.stats.probplot(result2.resid, plot = axA, fit=True)


# From above, we can see that middle part of the residuals of model 2 follow the red line, wwe can also see some outliers on the top right and down left of the plot. Comparing the qq plot of model 1 and model 2, the distribution of model 1 seems more normally distributed.

# In[54]:


cooks = result2.get_influence().cooks_distance[0]
plt.title("Cook's Distance Plot")
plt.ylabel("Cook's Distance")
plt.xlabel("index")
plt.scatter(range(len(cooks)), cooks)
plt.grid()


# From above, we can see two outliers, we have discuessed outliers in Q2 and we decided to keep them because remove those are representative points. By including those points, our model shows more accuracy.

# In[55]:


# bootstrap
# resample with replacement each row
boot_age = []
boot_avgmin = []
boot_center = []
boot_draft = []
boot_points = []
boot_coll = []
boot_forward = []
boot_rebounds = []
#boot_draft2 = []
boot_interc = []
boot_adjR2 = []
n_boots = 1000
n_points = df.shape[0]
plt.figure()
for _ in range(n_boots):
 # sample the rows, same size, with replacement
    sample_df = final.sample(n=n_points, replace=True)
 # fit a linear regression
    y = np.log(sample_df['wage'])
    x = sample_df[['age','avgmin','draft','points','coll','forward','center','rebounds']]
    #x["draft**2"] = x["draft"]**2
    x = sm.add_constant(x)
    ols_model_temp = sm.OLS(y,x)
    results_temp = ols_model_temp.fit()

 
 # append coefficients
    boot_interc.append(results_temp.params[0])
    boot_age.append(results_temp.params[1])
    boot_avgmin.append(results_temp.params[2])
    boot_draft.append(results_temp.params[3])
    boot_points.append(results_temp.params[4])
    boot_coll.append(results_temp.params[5])
    boot_forward.append(results_temp.params[6])
    boot_center.append(results_temp.params[7])
    boot_rebounds.append(results_temp.params[8])
    #boot_draft2.append(results_temp.params[9])
    boot_adjR2.append(results_temp.rsquared_adj)


# In[56]:


results = [boot_interc,boot_age,boot_avgmin,boot_draft,boot_points,boot_coll,boot_forward,boot_center,boot_rebounds]
name = ['const','age','avgmin','draft','points','coll','forward','center','rebounds']
fig, axes = plt.subplots(5,2, figsize=(15, 30))
axes = axes.ravel()
for i in range(9):
    sns.histplot(results[i], alpha = 0.5, stat="density", ax = axes[i])
    title = 'Bootstrap Estimates: ' + name[i]
    axes[i].set_title(title, fontsize=16)
    axes[i].axvline(x=result2.params[i], color='red', linestyle='--')
    low, up = empirical_ci(result2.params[i], results[i], .95)
    axes[i].axvline(low, color = "lime", label='Lower Empirical CI')
    axes[i].axvline(up, color = "lime", label='Upper Empirical CI')
    axes[i].legend(loc='upper right')
    print('We can expect the coefficient of variable '+ name[i] +' to fall within',str(low),'and',str(up),'for 95% of the time.')


# In[125]:


# testing set performance and cross-validation performance

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score


y = np.log(final['wage'])
x = final[['age','avgmin','draft','points','coll','forward','center','rebounds']]

# Perform an OLS fit using all the data
regr = LinearRegression()
model = regr.fit(x,y)
regr.coef_
regr.intercept_

# Split the data into train  (70%)/test(30%) samples:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train the model:
regr = LinearRegression()
regr.fit(x_train, y_train)

# Make predictions based on the test sample
y_pred = regr.predict(x_test)

# Evaluate Performance

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Perform a 5-fold CV
# Use MSE as the scoring function (there are other options as shown here:
# https://scikit-learn.org/stable/modules/model_evaluation.html

regr = linear_model.LinearRegression()
scores = cross_val_score(regr, x, y, cv=5, scoring='neg_root_mean_squared_error')
print('5-Fold CV RMSE Scores:', scores)


# Although the log-linear regression model has a significant lower value of AIC and BIC, the outcome of Ramsey-RESET test and Lagrange Multiplier test still suggest that our model can be somehow improved. Therefore, we decided to try out a log-quadratic regression form in model 3.

# ## Model 3

# In[57]:


# model3:

y = np.log(final['wage'])
x = final[['age','avgmin','draft','points','coll','forward','center','rebounds']]
x["draft**2"] = x["draft"]**2
x = sm.add_constant(x)
model3 = sm.OLS(y,x)
result3 = model3.fit()
result3.summary()


# Comparing to model 2, the outcome of model 3 has higher $R^2$ value of 0.564 and lower AIC and BIC value.

# In[58]:


# Evaluate transformations of variables
from statsmodels.graphics.regressionplots import add_lowess
fig, axs = plt.subplots(1,7, figsize=(15, 6), facecolor='w', edgecolor='k',sharey = True)
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
predictor_category = ['age','avgmin','draft','points','coll','rebounds','draft**2']
for i in range(7):
    sm.graphics.plot_ccpr(result3, predictor_category[i],ax = axs[i])
    axs[i].set_title(str(predictor_category[i]))
    add_lowess(axs[i],frac = 0.5)


# From above plots, we can see that plots show even more linearity than in model 2, which shows log(wage) and draft^ 2 improves the model.

# In[59]:


# Test for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF dataframe
pd.Series([variance_inflation_factor(x.values, i)
 for i in range(x.shape[1])],
 index=x.columns)


# As we stated in model 1, although adding $draft^2$ would increase multicollinearity, we decided to delete neither of them since they both have significant correlation with the dependent variable.

# In[60]:


# Test for heteroskedasticity
name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(result3.resid, result3.model.exog)
print(blue("BP Results:",['bold']))
print(list(zip(name, test)))


# From above BP test, we see that p value is small that we reject the null hypothesis: variance = constant. Therefore, this model shows heteroskedasticity.

# In[61]:


# Test for model misspecification
test =  dg.linear_reset(result3, power=2,  test_type='fitted', use_f = True)

print(blue("Ramsey-RESET:",['bold']))
print(test)


# From above Ramsey RESET test, for 5% level of confidence, we can see that p value is so small that we reject tha null hypothesis. We should consider improving this model by including quadratic terms or interactions.

# In[62]:


# Cook’s distance Plot, Residuals Plot, QQ-Plot

fig, ax = plt.subplots(1,2,figsize=(12, 6))
sns.regplot(x=result3.fittedvalues, y=np.log(final['wage']), lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
ax[0].set(xlabel='Predicted', ylabel='Observed')

sns.regplot(x=result3.fittedvalues, y=result3.resid, lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
ax[1].set(xlabel='Predicted', ylabel='Residuals')


# From above, we can see that residuals are not random distributed and the variance is not constant.

# In[63]:


import scipy as sp
figA, axA = plt.subplots(figsize=(6,4))
_, (__, ___, r) = sp.stats.probplot(result3.resid, plot = axA, fit=True)


# From above, we can see that majority of the residuals of model 3 follow the red line, we can also see some outliers on the down left of the plot.

# In[64]:


cooks = result3.get_influence().cooks_distance[0]
plt.title("Cook's Distance Plot")
plt.ylabel("Cook's Distance")
plt.xlabel("index")
plt.scatter(range(len(cooks)), cooks)
plt.grid()


# From above, we can see one outlier, we have discuessed outliers in Q2 and we decided to keep them because remove those are representative points.

# In[65]:


# bootstrap
# resample with replacement each row
boot_age = []
boot_avgmin = []
boot_center = []
boot_draft = []
boot_points = []
boot_coll = []
boot_forward = []
boot_rebounds = []
boot_draft2 = []
boot_interc = []
boot_adjR2 = []
n_boots = 1000
n_points = df.shape[0]
plt.figure()
for _ in range(n_boots):
 # sample the rows, same size, with replacement
    sample_df = final.sample(n=n_points, replace=True)
 # fit a linear regression
    y = np.log(sample_df['wage'])
    x = sample_df[['age','avgmin','draft','points','coll','forward','center','rebounds']]
    x["draft**2"] = x["draft"]**2
    x = sm.add_constant(x)
    ols_model_temp = sm.OLS(y,x)
    results_temp = ols_model_temp.fit()

 
 # append coefficients
    boot_interc.append(results_temp.params[0])
    boot_age.append(results_temp.params[1])
    boot_avgmin.append(results_temp.params[2])
    boot_draft.append(results_temp.params[3])
    boot_points.append(results_temp.params[4])
    boot_coll.append(results_temp.params[5])
    boot_forward.append(results_temp.params[6])
    boot_center.append(results_temp.params[7])
    boot_rebounds.append(results_temp.params[8])
    boot_draft2.append(results_temp.params[9])
    boot_adjR2.append(results_temp.rsquared_adj)


# In[67]:


results = [boot_interc,boot_age,boot_avgmin,boot_draft,boot_points,boot_coll,boot_forward,boot_center,boot_rebounds,boot_draft2]
name = ['const','age','avgmin','draft','points','coll','forward','center','rebounds','draft^2']
fig, axes = plt.subplots(5,2, figsize=(15, 30))
axes = axes.ravel()
for i in range(10):
    sns.histplot(results[i], alpha = 0.5, stat="density", ax = axes[i])
    title = 'Bootstrap Estimates: ' + name[i]
    axes[i].set_title(title, fontsize=16)
    axes[i].axvline(x=result3.params[i], color='red', linestyle='--')
    low, up = empirical_ci(result3.params[i], results[i], .95)
    axes[i].axvline(low, color = "lime", label='Lower Empirical CI')
    axes[i].axvline(up, color = "lime", label='Upper Empirical CI')
    axes[i].legend(loc='upper right')
    print('We can expect the coefficient of variable '+ name[i] +' to fall within',str(low), 'and',str(up),'for 95% of the time.')


# In[136]:


# testing set performance and cross-validation performance

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score


y = np.log(final['wage'])
x = final[['age','avgmin','draft','points','coll','forward','center','rebounds']]
x["draft**2"] = x["draft"]**2

# Perform an OLS fit using all the data
regr = LinearRegression()
model = regr.fit(x,y)
regr.coef_
regr.intercept_

# Split the data into train  (70%)/test(30%) samples:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train the model:
regr = LinearRegression()
regr.fit(x_train, y_train)

# Make predictions based on the test sample
y_pred = regr.predict(x_test)

# Evaluate Performance

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Perform a 5-fold CV
# Use MSE as the scoring function (there are other options as shown here:
# https://scikit-learn.org/stable/modules/model_evaluation.html

regr = linear_model.LinearRegression()
scores = cross_val_score(regr, x, y, cv=5, scoring='neg_root_mean_squared_error')
print('5-Fold CV RMSE Scores:', scores)


# We can see that model 3 has the relatively lowest error out of 3 models shown above.

# ## In conclusion

# We decided to use Model 3 as our final regression model since it has the lowest AIC and BIC values.
# 
# According to the outcome of Model 3, NBA players' wages has a significant positive relationship with their age and average minutes played per game. This indicates that players with more time on court will receive higher salaries. Also, model 3 suggests that wage is significantly correlated with drafts in a negative way. In other words, Players with lower draft picks are more likely to receive higher salaries. Last but not least, older players tend to receive higher salaries than younger players(In part I, we found there is high multicollinearity between age and experience, so we decided to delete $exper$. Therefore, the coefficient of age could be influenced by experience).
# 
# The coefficient on points per game is insignificant, which is due to the fact that we have included average minutes played per game who takes over most of the effect on wage. The coefficient on years played in college, forward, center, and rebounds are also insignificant, meaning these factors are not going to affect salary.

# It is worth mentioning that we have tried other models by adding possible quadratic terms or interaction variables with actual meanings, but we failed to get a p value that is larger than .05 for every Ramsey-RESET test. Nonetheless, the Lagrange Multiplier test result for all those potential models also implies heteroskedasticity exists in those models. By this time, we believe that is due to the dataset we chose is not good enough, it may lack some important variables which could explain other factors that are correlated to players' wage. In fact, when NBA tries to select the most valuable player of a season, the association looks at various categories of statistics.Common sense tells us that a player's salary will never only be determined by age, average time per game and his draft postion. For instance, this dataset from Wooldridge did not collect the data on blocks or steals, which are very important when we evaluate a player's performance. What's more, players in different positions will focus on different fields, guards tend to have higher steals than forwards and centers, and centers will obviously have more rebounds and blocks than guards. All these factors will potentially affect our model's outcome, but we are constrained by the data and cannot further improve it.
# 
# What's more, the rules of the NBA draft also changed in 1989. Before 1989,NBA draft every year has 7 rounds, each has 20 picks, which gives 140 picks per year. After 1989, It has narrowed to 2 rounds, 30 picks per round, 60 drafts in total. Because of this, the distribution of Draft in our dataset is strongly skewed, and it certainly will influence the correlation between draft and wages.

# Overall, based on our model, in order to be better paid, we would recommend NBA players to: 
# 
# >1. be better prepared before the draft begins so that they can get picked eariler(smaller draft number) since every draft position ahead will increase annual salaries, and smaller draft has an increasing effect on percentage change in wage.
# >2. try to play as long as possible in every game. For every additional minute of average time per game, a player's annual salary will increase by $27.2$ percent.
# >3. retire as late as possible since each younger player will earn $58.5$ percent per year less than a player who is one year older.
# 
