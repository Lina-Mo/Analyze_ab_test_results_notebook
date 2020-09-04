#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from statistics import mean
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


# Load and view first few lines of dataset
df = pd.read_csv('ab_data.csv')
df.head()


# In[3]:


# Review the dataset non-null values and data types
df.info()


# b. Use the cell below to find the number of rows in the dataset.

# In[4]:


# Review numbers of rows and columns of the dataset
df.shape


# c. The number of unique users in the dataset.

# In[5]:


# Review unique users in the dataset
df.nunique()


# d. The proportion of users converted.

# In[6]:


# Review summary of converted column 
df.converted.value_counts()


# In[7]:


#Check the proportion of users converted
df.query('landing_page == "new_page"')['converted'].mean()


# e. The number of times the `new_page` and `treatment` don't match.

# In[8]:


# Check the count of times when new_page and tratment don't match
df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].shape[0]


# f. Do any of the rows have missing values?

# In[9]:


# Review the rows with missing values
null_data = df[df.isnull().any(axis=1)]
null_data.count()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  
# 
# **In this case we need to remove these rows or except from the new dataset created.**
# 
# a. Store your new dataframe in **df2**.

# In[10]:


# Create new dataset without rows when new_page and tratment don't match
df_mod = df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == True]
df2 = df_mod.copy()


# In[11]:


# Review the numbers of rows and columns of the new dataset
df2.shape


# In[12]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[13]:


# Check the unique user_id number
df2.nunique()


# **Answer: There are 290,584 unique user_id in the df2 dataset**

# b. There is one **user_id** repeated in **df2**.  What is it?

# In[14]:


# Review if dataset have dublicated user_id
df2.user_id.duplicated().sum()


# In[15]:


# Review dublicated user_id 
duplicaterow = df2[df2.user_id.duplicated()]
print(duplicaterow)


# c. What is the row information for the repeat **user_id**? 

# In[16]:


# Review the repeat user_id information
duplicaterows = df2[df2['user_id'] == 773192]
print(duplicaterows)


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[17]:


# Remove one dublicated row from the df2 dataset
df2.drop_duplicates('user_id', inplace=True)


# In[18]:


# Run check if duplicated was removed
df2.user_id.duplicated().sum()


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[19]:


# Compute probability of converting
df2['converted'].mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[20]:


# Compute control group converted rate
control_mean = df2.query('group == "control"')['converted'].mean()
control_mean


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[21]:


# Compute treatment group converted rate
treatment_mean = df2.query('group == "treatment"')['converted'].mean()
treatment_mean


# d. What is the probability that an individual received the new page?

# In[22]:


# Compute individual received new page probability
(df2['landing_page'] == 'new_page').mean()


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Answer:**
# 
# **Base on calculated propability we can make conclution, that is no sufficient evidence to conclude that the new treatment page leads to more conversions.** 
# 
# **There are very similar resuls of convertion rate for both treatment and control groups.**
# 
# **Probability that an individual received the new page are 0.50, equal chance that user receive old page or new page.**
# 
# **Propability of treatment group are 0.1188 less than the average probability of converting (0.1195).**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Answer: $$H_0: P_{new} - P_{old} <=0$$**

# **$$H_1: P_{new} - P_{old} > 0$$**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[23]:


# Calculate p_new under the null
p_new = df2['converted'].mean()
print(p_new)


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[24]:


# Calculate p_old under the null
p_old = df2['converted'].mean()
print(p_old)


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[25]:


# Calculate n_new of treatment group
n_new = df2[df2['group'] == 'treatment'].shape[0]
print(n_new)


# d. What is $n_{old}$, the number of individuals in the control group?

# In[26]:


# Calculate n_old of control group
n_old = df2[df2['group'] == 'control'].shape[0]
print(n_old)


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[27]:


# Draw samples from a binomial distribution of p_new
new_page_converted = np.random.binomial(1, p_new, n_new)
# calculate the mean
new_mean = new_page_converted.mean()
new_mean


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[28]:


# Draw samples from a binomial distribution of p_old
old_page_converted = np.random.binomial(1, p_old, n_old)
# calculate the mean
old_mean = old_page_converted.mean()
old_mean


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[29]:


# Calculate difference in sample proportions
new_mean - old_mean


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[30]:


# Run 10,000 samples
p_diffs = []
size = df2.shape[0]

for _ in range(10000):
    new_page_converted = np.random.binomial(1, p_new, n_new)
    old_page_converted = np.random.binomial(1, p_old, n_old)
    new_mean = new_page_converted.mean()
    old_mean = old_page_converted.mean()
    p_diffs.append(new_mean - old_mean)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[31]:


# Convert to numpy array
p_diffs = np.array(p_diffs)


# In[32]:


# Plot sampling distribution histogram
plt.hist(p_diffs, edgecolor = 'black');

plt.title('Sampling distribution of p_new and p_old mean difference')
plt.xlabel('Probability difference')
plt.ylabel('Frequency')


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[33]:


# Calculate proportion difference between two groups
obs_diff = treatment_mean - control_mean
obs_diff


# In[34]:


# Calculate probability a statistic higher that observed
(p_diffs > obs_diff).mean()


# In[35]:


# Plot sampling distribution
plt.hist(p_diffs, edgecolor = 'black');
# Plot the yellow line which shows the observed difference
plt.axvline(obs_diff, c='yellow', linewidth = 3);

plt.title('Sampling distribution of p_new and p_old mean difference')
plt.xlabel('Probability difference')
plt.ylabel('Frequency')


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Answer:**
# 
# **In the part j we computed p-value probability of obtaining a result at least as extreme as the current one, assuming null hypothesis is true.**
# 
# **The p-value is high (0.9017 > 0.05) and fail to reject null hypothese.**
# 
# **Base on p-value we can make conclution, that is no statistically significance between the new and old pages.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[36]:


import statsmodels.api as sm

convert_old = sum(df2.query("group == 'control'")['converted'])
convert_new = sum(df2.query("group == 'treatment'")['converted'])
n_old = len(df2.query("group == 'control'"))
n_new = len(df2.query("group == 'treatment'"))


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[37]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative = 'larger')
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Answer:**
# 
# **The z score is 1.31 and the p value is 0.95.**
# **The z score is less than the critical value of 1.644 and we can't reject the null hypothesis.**
# **The difference of conversion rates of old and new pages are not significant.**
# **The findings are the same as in parts j. and k.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Answer:**
# 
# **Logistic regression, because we predict something that has only two possible outcomes.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[38]:


# Add intercept
df2['intercept'] = 1
# create additional columns 
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
df2['old_page'] = pd.get_dummies(df2['landing_page'])['old_page']
df2.head()


# In[39]:


# Check the means if the table created correctly
print(df2['ab_page'].mean())
print(df2['old_page'].mean())


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[40]:


# Use logistic regression model for individuals converts
log_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[41]:


# Run the summary
results = log_mod.fit()
results.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>

# **Answer:**
# 
# **The p-value associated with ab_page is 0.1899.**
# 
# **In part II we used the one tailed test with 10,000 sampling and in this part we used the two tailed test of the logistic regression, so the p-value is different in both models.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Answer:**
# 
# **Multiple linear regression allows us to consider other factors into a regression model that helps determine which factors matter most, which can be ignored, and how those factors interact with each other. The results may lead to a more accurate data understanding.**
# 
# **Despite the advantages, the technique of multiple regression analysis can suffer from the serious limitations. Adding factors multiple not related factors can return misleading results.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns

# In[42]:


# Load and view first few lines of dataset
countries_data = pd.read_csv('countries.csv')
countries_data.head()


# In[43]:


# Join two datasets and review first lines
new_df = countries_data.set_index('user_id').join(df2.set_index('user_id'), how = 'inner')
new_df.head()


# In[44]:


# Review the Country column by counts
new_df['country'].value_counts()


# In[45]:


# Add dummies colums and review first lines
new_df[['UK', 'US', 'CA']] = pd.get_dummies(new_df['country'])
new_df.head()


# In[46]:


# Use logistic regression model adding countries
add = sm.Logit(new_df['converted'], new_df[['intercept', 'UK', 'US']])
# Run the summary
results = add.fit()
results.summary2()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[47]:


# Use logistic regression model adding ab_page
add2 = sm.Logit(new_df['converted'], new_df[['intercept', 'UK', 'US', 'ab_page']])
# Run the summary
results = add2.fit()
results.summary2()


# **Answer:**
# 
# **Based on the p-values of the UK with 0.1295 and US with 0.4573, which are greater than 0.05 concludes that the country factor has no significant effect on conversion.**

# <a id='conclusions'></a>
# ## Conclusions
# 
# **Based on multiple test results we can advise the company to not implement the new page of an e-commerce site and keep the old page. The new page does not  have a significant effect on the converted rate compared with the old page.**

# In[48]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# **Resources:**
# https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/
# https://online.stat.psu.edu/stat501/lesson/5/5.5
# https://pandas.pydata.org/pandas-docs/stable/user_guide
