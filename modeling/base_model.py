#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:56:35 2020

@author: kathleenyoung
"""
# Train a base linear regression model on the aggregated book and review data

# import packages
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random

# pre-covid data
pre = pd.read_csv("pre.csv")

# post covid data
post = pd.read_csv("post.csv")

# create total review count column
pre["total_reviews"] = pre["review_count 2019-03"] + pre["review_count 2019-04"] + pre["review_count 2019-05"] + pre["review_count 2019-06"]
post["total_reviews"] = post["review_count 2020-03"] + post["review_count 2020-04"] + post["review_count 2020-05"] + post["review_count 2020-06"]

# drop month count columns, index column
pre = pre.drop(["Unnamed: 0", "review_count 2019-03", "review_count 2019-04", "review_count 2019-05", "review_count 2019-05"], axis = 1)
post = post.drop(["Unnamed: 0", "review_count 2020-03", "review_count 2020-04", "review_count 2020-05", "review_count 2020-06"], axis = 1)

# build 2019 model
# X Y
Xpre = pre[['num_reviews', 'num_ratings', "avg_rating"]]
Ypre = pre['total_reviews']

# test train split
Xpre_train, Xpre_test, Ypre_train, Ypre_test = train_test_split(Xpre, Ypre, train_size=.75)

# fit model
reg_pre = LinearRegression().fit(Xpre_train, Ypre_train)




# build 2020 model
# X Y
Xpost = post[['num_reviews', 'num_ratings', "avg_rating"]]
Ypost = post['total_reviews']

# test train split
Xpost_train, Xpost_test, Ypost_train, Ypost_test = train_test_split(Xpost, Ypost, train_size=.75, random_state = 14)

# fit model
reg_post = LinearRegression().fit(Xpost_train, Ypost_train)

# MSE
# Train MSE precovid model on precovid data
train_mse_pre_pre = metrics.mean_squared_error(Ypre_train, reg_pre.predict(Xpre_train))
print("Training MSE pre-covid model, pre-covid data: " + str(train_mse_pre_pre))

# Test MSE precovid model on precovid data
test_mse_pre_pre = metrics.mean_squared_error(Ypre_test, reg_pre.predict(Xpre_test))
print("Testing MSE pre-covid model, pre-covid data: " + str(test_mse_pre_pre))

# MSE precovid model on postcovid data
mse_pre_post = metrics.mean_squared_error(Ypost_test, reg_pre.predict(Xpost_test))
print("MSE pre-covid model, post-covid data: " + str(mse_pre_post))


# r^2
# Train r^2 precovid model on precovid data
r2_pre_pre_train = metrics.r2_score(Ypre_train, reg_pre.predict(Xpre_train))
print("Training r^2 pre-covid model, pre-covid data: " + str(r2_pre_pre_train))

# Test r^2 precovid model on precovid data
r2_pre_pre = metrics.r2_score(Ypre_test, reg_pre.predict(Xpre_test))
print("Testing r^2 pre-covid model, pre-covid data: " + str(r2_pre_pre))

# r^2 precovid model on postcovid data
r2_pre_post = metrics.r2_score(Ypost_test, reg_pre.predict(Xpost_test))
print("r^2 pre-covid model, post-covid data: " + str(r2_pre_post))
