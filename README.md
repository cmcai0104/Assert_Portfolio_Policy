This ia a Assert Portfolio Policy based on Reinforcement Learning.

I want to build a Deep Reinforcement Learning Model for Asset allocation.

Background: I have 7 stock indexes from different markets, and I want to build a policy to produce the action 
 (likes whether to sell or buy index? which index? and how much?) by observing the market informations.
 
 Question 1: 
 
 I have two idea for the output of my policy. One is to produce a vector $w$ of length 8, 
 Each element $w_i$ represent the target ratio of the stocks we want to hold (7 stock indexes and 1 cash), 
 so I need to set $w_i>0, $ and $\sum_{i}^{8}w_i=1.$ How to implement? I just let the 
 Activation function in the last layer of neural network to be sigmoid and divide the sum
 in environment. Is this available? And it's not easy to deal with transaction process 
 if buy and sell fee exist.
 
 The two is also produce a vector $w$ of length 8, For each element $w_i$ represent sell percent 
 for stock i when $w_i$ is negative and buy percent of cash when $w_i$ is positive. 
 It can solve the problem I meet in idea one. But I will meet a new question is cash is finite. 
 I need to decide order of buy, in other words, which stock to buy first and buy which one later.
 
 Question 2:
 
Many papers tell me to produce Distributed parameters by policy then create the action 
by distribution (like: normal distribution). It makes that more difficult to control the action 
 satisfy the condition above.
 
 Thanks @NeilSlater for your reply, may be I have not express my question clearly. I have already written an 
 environment [env](https://github.com/cmcai0104/Assert_Portfolio_Policy/blob/master/environment/DNN_MarketEnv.py) by idea one. 
 The Observed data is just the daily data of market like [data](https://github.com/cmcai0104/Assert_Portfolio_Policy/blob/master/data/create_feature.csv).
 And here is my policy gradient [pg](https://github.com/cmcai0104/Assert_Portfolio_Policy/blob/master/PolicyGradient.py).
 
 I want to train a policy by reinforcement learning to improve or exceed the traditional assets portfolio policy.
 Which use the rolling quantile of PE(the Price Earnings Ratio) of Stocks indexs as the target ratio monthly. The PE is one column in my data.
 
My initial idea was naive. I want to use multiple dense layers to get the actions, and then get the evaluation function(means the probability of this action) by multiple dense layers after actions. 
But the result is totally decided by the initialization and the training process is also slowly. I think lack of explore strategy is the reason.
 