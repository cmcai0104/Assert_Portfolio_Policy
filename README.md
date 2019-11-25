This ia a Assert Portfolio Policy based on Reinforcement Learning.

I want to build a Deep Reinforcement Learning Model for Asset allocation.
Background: I have 7 stock indexes from different markets, and I want to create a policy to produce the action 
 (like sell or buy this index? and how much? ) by observing the market.
 
 Question 1: 
 
 I have two idea for the output of policy. One idea is to produce a vector $w$ of length 8, 
 Each element $w_i$ represent the target ratio of the stocks we want to hold, so I need to set $w_i>0, $ and $\sum_{i}^{8}w_i=1.$ 
 How to implement? I just let the Activation function in the last layer of neural network to be sigmoid and divide the sum
 in environment. Is this available? And it's not easy to deal with transaction process if buy and sell fee exist.
 
 The two is also produce a vector $w$ of length 8, For each element $w_i$ represent sell percent for stock i when $w_i$
 is negative and buy percent of cash when $w_i$ is positive. It can solve the problem I meet in idea one. But I will meet a new
 question is cash is finite. I need to decide order of buy, in other words, which stock to buy first.
 
 Question 2:
 
As I mentioned above, the action is continuous and many papers tell me need to produce Distributed parameters
 by policy and create the action by distribution. 
 