import numpy as np
import pandas as pd
import random
import gym
import math

df = pd.read_csv('data/create_feature.csv', index_col=0, header=0)
#df['date'] = df['date'].dt.date
df = df.set_index('trade_date')

class DNNMarketEnv(gym.Env):
    environment_name = 'Assert Portfolio'

    def __init__(self, df, Initial_Account_Balance=10000):
        super(DNNMarketEnv, self).__init__()
        self.id = "Assert Portfolio"
        self.df = df
        colnames = self.df.columns
        self.price_cols = colnames[[colname[-5:] == 'close' for colname in colnames]]
        print("对以下标的进行交易：", self.price_cols)
        self.action_dim = len(self.price_cols)+1 #？？？
        self.seed()

        self.reward_range = (0, np.inf)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, self.action_dim), dtype=np.float16)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, self.df.shape[1]), dtype=np.float16)

        self.Max_Steps = len(self.df)
        self.Max_Share_Price = self.df.max(axis=0)
        self.Max_Share_Price = np.power(10, np.ceil(np.log10(self.Max_Share_Price)))

        self.Initial_Account_Balance = Initial_Account_Balance
        self.balance = Initial_Account_Balance
        self.net_worth = Initial_Account_Balance
        self.max_net_worth = Initial_Account_Balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = random.randint(0, self.Max_Steps)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # 下一步观测
    def _next_observation(self):
        obs = np.array([self.df.iloc[self.current_step, :].values / self.Max_Share_Price,])
        return obs

    # 进行交易
    def _take_action(self, action):
        current_price = np.array(
            self.df.iloc[self.current_step, ]
            [self.price_cols])
        for i in range(len(self.df.shape[1])):
            amount = np.abs(action[i])
            shares_sold = np.ones(shape=action.shape)
            if action[i] < 0:
                shares_sold[i] = self.shares_held[i] * amount
                self.balance[i] += current_price[i] * shares_sold[i]
                self.shares_held[i] -= shares_sold[i]
            if action[i] > 0:
                total_possible = self.balance / current_price)
                shares_bought = int(total_possible * amount)
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price
                self.balance -= additional_cost
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought

    # 在环境中执行一步
    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        if self.current_step > self.Max_Steps:
            self.current_step = 0
        delay_modifier = self.current_step / self.Max_Steps
        reward = self.balance * delay_modifier
        done = self.net_worth <= 0
        obs = self._next_observation()
        return obs, reward, done, {}

    #重置环境状态至初始状态
    def reset(self, Initial_Account_Balance=10000):
        self.Initial_Account_Balance = Initial_Account_Balance
        self.balance = Initial_Account_Balance
        self.net_worth = Initial_Account_Balance
        self.max_net_worth = Initial_Account_Balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = random.randint(0, self.Max_Steps)
        return self._next_observation()

    #打印出环境
    def render(self, mode='human'):
        profit = self.net_worth - self.Initial_Account_Balance
        print(f'Step:{self.current_step}')
        print(f':{self.balance}')
        print(f':{self.shares_held}(:{self.total_shares_sold})')
        print(f':{self.cost_basis}(:{self.total_sales_value})')
        print(f'总市值:{self.net_worth}(最大市值:{self.max_net_worth})')
        print(f'盈利:{profit}')

    def close(self):
        pass