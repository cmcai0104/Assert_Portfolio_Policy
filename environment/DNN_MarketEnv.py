import numpy as np
import pandas as pd
import random
import gym
# import tensorflow as tf
# import math

df = pd.read_csv('data/create_feature.csv', index_col=0, header=0)
# df['date'] = df['date'].dt.date
df = df.set_index('trade_date')
df = df.dropna(axis=0, how='any')


class DNNMarketEnv(gym.Env):
    environment_name = 'Assert Portfolio'

    def __init__(self, df, initial_account_balance=10000, sell_fee=0., buy_fee=0.0015):
        super(DNNMarketEnv, self).__init__()
        self.id = "Assert Portfolio"
        self.df = df
        colnames = self.df.columns
        self.price_cols = colnames[[colname[-5:] == 'close' for colname in colnames]]
        print("对以下标的进行交易：", self.price_cols)
        self.action_dim = len(self.price_cols)+1
        self.seed()

        self.reward_range = (0, np.inf)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1, self.action_dim), dtype=np.float16)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, self.df.shape[1]), dtype=np.float16)

        self.Max_Steps = len(self.df)
        self.Max_Share_Price = self.df.max(axis=0)
        self.Max_Share_Price = np.power(10, np.ceil(np.log10(self.Max_Share_Price)))

        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.initial_account_balance = initial_account_balance                                         # 初始资金
        self.net_worth = initial_account_balance                                                       # 账户总市值
        self.max_net_worth = initial_account_balance                                                   # 账户最大市值
        self.shares_held = np.append(np.zeros(shape=self.action_dim-1), initial_account_balance)       # 持有股票份额
        self.current_step = random.randint(0, self.Max_Steps)                                          # 最大步数

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # 下一步观测
    def _next_observation(self):
        obs = np.array([self.df.iloc[self.current_step, :].values / self.Max_Share_Price])
        return obs

    # 进行交易
    def _take_action(self, action):
        current_price = np.array(self.df.iloc[self.current_step, ][self.price_cols])
        self.net_worth = np.sum(np.append(current_price, 1)*self.shares_held)
        hold_rate = (np.append(current_price, 1) * self.shares_held / self.net_worth)
        target_rate = (action.numpy() / action.numpy().sum())[0]
        para = np.zeros(len(hold_rate)-1)
        sell_index = np.where(hold_rate[:-1] > target_rate[:-1])
        buy_index = np.where(hold_rate[:-1] < target_rate[:-1])
        para[sell_index] = 1-self.sell_fee
        para[buy_index] = 1/(1-self.buy_fee)
        self.net_worth = ((hold_rate[:-1]*para).sum()+hold_rate[-1]) / \
                         ((target_rate[:-1]*para).sum()+target_rate[-1]) * self.net_worth
        self.shares_held = self.net_worth * target_rate / np.append(current_price, 1)

    # 在环境中执行一步
    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        if self.current_step >= self.Max_Steps:
            self.current_step = 0
        delay_modifier = self.current_step / self.Max_Steps
        reward = self.net_worth * delay_modifier
        done = self.net_worth <= 0
        obs = self._next_observation()
        return obs, reward, done, {}

    # 重置环境状态至初始状态
    def reset(self, initial_account_balance=10000):
        self.initial_account_balance = initial_account_balance
        self.net_worth = initial_account_balance
        self.max_net_worth = initial_account_balance
        self.shares_held = np.append(np.zeros(shape=self.action_dim - 1), initial_account_balance)
        self.current_step = random.randint(0, self.Max_Steps)
        return self._next_observation()

    # 打印出环境
    def render(self, mode='human'):
        profit = self.net_worth - self.initial_account_balance
        print(f'Step:{self.current_step}')
        # print(f':{self.balance}')
        print(f':{self.shares_held}')
        # print(f':{self.cost_basis}(:{self.total_sales_value})')
        print(f'总市值:{self.net_worth}(最大市值:{self.max_net_worth})')
        print(f'盈利:{profit}')

