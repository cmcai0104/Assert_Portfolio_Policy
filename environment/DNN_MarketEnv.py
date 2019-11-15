import numpy as np
import pandas as pd
import random
import gym
import warnings

# df = pd.read_csv('./data/create_feature.csv', index_col=0, header=0)
# df['trade_date'] = df['trade_date'].astype('datetime64')
# df = df[df['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
# df = df.set_index('trade_date')
# df = df.fillna(method='ffill', axis=0)
# df = df.dropna(axis=0, how='any')


class DNNMarketEnv(gym.Env):
    environment_name = 'Assert Portfolio'

    def __init__(self, df, initial_account_balance=10000., sell_fee=0., buy_fee=0.015):
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

        self.Max_Steps = len(self.df)-2
        self.Max_Share_Price = self.df.max(axis=0)
        self.Max_Share_Price = np.power(10, np.ceil(np.log10(self.Max_Share_Price)))

        self.buy_fee = buy_fee                                                                         # 购买费率
        self.sell_fee = sell_fee                                                                       # 赎回费率
        self.initial_account_balance = initial_account_balance                                         # 初始资金
        self.net_worth = initial_account_balance                                                       # 账户总市值
        self.max_net_worth = initial_account_balance                                                   # 账户最大市值
        self.shares_held = np.append(np.zeros(shape=self.action_dim-1), initial_account_balance)       # 持有股票份额
        #self.current_step = random.randint(0, self.Max_Steps)                                          # 现在位置
        self.current_step = 0
        self.start_date = self.df.index[self.current_step]

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # 下一步观测
    def _next_observation(self):
        obs = np.array([self.df.iloc[self.current_step, :].values / self.Max_Share_Price])
        return obs

    # 进行交易
    def _take_action(self, action):
        self.current_price = np.array(self.df.iloc[self.current_step, ][self.price_cols])
        self.shares_before = self.shares_held
        self.net_worth = np.sum(np.append(self.current_price, 1)*self.shares_held)
        hold_rate = (np.append(self.current_price, 1) * self.shares_held / self.net_worth)
        target_rate = (action.numpy() / action.numpy().sum())[0]
        para = np.zeros(len(hold_rate)-1)
        sell_index = np.where(hold_rate[:-1] > target_rate[:-1])
        buy_index = np.where(hold_rate[:-1] < target_rate[:-1])
        para[sell_index] = 1-self.sell_fee
        para[buy_index] = 1/(1-self.buy_fee)
        self.net_worth = ((hold_rate[:-1]*para).sum()+hold_rate[-1]) / \
                         ((target_rate[:-1]*para).sum()+target_rate[-1]) * self.net_worth
        self.shares_held = self.net_worth * target_rate / np.append(self.current_price, 1)
        self.hold_rate = hold_rate
        self.target_rate = target_rate
        self.sell_index = sell_index
        self.buy_index = buy_index
        self.para = para


    # 在环境中执行一步
    def step(self, action):
        obs = self._next_observation()
        self._take_action(action)
        self.current_step += 1
        self.next_price = np.array(self.df.iloc[self.current_step, ][self.price_cols])
        reward = np.sum(np.append(self.next_price, 1)*self.shares_held)
        done = self.current_step >= self.Max_Steps
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        return obs, reward, done, (self.current_price, self.next_price, self.shares_before)

    # 重置环境状态至初始状态
    def reset(self):
        self.net_worth = self.initial_account_balance
        self.max_net_worth = self.initial_account_balance
        self.shares_held = np.append(np.zeros(shape=self.action_dim - 1), self.initial_account_balance)
        #self.current_step = random.randint(0, self.Max_Steps)
        self.current_step = 0
        self.start_date = self.df.index[self.current_step]
        return self._next_observation()

    # 打印出环境
    def render(self, mode='human'):
        ret = self.net_worth / self.initial_account_balance * 100 - 100
        yea_ret = (self.net_worth/self.initial_account_balance)**(365/(self.df.index[self.current_step] - self.start_date).days)*100-100
        #print(f'start_date:{self.start_date}')
        #print(f'股票份额:{self.shares_held}')
        print(f'总市值:{round(self.net_worth,2)}(最大市值:{round(self.max_net_worth,2)})')
        print(f'累计收益率:{round(ret,2)}%')
        print(f'累计年化收益率:{round(yea_ret,2)}%')

