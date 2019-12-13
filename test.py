from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from environment.DNN_MarketEnv import DNNMarketEnv
import pandas as pd

df = pd.read_csv('./data/create_feature.csv', index_col=0, header=0)
df['trade_date'] = df['trade_date'].astype('datetime64')
df = df[df['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
df = df.set_index('trade_date')
df = df.fillna(method='ffill', axis=0)
colnames = df.columns
colnames = colnames[[col[:6] != '399016' for col in colnames]]
df = df[colnames]
df = df.dropna(axis=0, how='any')
price_col = colnames[[col[-5:] == 'price' for col in colnames]]
env = DNNMarketEnv(df)
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()