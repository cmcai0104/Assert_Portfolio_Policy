import numpy as np
import pandas as pd
import tushare as ts
import time
pro = ts.pro_api('0d52800e5aed61cb4188bfde75dceff83fef0a928a0363a12a3c27d2')
dates = pd.date_range(start='20040101', end=pd.datetime.today(), freq='D')
index_daily = []
for date in dates:
    trade_date = pd.datetime.strftime(date, '%Y%m%d')
    index_oneday = pro.index_dailybasic(trade_date=trade_date)
    if len(index_oneday)>0:
        index_daily.append(index_oneday)
    time.sleep(0.6)

index_daily =  pd.concat(index_daily, axis=0)
index_daily = index_daily[[(code[:6] not in ('399300','399905')) for code in index_daily['ts_code']]]
index_daily['trade_date'] = index_daily['trade_date'].astype('datetime64')
index_daily = index_daily.set_index(['ts_code', 'trade_date']).unstack(0)
colnames = index_daily.columns
colnames = [(tscode+'_'+index) for (index ,tscode) in colnames]
index_daily.columns = colnames
index_daily = index_daily.reset_index()
index_daily['trade_date'] = index_daily['trade_date'].dt.date

index_daily2 = pd.read_excel('./data/data.xlsx', sheet_name=0, index_col=False, header=0)
index_daily2['trade_date'] = index_daily2['trade_date'].dt.date
index_daily = index_daily.merge(index_daily2, how='outer', on='trade_date')
index_daily = index_daily.sort_values(by='trade_date', ascending=True)
index_daily.set_index('trade_date')
#index_daily = pd.concat([index_daily2, index_daily], axis=0)
index_daily.to_csv('./data/create_feature.csv')

#'000001.SH_pe',上证综指      '000005.SH_pe',商业指数
# '000006.SH_pe',地产指数     '000016.SH_pe', 上证50
#'000300.SH_pe', 沪深300      '000905.SH_pe', 中证500
# '399001.SZ_pe', 深圳成指    '399005.SZ_pe', 中小板指数
#'399006.SZ_pe', 创业板指     '399016.SZ_pe', 深证创新
#colnames = index_daily.columns
#a = index_daily[colnames[[colname[:12] in ('000001.SH_pe','000300.SH_pe','000905.SH_pe','399006.SZ_pe') for colname in colnames]]]
#colnames = index_daily2.columns
#b = index_daily2[colnames[[colname[:12] in ('000001.SH_pe','000300.SH_pe','000905.SH_pe','399006.SZ_pe') for colname in colnames]]]
#c = pd.concat([a, b], axis=1)
