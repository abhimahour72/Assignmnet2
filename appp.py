import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
st.title("Stocks App")
st.write("Here in this app, We can invest any amount of money in the NIfty 50 stocks equally and get the equity curve. We can also find out the top performing stocks by giving the time period to access their performance and invest in them and get the equity curve. Also, I have included the equity curve for the NIFTY50 index to compare. Using this app, we can access the Nifty 50 stocks and invest accordingly.")
sim_start = st.text_input('Please input the start date in format (yyyy-mm-dd)')
end_date = st.text_input('Please input the end date in format (yyyy-mm-dd)')
n_days_measure_perf = st.number_input(
    'Please enter the number of days to calculate performance of stocks', min_value=1, max_value=200, value=100, step=10)
top_n_stocks = st.number_input(
    'please enter the number of top stocks you want', min_value=1, max_value=50, value=10, step=1)
in_eq = st.number_input(
    'please enter the initial investment amount', min_value=10000, max_value=1000000000, value=1000000, step=10000)
print(sim_start)
print(end_date)
print(n_days_measure_perf)
print(top_n_stocks)
print(in_eq)
fg = pd.read_csv("data_stocks.csv", header=[
                 0, 1], index_col=0, skipinitialspace=True)
str_list = ["DIVISLAB.NS", "SUNPHARMA.NS", "CIPLA.NS", "TATASTEEL.NS", "ITC.NS", "INFY.NS", "HCLTECH.NS",
            "ONGC.NS", "NESTLEIND.NS", "HINDUNILVR.NS", "HEROMOTOCO.NS", "SBILIFE.NS", "TCS.NS", "MARUTI.NS",
            "BRITANNIA.NS", "WIPRO.NS", "DRREDDY.NS", "ASIANPAINT.NS", "BAJAJ-AUTO.NS", "ULTRACEMCO.NS",
            "TECHM.NS", "HDFCLIFE.NS", "BPCL.NS", "LT.NS", "JSWSTEEL.NS", "BHARTIARTL.NS", "COALINDIA.NS",
            "EICHERMOT.NS", "KOTAKBANK.NS", "RELIANCE.NS", "TATAMOTORS.NS", "TITAN.NS", "ICICIBANK.NS",
            "GRASIM.NS", "SHREECEM.NS", "HDFC.NS", "AXISBANK.NS", "HDFCBANK.NS", "TATACONSUM.NS", "BAJFINANCE.NS",
            "INDUSINDBK.NS", "M&M.NS", "BAJAJFINSV.NS", "SBIN.NS", "UPL.NS", "NTPC.NS", "ADANIPORTS.NS",
            "APOLLOHOSP.NS", "HINDALCO.NS", "POWERGRID.NS", "^NSEI"]
# fgh = fk.loc[sim_start:end_date]

# fk = fgh
st.text("Equity Curves")
fg = fg.drop('Equity Curve', axis=1)
fk = fg.loc[sim_start:end_date]
NSEI = fk.xs('NSEI', axis=1, level=1, drop_level=True)
fk.drop(labels='NSEI', axis=1, index=None, columns=None, level=1, inplace=True)
# for i in range(2,len(str_list)+2):
#   print(i)
#   fg.iloc[:,{i+(2*len(str_list))}] = 20000/fg.iloc[:,i].values[0]
# fg['Qty']
Amount1 = in_eq/50

for i in range(len(str_list)):
    # print(i)
    fk.iloc[:, {i+(2*(len(str_list)-1))}] = Amount1/fk.iloc[:, i].values[0]
    # print(fk.iloc[:,i].values[0])
    # print(fk.iloc[:,{(2*len(str_list))}])
    # if(20000/fk.iloc[:,i].values[0] != 0.0):
    # fg.iloc[:,{i+(2*len(str_list))}] = 0.0
dailyvalue = fk['close price']*fk['Qty']
# print(dailyvalue)
fk['Daily Value'] = dailyvalue
summ = fk['Daily Value'].sum(axis=1)
fk['Equity Curve'] = summ
for_volatility_eq = fk
len_equal = len(fk)/252
CAGR_equal = (((for_volatility_eq['Equity Curve'].tail(
    1).values[0]/for_volatility_eq['Equity Curve'].head(1).values[0])**len_equal)-1)*100
for_volatility_eq['for volatility'] = for_volatility_eq['Equity Curve'].shift(
    1)
for_volatility_eq['daily_return'] = (
    (for_volatility_eq['Equity Curve'] / for_volatility_eq['for volatility'])-1)

volatility_eq = ((for_volatility_eq['daily_return'].std())**(1/252))*100
sharpy_eq = (for_volatility_eq['daily_return'].mean(
)/for_volatility_eq['daily_return'].std())**(1/252)
all_eq = np.array([CAGR_equal, volatility_eq, sharpy_eq])
fg.drop(labels='NSEI', axis=1, index=None, columns=None, level=1, inplace=True)
fg2 = fg['close price'].loc[:sim_start]

fg2.drop(index=fg2.index[-1],
         axis=0,
         inplace=True)
closing2 = fg2
# closing2
# closing2 = closing2.droplevel(level=0,axis=1)
closing2 = closing2.tail(n_days_measure_perf)
fgff = (closing2.iloc[-1] / closing2.iloc[-100])-1
Dataa = fgff.sort_values(ascending=False)
# Dataa
Dataa = Dataa.head(top_n_stocks)
dddddd = Dataa.index.tolist()
dddddf = np.array(dddddd)
# Dataa
closing2 = closing2[closing2.columns.intersection(dddddd)]
# closing2
hjk = pd.DataFrame()
hjk2 = pd.DataFrame()
hjk3 = pd.DataFrame()
for i, word in enumerate(dddddd):
    #print(fk["open price", word])
    # print(fk["open price",word])
    hjk["open price", word] = fk["open price", word]
    hjk2["close price", word] = fk["close price", word]
    #hjk3["Qty", word] = np.ones(length2)


pdlist = [hjk, hjk2]
new_df = pd.concat(pdlist, axis=1)
new_df2 = new_df
# new_df2
Amount2 = in_eq/len(dddddf)
for i, word in enumerate(dddddf):
    new_df2['Qty', word] = Amount2/new_df.iloc[:, i].values[0]
    # print(fk.iloc[:,i].values[0])
for i, word in enumerate(dddddf):
    new_df2['Daily Value', word] = new_df2['Qty', word] * \
        new_df2['close price', word]
new_df2['Equity Curve'] = new_df2.iloc[:, -10:].sum(axis=1)
for_volatility = new_df2
len_perf = len(new_df2)/252
CAGR_perf = (((new_df2['Equity Curve'].tail(1).values[0] /
             new_df2['Equity Curve'].head(1).values[0])**len_perf)-1)*100
for_volatility['for volatility'] = for_volatility['Equity Curve'].shift(1)
for_volatility['daily_return'] = (
    (for_volatility['Equity Curve'] / for_volatility['for volatility'])-1)

volatility_perf = ((for_volatility['daily_return'].std())**(1/252))*100
sharpy_perf = (for_volatility['daily_return'].mean(
)/for_volatility['daily_return'].std())**(1/252)
all_perf = np.array([CAGR_perf, volatility_perf, sharpy_perf])
NSEI['Qty'] = in_eq/NSEI.iloc[:, 0].values[0]
NSEI['Daily Value'] = NSEI['close price']*NSEI['Qty']
NSEI['Equity_Curve'] = NSEI['Daily Value']
NSEI = NSEI[NSEI.Equity_Curve != 0]
#pd.set_option('display.max_rows', None)
nifty_vol = NSEI
len_nifty = len(nifty_vol)/252
CAGR_nifty = (((nifty_vol['Equity_Curve'].tail(
    1).values[0]/nifty_vol['Equity_Curve'].head(1).values[0])**len_nifty)-1)*100
nifty_vol['for volatility'] = nifty_vol['Equity_Curve'].shift(1)
nifty_vol['daily_return'] = (
    (nifty_vol['Equity_Curve'] / nifty_vol['for volatility'])-1)

volatility_nifty = ((nifty_vol['daily_return'].std())**(1/252))*100
sharpy_nifty = (nifty_vol['daily_return'].mean() /
                nifty_vol['daily_return'].std())**(1/252)
all_nifty = np.array([CAGR_nifty, volatility_nifty, sharpy_nifty])
all_stats = np.vstack((all_eq, all_nifty, all_perf))
all_stats_pd = pd.DataFrame(all_stats, index=[
                            'Equal_alloc_buy_hold', 'Nifty', 'performance_strat'], columns=['CAGR %', 'volatility %', 'sharpe'])
#dfffff = fk['Equity Curve']
#st.line_chart(fk['Equity Curve'], new_df2['Equity Curve'])
# NSEI['Equity_Curve'].plot(color='red')
# plt.legend(['Equal alloc buy hold', 'performance_strat', 'nifty'])
# plt.show()
print(all_stats_pd)
print(dddddf)
new_forgraph = pd.DataFrame()
new_forgraph['Equal_invest'] = fk['Equity Curve']
new_forgraph['perf'] = new_df2['Equity Curve']
new_forgraph['NSEI'] = NSEI['Equity_Curve']
st.line_chart(new_forgraph)
st.text('Stats')
st.write(all_stats_pd)
st.text('Top selected stocks')
st.write(dddddf)


sim_start = st.text_input('Please input the start date in format (yyyy-mm-dd)')
end_date = st.text_input('Please input the end date in format (yyyy-mm-dd)')
n_days_measure_perf = st.number_input(
    'Please enter the number of days to calculate performance of stocks', min_value=1, max_value=200, value=100, step=10)
top_n_stocks = st.number_input(
    'please enter the number of top stocks you want', min_value=1, max_value=50, value=10, step=1)
in_eq = st.number_input(
    'please enter the initial investment amount', min_value=10000, max_value=1000000000, value=1000000, step=10000)
print(sim_start)
print(end_date)
print(n_days_measure_perf)
print(top_n_stocks)
print(in_eq)
fg = pd.read_csv("data_stocks.csv", header=[
                 0, 1], index_col=0, skipinitialspace=True)
str_list = ["DIVISLAB.NS", "SUNPHARMA.NS", "CIPLA.NS", "TATASTEEL.NS", "ITC.NS", "INFY.NS", "HCLTECH.NS",
            "ONGC.NS", "NESTLEIND.NS", "HINDUNILVR.NS", "HEROMOTOCO.NS", "SBILIFE.NS", "TCS.NS", "MARUTI.NS",
            "BRITANNIA.NS", "WIPRO.NS", "DRREDDY.NS", "ASIANPAINT.NS", "BAJAJ-AUTO.NS", "ULTRACEMCO.NS",
            "TECHM.NS", "HDFCLIFE.NS", "BPCL.NS", "LT.NS", "JSWSTEEL.NS", "BHARTIARTL.NS", "COALINDIA.NS",
            "EICHERMOT.NS", "KOTAKBANK.NS", "RELIANCE.NS", "TATAMOTORS.NS", "TITAN.NS", "ICICIBANK.NS",
            "GRASIM.NS", "SHREECEM.NS", "HDFC.NS", "AXISBANK.NS", "HDFCBANK.NS", "TATACONSUM.NS", "BAJFINANCE.NS",
            "INDUSINDBK.NS", "M&M.NS", "BAJAJFINSV.NS", "SBIN.NS", "UPL.NS", "NTPC.NS", "ADANIPORTS.NS",
            "APOLLOHOSP.NS", "HINDALCO.NS", "POWERGRID.NS", "^NSEI"]
# fgh = fk.loc[sim_start:end_date]

# fk = fgh
st.text("Equity Curves")
fg = fg.drop('Equity Curve', axis=1)
fk = fg.loc[sim_start:end_date]
NSEI = fk.xs('NSEI', axis=1, level=1, drop_level=True)
fk.drop(labels='NSEI', axis=1, index=None, columns=None, level=1, inplace=True)
# for i in range(2,len(str_list)+2):
#   print(i)
#   fg.iloc[:,{i+(2*len(str_list))}] = 20000/fg.iloc[:,i].values[0]
# fg['Qty']
Amount1 = in_eq/50

for i in range(len(str_list)):
    # print(i)
    fk.iloc[:, {i+(2*(len(str_list)-1))}] = Amount1/fk.iloc[:, i].values[0]
    # print(fk.iloc[:,i].values[0])
    # print(fk.iloc[:,{(2*len(str_list))}])
    # if(20000/fk.iloc[:,i].values[0] != 0.0):
    # fg.iloc[:,{i+(2*len(str_list))}] = 0.0
dailyvalue = fk['close price']*fk['Qty']
# print(dailyvalue)
fk['Daily Value'] = dailyvalue
summ = fk['Daily Value'].sum(axis=1)
fk['Equity Curve'] = summ
for_volatility_eq = fk
len_equal = len(fk)/252
CAGR_equal = (((for_volatility_eq['Equity Curve'].tail(
    1).values[0]/for_volatility_eq['Equity Curve'].head(1).values[0])**len_equal)-1)*100
for_volatility_eq['for volatility'] = for_volatility_eq['Equity Curve'].shift(
    1)
for_volatility_eq['daily_return'] = (
    (for_volatility_eq['Equity Curve'] / for_volatility_eq['for volatility'])-1)

volatility_eq = ((for_volatility_eq['daily_return'].std())**(1/252))*100
sharpy_eq = (for_volatility_eq['daily_return'].mean(
)/for_volatility_eq['daily_return'].std())**(1/252)
all_eq = np.array([CAGR_equal, volatility_eq, sharpy_eq])
fg.drop(labels='NSEI', axis=1, index=None, columns=None, level=1, inplace=True)
fg2 = fg['close price'].loc[:sim_start]

fg2.drop(index=fg2.index[-1],
         axis=0,
         inplace=True)
closing2 = fg2
# closing2
# closing2 = closing2.droplevel(level=0,axis=1)
closing2 = closing2.tail(n_days_measure_perf)
fgff = (closing2.iloc[-1] / closing2.iloc[-100])-1
Dataa = fgff.sort_values(ascending=False)
# Dataa
Dataa = Dataa.head(top_n_stocks)
dddddd = Dataa.index.tolist()
dddddf = np.array(dddddd)
# Dataa
closing2 = closing2[closing2.columns.intersection(dddddd)]
# closing2
hjk = pd.DataFrame()
hjk2 = pd.DataFrame()
hjk3 = pd.DataFrame()
for i, word in enumerate(dddddd):
    #print(fk["open price", word])
    # print(fk["open price",word])
    hjk["open price", word] = fk["open price", word]
    hjk2["close price", word] = fk["close price", word]
    #hjk3["Qty", word] = np.ones(length2)


pdlist = [hjk, hjk2]
new_df = pd.concat(pdlist, axis=1)
new_df2 = new_df
# new_df2
Amount2 = in_eq/len(dddddf)
for i, word in enumerate(dddddf):
    new_df2['Qty', word] = Amount2/new_df.iloc[:, i].values[0]
    # print(fk.iloc[:,i].values[0])
for i, word in enumerate(dddddf):
    new_df2['Daily Value', word] = new_df2['Qty', word] * \
        new_df2['close price', word]
new_df2['Equity Curve'] = new_df2.iloc[:, -10:].sum(axis=1)
for_volatility = new_df2
len_perf = len(new_df2)/252
CAGR_perf = (((new_df2['Equity Curve'].tail(1).values[0] /
             new_df2['Equity Curve'].head(1).values[0])**len_perf)-1)*100
for_volatility['for volatility'] = for_volatility['Equity Curve'].shift(1)
for_volatility['daily_return'] = (
    (for_volatility['Equity Curve'] / for_volatility['for volatility'])-1)

volatility_perf = ((for_volatility['daily_return'].std())**(1/252))*100
sharpy_perf = (for_volatility['daily_return'].mean(
)/for_volatility['daily_return'].std())**(1/252)
all_perf = np.array([CAGR_perf, volatility_perf, sharpy_perf])
NSEI['Qty'] = in_eq/NSEI.iloc[:, 0].values[0]
NSEI['Daily Value'] = NSEI['close price']*NSEI['Qty']
NSEI['Equity_Curve'] = NSEI['Daily Value']
NSEI = NSEI[NSEI.Equity_Curve != 0]
#pd.set_option('display.max_rows', None)
nifty_vol = NSEI
len_nifty = len(nifty_vol)/252
CAGR_nifty = (((nifty_vol['Equity_Curve'].tail(
    1).values[0]/nifty_vol['Equity_Curve'].head(1).values[0])**len_nifty)-1)*100
nifty_vol['for volatility'] = nifty_vol['Equity_Curve'].shift(1)
nifty_vol['daily_return'] = (
    (nifty_vol['Equity_Curve'] / nifty_vol['for volatility'])-1)

volatility_nifty = ((nifty_vol['daily_return'].std())**(1/252))*100
sharpy_nifty = (nifty_vol['daily_return'].mean() /
                nifty_vol['daily_return'].std())**(1/252)
all_nifty = np.array([CAGR_nifty, volatility_nifty, sharpy_nifty])
all_stats = np.vstack((all_eq, all_nifty, all_perf))
all_stats_pd = pd.DataFrame(all_stats, index=[
                            'Equal_alloc_buy_hold', 'Nifty', 'performance_strat'], columns=['CAGR %', 'volatility %', 'sharpe'])
#dfffff = fk['Equity Curve']
#st.line_chart(fk['Equity Curve'], new_df2['Equity Curve'])
# NSEI['Equity_Curve'].plot(color='red')
# plt.legend(['Equal alloc buy hold', 'performance_strat', 'nifty'])
# plt.show()
print(all_stats_pd)
print(dddddf)
new_forgraph = pd.DataFrame()
new_forgraph['Equal_invest'] = fk['Equity Curve']
new_forgraph['perf'] = new_df2['Equity Curve']
new_forgraph['NSEI'] = NSEI['Equity_Curve']
st.line_chart(new_forgraph)
st.text('Stats')
st.write(all_stats_pd)
st.text('Top selected stocks')
st.write(dddddf)
