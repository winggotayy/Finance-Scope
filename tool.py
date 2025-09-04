import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from matplotlib.ticker import MaxNLocator
#from prettytable import PrettyTable
#from blessed import Terminal
import time
from datetime import datetime, timedelta
import numpy as np
import mplfinance as mpf

from typing import Optional
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from typing import Union, Any
from sklearn.linear_model import LinearRegression
import seaborn as sns
from typing import List


# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

font_path = './fonts/SimHei.ttf'
font_prop = fm.FontProperties(fname=font_path)


tushare_token = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(tushare_token)



def get_last_year_date(date_str: str = '') -> str:
    """
        This function takes a date string in the format YYYYMMDD and returns the date string one year prior to the input date.

        Args:
        - date_str: string, the input date in the format YYYYMMDD

        Returns:
        - string, the date one year prior to the input date in the format YYYYMMDD
        """
    dt = datetime.strptime(date_str, '%Y%m%d')
    # To calculate the date one year ago
    one_year_ago = dt - timedelta(days=365)

    # To format the date as a string
    one_year_ago_str = one_year_ago.strftime('%Y%m%d')

    return one_year_ago_str


def get_adj_factor(stock_code: str = '', start_date: str = '', end_date: str = '') -> pd.DataFrame:
    # Get stock price adjustment factors. Retrieve the stock price adjustment factors for a single stock's entire historical data or for all stocks on a single trading day.
    # The input includes the stock code, start date, end date, and trading date, all in string format with the date in the YYYYMMDD format
    # The return value is a dataframe containing the stock code, trading date, and adjustment factor
    # ts_code	str	股票代码
    # adj_factor	float	复权因子
    """
       This function retrieves the adjusted stock prices for a given stock code and date range.

       Args:
       - stock_code: string, the stock code to retrieve data for
       - start_date: string, the start date in the format YYYYMMDD
       - end_date: string, the end date in the format YYYYMMDD

       Returns:
       - dataframe, a dataframe containing the stock code, trade date, and adjusted factor

       This will retrieve the adjusted stock prices for the stock with code '000001.SZ' between the dates '20220101' and '20220501'.
       """
    df = pro.adj_factor(**{
        "ts_code": stock_code,
        "trade_date": "",
        "start_date": start_date,
        "end_date": end_date,
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "adj_factor"
    ])

    return df

def get_stock_code(stock_name: str) -> str:
    # Retrieve the stock code of a given stock name. If we call get_stock_code('贵州茅台'), it will return '600519.SH'.


    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    try:
        code = df.loc[df.name==stock_name].ts_code.iloc[0]
        return code
    except:
        return None




def get_stock_name_from_code(stock_code: str) -> str:
    """
        Reads a local file to retrieve the stock name from a given stock code.

        Args:
        - stock_code (str): The code of the stock.

        Returns:
        - str: The stock name of the given stock code.
        """
    # For example,if we call get_stock_name_from_code('600519.SH'), it will return '贵州茅台'.


    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    name = df.loc[df.ts_code == stock_code].name.iloc[0]

    return name

def get_stock_prices_data(stock_name: str='', start_date: str='', end_date: str='', freq:str='daily') -> pd.DataFrame:
    """
        Retrieves the daily/weekly/monthly price data for a given stock code during a specific time period. get_stock_prices_data('贵州茅台','20200120','20220222','daily')

        Args:
        - stock_name (str)
        - start_date (str): The start date in the format 'YYYYMMDD'.
        - end_date (str): The end date in 'YYYYMMDD'.
        - freq (str): The frequency of the price data, can be 'daily', 'weekly', or 'monthly'.

        Returns:
        - pd.DataFrame: A dataframe that contains the daily/weekly/monthly data. The output columns contain stock_code, trade_date, open, high, low, close, pre_close(昨天收盘价), change(涨跌额), pct_chg(涨跌幅),vol(成交量),amount(成交额)
        """

    stock_code = get_stock_code(stock_name)

    if freq == 'daily':
        stock_data = pro.daily(**{
            "ts_code": stock_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "offset": "",
            "limit": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])

    elif freq == 'weekly':
        stock_data = pro.weekly(**{
            "ts_code": stock_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
    elif freq == 'monthly':
        stock_data = pro.monthly(**{
            "ts_code": stock_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])


    adj_f = get_adj_factor(stock_code, start_date, end_date)
    stock_data = pd.merge(stock_data, adj_f, on=['ts_code', 'trade_date'])
    # Multiply the values of open, high, low, and close by their corresponding adjustment factors.
    # To obtain the adjusted close price
    stock_data[['open', 'high', 'low', 'close']] *= stock_data['adj_factor'].values.reshape(-1, 1)

    #stock_data.rename(columns={'vol': 'volume'}, inplace=True)
    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    stock_data_merged = pd.merge(stock_data, df, on='ts_code')
    stock_data_merged.rename(columns={'ts_code': 'stock_code'}, inplace=True)
    stock_data_merged.rename(columns={'name': 'stock_name'}, inplace=True)
    stock_data_merged = stock_data_merged.sort_values(by='trade_date', ascending=True)  # To sort the DataFrame by date in ascending order
    return stock_data_merged



def get_stock_technical_data(stock_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
        Retrieves the daily technical data of a stock including macd turnover rate, volume, PE ratio, etc. Those technical indicators are usually plotted as subplots in a k-line chart.

        Args:
            stock_name (str):
            start_date (str): Start date "YYYYMMDD"
            end_date (str): End date "YYYYMMDD"

        Returns:
            pd.DataFrame: A DataFrame containing the technical data of the stock,
            including various indicators such as ts_code, trade_date, close, macd_dif, macd_dea, macd, kdj_k, kdj_d, kdj_j, rsi_6, rsi_12, boll_upper, boll_mid, boll_lower, cci, turnover_rate, turnover_rate_f, volume_ratio, pe_ttm(市盈率), pb(市净率), ps_ttm, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv

    """

    # Technical factors
    stock_code = get_stock_code(stock_name)
    stock_data1 = pro.stk_factor(**{
        "ts_code": stock_code,
        "start_date": start_date,
        "end_date": end_date,
        "trade_date": '',
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "close",
        "macd_dif",
        "macd_dea",
        "macd",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "rsi_6",
        "rsi_12",
        "rsi_24",
        "boll_upper",
        "boll_mid",
        "boll_lower",
        "cci"
    ])
    # Trading factors
    stock_data2 = pro.daily_basic(**{
        "ts_code": stock_code,
        "trade_date": '',
        "start_date": start_date,
        "end_date": end_date,
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",  #
        "trade_date",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv"
    ])

    #
    stock_data = pd.merge(stock_data1, stock_data2, on=['ts_code', 'trade_date'])
    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    stock_data_merged = pd.merge(stock_data, df, on='ts_code')
    stock_data_merged = stock_data_merged.sort_values(by='trade_date', ascending=True)

    stock_data_merged.drop(['symbol'], axis=1, inplace=True)

    stock_data_merged.rename(columns={'ts_code': 'stock_code'}, inplace=True)
    stock_data_merged.rename(columns={'name': 'stock_name'}, inplace=True)

    return stock_data_merged




def plot_stock_data(stock_data: pd.DataFrame, ax: Optional[plt.Axes] = None, figure_type: str = 'line', title_name: str ='') -> plt.Axes:

    """
    This function plots stock data.

    Args:
    - stock_data: pandas DataFrame, the stock data to plot. The DataFrame should contain three columns:
        - Column 1: trade date in 'YYYYMMDD'
        - Column 2: Stock name or code (string format)
        - Column 3: Index value (numeric format)
        The DataFrame can be time series data or cross-sectional data. If it is time-series data, the first column represents different trade time, the second column represents the same name. For cross-sectional data, the first column is the same, the second column contains different stocks.

    - ax: matplotlib Axes object, the axes to plot the data on
    - figure_type: the type of figure (either 'line' or 'bar')
    - title_name

    Returns:
    - matplotlib Axes object, the axes containing the plot
    """

    index_name = stock_data.columns[2]
    name_list = stock_data.iloc[:,1]
    date_list = stock_data.iloc[:,0]
    if name_list.nunique() == 1 and date_list.nunique() != 1:
        # Time Series Data
        unchanged_var = name_list.iloc[0]   # stock name
        x_dim = date_list                   # tradingdate
        x_name = stock_data.columns[0]

    elif name_list.nunique() != 1 and date_list.nunique() == 1:
        # Cross-sectional Data
        unchanged_var = date_list.iloc[0]    # tradingdate
        x_dim = name_list                    # stock name
        x_name = stock_data.columns[1]

        data_size = x_dim.shape[0]



    start_x_dim, end_x_dim = x_dim.iloc[0], x_dim.iloc[-1]

    start_y = stock_data.iloc[0, 2]
    end_y = stock_data.iloc[-1, 2]


    def generate_random_color():
        r = random.randint(0, 255)/ 255.0
        g = random.randint(0, 100)/ 255.0
        b = random.randint(0, 255)/ 255.0
        return (r, g, b)

    color = generate_random_color()
    if ax is None:
        _, ax = plt.subplots()

    if figure_type =='line':
        #

        ax.plot(x_dim, stock_data.iloc[:, 2], label = unchanged_var+'_' + index_name, color=color,linewidth=3)
        #
        plt.scatter(x_dim, stock_data.iloc[:, 2], color=color,s=3)  # Add markers to the data points

        #
        #ax.scatter(x_dim, stock_data.iloc[:, 2],label = unchanged_var+'_' + index_name, color=color, s=3)
        #

        ax.annotate(unchanged_var + ':' + str(round(start_y, 2)) + ' @' + start_x_dim, xy=(start_x_dim, start_y),
                    xytext=(start_x_dim, start_y),
                    textcoords='data', fontsize=14,color=color, horizontalalignment='right',fontproperties=font_prop)

        ax.annotate(unchanged_var + ':' + str(round(end_y, 2)) +' @' + end_x_dim, xy=(end_x_dim, end_y),
                    xytext=(end_x_dim, end_y),
                    textcoords='data', fontsize=14, color=color, horizontalalignment='left',fontproperties=font_prop)


    elif figure_type == 'bar':
        ax.bar(x_dim, stock_data.iloc[:, 2], label = unchanged_var + '_' + index_name, width=0.3, color=color)
        ax.annotate(unchanged_var + ':' + str(round(start_y, 2)) + ' @' + start_x_dim, xy=(start_x_dim, start_y),
                    xytext=(start_x_dim, start_y),
                    textcoords='data', fontsize=14, color=color, horizontalalignment='right',fontproperties=font_prop)

        ax.annotate(unchanged_var + ':' + str(round(end_y, 2)) + ' @' + end_x_dim, xy=(end_x_dim, end_y),
                    xytext=(end_x_dim, end_y),
                    textcoords='data', fontsize=14, color=color, horizontalalignment='left',fontproperties=font_prop)

    plt.xticks(x_dim,rotation=45)                                                  #
    ax.xaxis.set_major_locator(MaxNLocator( integer=True, prune=None, nbins=100))  #


    plt.xlabel(x_name, fontproperties=font_prop,fontsize=18)
    plt.ylabel(f'{index_name}', fontproperties=font_prop,fontsize=16)
    ax.set_title(title_name , fontproperties=font_prop,fontsize=16)
    plt.legend(prop=font_prop)  # 显示图例
    fig = plt.gcf()
    fig.set_size_inches(18, 12)

    return ax

# add: 柱状图
def plot_bar_chart(stock_data: pd.DataFrame, ax: Optional[plt.Axes] = None, title_name: str = '') -> plt.Axes:
    """
    绘制柱状图
    :param stock_data: pandas DataFrame，股票数据
    :param ax: matplotlib Axes对象
    :param title_name: 图表标题
    :return: matplotlib Axes对象
    """
    index_name = stock_data.columns[2]
    name_list = stock_data.iloc[:, 1]
    date_list = stock_data.iloc[:, 0]

    if name_list.nunique() == 1 and date_list.nunique() != 1:
        # 时间序列数据
        unchanged_var = name_list.iloc[0]
        x_dim = date_list
        x_name = stock_data.columns[0]
    elif name_list.nunique() != 1 and date_list.nunique() == 1:
        # 横截面数据
        unchanged_var = date_list.iloc[0]
        x_dim = name_list
        x_name = stock_data.columns[1]

    start_x_dim, end_x_dim = x_dim.iloc[0], x_dim.iloc[-1]
    start_y = stock_data.iloc[0, 2]
    end_y = stock_data.iloc[-1, 2]

    def generate_random_color():
        r = random.randint(0, 255) / 255.0
        g = random.randint(0, 100) / 255.0
        b = random.randint(0, 255) / 255.0
        return (r, g, b)

    color = generate_random_color()
    if ax is None:
        _, ax = plt.subplots()

    ax.bar(x_dim, stock_data.iloc[:, 2], label=f'{unchanged_var}_{index_name}', width=0.3, color=color)

    ax.annotate(f'{unchanged_var}: {round(start_y, 2)} @ {start_x_dim}', xy=(start_x_dim, start_y),
                xytext=(start_x_dim, start_y),
                textcoords='data', fontsize=14, color=color, horizontalalignment='right', fontproperties=font_prop)

    ax.annotate(f'{unchanged_var}: {round(end_y, 2)} @ {end_x_dim}', xy=(end_x_dim, end_y),
                xytext=(end_x_dim, end_y),
                textcoords='data', fontsize=14, color=color, horizontalalignment='left', fontproperties=font_prop)

    plt.xticks(x_dim, rotation=45)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune=None, nbins=100))

    plt.xlabel(x_name, fontproperties=font_prop, fontsize=18)
    plt.ylabel(index_name, fontproperties=font_prop, fontsize=16)
    ax.set_title(title_name, fontproperties=font_prop, fontsize=16)
    plt.legend(prop=font_prop)
    fig = plt.gcf()
    fig.set_size_inches(18, 12)

    return ax

# add: 饼图
def plot_pie_chart(data: pd.DataFrame,
                   labels_col: str,
                   values_col: str,
                   ax: Optional[plt.Axes] = None,
                   title: str = '') -> plt.Axes:
    """
    Plot a pie chart with improved readability.
    
    Args:
        data (pd.DataFrame): DataFrame containing labels and values
        labels_col (str): Column name for labels
        values_col (str): Column name for values
        ax (plt.Axes, optional): Matplotlib axes
        title (str): Chart title
    
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))  # 增加图表尺寸

    # 计算小于3%的切片
    # small_slices = data[values_col] < 3
    explode = [0.1 if val < 3 else 0 for val in data[values_col]]

    # 绘制饼图，将百分比标签放到饼图外面
    wedges, texts, autotexts = ax.pie(
        data[values_col],
        labels=[''] * len(data),  # 空标签
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=1.2,    # 将百分比标签移到饼图外面
        radius=0.7,         # 减小饼图半径以留出更多空间
        labeldistance=1.1,   # 标签距离
        explode=explode
    )
    
    # 设置百分比标签的样式
    plt.setp(autotexts, size=10, weight="bold")
    
    # 添加图例，使用实际的标签
    legend_title = labels_col.replace('_', ' ').title()  # 将列名转换为标题格式
    legend = ax.legend(
        wedges,
        data[labels_col],
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9
    )
    
    # 设置图例标题的字体大小
    legend.get_title().set_fontsize(12)
    legend.get_title().set_weight('bold')
    
    # 设置主标题，增加与饼图的距离
    ax.set_title(title, fontproperties=font_prop, pad=40, fontsize=14, weight='bold', y=1.05)

    # 保持饼图为圆形
    ax.axis('equal')
    
    # 调整布局以适应图例和标题
    plt.tight_layout()
    
    return ax

def query_fund_Manager(Manager_name: str) -> pd.DataFrame:
    # 代码fund_code,公告日期ann_date,基金经理名字name,性别gender,出生年份birth_year,学历edu,国籍nationality,开始管理日期begin_date,结束日期end_date,简历resume
    """
        Retrieves information about a fund manager.

        Args:
            Manager_name (str): The name of the fund manager.

        Returns:
            df (DataFrame): A DataFrame containing the fund manager's information, including the fund codes, announcement dates,
                            manager's name, gender, birth year, education, nationality, start and end dates of managing funds,
                            and the manager's resume.
    """

    df = pro.fund_manager(**{
        "ts_code": "",
        "ann_date": "",
        "name": Manager_name,
        "offset": "",
        "limit": ""
    }, fields=[
        "ts_code",
        "ann_date",
        "name",
        "gender",
        "birth_year",
        "edu",
        "nationality",
        "begin_date",
        "end_date",
        "resume"
    ])
    #
    df.rename(columns={'ts_code': 'fund_code'}, inplace=True)
    # To query the fund name based on the fund code and store it in a new column called fund_name, while removing the rows where the fund name is not found
    df['fund_name'] = df['fund_code'].apply(lambda x: query_fund_name_or_code('', x))
    df.dropna(subset=['fund_name'], inplace=True)
    df.rename(columns={'name': 'manager_name'}, inplace=True)
    #
    df_out = df[['fund_name','fund_code','ann_date','manager_name','begin_date','end_date']]

    return df_out

def plot_scatter(data: pd.DataFrame,
                x_col: str,
                y_col: str,
                hue_col: Optional[str] = None,
                title: str = '',
                xlabel: str = '',
                ylabel: str = '',
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    # debug
    print('Debug: data.columns', data.columns)
    """
    创建散点图，用于比较两个变量之间的关系。

    Args:
        data: 包含数据的DataFrame
        x_col: x轴的列名
        y_col: y轴的列名
        hue_col: 可选的颜色分类列名
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        ax: 可选的matplotlib Axes对象

    Returns:
        matplotlib Axes对象
    """
    # 替换data的列名
    if x_col == 'close_price':
        x_col = 'close'
    if y_col == 'volume':
        y_col = 'vol'
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建散点图
    if hue_col is not None:
        sns.scatterplot(data=data,
                       x=x_col,
                       y=y_col,
                       hue=hue_col,
                       palette='viridis',
                       s=100,
                       alpha=0.7,
                       ax=ax)
    else:
        sns.scatterplot(data=data,
                       x=x_col,
                       y=y_col,
                       s=100,
                       alpha=0.7,
                       ax=ax)
    
    # 设置图表属性
    ax.set_title(title, fontproperties=font_prop, fontsize=14)
    ax.set_xlabel(xlabel, fontproperties=font_prop, fontsize=12)
    ax.set_ylabel(ylabel, fontproperties=font_prop, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 如果指定了hue_col，添加颜色条
    if hue_col is not None:
        norm = plt.Normalize(data[hue_col].min(), data[hue_col].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label=hue_col)
    
    plt.tight_layout()
    return ax

def test_stock_scatter_plot():
    """
    测试股票价格和成交量的散点图
    """
    # 获取贵州茅台2024年的数据
    stock_name = '贵州茅台'
    start_date = '20240101'
    end_date = '20240331'  # 可以根据需要调整日期
    
    # 获取股票数据
    stock_data = get_stock_prices_data(stock_name, start_date, end_date, 'daily')
    
    # 创建散点图
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_scatter(
        data=stock_data,
        x_col='close',  # 收盘价
        y_col='vol',    # 成交量
        title=f'{stock_name}股价与成交量关系图',
        xlabel='收盘价（元）',
        ylabel='成交量（手）',
        ax=ax
    )
    
    plt.show()



def plot_field_distribution(field_name: str, top_k: int = 10):
    """
    绘制A股某字段分布饼图
    
    Args:
        field_name: 字段名称, 例如'industry'
        top_k: 前k个字段

    Returns:
        DataFrame with columns:
        - field_name: 行业名称
        - count: 数量
    """
    # ================== 示例1：A股行业分布 ==================
    df = pro.stock_basic(exchange='', list_status='L', fields=f'ts_code,{field_name}')
    df_dist = df[field_name].value_counts().head(top_k)
    return df_dist

def fund_type_basedon_market(market: str = '', top_k: int = None) -> pd.DataFrame:
    """
    分析各市场类型的股票分布情况
    
    Args:
        market: 市场类型, 可选值为:
                - E: 场内
                - O: 场外
        top_k: 展示前k个类型,默认为None表示展示所有
    
    Returns:
        DataFrame with columns:
        - fund_type: 股票类型
        - count: 数量
        - ratio: 占比(%)
    """
    df = pro.fund_basic(**{
        "ts_code": "",
        "market": market,
        "update_flag": "",
        "offset": "",
        "limit": "",
        "status": "",
        "name": ""
    }, fields=[
        "ts_code",
        "name",
        "management",
        "custodian",
        "fund_type",
        "found_date",
        "due_date",
        "list_date",
        "issue_date",
        "delist_date",
        "issue_amount",
        "m_fee",
        "c_fee",
        "duration_year",
        "p_value",
        "min_amount",
        "exp_return",
        "benchmark",
        "status",
        "invest_type",
        "type",
        "trustee",
        "purc_startdate",
        "redm_startdate",
        "market"
    ])
    # 把market列作转换，场内转为E，场外转为O
    df['market'] = df['market'].map({'场内': 'E', '场外': 'O'})
    # 如果df为空则返回空DataFrame
    if df is None or df.empty:
        return pd.DataFrame(columns=['fund_type', 'count', 'ratio'])
        
    # 统计各基金类型的数量
    type_counts = df['fund_type'].value_counts()
    
    # 计算百分比
    type_ratio = (type_counts / len(df) * 100).round(2)
    
    # 构建结果DataFrame
    result_df = pd.DataFrame({
        'fund_name': type_counts.index,
        'count': type_counts.values,
        'fund_ratio': type_ratio.values
    })
    
    # 如果指定了top_k,只取前k个
    if top_k is not None:
        result_df = result_df.head(top_k)
        
    return result_df
    
# add: 获取上市公司前十大股东数据，包括持有数量和比例等信息
def get_top10_holders(fund_code: str, start_date: str = '', end_date: str = '') -> pd.DataFrame:
    """
    获取上市公司前十大股东数据，包括持有数量和比例等信息
    
    Args:
        fund_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    Returns:
        DataFrame with columns:
        - holder_name: 股东名称
        - hold_ratio: 持股比例(%)
        - hold_amount: 持股数量(股)
    """
    try:
        df = pro.top10_holders(**{
            "ts_code": fund_code,
            "period": "",
            "ann_date": "",
            "start_date": "",
            "end_date": ""
        }, fields=[
            "ts_code",
            "ann_date",
            "end_date",	
            "holder_name",	
            "hold_amount",	
            "hold_ratio",	
            "hold_float_ratio",
            "hold_change",	
            "holder_type"	
        ])

        if df is None or df.empty:
            # 如果找不到股票代码，则说明是股票名称
            df_stock_basic = pd.read_csv("tushare_stock_basic_20230421210721.csv")
            # print("df_stock_basic:",df_stock_basic)
            ts_code = df_stock_basic[df_stock_basic['name'] == fund_code]['ts_code']
            if not ts_code.empty:
                return get_top10_holders(ts_code.values[0])
            print(f"Warning: No data found for stock {fund_code}")
            return pd.DataFrame(columns=['holder_name', 'hold_ratio', 'hold_amount'])
        
        # 将holder_name进行去重
        df = df.drop_duplicates(subset=['holder_name'])

        # 重命名列以更好地支持饼图展示
        df = df.rename(columns={
            'holder_name': 'holder_name',
            'hold_ratio': 'hold_ratio',
            'hold_amount': 'hold_amount'
        })

        # 只展示上面的3列
        df = df[['holder_name', 'hold_ratio', 'hold_amount']]

        # 确保持股比例是数值类型
        df['hold_ratio'] = pd.to_numeric(df['hold_ratio'], errors='coerce')
        
        # 按持股比例降序排序
        df = df.sort_values('hold_ratio', ascending=False)
        
        # 只保留前10大股东
        df = df.head(10)
        
        return df
        
    except Exception as e:
        print(f"Error getting top 10 holders for {fund_code}: {str(e)}")
        return pd.DataFrame(columns=['holder_name', 'hold_ratio', 'hold_amount'])

# def save_stock_prices_to_csv(stock_prices: pd.DataFrame, stock_name: str, file_path: str) -> None:
#
#     """
#         Saves the price data of a specific stock symbol during a specific time period to a local CSV file.
#
#         Args:
#         - stock_prices (pd.DataFrame): A pandas dataframe that contains the daily price data for the given stock symbol during the specified time period.
#         - stock_name (str): The name of the stock.
#         - file_path (str): The file path where the CSV file will be saved.
#
#         Returns:
#         - None: The function only saves the CSV file to the specified file path.
#     """
#     # The function checks if the directory to save the CSV file exists and creates it if it does not exist.
#     # The function then saves the price data of the specified stock symbol during the specified time period to a local CSV file with the name {stock_name}_price_data.csv in the specified file path.
#
#
#     if not os.path.exists(file_path):
#         os.makedirs(file_path)
#
#
#     file_path = f"{file_path}{stock_name}_stock_prices.csv"
#     stock_prices.to_csv(file_path, index_label='Date')
#     print(f"Stock prices for {stock_name} saved to {file_path}")


def calculate_stock_index(stock_data: pd.DataFrame, index:str='close') -> pd.DataFrame:
    """
        Calculate a specific index of a stock based on its price information.

        Args:
            stock_data (pd.DataFrame): DataFrame containing the stock's price information.
            index (str, optional): The index to calculate. The available options depend on the column names in the
                input stock price data. Additionally, there are two special indices: 'candle_K' and 'Cumulative_Earnings_Rate'.

        Returns:
            DataFrame containing the corresponding index data of the stock. In general, it includes three columns: 'trade_date', 'name', and the corresponding index value.
            Besides, if index is 'candle_K', the function returns the DataFrame containing 'trade_date', 'Open', 'High', 'Low', 'Close', 'Volume','name' column.
            If index is a technical index such as 'macd' or a trading index likes 'pe_ttm', the function returns the DataFrame with corresponding columns.
        """


    if 'stock_name' not in  stock_data.columns and 'index_name' in stock_data.columns:
        stock_data.rename(columns={'index_name': 'stock_name'}, inplace=True)
    #
    index = index.lower()
    if index=='Cumulative_Earnings_Rate' or index =='Cumulative_Earnings_Rate'.lower() :
        stock_data[index] = (1 + stock_data['pct_chg'] / 100.).cumprod() - 1.
        stock_data[index] = stock_data[index] * 100.
        if 'stock_name' in stock_data.columns :
           selected_index = stock_data[['trade_date', 'stock_name', index]].copy()
        #
        if 'fund_name' in stock_data.columns:
            selected_index = stock_data[['trade_date', 'fund_name', index]].copy()
        return selected_index

    elif index == 'candle_K' or index == 'candle_K'.lower():
        #tech_df = tech_df.drop(['name', 'symbol', 'industry', 'area','market','list_date','ts_code','close'], axis=1)
        # Merge two DataFrames based on the 'trade_date' column.

        stock_data = stock_data.rename(
            columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                     'vol': 'Volume'})
        selected_index = stock_data[['trade_date', 'Open', 'High', 'Low', 'Close', 'Volume','stock_name']].copy()
        return selected_index

    elif index =='macd':
        selected_index = stock_data[['trade_date','macd','macd_dea','macd_dif']].copy()
        return selected_index

    elif index =='rsi':
        selected_index = stock_data[['trade_date','rsi_6','rsi_12']].copy()
        return selected_index

    elif index =='boll':
        selected_index = stock_data[['trade_date', 'boll_upper', 'boll_lower','boll_mid']].copy()
        return selected_index

    elif index =='kdj':
        selected_index = stock_data[['trade_date', 'kdj_k', 'kdj_d','kdj_j']].copy()
        return selected_index

    elif index =='cci':
        selected_index = stock_data[['trade_date', 'cci']].copy()
        return selected_index

    elif index == '换手率':
        selected_index = stock_data[['trade_date', 'turnover_rate','turnover_rate_f']].copy()
        return selected_index

    elif index == '市值':
        selected_index = stock_data[['trade_date', 'total_mv','circ_mv']].copy()
        return selected_index


    elif index in stock_data.columns:
        stock_data = stock_data

        if 'stock_name' in stock_data.columns :
           selected_index = stock_data[['trade_date', 'stock_name', index]].copy()

        if 'fund_name' in stock_data.columns:
            selected_index = stock_data[['trade_date', 'fund_name', index]].copy()
        # Except for candlestick chart and technical indicators, the remaining outputs consist of three columns: date, name, and indicator.
        return selected_index
    
    print("Debug: input1.columns =", input.columns) # debug



def rank_index_cross_section(stock_data: pd.DataFrame, Top_k: int = -1, ascending: bool = False) -> pd.DataFrame:
    """
        Sort the cross-sectional data based on the given index.

        Args:
            stock_data : DataFrame containing the cross-sectional data. It should have three columns, and the last column represents the variable to be sorted.
            Top_k : The number of data points to retain after sorting. (Default: -1, which retains all data points)
            ascending: Whether to sort the data in ascending order or not. (Default: False)

        Returns:
            stock_data_selected : DataFrame containing the sorted data. It has the same structure as the input DataFrame.
        """

    index = stock_data.columns[-1]
    stock_data = stock_data.sort_values(by=index, ascending=ascending)
    #stock_data_selected = stock_data[['trade_date','stock_name', index]].copy()
    stock_data_selected = stock_data[:Top_k]
    stock_data_selected = stock_data_selected.drop_duplicates(subset=['stock_name'], keep='first')
    return stock_data_selected


def get_company_info(stock_name: str='') -> pd.DataFrame:
    # ts_code: str	股票代码,  exchange:str	交易所代码SSE上交所 SZSE深交所, chairman:str 法人代表, manager:str 总经理, secretary:str	董秘 # reg_capital:float	注册资本, setup_date:str 注册日期, province:str 所在省份 ,city:str 所在城市
    # introduction:str 公司介绍, website:str 公司主页 , email:str	电子邮件, office:str 办公室 # ann_date: str 公告日期, business_scope:str 经营范围, employees:int	员工人数, main_business:str 主要业务及产品
    """
            This function retrieves company information including stock code, exchange, chairman, manager, secretary,
            registered capital, setup date, province, city, website, email, employees, business scope, main business,
            introduction, office, and announcement date.

            Args:
            - stock_name (str): The name of the stock.

            Returns:
            - pd.DataFrame: A DataFrame that contains the company information.
    """

    stock_code = get_stock_code(stock_name)
    df = pro.stock_company(**{
        "ts_code": stock_code,"exchange": "","status": "", "limit": "","offset": ""
    }, fields=[
        "ts_code","exchange","chairman", "manager","secretary", "reg_capital","setup_date", "province","city",
        "website", "email","employees","business_scope","main_business","introduction","office", "ann_date"
    ])


    en_to_cn = {
        'ts_code': '股票代码',
        'exchange': '交易所代码',
        'chairman': '法人代表',
        'manager': '总经理',
        'secretary': '董秘',
        'reg_capital': '注册资本',
        'setup_date': '注册日期',
        'province': '所在省份',
        'city': '所在城市',
        'introduction': '公司介绍',
        'website': '公司主页',
        'email': '电子邮件',
        'office': '办公室',
        'ann_date': '公告日期',
        'business_scope': '经营范围',
        'employees': '员工人数',
        'main_business': '主要业务及产品'
    }

    df.rename(columns=en_to_cn, inplace=True)
    df.insert(0, '股票名称', stock_name)
    # for column in df.columns:
    #     print(f"[{column}]: {df[column].values[0]}")


    return df




def get_Financial_data_from_time_range(stock_name:str, start_date:str, end_date:str, financial_index:str='') -> pd.DataFrame:
    # start_date='20190101',end_date='20221231',financial_index='roe', The returned data consists of the ROE values for the entire three-year period from 2019 to 2022.
    # To query quarterly or annual financial report data for a specific moment, "yyyy0331"为一季报,"yyyy0630"为半年报,"yyyy0930"为三季报,"yyyy1231"为年报,例如get_Financial_data_from_time_range("600519.SH", "20190331", "20190331", "roe") means to query the return on equity (ROE) data from the first quarter of 2019,
    #  # current_ratio	流动比率 # quick_ratio	速动比率 # netprofit_margin	销售净利率 # grossprofit_margin	销售毛利率 # roe	净资产收益率 # roe_dt	净资产收益率(扣除非经常损益)
    # roa	总资产报酬率 # debt_to_assets 资产负债率 # roa_yearly	年化总资产净利率  # q_dtprofit	扣除非经常损益后的单季度净利润 # q_eps	每股收益(单季度)
    # q_netprofit_margin	销售净利率(单季度) # q_gsprofit_margin	销售毛利率(单季度) # basic_eps_yoy 基本每股收益同比增长率(%) # netprofit_yoy	归属母公司股东的净利润同比增长率(%)   # q_netprofit_yoy	归属母公司股东的净利润同比增长率(%)(单季度) # q_netprofit_qoq	归属母公司股东的净利润环比增长率(%)(单季度) # equity_yoy	净资产同比增长率
    """
        Retrieves the financial data for a given stock within a specified date range.

        Args:
            stock_name (str): The stock code.
            start_date (str): The start date of the data range in the format "YYYYMMDD".
            end_date (str): The end date of the data range in the format "YYYYMMDD".
            financial_index (str, optional): The financial indicator to be queried.

        Returns:
            pd.DataFrame: A DataFrame containin financial data for the specified stock and date range.

"""
    stock_code = get_stock_code(stock_name)
    stock_data = pro.fina_indicator(**{
        "ts_code": stock_code,
        "ann_date": "",
        "start_date": start_date,
        "end_date": end_date,
        "period": '',
        "update_flag": "1",
        "limit": "",
        "offset": ""
    }, fields=["ts_code", "end_date", financial_index])

    #stock_name = get_stock_name_from_code(stock_code)
    stock_data['stock_name'] = stock_name
    stock_data = stock_data.sort_values(by='end_date', ascending=True)  # 按照日期升序排列
    # 把end_data列改名为trade_date
    stock_data.rename(columns={'end_date': 'trade_date'}, inplace=True)
    stock_financial_data = stock_data[['stock_name', 'trade_date', financial_index]]
    return stock_financial_data


def get_GDP_data(start_quarter:str='', end_quarter:str='', index:str='gdp_yoy') -> pd.DataFrame:
    # The available indicators for query include the following 9 categories: # gdp GDP累计值（亿元）# gdp_yoy 当季同比增速（%）# pi 第一产业累计值（亿元）# pi_yoy 第一产业同比增速（%）# si 第二产业累计值（亿元）# si_yoy 第二产业同比增速（%）# ti 第三产业累计值（亿元） # ti_yoy 第三产业同比增速（%）
    """
        Retrieves GDP data for the chosen index and specified time period.

        Args:
        - start_quarter (str): The start quarter of the query, in YYYYMMDD format.
        - end_quarter (str): The end quarter, in YYYYMMDD format.
        - index (str): The specific GDP index to retrieve. Default is `gdp_yoy`.

        Returns:
        - pd.DataFrame: A pandas DataFrame with three columns: `quarter`, `country`, and the selected `index`.
        """

    # The output is a DataFrame with three columns:
    # the first column represents the quarter (quarter), the second column represents the country (country), and the third column represents the index (index).
    df = pro.cn_gdp(**{
        "q":'',
        "start_q": start_quarter,
        "end_q": end_quarter,
        "limit": "",
        "offset": ""
    }, fields=[
        "quarter",
        "gdp",
        "gdp_yoy",
        "pi",
        "pi_yoy",
        "si",
        "si_yoy",
        "ti",
        "ti_yoy"
    ])
    df = df.sort_values(by='quarter', ascending=True)  #
    df['country'] = 'China'
    df = df[['quarter', 'country', index]].copy()


    return df

def get_cpi_ppi_currency_supply_data(start_month: str = '', end_month: str = '', type: str = 'cpi', index: str = '') -> pd.DataFrame:
    # The query types (type) include three categories: CPI, PPI, and currency supply. Each type corresponds to different indices.
    # Specifically, CPI has 12 indices, PPI has 30 indices, and currency supply has 9 indices.
    # The output is a DataFrame table with three columns: the first column represents the month (month), the second column represents the country (country), and the third column represents the index (index).

    # type='cpi',monthly CPI data include the following 12 categories:
    # nt_val	全国当月值 # nt_yoy	全国同比（%）# nt_mom	全国环比（%）# nt_accu	全国累计值# town_val	城市当月值# town_yoy	城市同比（%）# town_mom	城市环比（%）# town_accu	城市累计值# cnt_val	农村当月值# cnt_yoy	农村同比（%）# cnt_mom	农村环比（%）# cnt_accu	农村累计值

    # type = 'ppi', monthly PPI data include the following 30 categories:
    # ppi_yoy	PPI：全部工业品：当月同比
    # ppi_mp_yoy    PPI：生产资料：当月同比
    # ppi_mp_qm_yoy	PPI：生产资料：采掘业：当月同比
    # ppi_mp_rm_yoy	PPI：生产资料：原料业：当月同比
    # ppi_mp_p_yoy	PPI：生产资料：加工业：当月同比
    # ppi_cg_yoy	PPI：生活资料：当月同比
    # ppi_cg_f_yoy	PPI：生活资料：食品类：当月同比
    # ppi_cg_c_yoy	PPI：生活资料：衣着类：当月同比
    # ppi_cg_adu_yoy	PPI：生活资料：一般日用品类：当月同比
    # ppi_cg_dcg_yoy	PPI：生活资料：耐用消费品类：当月同比
    # ppi_mom	PPI：全部工业品：环比
    # ppi_mp_mom	PPI：生产资料：环比
    # ppi_mp_qm_mom	PPI：生产资料：采掘业：环比
    # ppi_mp_rm_mom	PPI：生产资料：原料业：环比
    # ppi_mp_p_mom	PPI：生产资料：加工业：环比
    # ppi_cg_mom	PPI：生活资料：环比
    # ppi_cg_f_mom	PPI：生活资料：食品类：环比
    # ppi_cg_c_mom	PPI：生活资料：衣着类：环比
    # ppi_cg_adu_mom	PPI：生活资料：一般日用品类：环比
    # ppi_cg_dcg_mom		PPI：生活资料：耐用消费品类：环比
    # ppi_accu		PPI：全部工业品：累计同比
    # ppi_mp_accu		PPI：生产资料：累计同比
    # ppi_mp_qm_accu		PPI：生产资料：采掘业：累计同比
    # ppi_mp_rm_accu		PPI：生产资料：原料业：累计同比
    # ppi_mp_p_accu	    PPI：生产资料：加工业：累计同比
    # ppi_cg_accu	PPI：生活资料：累计同比
    # ppi_cg_f_accu		PPI：生活资料：食品类：累计同比
    # ppi_cg_c_accu		PPI：生活资料：衣着类：累计同比
    # ppi_cg_adu_accu	PPI：生活资料：一般日用品类：累计同比
    # ppi_cg_dcg_accu	PPI：生活资料：耐用消费品类：累计同比

    # type = 'currency_supply', monthly currency supply data include the following 9 categories:
    # m0  M0（亿元）# m0_yoy  M0同比（%）# m0_mom  M0环比（%）# m1  M1（亿元）# m1_yoy  M1同比（%）# m1_mom  M1环比（%）# m2  M2（亿元）# m2_yoy  M2同比（%）# m2_mom  M2环比（%）

    """
        This function is used to retrieve China's monthly CPI (Consumer Price Index), PPI (Producer Price Index),
        and monetary supply data published by the National Bureau of Statistics,
        and return a DataFrame table containing month, country, and index values.
        The function parameters include start month, end month, query type, and query index.
        For query indexes that are not within the query range, the default index for the corresponding type is returned.

        Args:
        - start_month (str): start month of the query, in the format of YYYYMMDD.
        - end_month (str):end month in YYYYMMDD
        - type (str): required parameter, query type, including three types: cpi, ppi, and currency_supply.
        - index (str): optional parameter, query index, the specific index depends on the query type.
        If the query index is not within the range, the default index for the corresponding type is returned.

        Returns:
        - pd.DataFrame: DataFrame type, including three columns: month, country, and index value.
        """

    if type == 'cpi':

        df = pro.cn_cpi(**{
            "m": '',
            "start_m": start_month,
            "end_m": end_month,
            "limit": "",
            "offset": ""
        }, fields=[
            "month", "nt_val","nt_yoy", "nt_mom","nt_accu", "town_val", "town_yoy",  "town_mom",
            "town_accu", "cnt_val", "cnt_yoy", "cnt_mom", "cnt_accu"])
        # If the index is not within the aforementioned range, the index is set as "nt_yoy".
        if index not in df.columns:
            index = 'nt_yoy'


    elif type == 'ppi':
        df = pro.cn_ppi(**{
            "m": '',
            "start_m": start_month,
            "end_m": end_month,
            "limit": "",
            "offset": ""
        }, fields=[
            "month", "ppi_yoy", "ppi_mp_yoy", "ppi_mp_qm_yoy", "ppi_mp_rm_yoy", "ppi_mp_p_yoy", "ppi_cg_yoy",
            "ppi_cg_f_yoy", "ppi_cg_c_yoy", "ppi_cg_adu_yoy", "ppi_cg_dcg_yoy",
            "ppi_mom", "ppi_mp_mom", "ppi_mp_qm_mom", "ppi_mp_rm_mom", "ppi_mp_p_mom", "ppi_cg_mom", "ppi_cg_f_mom",
            "ppi_cg_c_mom", "ppi_cg_adu_mom", "ppi_cg_dcg_mom",
            "ppi_accu", "ppi_mp_accu", "ppi_mp_qm_accu", "ppi_mp_rm_accu", "ppi_mp_p_accu", "ppi_cg_accu",
            "ppi_cg_f_accu", "ppi_cg_c_accu", "ppi_cg_adu_accu", "ppi_cg_dcg_accu"
        ])
        if index not in df.columns:
            index = 'ppi_yoy'

    elif type == 'currency_supply':
        df = pro.cn_m(**{
            "m": '',
            "start_m": start_month,
            "end_m": end_month,
            "limit": "",
            "offset": ""
        }, fields=[
            "month", "m0",  "m0_yoy","m0_mom", "m1",
            "m1_yoy",  "m1_mom", "m2", "m2_yoy", "m2_mom"])
        if index not in df.columns:
            index = 'm2_yoy'


    df = df.sort_values(by='month', ascending=True)  #
    df['country'] = 'China'
    df = df[['month', 'country', index]].copy()
    return df

def predict_next_value(df: pd.DataFrame, pred_index: str = 'nt_yoy', pred_num:int = 1. ) -> pd.DataFrame:
    """
    Predict the next n values of a specific column in the DataFrame using linear regression.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
            pred_index (str): The name of the column to predict.
            pred_num (int): The number of future values to predict.

        Returns:
        pandas.DataFrame: The DataFrame with the predicted values appended to the specified column
                          and other columns filled as pred+index.
        """
    input_array = df[pred_index].values

    # Convert the input array into the desired format.
    x = np.array(range(len(input_array))).reshape(-1, 1)
    y = input_array.reshape(-1, 1)

    # Train a linear regression model.
    model = LinearRegression()
    model.fit(x, y)

    # Predict the future n values.
    next_indices = np.array(range(len(input_array), len(input_array) + pred_num)).reshape(-1, 1)
    predicted_values = model.predict(next_indices).flatten()

    for i, value in enumerate(predicted_values, 1):
        row_data = {pred_index: value}
        for other_col in df.columns:
            if other_col != pred_index:
                row_data[other_col] = 'pred' + str(i)
        df = df.append(row_data, ignore_index=True)

        # Return the updated DataFrame
    return df






def get_latest_new_from_web(src: str = 'sina') -> pd.DataFrame:

    # 新浪财经	sina	获取新浪财经实时资讯
    # 同花顺	    10jqka	同花顺财经新闻
    # 东方财富	eastmoney	东方财富财经新闻
    # 云财经	    yuncaijing	云财经新闻
    """
    Retrieves the latest news data from major news websites, including Sina Finance, 10jqka, Eastmoney, and Yuncaijing.

    Args:
        src (str): The name of the news website. Default is 'sina'. Optional parameters include: 'sina' for Sina Finance,
        '10jqka' for 10jqka, 'eastmoney' for Eastmoney, and 'yuncaijing' for Yuncaijing.

    Returns:
        pd.DataFrame: A DataFrame containing the news data, including two columns for date/time and content.
    """

    df = pro.news(**{
        "start_date": '',
        "end_date": '',
        "src": src,
        "limit": "",
        "offset": ""
    }, fields=[
        "datetime",
        "content",
    ])
    df = df.apply(lambda x: '[' + x.name + ']' + ': ' + x.astype(str))
    return df




def get_index_constituent(index_name: str = '', start_date:str ='', end_date:str ='') -> pd.DataFrame:
    """
        Query the constituent stocks of basic index (中证500) or a specified SW (申万) industry index

        args:
             index_name: the name of the index.
             start_date: the start date in "YYYYMMDD".
             end_date:  the end date in "YYYYMMDD".

        return:
            A pandas DataFrame containing the following columns:
            index_code
            index_name
            stock_code: the code of the constituent stock.
            stock_name:  the name of the constituent stock.
            weight: the weight of the constituent stock.
    """

    if '申万' in index_name:
        if '申万一级行业' in index_name:
            # index_name取后面的名字
            index_name = index_name[6:]
            df1 = pd.read_csv('SW2021_industry_L1.csv')
            index_code = df1[df1['industry_name'] == index_name]['index_code'].iloc[0]
        elif '申万二级行业' in index_name:
            index_name = index_name[6:]
            df1 = pd.read_csv('SW2021_industry_L2.csv')
            index_code = df1[df1['industry_name'] == index_name]['index_code'].iloc[0]
        elif '申万三级行业' in index_name:
            index_name = index_name[6:]
            df1 = pd.read_csv('SW2021_industry_L3.csv')
            index_code = df1[df1['industry_name'] == index_name]['index_code'].iloc[0]

        print('The industry code for ', index_name, ' is: ', index_code)

        # 拉取数据
        df = pro.index_member(**{
            "index_code": index_code ,  #'851251.SI'
            "is_new": "",
            "ts_code": "",
            "limit": "",
            "offset": ""
        }, fields=[
            "index_code",
            "con_code",
            "in_date",
            "out_date",
            "is_new",
            "index_name",
            "con_name"
        ])
        #
        # For each stock, filter the start_date and end_date that are between in_date and out_date.
        df = df[(df['in_date'] <= start_date)]
        df = df[(df['out_date'] >= end_date) | (df['out_date'].isnull())]



        df.rename(columns={'con_code': 'stock_code'}, inplace=True)

        df.rename(columns={'con_name': 'stock_name'}, inplace=True)
        #
        df['weight'] = np.nan

        df = df[['index_code', "index_name", 'stock_code', 'stock_name','weight']]

    else: # 宽基指数
        df1 = pro.index_basic(**{
            "ts_code": "",
            "market": "",
            "publisher": "",
            "category": "",
            "name": index_name,
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "name",
        ])

        index_code = df1["ts_code"][0]
        print(f'index_code for basic index {index_name} is {index_code}')


        # Step 2: Retrieve the constituents of an index based on the index code and given date.
        df = pro.index_weight(**{
            "index_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "index_code",
            "con_code",
            "trade_date",
            "weight"
        ])
        # df = df.sort_values(by='trade_date', ascending=True)  #
        df['index_name'] = index_name
        last_day = df['trade_date'][0]
        #  for the last trading day
        df = df[df['trade_date'] == last_day]
        df_stock = pd.read_csv('tushare_stock_basic_20230421210721.csv')
        # Merge based on the stock code.
        df = pd.merge(df, df_stock, how='left', left_on='con_code', right_on='ts_code')
        # df.rename(columns={'name_y': 'name'}, inplace=True)
        df = df.drop(columns=['symbol', 'area', 'con_code'])
        df.sort_values(by='weight', ascending=False, inplace=True)
        df.rename(columns={'name': 'stock_name'}, inplace=True)
        df.rename(columns={'ts_code': 'stock_code'}, inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        #
        df = df[['index_code', "index_name", 'stock_code', 'stock_name', 'weight']]

    return df

# Determine whether the given name is a stock or a fund.,
def is_fund(ts_name: str = '') -> bool:
    # call  get_stock_code()和query_fund_name_or_code()
    if get_stock_code(ts_name) is not None and query_fund_name_or_code(ts_name) is None:
        return False
    elif get_stock_code(ts_name) is None and query_fund_name_or_code(ts_name) is not None:
        return True




def calculate_earning_between_two_time(stock_name: str = '', start_date: str = '', end_date: str = '', index: str = 'close') -> float:
    """
        Calculates the rate of return for a specified stock/fund between two dates.

        Args:
            stock_name: stock_name or fund_name
            start_date
            end_date
            index (str): The index used to calculate the stock return, including 'open' and 'close'.

        Returns:
            float: The rate of return for the specified stock between the two dates.
    """
    if is_fund(stock_name):
        fund_code = query_fund_name_or_code(stock_name)
        stock_data = query_fund_data(fund_code, start_date, end_date)
        if index =='':
            index = 'adj_nav'
    else:
        stock_data = get_stock_prices_data(stock_name, start_date, end_date,'daily')
    try:
        end_price = stock_data.iloc[-1][index]
        start_price = stock_data.iloc[0][index]
        earning = cal_dt(end_price, start_price)
        # earning = round((end_price - start_price) / start_price * 100, 2)
    except:
        print(ts_code,start_date,end_date)
        print('##################### 该股票没有数据 #####################')
        return None
    # percent = earning * 100
    # percent_str = '{:.2f}%'.format(percent)

    return  earning


def loop_rank(df: pd.DataFrame,  func: callable, *args, **kwargs) -> pd.DataFrame:
    """
        It iteratively applies the given function to each row and get a result using function. It then stores the calculated result in 'new_feature' column.

        Args:
        df: DataFrame with a single column
        func : The function to be applied to each row: func(row, *args, **kwargs)
        *args: Additional positional arguments for `func` function.
        **kwargs: Additional keyword arguments for `func` function.

        Returns:
        pd.DataFrame: A output DataFrame with three columns: the constant column, input column, and new_feature column.
                     The DataFrame is sorted based on the new_feature column in descending order.

        """
    df['new_feature'] = None
    loop_var = df.columns[0]
    for _, row in df.iterrows():
        res  = None
        var = row[loop_var]                                         #

        if var is not None:
            if loop_var == 'stock_name':
                stock_name = var
            elif loop_var == 'stock_code':
                stock_name = get_stock_name_from_code(var)
            elif loop_var == 'fund_name':
                stock_name = var
            elif loop_var == 'fund_code':
                stock_name = query_fund_name_or_code('',var)
            time.sleep(0.4)
            try:
                res = func(stock_name, *args, **kwargs)             #
            except:
                raise ValueError('#####################Error for func#####################')
            # res represents the result obtained for the variable. For example, if the variable is a stock name, res could be the return rate of that stock over a certain period or a specific feature value of that stock. Therefore, res should be a continuous value.
            # If the format of res is a float, then it can be used directly. However, if res is in DataFrame format, you can retrieve the value corresponding to the index.
            if isinstance(res, pd.DataFrame) and not res.empty:
                #
                try:
                    res = round(res.loc[:,args[-1]][0], 2)
                    df.loc[df[loop_var] == var, 'new_feature'] = res
                except:
                    raise ValueError('##################### Error ######################')
            elif isinstance(res, float): #
                res = res
                df.loc[df[loop_var] == var, 'new_feature'] = res
            print(var, res)


    # Remove the rows where the new_feature column is empty.
    df = df.dropna(subset=['new_feature'])
    stock_data = df.sort_values(by='new_feature', ascending=False)
    #
    stock_data.insert(0, 'unchanged', loop_var)
    stock_data = stock_data.loc[:,[stock_data.columns[0], loop_var, 'new_feature']]

    return stock_data

def output_mean_median_col(data: pd.DataFrame, col: str = 'new_feature') -> float:
    # It calculates the mean and median value for the specified column.

    mean = round(data[col].mean(), 2)
    median = round(data[col].median(), 2)
    #
    #print(title, mean)
    return (mean, median)



def output_weighted_mean_col(data: pd.DataFrame, col: str, weight_col: pd.Series) -> float:

    """
        Calculates the weighted mean of a column and returns the result as a float.

        Args:
            data (pd.DataFrame): The input cross-sectional or time-series data containing the feature columns.
            col (str): The name of the feature column to calculate the weighted mean for.
            weight_col (pd.Series): The weights used for the calculation, as a pandas Series.

        Returns:
            float: The weighted mean of the specified feature column.
        """

    weighted_mean = round(np.average(data[col], weights = weight_col)/100., 2)
    return weighted_mean



def get_index_data(index_name: str = '', start_date: str = '', end_date: str = '', freq: str = 'daily') -> pd.DataFrame:
    """
        This function retrieves daily, weekly, or monthly data for a given stock index.

        Arguments:
        - index_name: Name of the index
        - start_date: Start date in 'YYYYMMDD'
        - end_date: End date in 'YYYYMMDD'
        - freq: Frequency 'daily', 'weekly', or 'monthly'

        Returns:
        A DataFrame containing the following columns:
        trade_date, ts_code, close, open, high, low, pre_close: Previous day's closing price, change(涨跌额), pct_chg(涨跌幅), vol(成交量), amount(成交额), name: Index Name
        """
    df1 = pro.index_basic(**{
        "ts_code": "",
        "market": "",
        "publisher": "",
        "category": "",
        "name": index_name,
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "name",
    ])

    index_code = df1["ts_code"][0]
    print(f'index_code for index {index_name} is {index_code}')
    #
    if freq == 'daily':
        df = pro.index_daily(**{
            "ts_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "trade_date",
            "ts_code",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
    elif freq == 'weekly':
        df = pro.index_weekly(**{
            "ts_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "trade_date",
            "ts_code",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
    elif freq == 'monthly':
        df = pro.index_monthly(**{
            "ts_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "trade_date",
            "ts_code",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])

    df = df.sort_values(by='trade_date', ascending=True)  #
    df['index_name'] = index_name
    return df





def get_north_south_money(start_date: str = '', end_date: str = '', trade_date: str = '') -> pd.DataFrame:
    #
    # trade_date: 交易日期
    # ggt_ss:	港股通（上海）
    # ggt_sz:	港股通（深圳）
    # hgt:	沪股通（亿元）
    # sgt:	深股通（亿元）
    # north_money:	北向资金（亿元）= hgt + sgt
    # south_money:	南向资金（亿元）= ggt_ss + ggt_sz
    # name:  固定为'A-H',代表A股和H股
    # accumulate_north_money: 累计北向资金流入
    # accumulate_south_money: 累计南向资金流入


    month_df = pro.moneyflow_hsgt(**{
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
        "limit": "",
        "offset": ""
    }, fields=[
        "trade_date",
        "ggt_ss",
        "ggt_sz",
        "hgt",
        "sgt",
        "north_money",
        "south_money"
    ])

    month_df[['ggt_ss','ggt_sz','hgt','sgt','north_money','south_money']] = month_df[['ggt_ss','ggt_sz','hgt','sgt','north_money','south_money']]/100.0
    month_df = month_df.sort_values(by='trade_date', ascending=True)  #
    month_df['stock_name'] = 'A-H'
    month_df['accumulate_north_money'] = month_df['north_money'].cumsum()
    month_df['accumulate_south_money'] = month_df['south_money'].cumsum()
    return month_df



def plot_k_line(stock_data: pd.DataFrame, title: str = '') -> plt.Axes:
    """
    绘制K线图，仅显示K线和移动平均线

    Args:
        stock_data : A pandas DataFrame containing the stock price information
        title : The title of the K-line chart.

    Returns:
        axes : The axes of the K-line chart.
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    stock_data.rename(columns={'vol': 'volume'}, inplace=True)
    
    # 转换日期格式
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], format='%Y%m%d')
    stock_data.set_index('trade_date', inplace=True)

    # 设置K线图样式（红涨黑跌）
    custom_style = mpf.make_marketcolors(up='r', down='k', inherit=True)
    china_style = mpf.make_mpf_style(marketcolors=custom_style)

    # 创建图形
    fig, axes = mpf.plot(
        stock_data, 
        type='candle',
        #title=title,
        mav=(5, 10, 20),  # 显示5、10、20日均线
        mavcolors=['red', 'green', 'blue'],  # 设置均线颜色
        style=china_style,
        volume=False,  # 不显示成交量
        returnfig=True,
        figsize=(20, 10)  # 由于不显示下面的图，所以可以调整整体高度
    )

    # 添加均线图例
    mav_labels = ['5-day MA', '10-day MA', '20-day MA']
    legend_lines = [plt.Line2D([0], [0], color=color, lw=2) 
                   for color in ['red', 'green', 'blue']]
    axes[0].legend(legend_lines, mav_labels)

    # 添加网格
    axes[0].grid(True)

    plt.show()
    return axes[0]

def test_plot_k_line():
    stock_data = get_stock_prices_data('贵州茅台', '20230101', '20231231', 'daily')
    stock_data.rename(columns={'vol': 'volume'}, inplace=True)
    plot_k_line(stock_data, '贵州茅台2023年1月1日到2023年12月31日K线图')    
    #plt.show()  # 添加这行来显示图形

def cal_dt(num_at_time_2: float = 0.0, num_at_time_1: float = 0.0) -> float:
    """
        This function calculates the percentage change of a metric from one time to another.

        Args:
        - num_at_time_2: the metric value at time 2 (end time)
        - num_at_time_1: the metric value at time 1 (start time)

        Returns:
        - float: the percentage change of the metric from time 1 to time 2

        """
    if num_at_time_1 == 0:
        num_at_time_1 = 0.0000000001
    return round((num_at_time_2 - num_at_time_1) / num_at_time_1, 4)


def query_fund_info(fund_code: str = '') -> pd.DataFrame:
    #
    # fund_code	str	Y	基金代码 # fund_name	str	Y	简称 # management	str	Y	管理人 # custodian	str	Y	托管人 # fund_type	str	Y	投资类型 # found_date	str	Y	成立日期 # due_date	str	Y	到期日期 # list_date	str	Y	上市时间 # issue_date	str	Y	发行日期 # delist_date	str	Y	退市日期 # issue_amount	float	Y	发行份额(亿) # m_fee	float	Y	管理费 # c_fee	float	Y	托管费
    # duration_year	float	Y	存续期 # p_value	float	Y	面值 # min_amount	float	Y	起点金额(万元) # benchmark	str	Y	业绩比较基准 # status	str	Y	存续状态D摘牌 I发行 L已上市 # invest_type	str	Y	投资风格 # type	str	Y	基金类型 # purc_startdate	str	Y	日常申购起始日 # redm_startdate	str	Y	日常赎回起始日 # market	str	Y	E场内O场外
    """
        Retrieves information about a fund based on the fund code.

        Args:
            fund_code (str, optional): Fund code. Defaults to ''.

        Returns:
            df (DataFrame): A DataFrame containing various information about the fund, including fund code, fund name,
                            management company, custodian company, investment type, establishment date, maturity date,
                            listing date, issuance date, delisting date, issue amount, management fee, custodian fee,
                            fund duration, face value, minimum investment amount, benchmark, fund status, investment style,
                            fund type, start date for daily purchases, start date for daily redemptions, and market type.
                            The column 'ts_code' is renamed to 'fund_code', and 'name' is renamed to 'fund_name' in the DataFrame.
        """
    df = pro.fund_basic(**{
        "ts_code": fund_code,
        "market": "",
        "update_flag": "",
        "offset": "",
        "limit": "",
        "status": "",
        "name": ""
    }, fields=[
        "ts_code",
        "name",
        "management",
        "custodian",
        "fund_type",
        "found_date",
        "due_date",
        "list_date",
        "issue_date",
        "delist_date",
        "issue_amount",
        "m_fee",
        "c_fee",
        "duration_year",
        "p_value",
        "min_amount",
        "benchmark",
        "status",
        "invest_type",
        "type",
        "purc_startdate",
        "redm_startdate",
        "market"
    ])
    #
    df.rename(columns={'ts_code': 'fund_code'}, inplace=True)
    df.rename(columns={'name': 'fund_name'}, inplace=True)
    return df

def query_fund_data(fund_code: str = '', start_date: str = '', end_date: str = '') -> pd.DataFrame:
    #
    # ts_code	str	Y	TS代码 # ann_date	str	Y	公告日期 # nav_date	str	Y	净值日期 # unit_nav	float	Y	单位净值 # accum_nav	float	Y	累计净值
    # accum_div	float	Y	累计分红 # net_asset	float	Y	资产净值 # total_netasset	float	Y	合计资产净值 # adj_nav	float	Y	复权单位净值  pct_chg 每日涨跌幅
    """
        Retrieves fund data based on the fund code, start date, and end date.

        Args:
            fund_code (str, optional): Fund code. Defaults to ''.
            start_date (str, optional): Start date in YYYYMMDD format. Defaults to ''.
            end_date (str, optional): End date in YYYYMMDD format. Defaults to ''.

        Returns:
            df (DataFrame): A DataFrame containing fund data, including TS code, announcement date, net asset value date,
                            unit net asset value, accumulated net asset value, accumulated dividends, net asset value,
                            total net asset value, adjusted unit net asset value, and fund name. The 'ts_code' column is renamed
                            to 'fund_code', 'nav_date' is renamed to 'trade_date', and the DataFrame is sorted by the trade date
                            in ascending order. If the fund code does not exist, None is returned.
        """
    df = pro.fund_nav(**{
        "ts_code": fund_code,
        "nav_date": "",
        "offset": "",
        "limit": "",
        "market": "",
        "start_date": start_date,
        "end_date": end_date
    }, fields=[
        "ts_code",
        "ann_date",
        "nav_date",
        "unit_nav",
        "accum_nav",
        "accum_div",
        "net_asset",
        "total_netasset",
        "adj_nav",
        "update_flag"
    ])
    try:
        fund_name= query_fund_name_or_code(fund_code=fund_code)
        df['fund_name'] = fund_name
        #
        df.rename(columns={'ts_code': 'fund_code'}, inplace=True)
        df.rename(columns={'nav_date': 'trade_date'}, inplace=True)
        df.sort_values(by='trade_date', ascending=True, inplace=True)
    except:
        print(fund_code,'基金代码不存在')
        return None
    #
    df['pct_chg'] = df['adj_nav'].pct_change()
    #
    df.loc[0, 'pct_chg'] = 0.0


    return df

def query_fund_name_or_code(fund_name: str = '', fund_code: str = '') -> str:
    #
    """
        Retrieves the fund code based on the fund name or Retrieves the fund name based on the fund code.

        Args:
            fund_name (str, optional): Fund name. Defaults to ''.
            fund_code (str, optional): Fund code. Defaults to ''.

        Returns:
            code or name: Fund code if fund_name is provided and fund_code is empty. Fund name if fund_code is provided and fund_name is empty.
        """


    #df = pd.read_csv('./tushare_fund_basic_20230508193747.csv')
    # Query the fund code based on the fund name.
    if fund_name != '' and fund_code == '':
        #
        df = pd.read_csv('./tushare_fund_basic_all.csv')
        #
        # df = pro.fund_basic(**{
        #     "ts_code": "",
        #     "market": "",
        #     "update_flag": "",
        #     "offset": "",
        #     "limit": "",
        #     "status": "",
        #     "name": fund_name
        # }, fields=[
        #     "ts_code",
        #     "name"
        # ])
        try:
            #
            code = df[df['name'] == fund_name]['ts_code'].values[0]
        except:
            #print(fund_name,'基金名称不存在')
            return None
        return code
    # Query the fund name based on the fund code.
    if fund_code != '' and fund_name == '':
        df = pd.read_csv('./tushare_fund_basic_all.csv')
        try:
            name = df[df['ts_code'] == fund_code]['name'].values[0]
        except:
            #print(fund_code,'基金代码不存在')
            return None
        return name



def print_save_table(df: pd.DataFrame, title_name: str = '', save: bool = False, file_path: str = './output/') -> pd.DataFrame:
    """
    打印并保存DataFrame到CSV文件。

    Args:
        df: 要打印和保存的DataFrame
        title_name: 保存的文件名（不含扩展名）
        save: 是否保存到CSV文件
        file_path: 保存文件的路径

    Returns:
        DataFrame: 返回原始DataFrame
    """
    # 确保输出目录存在
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 打印DataFrame信息
    print(f"\n{'='*50}")
    print(f"DataFrame信息: {title_name}")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {', '.join(df.columns)}")
    print(f"{'='*50}\n")

    # 打印DataFrame预览
    print("数据预览:")
    print(df.head())
    print("\n")

    # 保存到CSV文件
    if save:
        # 生成文件名（如果未提供title_name，使用时间戳）
        if not title_name:
            title_name = f"data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 确保文件名以.csv结尾
        if not title_name.endswith('.csv'):
            title_name += '.csv'
            
        # 构建完整文件路径
        full_path = os.path.join(file_path, title_name)
        
        # 保存文件
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {full_path}")

    return df



#
def merge_indicator_for_same_stock(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
        Merges two DataFrames (two indicators of the same stock) based on common names for same stock. Data from two different stocks cannot be merged

        Args:
            df1: DataFrame contains some indicators for stock A.
            df2: DataFrame contains other indicators for stock A.

        Returns:
            pd.DataFrame: The merged DataFrame contains two different indicators.
    """
    if len(set(df1.columns).intersection(set(df2.columns))) > 0:
        # If there are identical column names, merge the two DataFrames based on the matching column names.
        #
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        #
        df = pd.merge(df1, df2, on=common_cols)
        return  df
    else:
        #
        raise ValueError('The two dataframes have no columns in common.')

def select_value_by_column(df1:pd.DataFrame, col_name: str = '', row_index: int = -1) -> Union[pd.DataFrame, Any]:
    """
        Selects a specific column or a specific value within a DataFrame.

        Args:
            df1: The input DataFrame.
            col_name: The name of the column to be selected.
            row_index: The index of the row to be selected.

        Returns:
            Union[pd.DataFrame, Any]. row_index=-1: df1[col_name].to_frame() or df1[col_name][row_index]
    """
    if row_index == -1:
        #
        return df1[col_name].to_frame()
    else:
        #
        return df1[col_name][row_index]





import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 假设这里定义了 font_prop，用于设置标题字体
# font_prop = FontProperties(fname='SimHei.ttf')  # Mac 示例，Windows 可用 'SimHei'

# 引入你的函数
# from your_module import plot_pie_chart  # 如果你是从模块导入

# 示例数据
def test_plot_pie_chart():
    data = pd.DataFrame({
        '类别': ['苹果', '香蕉', '橘子', '西瓜'],
        '销量': [30, 15, 45, 10]
    })

    # 创建图形并绘制饼图
    fig, ax = plt.subplots()
    plot_pie_chart(data, labels_col='类别', values_col='销量', ax=ax, title='水果销量分布')

    # 显示图表
    plt.show()


def plot_correlation_heatmap(data1: pd.DataFrame, data2: pd.DataFrame, data3: pd.DataFrame,
                           title: str = '',
                           figsize: tuple = (10, 8),
                           cmap: str = 'RdYlGn',
                           annot: bool = True) -> plt.Axes:
    """
    绘制相关性热力图

    Args:
        data: 包含需要计算相关性的数据的DataFrame
        title: 图表标题
        figsize: 图表大小，默认(10, 8)
        cmap: 热力图配色方案，默认'RdYlGn'
        annot: 是否在热力图上显示具体数值，默认True

    Returns:
        matplotlib.axes.Axes: 图表对象
    """
    # 创建一个空的DataFrame来存储收盘价数据
    close_data = pd.DataFrame()

    # 将每个股票数据添加为新的列
    close_data[data1['stock_name'].iloc[0]] = data1.set_index('trade_date')['close']
    close_data[data2['stock_name'].iloc[0]] = data2.set_index('trade_date')['close']
    close_data[data3['stock_name'].iloc[0]] = data3.set_index('trade_date')['close']

    # 删除包含缺失值的行
    close_data = close_data.dropna()

    # 计算相关性矩阵
    corr_matrix = close_data.corr()
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制热力图，添加 linewidths=0 去除网格线
    ax = sns.heatmap(corr_matrix,
                     annot=annot,
                     cmap=cmap,
                     annot_kws={"size": 12},
                     fmt='.2f',
                     square=True,
                     center=0,
                     linewidths=0)
    
    # 设置标题和样式
    plt.title(title, pad=20, fontsize=16, fontproperties=font_prop)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 新增代码：关闭所有坐标轴边框线
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    # 调整布局
    plt.tight_layout()
    
    return ax

def test_correlation_heatmap():
    """
    测试相关性热力图绘制功能
    """
    # 获取三只股票的日线数据
    stock_codes = ['贵州茅台', '平安银行', '招商银行']
    start_date = '20230101'
    end_date = '20231231'
    
    stock_data_1 = get_stock_prices_data(stock_codes[0], start_date, end_date, 'daily') 
    stock_data_2 = get_stock_prices_data(stock_codes[1], start_date, end_date, 'daily')
    stock_data_3 = get_stock_prices_data(stock_codes[2], start_date, end_date, 'daily')
    
    # 绘制相关性热力图
    plot_correlation_heatmap(
        data1=stock_data_1,
        data2=stock_data_2,
        data3=stock_data_3,
        title='个股收盘价相关性热力图',
        figsize=(10, 8),
        cmap='RdYlGn',
        annot=True
    )
    
    plt.show()


if __name__ == '__main__':
    #test_plot_pie_chart()
    #test_correlation_heatmap()
    # test_plot_k_line()
    res = get_latest_new_from_web("sina")
    print(res)
    








