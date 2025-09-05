# -*- coding: utf-8 -*-
import requests
import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from snownlp import SnowNLP
import os
from datetime import datetime, timedelta


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出文件夹
OUTPUT_DIR = "analysis_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ---------------------- 1. 数据爬取（以NewsAPI为例） ----------------------
def fetch_news(keyword="股票"):
    """通过NewsAPI获取金融新闻"""
    language = "zh"
    days = 30
    # 计算时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "apiKey": os.getenv("NEWSAPI_KEY"),
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": 100,  # 最大100条/次
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d")
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        # 提取标题和内容
        data = [{
            "title": article["title"],
            "content": article["content"],
            "date": article["publishedAt"][:10]
        } for article in articles]
        df = pd.DataFrame(data)
        print(f"获取到 {len(df)} 条新闻，时间范围：{start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        return df
    else:
        print("API请求失败:", response.status_code)
        return pd.DataFrame()
# 将fetch_news函数添加到tool_lib/tool_text_web.json中
# fetch_news

# ---------------------- 2. 数据预处理 ----------------------
#def preprocess_text(text):
#    """文本清洗与分词"""
    # 清洗
#    text = text.replace("\r", "").replace("\n", "").replace(" ", "")
    # 加载金融停用词表（需自行准备或从网上下载）
#    stopwords = set(["的", "了", "和", "在", "是", ...])  # 示例
    # 分词+过滤
#    words = jieba.cut(text)
#    return " ".join([word for word in words if word not in stopwords and len(word) > 1])

def preprocess_text(df, input_column="content", output_column="cleaned_content"):
    """
    对DataFrame指定列进行文本预处理（清洗+分词），并生成新列。
    参数:
        df (pd.DataFrame): 输入数据
        input_column (str): 要处理的原始列名
        output_column (str): 处理后的新列名
    返回:
        pd.DataFrame: 添加了预处理列的新DataFrame
    """
    # 停用词表（建议从文件加载）
    stopwords = set(["的", "了", "和", "在", "是", "也", "就", "都", "与", "上", "下", "对", "等"])

    def clean_and_cut(text):
        if not isinstance(text, str):
            return ""
        text = text.replace("\r", "").replace("\n", "").replace(" ", "")
        words = jieba.cut(text)
        return " ".join([w for w in words if w not in stopwords and len(w) > 1])

    df[output_column] = df[input_column].apply(clean_and_cut)
    return df

# ---------------------- 3. 情感分析 ----------------------
def sentiment_analysis(df):
    """使用SnowNLP进行情感分析"""
    df["sentiment"] = df["content"].apply(lambda x: SnowNLP(x).sentiments)
    # 分类：正面 (>0.6), 中性 (0.4-0.6), 负面 (<0.4)
    df["sentiment_label"] = df["sentiment"].apply(
        lambda x: "正面" if x > 0.6 else "负面" if x < 0.4 else "中性"
    )
    return df

# ---------------------- 5. 可视化结果 ----------------------
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def visualize_results(df: pd.DataFrame, 
                               output_prefix: str = '',
                               pie_ax: Optional[plt.Axes] = None, 
                               trend_ax: Optional[plt.Axes] = None
                              ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    在一张图中绘制情感分布图（饼图）和情感得分时间趋势图（折线图）

    Args:
        df (pd.DataFrame): 包含情感分析结果的数据框
        output_prefix (str): 图表标题前缀

    Returns:
        Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]: Figure对象和包含两个Axes的元组
    """
    import matplotlib.pyplot as plt

    # 输入验证
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是pandas DataFrame，而不是 " + str(type(df)))
    
    required_columns = ["sentiment_label", "date", "sentiment"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")

    # 创建共享的Figure
    fig, (pie_ax, trend_ax) = plt.subplots(1, 2, figsize=(16, 6))  # 一行两个子图

    # 饼图
    sentiment_counts = df["sentiment_label"].value_counts()
    pie_ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
    pie_ax.set_title(output_prefix + "情感分布")
    pie_ax.axis('equal')

    # 折线图
    df["date"] = pd.to_datetime(df["date"])
    daily_sentiment = df.groupby("date")["sentiment"].mean()
    daily_sentiment.plot(kind="line", marker="o", ax=trend_ax)
    trend_ax.set_title(output_prefix + "情感得分时间趋势")
    trend_ax.set_xlabel("日期")
    trend_ax.set_ylabel("平均情感得分")
    trend_ax.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_sentiment_analysis.png"))

    return fig, (pie_ax, trend_ax)


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 1. 替换为你的NewsAPI密钥（注册地址：https://newsapi.org/）
    API_KEY = "6900df901f2346f0a1822ccea28c6d52"
    
    # 获取用户输入
    print("\n关键词输入说明：")
    print("- 可以输入多个关键词，用空格分隔")
    print("- 例如：'特斯拉 股票' 或 '比亚迪 销量 股价'")
    print("- 直接回车将使用默认关键词'特斯拉'\n")
    
    keyword = input("请输入要搜索的关键词：") or "特斯拉"
    # 处理关键词，确保多个关键词之间只有一个空格
    keyword = " ".join(keyword.split())
    
    days = input("请输入要查询的天数（默认为30天）：") or "30"
    try:
        days = int(days)
        if days <= 0:
            print("天数必须大于0，将使用默认值30天")
            days = 30
        elif days > 100:
            print("由于API限制，最多只能查询100天的数据，将使用100天")
            days = 100
    except ValueError:
        print("输入的天数无效，将使用默认值30天")
        days = 30
    
    print(f"\n开始获取关于 '{keyword}' 的{days}天新闻数据...")
    
    # 2. 爬取新闻数据
    df = fetch_news(keyword=keyword)
    if df.empty:
        print("未获取到数据，请检查API密钥或网络连接！")
        exit()
    
    # 3. 数据预处理
    print("正在进行文本预处理...")
    df = preprocess_text(df)
    
    # 4. 情感分析
    print("正在进行情感分析...")
    df = sentiment_analysis(df)
    
    # 5. 可视化
    print("正在生成可视化图表...")
    output_prefix = "_".join(keyword.split()[:2])  # 使用前两个关键词作为文件名前缀
    pie_fig, trend_fig = visualize_results(df, output_prefix)
    
    # 6. 保存结果
    csv_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_news_analysis.csv")
    print("\n分析完成！结果已保存到以下文件：")
    print(f"- {csv_file}（详细数据）")
    print(f"- pie_fig（情感分布饼图）")
    print(f"- trend_fig（情感趋势图）")
    print(f"\n所有文件都已保存到 '{OUTPUT_DIR}' 文件夹中。")
