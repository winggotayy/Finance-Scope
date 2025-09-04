import streamlit as st
from main import streamlit_interface, available_models, example_stock, example_economic, example_fund, example_company
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from io import BytesIO

def set_page_style():
    """设置页面样式"""
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7f9;
            padding: 0.5rem 1rem;
        }
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .css-1d391kg {
            padding: 0.5rem 0.5rem 3rem;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stSelectbox {
            border-radius: 8px;
        }
        .stTab {
            border-radius: 4px 4px 0 0;
        }
        h1 {
            color: #1E3A8A;
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            margin-bottom: 1rem !important;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            letter-spacing: 1px;
        }
        h2 {
            color: #2563EB;
            font-weight: 600 !important;
        }
        h3 {
            color: #3B82F6;
            font-weight: 500 !important;
        }
        .subtitle {
            font-size: 1rem;
            color: #4B5563;
            text-align: center;
            margin-bottom: 2rem;
            line-height: 1.6;
            font-weight: 500;
        }
        .success-box {
            padding: 1rem;
            border-radius: 8px;
            background-color: #ECFDF5;
            border: 1px solid #34D399;
            color: #065F46;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 8px;
            background-color: #FEF3C7;
            border: 1px solid #FBBF24;
            color: #92400E;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px 4px 0 0;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem 1rem;
            background-color: #f8f9fa;
            border-top: 1px solid #e9ecef;
            font-size: 0.9rem;
            color: #6c757d;
            display: flex;
            align-items: center;
            height: 60px;
        }
        .footer p {
            margin: 0;
            line-height: 1.5;
            padding-left: 1rem;
        }
        .footer a {
            color: #2563EB;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

def show_footer():
    """显示页脚信息"""
    st.markdown("""
        <div class="footer">
            <p>📊 数据来源：<a href="https://tushare.pro/" target="_blank">Tushare Pro</a></p>
            <p>📊 联网新闻来源：<a href="https://newsapi.org/" target="_blank">NewsAPI</a></p>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="FinanceScope | 金融数据分析助手",
        page_icon="��",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    set_page_style()

    # 标题区域
    st.markdown("<h1>FinanceScope | 金融数据分析助手</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p class='subtitle'>
            一个集成多源数据的智能金融分析平台，支持查看股票、基金、经济指标等多类金融信息，<br/>
                帮助用户快速了解市场动向，做出更明智的投资决策。
        </p>
    """, unsafe_allow_html=True)
#    一个简洁易用的金融数据分析平台，支持查看股票、基金和经济指标等多类数据。<br/>
#            帮助用户快速了解市场动向，做出更明智的投资决策。
    # 创建两列布局，调整宽度比例
    left_col, right_col = st.columns([1.2, 2.8])  # 增加左侧栏宽度比例

    # 左侧列：输入区域
    with left_col:
        st.markdown("### 🎯 模型配置")
        
        # 初始化 session state
        if 'model_confirmed' not in st.session_state:
            st.session_state.model_confirmed = False
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'openai_key' not in st.session_state:
            st.session_state.openai_key = None
        if 'analysis_mode' not in st.session_state:
            st.session_state.analysis_mode = "金融分析"

        # 模型选择区域
        with st.container():
            selected_model = st.selectbox(
                "🤖 选择分析模型",
                available_models,
                index=0,
                key='model_selector'
            )
            
            if selected_model == "gpt-3.5":
                openai_key = st.text_input("🔑 OpenAI API Key", type="password")
            else:
                openai_key = None

            # 确认按钮
            confirm_col1, confirm_col2 = st.columns([1.5, 2.5])  # 增加确认按钮列的宽度比例
            with confirm_col1:
                if st.button("确认", type="primary", use_container_width=True):  # 使用容器宽度
                    st.session_state.model_confirmed = True
                    st.session_state.current_model = selected_model
                    st.session_state.openai_key = openai_key
            
            with confirm_col2:
                if st.session_state.model_confirmed:
                    st.markdown(f"<div class='success-box'>✓ 当前模型: {st.session_state.current_model}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-box'>⚠️ 请确认模型选择</div>", unsafe_allow_html=True)

        instruction_disabled = not st.session_state.model_confirmed

        st.markdown("---")
        st.markdown("### 📈 分析模式")
        
        # 分析模式选择
        analysis_mode = st.radio(
            "选择分析模式",
            ["数据分析", "文本分析"],
            horizontal=True,
            key="analysis_mode_selector"
        )
        st.session_state.analysis_mode = analysis_mode

        if analysis_mode == "数据分析":
            # 创建金融分析选项卡
            tab1, tab2, tab3, tab4 = st.tabs(["📊 股票", "🏦 经济", "💰 基金", "🏢 公司"])
            
            with tab1:
                stock_command = st.selectbox("选择股票分析指令", example_stock)
                if st.button("执行分析", key="stock_btn", disabled=instruction_disabled):
                    process_instruction(st.session_state.current_model, stock_command, st.session_state.openai_key, right_col)

            with tab2:
                economic_command = st.selectbox("选择经济分析指令", example_economic)
                if st.button("执行分析", key="economic_btn", disabled=instruction_disabled):
                    process_instruction(st.session_state.current_model, economic_command, st.session_state.openai_key, right_col)

            with tab3:
                fund_command = st.selectbox("选择基金分析指令", example_fund)
                if st.button("执行分析", key="fund_btn", disabled=instruction_disabled):
                    process_instruction(st.session_state.current_model, fund_command, st.session_state.openai_key, right_col)

            with tab4:
                company_command = st.selectbox("选择公司分析指令", example_company)
                if st.button("执行分析", key="company_btn", disabled=instruction_disabled):
                    process_instruction(st.session_state.current_model, company_command, st.session_state.openai_key, right_col)

        else:  # 文本分析模式
            # 文本分析选项
            col1, col2 = st.columns([1, 1])
            with col1:
                keyword = st.text_input("输入分析关键词", placeholder="例如：特斯拉、比亚迪、新能源...")
            
            with col2:
                analysis_type = st.selectbox(
                    "分析类型",
                    [
                        "公司相关",
                        "行业主题",
                        "市场热点",
                        "宏观经济",
                        "投资主题"
                    ]
                )
                st.markdown("##### 示例关键词")
                st.markdown("""
                - 公司：特斯拉、营收、销量
                - 行业：新能源、芯片、政策
                - 市场：ChatGPT、概念股
                - 宏观：美联储加息、GDP、通货膨胀
                - 投资：基金、ETF、预测
                """)
            
            if keyword:
                # 根据选择的分析类型生成不同的指令模板
                instruction_templates = {
                    "公司相关": "联网搜索关键词：'{keyword}'，并进行情感分析",
                    "行业主题": "联网搜索关键词：'{keyword}'，并进行情感分析",
                    "市场热点": "联网搜索关键词：'{keyword}'，并进行情感分析",
                    "宏观经济": "联网搜索关键词：'{keyword}'，并进行情感分析",
                    "投资主题": "联网搜索关键词：'{keyword}'，并进行情感分析"
                }
                
                # 生成完整指令
                text_command = instruction_templates[analysis_type].format(
                    keyword=keyword
                )
                
                # 显示生成的指令
                st.info(f"📝 将执行指令: {text_command}")
                
                if st.button("开始分析", key="text_btn", disabled=instruction_disabled):
                    # 创建占位符用于显示搜索状态
                    search_status = st.empty()
                    search_status.info("🌐 正在进行联网搜索，获取最新相关信息...")
                    
                    try:
                        process_instruction(st.session_state.current_model, text_command, st.session_state.openai_key, right_col)
                    finally:
                        # 清除搜索状态提示
                        search_status.empty()
            else:
                st.warning("请输入分析关键词")

        # 自定义指令输入
        st.markdown("---")
        st.markdown("### ✨ 自定义分析")
        custom_instruction = st.text_area("输入自定义指令", height=100)
        if st.button("开始分析", key="custom_btn", disabled=instruction_disabled, type="primary"):
            if custom_instruction:
                process_instruction(st.session_state.current_model, custom_instruction, st.session_state.openai_key, right_col)
            else:
                st.warning("请输入分析指令")

        # 系统状态（放在折叠面板中）
        with st.expander("⚙️ 系统状态", expanded=False):
            st.info(f"模型状态: {'已确认' if st.session_state.model_confirmed else '未确认'}")
            st.info(f"当前模型: {st.session_state.current_model or '未选择'}")
            if st.button("🔄 重置设置", type="secondary"):
                st.session_state.model_confirmed = False
                st.session_state.current_model = None
                st.session_state.openai_key = None
                st.rerun()

    # 显示页脚
    show_footer()

def process_instruction(model, instruction, openai_key, right_col):
    """处理指令并在右侧显示结果"""
    with right_col:
        st.markdown("### 📊 分析结果")
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 显示当前执行的指令
        st.info(f"📝 当前指令: {instruction}")

        try:
            # 更新状态
            status_text.markdown("⏳ 正在初始化...")
            progress_bar.progress(10)

            # 调用处理函数
            status_text.markdown("🔄 正在处理数据...")
            progress_bar.progress(30)
            
            output_text, image, summary, df = streamlit_interface(model, instruction, openai_key)

            # 显示处理过程
            status_text.markdown("📊 正在生成结果...")
            progress_bar.progress(60)

            # 创建结果标签页
            result_tabs = st.tabs(["📋 处理详情", "📈 可视化结果", "📊 数据表格"])
            
            with result_tabs[0]:
                st.markdown("#### 处理过程")
                st.code(output_text)
                
                st.markdown("#### 结果总结")
                st.info(summary)
            
            with result_tabs[1]:
                if image:
                    st.markdown("#### 可视化图表")
                    # 显示两张图表
                    st.image(image)
                    
                    # 添加图表下载按钮
                    if image:
                        # 将图表转换为字节
                        buf = BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                        buf.seek(0)
                        
                        # 创建下载按钮，使用唯一key
                        st.download_button(
                            label="📥 下载图表",
                            data=buf,
                            file_name=f"financeScope_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            key=f"chart_download_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                else:
                    st.info("💡 此次分析未生成图表")

            with result_tabs[2]:
                if df is not None and not df.empty:
                    st.markdown("#### 数据表格")
                    st.markdown("预览（前10行）：")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.markdown(f"📊 完整数据集: {len(df)} 行, {len(df.columns)} 列")
                    
                    # 下载按钮
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下载完整数据",
                        data=csv,
                        file_name=f"financeScope_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("💡 此次分析未生成数据表格")

            # 完成进度条
            progress_bar.progress(100)
            status_text.markdown("✅ 分析完成！")

        except Exception as e:
            st.error(f"❌ 处理过程中出现错误: {str(e)}")
            progress_bar.empty()
            status_text.markdown("⚠️ 处理失败")
            raise e

if __name__ == "__main__":
    main()
