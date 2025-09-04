# 导入tushare
import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from lab_gpt4_call import send_chat_request,send_chat_request_Azure,send_official_call
from lab_llms_call import send_chat_request_fin_r1, send_chat_request_qwen, send_chat_request_deepseek, send_chat_request_glm,send_chat_request_chatglm3_6b,send_chat_request_chatglm_6b
# from lab_llm_local_call import send_chat_request_internlm_chat
#import ast
import re
from tool import *
from tool_text_analysis import *
import tiktoken
import concurrent.futures
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import queue
from threading import Thread
import time  # Add this at the top with other imports
#plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['SimHei'] # fix: font missing
plt.rcParams['axes.unicode_minus'] = False


import openai


# To override the Thread method
class MyThread(Thread):

    def __init__(self, target, args):
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result




def parse_and_exe(call_dict, result_buffer, parallel_step: str='1'):
    """
    Parse the input and call the corresponding function to obtain the result.
    :param call_dict: dict, including arg, func, and output
    :param result_buffer: dict, storing the corresponding intermediate results
    :param parallel_step: int, parallel step
    :return: Returns func(arg) and stores the corresponding result in result_buffer.
    """
    arg_list = call_dict['arg' + parallel_step]
    replace_arg_list = [result_buffer[item][0] if isinstance(item, str) and ('result' in item or 'input' in item) else item for item in arg_list]  # 参数
    func_name = call_dict['function' + parallel_step]             #
    output = call_dict['output' + parallel_step]                  #
    desc = call_dict['description' + parallel_step]               #
    if func_name == 'loop_rank':
        replace_arg_list[1] = eval(replace_arg_list[1])
    result = eval(func_name)(*replace_arg_list)
    result_buffer[output] = (result, desc)                        #    'result1': (df1, desc)
    return result_buffer

def load_tool_and_prompt(tool_lib, tool_prompt ):
    '''
    Read two JSON files.
    :param tool_lib: Tool description
    :param tool_prompt: Tool prompt
    :return: Flattened prompt
    '''
    #

    with open(tool_lib, 'r') as f:
        tool_lib = json.load(f)

    with open(tool_prompt, 'r') as f:
        #
        tool_prompt = json.load(f)

    for key, value in tool_lib.items():
        tool_prompt["Function Library:"] = tool_prompt["Function Library:"] + key + " " + value+ '\n\n'


    prompt_flat = ''

    for key, value in tool_prompt.items():
        prompt_flat = prompt_flat + key +'  '+ value + '\n\n'


    return prompt_flat

# callback function
intermediate_results = queue.Queue()  # Create a queue to store intermediate results.

def add_to_queue(intermediate_result):
    intermediate_results.put(f"After planing, the intermediate result is {intermediate_result}")



def check_RPM(run_time_list, new_time, max_RPM=1):
    # Check if there are already 3 timestamps in the run_time_list, with a maximum of 3 accesses per minute.
    # False means no rest is needed, True means rest is needed.
    if len(run_time_list) < 3:
        run_time_list.append(new_time)
        return 0
    else:
        if (new_time - run_time_list[0]).seconds < max_RPM:
            # Calculate the required rest time.
            sleep_time = 60 - (new_time - run_time_list[0]).seconds
            print('sleep_time:', sleep_time)
            run_time_list.pop(0)
            run_time_list.append(new_time)
            return sleep_time
        else:
            run_time_list.pop(0)
            run_time_list.append(new_time)
            return 0

def run(model, instruction, add_to_queue=None, send_chat_request_Azure = send_official_call, openai_key = '', api_base='', engine=''):
    # Start timing for core processing
    core_start_time = time.time()
    
    output_text = ''
    ################################# Step-1:Task select ###########################################
    intent_start_time = time.time()
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d")
    # If the time has not exceeded 3 PM, use yesterday's data.
    if current_time.hour < 15:
        formatted_time = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")

    print('===============================Intent Detecting===========================================')
    with open('./prompt_lib/prompt_intent_detection.json', 'r') as f:
        prompt_task_dict = json.load(f)
    prompt_intent_detection = ''
    for key, value in prompt_task_dict.items():
        prompt_intent_detection = prompt_intent_detection + key + ": " + value+ '\n\n'

    prompt_intent_detection = prompt_intent_detection + '\n\n' + 'Instruction:' + '今天的日期是'+ formatted_time +', '+ instruction + ' ###New Instruction: '
    
    response = send_chat_request(model,prompt_intent_detection, openai_key=openai_key, api_base=api_base, engine=engine)
    if model in ["qwen-chat-72b", "deepseek-r1", "fin-r1"]:
        response = re.sub(r"^.*?step1=", "step1=", response, flags=re.DOTALL)
    new_instruction = response
    intent_end_time = time.time()
    intent_duration = intent_end_time - intent_start_time
    print(f"\n意图识别阶段耗时: {intent_duration:.2f} 秒")
    
    output_text = output_text + '\n======Intent Detecting Stage=====\n\n'
    output_text = output_text + new_instruction +'\n\n'

    if add_to_queue is not None:
        add_to_queue(output_text)

    event_happen = True
    print('===============================Task Planing===========================================')
    planning_start_time = time.time()
    output_text= output_text + '=====Task Planing Stage=====\n\n'

    with open('./prompt_lib/prompt_task.json', 'r') as f:
        prompt_task_dict = json.load(f)
    prompt_task = ''
    for key, value in prompt_task_dict.items():
        prompt_task = prompt_task + key + ": " + value+ '\n\n'

    prompt_task = prompt_task + '\n\n' + 'Instruction:' + new_instruction + ' ###Plan:'
    
    response = send_chat_request(model, prompt_task, openai_key=openai_key,api_base=api_base,engine=engine)

    task_select = response
    pattern = r"(task\d+=)(\{[^}]*\})" # 模型适配
    matches = re.findall(pattern, task_select)
    task_plan = {}
    for task in matches:
        task_step, task_select = task
        task_select = task_select.replace("'", "\"")  # Replace single quotes with double quotes.
        task_select = json.loads(task_select)
        task_name = list(task_select.keys())[0]
        task_instruction = list(task_select.values())[0]

        task_plan[task_name] = task_instruction

    planning_end_time = time.time()
    planning_duration = planning_end_time - planning_start_time
    print(f"任务规划阶段耗时: {planning_duration:.2f} 秒")

    # task_plan
    for key, value in task_plan.items():
        print(key, ':', value)
        output_text = output_text + key + ': ' + str(value) + '\n'

    output_text = output_text +'\n'
    if add_to_queue is not None:
        add_to_queue(output_text)

    ################################# Step-2:Tool select and use ###########################################
    tool_start_time = time.time()
    print('===============================Tool select and using Stage===========================================')
    output_text = output_text + '======Tool select and using Stage======\n\n'
    # Read the task_select JSON file name.
    task_name = list(task_plan.keys())[0].split('_task')[0]
    task_instruction = list(task_plan.values())[0]

    tool_lib = './tool_lib/' + 'tool_' + task_name + '.json'
    tool_prompt = './prompt_lib/' + 'prompt_' + task_name + '.json'
    prompt_flat = load_tool_and_prompt(tool_lib, tool_prompt)
    prompt_flat = prompt_flat + '\n\n' +'Instruction :'+ task_instruction+ ' ###Function Call'

    response = send_chat_request(model, prompt_flat, openai_key=openai_key,api_base=api_base, engine=engine)
    if '###' in response:
        call_steps = response.replace("###","")
    else:
        call_steps = response  
    pattern = r"(?i)(step\d+\s*=\s*)?(\{[^}]*\})" # 模型适配
    matches = re.findall(pattern, call_steps)
 
    result_buffer = {}                
    output_buffer = []                
    
    for match in matches:
        step, content = match
        content = content.replace("'", "\"")  # Replace single quotes with double quotes.
        print('==================')
        print("\n\nstep:", step)
        print('content:',content)
        call_dict = json.loads(content)
        print('It has parallel steps:', len(call_dict) / 4)
        output_text = output_text + step + ': ' + str(call_dict) + '\n\n'

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(parse_and_exe, call_dict, result_buffer, str(parallel_step))
                       for parallel_step in range(1, int(len(call_dict) / 4) + 1)}

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    print('parallel step:', idx+1)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        if step == matches[-1][0]:
            for parallel_step in range(1, int(len(call_dict) / 4) + 1):
                output_buffer.append(call_dict['output' + str(parallel_step)])
    
    tool_end_time = time.time()
    tool_duration = tool_end_time - tool_start_time
    print(f"工具调用阶段耗时: {tool_duration:.2f} 秒")
    
    output_text = output_text + '\n'
    if add_to_queue is not None:
        add_to_queue(output_text)

    ################################# Step-3:visualization ###########################################
    viz_start_time = time.time()
    print('===============================Visualization Stage===========================================')
    output_text = output_text + '======Visualization Stage====\n\n'
    task_name = 'visualization'
    task_instruction = list(task_plan.values())[1]

    tool_lib = './tool_lib/' + 'tool_' + task_name + '.json'
    tool_prompt = './prompt_lib/' + 'prompt_' + task_name + '.json'

    result_buffer_viz={}
    Previous_result = {}
    for output_name in output_buffer:
        if output_name in result_buffer:
            rename = 'input'+ str(output_buffer.index(output_name)+1)
            Previous_result[rename] = result_buffer[output_name][1]
            result_buffer_viz[rename] = result_buffer[output_name]
        else:
            print(f"Warning: Expected result {output_name} not found in result_buffer")

    prompt_flat = load_tool_and_prompt(tool_lib, tool_prompt)
    prompt_flat = prompt_flat + '\n\n' +'Instruction: '+ task_instruction + ', Previous_result: '+ str(Previous_result) + ' ###Function Call'

    response = send_chat_request(model, prompt_flat, openai_key=openai_key, api_base=api_base, engine=engine)
    if '###' in response:
        call_steps, _ = response.split('###') 
    else:
        call_steps = response
    
    if response.startswith("{"):
        response = "step1=" + response
    call_steps = response
    pattern = r"(step\d+=)(\{[^}]*\})" 
    matches = re.findall(pattern, call_steps)
    for match in matches:
        step, content = match
        content = content.replace("'", "\"")  # Replace single quotes with double quotes.
        content = content.replace("None","null")
        print('==================')
        print("\n\nstep:", step)
        print('content:',content)
        call_dict = json.loads(content)
        print('It has parallel steps:', len(call_dict) / 4)
        result_buffer_viz = parse_and_exe(call_dict, result_buffer_viz, parallel_step = '' )
        output_text = output_text + step + ': ' + str(call_dict) + '\n\n'

# 性能测试
    viz_end_time = time.time()
    viz_duration = viz_end_time - viz_start_time
    print(f"可视化阶段耗时: {viz_duration:.2f} 秒")

    if add_to_queue is not None:
        add_to_queue(output_text)

    finally_output = list(result_buffer_viz.values())

    # Calculate total core processing time
    core_end_time = time.time()
    core_duration = core_end_time - core_start_time
    
    # Print detailed performance metrics
    print("\n性能测试详细结果:")
    print(f"意图识别阶段耗时: {intent_duration:.2f} 秒")
    print(f"任务规划阶段耗时: {planning_duration:.2f} 秒")
    print(f"工具调用阶段耗时: {tool_duration:.2f} 秒")
    print(f"可视化阶段耗时: {viz_duration:.2f} 秒")
    print(f"核心处理总耗时: {core_duration:.2f} 秒")
    print(f"平均每分钟处理请求数: {60/core_duration:.2f} 次")
    
    # Rest of the visualization and output code...
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ################################# Step-3:visualization ###########################################
    print('===============================Visualization Stage===========================================')
    output_text = output_text + '======Visualization Stage====\n\n'
    # task_name = list(task_plan.keys())[1].split('_task')[0] # visualization_task
    task_name = 'visualization'
    task_instruction = list(task_plan.values())[1] #''


    tool_lib = './tool_lib/' + 'tool_' + task_name + '.json'
    tool_prompt = './prompt_lib/' + 'prompt_' + task_name + '.json'

    # print("Current result_buffer keys:", result_buffer.keys()) # debug
    result_buffer_viz={}
    Previous_result = {}
    for output_name in output_buffer:
        if output_name in result_buffer:  # Add check for key existence
            rename = 'input'+ str(output_buffer.index(output_name)+1)
            Previous_result[rename] = result_buffer[output_name][1]
            result_buffer_viz[rename] = result_buffer[output_name]
        else:
            print(f"Warning: Expected result {output_name} not found in result_buffer")

    prompt_flat = load_tool_and_prompt(tool_lib, tool_prompt)
    prompt_flat = prompt_flat + '\n\n' +'Instruction: '+ task_instruction + ', Previous_result: '+ str(Previous_result) + ' ###Function Call'

    # current_time = datetime.datetime.now()
    # sleep_time = check_RPM(run_time, current_time)
    # if sleep_time > 0:
    #     time.sleep(sleep_time)

    # response = send_chat_request("qwen-chat-72b", prompt_flat)
    # print("prompt"+prompt_flat)
    response = send_chat_request(model, prompt_flat, openai_key=openai_key, api_base=api_base, engine=engine)
    if '###' in response:
        call_steps, _ = response.split('###') 
    else:
        call_steps = response
    #print("response:\n",response) # debug 
    # debug for piechart
    if response.startswith("{"):
        response = "step1=" + response
    #print("new response:\n",response) # debug
    # 重新赋予 call_steps 值
    call_steps = response
    pattern = r"(step\d+=)(\{[^}]*\})" 
    matches = re.findall(pattern, call_steps)
    for match in matches:
        step, content = match
        content = content.replace("'", "\"")  # Replace single quotes with double quotes.
        content = content.replace("None","null")
        print('==================')
        print("\n\nstep:", step)
        print('content:',content)
        call_dict = json.loads(content)
        print('It has parallel steps:', len(call_dict) / 4)
        result_buffer_viz = parse_and_exe(call_dict, result_buffer_viz, parallel_step = '' )
        output_text = output_text + step + ': ' + str(call_dict) + '\n\n'

    if add_to_queue is not None:
        add_to_queue(output_text)

    finally_output = list(result_buffer_viz.values()) # plt.Axes

    # add: save the visualization result to the result folder
    # 确保输出目录存在
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame()
    str_out = output_text + 'Finally result: '
    for ax in finally_output:
        safe_name = re.sub(r'[\\/*?:"<>|]', '_', ax[1])  # 为文件名生成安全的字符串


        if isinstance(ax[0], plt.Axes): 
            #print('is plot')
            # 设置网格，并获取图形对象
            plt.grid()
            fig = ax[0].get_figure()
            plt.show()
            
            # 构造文件名，并保存图像为 PNG 文件
            filename = f"plot_{idx}_{safe_name}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, bbox_inches='tight', dpi=150)

            str_out = str_out + ax[1]+ ':' + 'plt.Axes' + '\n\n'
                
        elif isinstance(ax[0], pd.DataFrame):
            df = ax[0]
            #print('is dataframe')
            
            # 构造 CSV 文件名，并保存 DataFrame 为 CSV 文件
            csv_filename = f"data_{idx}_{safe_name}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_filepath, index=False)

            # 将 DataFrame 的前 10 行转换为图像保存
            df_head = df.head(10)
            # print("df_head:",df_head)
            fig, ax_table = plt.subplots(figsize=(min(12, len(df.columns)*1.5), 2 + 0.5*len(df_head)))
            ax_table.axis('tight')
            ax_table.axis('off')
            table = ax_table.table(cellText=df_head.values, colLabels=df_head.columns, loc='center')
            # 构造 DataFrame 图片文件名，并保存
            df_png_filename = f"data_{idx}_{safe_name}_head.png"
            df_png_filepath = os.path.join(output_dir, df_png_filename)
            plt.savefig(df_png_filepath, bbox_inches='tight', dpi=150)
            plt.close(fig)  # 关闭当前图形
            
            str_out = str_out + ax[1]+ ':' + 'pd.DataFrame' + '\n\n'
        elif isinstance(ax[1], tuple):
            #print('is tuple')
            fig, (pie_ax, trend_ax) = visualize_results(df, safe_name)
            # 保存饼图和趋势图
            fig.savefig(os.path.join(output_dir, f"plot_{idx}_{safe_name}.png"), bbox_inches='tight', dpi=150)

            str_out = str_out + str(ax[1])+ ':' + str(ax[0]) + '\n\n'
        else:
            str_out = str_out + str(ax[1])+ ':' + str(ax[0]) + '\n\n'


    #
    print('===============================Summary Stage===========================================')
    output_prompt = "请用第一人称总结一下整个任务规划和解决过程,并且输出结果,用[Task]表示每个规划任务,用\{function\}表示每个任务里调用的函数." + \
                    "示例1:###我用将您的问题拆分成两个任务,首先第一个任务[stock_task],我依次获取五粮液和贵州茅台从2013年5月20日到2023年5月20日的净资产回报率roe的时序数据. \n然后第二个任务[visualization_task],我用折线图绘制五粮液和贵州茅台从2013年5月20日到2023年5月20日的净资产回报率,并计算它们的平均值和中位数. \n\n在第一个任务中我分别使用了2个工具函数\{get_stock_code\},\{get_Financial_data_from_time_range\}获取到两只股票的roe数据,在第二个任务里我们使用折线图\{plot_stock_data\}工具函数来绘制他们的roe十年走势,最后并计算了两只股票十年ROE的中位数\{output_median_col\}和均值\{output_mean_col\}.\n\n最后贵州茅台的ROE的均值和中位数是\{\},{},五粮液的ROE的均值和中位数是\{\},\{\}###" + \
                    "示例2:###我用将您的问题拆分成两个任务,首先第一个任务[stock_task],我依次获取20230101到20230520这段时间北向资金每日净流入和每日累计流入时序数据,第二个任务是[visualization_task],因此我在同一张图里同时绘制北向资金20230101到20230520的每日净流入柱状图和每日累计流入的折线图 \n\n为了完成第一个任务中我分别使用了2个工具函数\{get_north_south_money\},\{calculate_stock_index\}分别获取到北上资金的每日净流入量和每日的累计净流入量,第二个任务里我们使用折线图\{plot_stock_data\}绘制来两个指标的变化走势.\n\n最后我们给您提供了包含两个指标的折线图和数据表格." + \
                    "示例3:###我用将您的问题拆分成两个任务,首先第一个任务[economic_task],我爬取了上市公司贵州茅台和其主营业务介绍信息. \n然后第二个任务[visualization_task],我用表格打印贵州茅台及其相关信息. \n\n在第一个任务中我分别使用了1个工具函数\{get_company_info\} 获取到贵州茅台的公司信息,在第二个任务里我们使用折线图\{print_save_table\}工具函数来输出表格.\n"
    # output_result = send_chat_request("qwen-chat-72b", output_prompt + str_out + '###')
    output_result = send_chat_request(model, output_prompt + str_out + '###', openai_key=openai_key, api_base=api_base,engine=engine)
    print(output_result)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    #
    #
    image = Image.open(buf)


    return output_text, image, output_result, df


def send_chat_request(model, prompt, send_chat_request_Azure = send_official_call, openai_key = '', api_base='', engine=''):
    '''
    Send request to LLMs(gpt, qwen-chat-72b, glm-3-turbo...)
    :param model: the name of llm
    :param prompt: prompt
    :param send_chat_request_Azure(for gpt call)
    :param openai_key(for gpt call)
    :param api_base(for gpt call)
    :param engine(for gpt call)
    :return response: the response of llm
    '''
    if model=="gpt-3.5":
        response = send_chat_request_Azure(prompt, openai_key=openai_key, api_base=api_base, engine=engine)
    elif model=="qwen-chat-72b":
        response = send_chat_request_qwen(prompt)# please set your api_key in lab_llms_call.py 
    elif model == "deepseek-r1":
        response = send_chat_request_deepseek(prompt)
    elif model == "Fin-R1":
        response = send_chat_request_fin_r1(prompt)
    # elif model=="glm-3-turbo":
    #     response = send_chat_request_glm(prompt)# please set your api_key in lab_llms_call.py 
    # Currently, smaller LLMs are unsupported
    # elif model =="chatglm3-6b":
    #     response = send_chat_request_chatglm3_6b(prompt)# please set your api_key in lab_llms_call.py 
    # If you want to call the llm from local, you can try the following: internlm-chat-7b
    # elif model=="internlm-chat-7b":
    #     response = send_chat_request_internlm_chat(prompt)  
    return response


# 可用模型列表
available_models = ["gpt-3.5", "qwen-chat-72b", "deepseek-r1", "Fin-R1"]

# debugging: compare
# 示例指令选项
example_stock = [
    '我想看看中国软件的2019年1月12日到2019年02月12日的收盘价的走势图', # 折线图
    '今年上证50所有成分股的收益率是多少', # 柱状图 # qwen-chat-72b
    '请绘制贵州茅台从2024年一季度的股价涨跌幅(收益率)的柱状图', # 柱状图 
    '请绘制2024年贵州茅台股东持股比例饼图', # 饼图
    '请绘制2024年第一季度平安银行的收盘价和成交量的散点图', # 散点图
    '绘制工商银行近一年的k线图',    # k线图
    '给我画一下可孚医疗2024年年中到今天的股价', # 折线图
    '请绘制近三个月贵州茅台的股价走势图', # 折线图
    '北向资金2023年的每日流入和累计流入', 
    '看一下近三年宁德时代和贵州茅台的pb变化', 
    '画一下五粮液和泸州老窖从2021年年初到2024年年中的收益率走势',
    #'成都银行近一年的k线图和kdj指标',
    '比较下沪深300,创业板指,中证1000指数今年的收益率'
]

example_economic = [
    '我想看看中国近十年gdp的走势', # qwen-chat-72b # deepseek-r1
    '我想看看现在的新闻或者最新的消息', # df
    '预测中国未来12个季度的GDP增速', # df
    '中国过去十年的cpi走势是什么', # 折线图
    '过去五年中国的货币供应量走势,并且打印保存' # 折线图   
]

example_fund = [
    '请绘制场内交易市场的6大基金分布饼图', # 饼图 # market = 'E'
    '请绘制场外交易市场的6大基金分布饼图', # 饼图 # market = 'O' # not supported for less than 3%
    '易方达的张坤管理了几个基金', # df
    '基金经理周海栋名下的所有基金今年的收益率情况', # 柱状图 # debugging to bar chart(not yet)
    '我想看看周海栋管理的华商优势行业的近三年来的的净值曲线',
    '比较下华商优势行业和易方达蓝筹精选这两只基金的近三年的收益率'
]

example_company = [
    '介绍下贵州茅台,这公司是干什么的,主营业务是什么', # gpt: line # deepseek-r1: df
    '请获取2023年1月1日到2023年12月31日贵州茅台、平安银行和招商银行个股收盘价相关性，用热力图展示', # 热力图
    '我想比较下工商银行和贵州茅台近十年的净资产回报率',
    '今年一季度上证50的成分股的归母净利润同比增速分别是' # 柱状图 # debugging to bar chart(not yet)
]

def streamlit_interface(model: str, instruction: str, openai_key: str = None) -> tuple:
    """
    Streamlit 界面的驱动函数
    
    Args:
        model (str): 选择的模型名称
        instruction (str): 用户输入的指令
        openai_key (str, optional): OpenAI API key，仅在使用 GPT 模型时需要

    Returns:
        tuple: (output_text, image, summary, df)
            - output_text: 处理过程的文本输出
            - image: 可视化结果图像
            - summary: 结果总结
            - df: 数据表格（如果有）
    """
    # 设置 GPT 相关参数
    if model == "gpt-3.5":
        openai_call = send_official_call
    else:
        openai_call = None
        openai_key = None

    # 运行模型并获取结果
    output_text, image, summary, df = run(
        model, 
        instruction, 
        send_chat_request_Azure=openai_call,
        openai_key=openai_key,
        api_base='',
        engine=''
    )

    return output_text, image, summary, df

from text_web import *
from text_pdf import *

def main():
    # Start timing
    start_time = time.time()
    
    print("可用模型列表:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    # 用户选择模型
    model_index = int(input("请选择一个模型（输入编号）: ")) - 1
    if model_index < 0 or model_index >= len(available_models):
        print("输入错误，请输入有效编号。")
        return
    model = available_models[model_index]
    
    # 如果选择的模型是 "gpt"，自动输入 openai_call 和 openai_key
    if model == "gpt-3.5":
        openai_call = send_official_call
        openai_key = os.getenv("OPENAI_KEY")  # 请替换为真实密钥
    else:
        # 如果选择的是其他模型，可以按需设置其他的 API 参数
        openai_call = None
        openai_key = None

    # 展示指令选项
    print("\n以下是一些指令示例，你可以选择一个指令或输入自己的指令：")
    print("\n股票相关指令:")
    for idx, command in enumerate(example_stock, 1):
        print(f"{idx}. {command}")
    
    print("\n经济相关指令:")
    for idx, command in enumerate(example_economic, 1):
        print(f"{idx}. {command}")
    
    print("\n基金相关指令:")
    for idx, command in enumerate(example_fund, 1):
        print(f"{idx}. {command}")
    
    print("\n公司相关指令:")
    for idx, command in enumerate(example_company, 1):
        print(f"{idx}. {command}")

    print("\n您也可以选择文本分析功能, 请选择文本类型:")
    # 选择搜索类型
    search_type = input("请选择搜索类型(1. 联网搜索：情感分析 2. 上传pdf文件：文本摘要): ")
    if search_type == "1":
        # 联网搜索
        keyword = input("请输入关键词: ")
        instruction = f"请联网搜索关键词: {keyword}, 并进行情感分析"
    elif search_type == "2":
        # 上传pdf文件
        pdf_file = input("请上传文件: ")
        instruction = f"请上传文件: {pdf_file}，并进行文本摘要"
    else:
        # 用户输入指令
        instruction = input("\n请输入您的指令: ")
    
    # 运行模型
    output, image, df, output_result = run(model, instruction, send_chat_request_Azure=openai_call, openai_key=openai_key, api_base='', engine='')
    print(output_result)
    
    # End timing and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Print performance metrics
    print("\n性能测试结果:")
    print(f"总运行时间: {duration:.2f} 秒")
    print(f"平均每分钟处理时间: {60/duration:.2f} 次请求")
    #plt.show()

if __name__ == "__main__":
    main()
    

# instruction ='请获取2023年1月1日到2023年12月31日贵州茅台、平安银行和招商银行个股收盘价相关性，用热力图展示'
# instruction ='绘制工商银行近一年的k线图' # k线图

