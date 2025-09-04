from http import HTTPStatus
import dashscope
from zhipuai import ZhipuAI

import openai
import os

def send_chat_request_fin_r1(query):
#    client = OpenAI(
#        base_url="https://ai.gitee.com/v1",
#        api_key="QIMVZR3STYNKLIS9SX6SCAHWRWOL4PABX8JZJ61I",
#        default_headers={"X-Failover-Enabled":"true"},
#    )

    openai.api_key  =  os.getenv("OPENAI_KEY")
    openai.api_base = "https://ai.gitee.com/v1"

#    response = client.chat.completions.create(
#        model="Fin-R1",
#        stream=True,
#        max_tokens=2048,
#        temperature=0.7,
#        top_p=0.8,
#        extra_body={
#            "top_k": 50,
#        },
#        frequency_penalty=1.05,
#        messages=[
#            {
#                "role": "system",
#                "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
#            },
#            {
#                "role": "user",
#                "content": "Can you please let us know more details about your "
#            }
#        ],
#    )
    

    response = openai.ChatCompletion.create(
        # engine="gpt35",
        model="Fin-R1",
        messages = [{"role": "system", "content": "You are an useful AI assistant that helps people solve the problem step by step."},
                  {"role": "user", "content": "" + query}],
        #max_tokens=max_token_num,
        temperature=0.1,
        top_p=0.8,
        frequency_penalty=0,
        # presence_penalty=0,
        # stop=None
        )

    data_res = response['choices'][0]['message']['content']
    return data_res

dashscope.api_key='<your api key>'
def send_chat_request_qwen(query):
    '''
    You can generate API keys in https://bailian.console.aliyun.com/
    '''
    messages = [
        {'role': 'system', 'content': 'You are an useful AI assistant that helps people solve the problem step by step.'},
        #{'role': 'system', 'content': 'You are an useful stock analyst that helps people solve the problem step by step. Please always respond in JSON format.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        'qwen-72b-chat',
        messages=messages,
        result_format='message'
    )
    if response.status_code == HTTPStatus.OK:
        data_res = response['output']['choices'][0]['message']['content']
        #print(data_res)
        return data_res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        
def send_chat_request_deepseek(query):
    '''
    You can generate API keys in https://bailian.console.aliyun.com/
    '''
    messages = [
        {'role': 'system', 'content': 'You are an useful AI assistant that helps people solve the problem step by step.'},
        #{'role': 'system', 'content': 'You are an useful stock analyst that helps people solve the problem step by step. Please always respond in JSON format.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        'deepseek-r1',
        messages=messages,
        result_format='message'
    )
    if response.status_code == HTTPStatus.OK:
        data_res = response['output']['choices'][0]['message']['content']
        #print(data_res)
        return data_res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        
def send_chat_request_chatglm3_6b(query):
    '''
    You can generate API keys in https://bailian.console.aliyun.com/
    '''
    messages = [
        {'role': 'system', 'content': 'You are an useful AI assistant that helps people solve the problem step by step.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        'chatglm3-6b',
        messages=messages,
        result_format='message',  
    )
    if response.status_code == HTTPStatus.OK:
        data_res = response['output']['choices'][0]['message']['content']
        print(data_res)
        return data_res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        

def send_chat_request_chatglm_6b(query):
    '''
    You can generate API keys in https://bailian.console.aliyun.com/
    '''
    messages = [
        {'role': 'system', 'content': 'You are an useful AI assistant that helps people solve the problem step by step.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        'chatglm-6b-v2',
        messages=messages,
        result_format='message',  
    )
    if response.status_code == HTTPStatus.OK:
        data_res = response['output']['choices'][0]['message']['content']
        # print(data_res)
        return data_res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
   
	
client = ZhipuAI(api_key="<your api key>") 
def send_chat_request_glm(query):
    '''
    You can generate API keys in https://open.bigmodel.cn/
    '''
    response = client.chat.completions.create(
        model="glm-3-turbo",  
        messages=[
        {'role': 'system', 'content': '你是一个有用的人工智能助手，帮助人们逐步解决问题.'},
        {'role': 'user', 'content': query}],
    )
    response=response.choices[0].message.content
    return response


      
if __name__ == '__main__':
    # response=send_chat_request_qwen("hello")
    response=send_chat_request_fin_r1("你好")
    print(response)