import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.callbacks import BaseCallbackHandler
import tiktoken
from callbacks import TokenAnalysisCallback 
from langchain.globals import set_debug
set_debug(True)

# 初始化 DeepSeek LLM
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # 确保环境变量已设置
llm = ChatOpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0,
)

# 初始化数据库连接
db = SQLDatabase.from_uri("postgresql://test_ai:LMY2025..@172.16.19.203:5432/test_dw_ai")

# 创建 SQL Agent
agent = create_sql_agent(llm=llm, db=db, agent_type="openai-tools")

# 用户交互循环
while True:
    user_input = input("\n请输入您想查询的问题（输入 'exit' 退出）: ")
    
    if user_input.lower() == 'exit':
        print("退出程序...")
        break
        
    try:
        # 初始化 callback（每次查询重新创建，避免数据污染）
        callback = TokenAnalysisCallback()
        
        # 执行查询，并传入 callback 以捕获 Token 信息
        result = agent.invoke(
            {"input": user_input},
            {"callbacks": [callback]}  # 确保 callback 被正确传入
        )
        
        print("\n查询结果:")
        print(result["output"])
        
        # 可选：打印完整的 Token 数据（如需要存储到数据库）
        db_data = callback.get_db_data()
        print("\nToken 数据（可存储到数据库）:")
        print(f"请求 Token 数: {db_data['request_token_count']}")
        print(f"响应 Token 数: {db_data['response_token_count']}")
        
    except Exception as e:
        print(f"查询出错: {e}")