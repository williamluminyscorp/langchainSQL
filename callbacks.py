import tiktoken
from langchain_core.callbacks import BaseCallbackHandler


class TokenAnalysisCallback(BaseCallbackHandler):
    def __init__(self, encoding_name="cl100k_base"):
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.db_data = {
            "request": None,
            "response": None,
            "request_token_count": 0,
            "response_token_count": 0,
        }
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        combined_prompt = "\n".join(prompts)
        tokens = self.encoder.encode(combined_prompt)
        
        self.db_data["request"] = combined_prompt
        self.db_data["request_token_count"] = len(tokens)
        
        # 打印请求信息和内容
        print("\n=== 请求信息 ===")
        print(f"请求内容:\n{combined_prompt}")  # 打印完整请求内容
        print(f"Token 数量（请求）: {len(tokens)}")
    
    def on_llm_end(self, response, **kwargs):
        if response.generations:
            result_text = response.generations[0][0].text
            tokens = self.encoder.encode(result_text)
            
            self.db_data["response"] = result_text
            self.db_data["response_token_count"] = len(tokens)
            
            # 打印响应信息
            print("=== 响应信息 ===")
            print(f"响应内容:\n{result_text}")  # 打印完整响应内容
            print(f"Token 数量（响应）: {len(tokens)}")
            print("------------------")
    
    def get_db_data(self):
        return self.db_data