from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import json
import requests
from openai import OpenAI


class ModelConfig:
    """模型配置"""
    def __init__(self, name: str, max_input_tokens: int, max_output_tokens: int,
                 doc_budget_tokens: int = 2400,
                 min_sec_quota: int=80,
                 max_sec_quota: int=320,
                 keep_per_section: int=6):
        self.name = name
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.hard_cap = max_input_tokens - max_output_tokens
         # 预算相关参数
        self.doc_budget_tokens = doc_budget_tokens
        self.min_sec_quota = min_sec_quota
        self.max_sec_quota = max_sec_quota
        self.keep_per_section = keep_per_section


class LLMConfig:
    """统一的 LLM 配置"""
    # 模型配置字典
    MODEL_CONFIGS = {
        "gpt-4o": ModelConfig("gpt-4o", 8192, 1024),
        "claude-3-5-sonnet": ModelConfig("claude-3-5-sonnet", 128000, 4096),
        "deepseek-reasoner": ModelConfig("deepseek-reasoner", 32768, 2048),
        "qwen2-5-72b-instruct": ModelConfig("qwen2-5-72b-instruct", 32768, 2048)
    }
    # 模型配置字典，包含每个模型的预算参数
    MODEL_CONFIGS = {
        "gpt-4o": ModelConfig(
            name="gpt-4o", # 1280000
            max_input_tokens=8192,
            max_output_tokens=1024,
            doc_budget_tokens=3200,
            min_sec_quota=80,
            max_sec_quota=320,
            keep_per_section=6
        ),
        "claude-3-5-sonnet": ModelConfig(
            name="claude-3-5-sonnet",  # 200000
            max_input_tokens=128000,
            max_output_tokens=4096,
            doc_budget_tokens=4800,  # 更大的文档预算
            min_sec_quota=160,       # 更大的最小配额
            max_sec_quota=640,       # 更大的最大配额
            keep_per_section=12      # 保留更多的每节内容
        ),
        "deepseek-reasoner": ModelConfig(
            name="deepseek-reasoner", # 64000
            max_input_tokens=32768,
            max_output_tokens=2048,
            doc_budget_tokens=3600,
            min_sec_quota=120,
            max_sec_quota=480,
            keep_per_section=8
        ),
        "qwen2-5-72b-instruct": ModelConfig(
            name="qwen2-5-72b-instruct", # 16000 
            max_input_tokens=32768,
            max_output_tokens=2048,
            doc_budget_tokens=3600,
            min_sec_quota=120,
            max_sec_quota=480,
            keep_per_section=8
        ),
       "gpt-5": ModelConfig(
            name="gpt-5",  # 400000
            max_input_tokens=256000,   
            max_output_tokens=2048,
            doc_budget_tokens=4000,
            min_sec_quota=80,
            max_sec_quota=400,
            keep_per_section=8
        ),
    }
     
    def __init__(self, api_key: str, base_url: str,model_name: str= "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name.lower()
        
         # 获取模型配置
        self.model_config = self.MODEL_CONFIGS.get(self.model_name)
        if not self.model_config:
            raise ValueError(f"不支持的模型: {model_name}")


        self.openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        ) if "gpt" in self.model_name else None
    # 属性访问器
    @property
    def max_input_tokens(self) -> int:
        return self.model_config.max_input_tokens

    @property
    def max_output_tokens(self) -> int:
        return self.model_config.max_output_tokens

    @property
    def hard_cap(self) -> int:
        return self.model_config.hard_cap

class BaseLLM(ABC):
    """LLM 基类"""
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def format_messages(self, system_prompt: str, developer_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """格式化消息"""
        pass

    @abstractmethod
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """调用模型"""
        pass

class GPT4oLLM(BaseLLM):
    def format_messages(self, system_prompt: str, developer_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            # {"role": "system", "content": f"{system_prompt}\n\n--- Developer Guidelines ---\n{developer_prompt}"},
            {"role":"system","content":system_prompt},
            {"role":"developer","content":developer_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        response = self.config.openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0
        )
        return response.choices[0].message.content
class GPT5LLM(BaseLLM):
    def format_messages(self, system_prompt: str, developer_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ]
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        response = self.config.openai_client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    
class SonnetLLM(BaseLLM):
    def format_messages(self, system_prompt: str, developer_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": f"{system_prompt}\n\n--- Developer Guidelines ---\n{developer_prompt}"},
            {"role": "user", "content": user_prompt}
        ]
    
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str: 
        response = requests.post(
            f"{self.config.base_url}chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0
            }
        )
        response_json = json.loads(response.text)
        content = response_json["choices"][0]["message"]["content"]
        return content.strip()

class DeepseekLLM(BaseLLM):
    def format_messages(self, system_prompt: str, developer_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": developer_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        response = requests.post(
            f"{self.config.base_url}chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-reasoner",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0
            }
        )
        response_json = json.loads(response.text)
        content = response_json["choices"][0]["message"]["content"]
        return content.strip()

class QwenLLM(BaseLLM):
    def format_messages(self, system_prompt: str, developer_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": f"{system_prompt}\n\n--- Developer Guidelines ---\n{developer_prompt}"},
            {"role": "user", "content": user_prompt}
        ]
    
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        response = requests.post(
            f"{self.config.base_url}chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Qwen2.5-72B-Instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0
            }
        )
        response_json = json.loads(response.text)
        content = response_json["choices"][0]["message"]["content"]
        return content.strip()

def create_llm(model_name: str, config: LLMConfig) -> BaseLLM:
    """LLM 工厂函数"""
    llm_map = {
        "gpt-4o": GPT4oLLM,
        "claude-3-5-sonnet": SonnetLLM,
        "deepseek-reasoner": DeepseekLLM,
        "qwen2-5-72b-instruct": QwenLLM,
        "gpt-5": GPT5LLM    
    }
    llm_class = llm_map.get(model_name.lower())
    if not llm_class:
        raise ValueError(f"不支持的模型: {model_name}")
    return llm_class(config)