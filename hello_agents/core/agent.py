"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config

class Agent(ABC):
    """Agent基类"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """初始化 Agent
        
        Args:
            name: Agent 名称
            llm: LLM Client
            system_prompt: 系统提示词
            config: HelloAgents 配置
        """
        self.name = name  
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []  # 消息历史记录
    
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent"""
        pass
    
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
    
    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()
    
    def clear_history(self):
        """清空历史记录"""
        self._history.clear()
    
    def __str__(self) -> str:
        return f"Agent(name={self.name})"