"""HelloAgents统一 LLM 接口"""

import os
from openai import OpenAI
from typing import Optional, List, Dict, Iterator

class NanoAgentLLM:
    """
    NanoAgent 统一 LLM 客户端
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化 LLM 客户端

        参数优先级: 传入参数 > 环境变量

        Args:
            model: 模型名称, 默认从 LLM_MODEL_ID 读取
            api_key: API 密钥, 默认从 LLM_API_KEY 读取
            base_url: 服务地址, 默认从 LLM_BASE_URL 读取
            timeout: 超时时间 (秒), 默认从 LLM_TIMEOUT 读取, 默认 60 秒
            temperature: 温度参数, 默认 0.7
            max_tokens: 最大 token 数
        """

        # 加载配置
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # 检查必填参数
        if not self.model:
            raise ValueError("模型名称未设置")

        if not self.api_key:
            raise ValueError("API Key 未设置")
        
        if not self.base_url:
            raise ValueError("服务 URL 未设置")

        # 创建客户端
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """非流式调用 LLM, 返回完整响应, 适用于不需要流式输出的场景"""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"❌ 调用 LLM API 时发生错误: {e}")
            raise


    def think(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用大语言模型进行思考, 并返回流式响应.
        这是主要的调用方法, 默认使用流式响应以获得更好的用户体验.

        Args:
            messages: 消息列表
            temperature: 温度参数, 如果未提供则使用初始化时的值

        Returns:
            str: 模型的完整响应文本
        """
        print(f"🧠 正在调用 {self.model} 模型...")

        # 准备参数
        if temperature is None:
            temperature = self.temperature
               
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **self.kwargs
            )
        
            print("✅ 大语言模型响应成功:")

            # 处理流式响应
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    # print(content, end="", flush=True)
                    yield content
            # print()  # 在流式输出结束后换行
            
        except Exception as e:
            print(f"❌ 调用 LLM API 时发生错误: {e}")
            raise
    
    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        流式调用 LLM 的别名方法, 与 think 方法功能相同.
        """
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)