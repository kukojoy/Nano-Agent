"""消息系统"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal
from datetime import datetime

# 定义消息角色类型, 包括系统 (system), AI (assistant), 用户 (user), 工具 (tool)
MessageRole = Literal["system", "assistant", "user", "tool"]

class Message(BaseModel):
    """消息类
    
    Args:
        content: 消息内容
        role: 消息角色
        timestamp: 消息时间戳, 默认为当前时间
        metadata: 可选的元数据字典, 用于存储额外信息
    """

    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 (OpenAI API 格式)"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
