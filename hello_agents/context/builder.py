"""ContextBuilder - GSSC流水线实现

实现 Gather-Select-Structure-Compress 上下文构建流程:
1. Gather: 从多个数据源收集候选信息 (历史对话/Memory/RAG/工具结果)
2. Select: 基于优先级, 相关性, 多样性对候选信息进行筛选
3. Structure: 将筛选后的信息组织成结构化的上下文
4. Compress: 在 Token 预算内压缩与优化
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken
import math

from ..core.message import Message
from ..tools import MemoryTool, RAGTool

def count_tokens(text: str) -> int:
    """计算文本token数 (使用tiktoken)"""
    try:
        # TODO: 这个是啥? tokenizer吗?
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 降级方案: 粗略估算 (1 token ≈ 4 字符)
        return len(text) // 4
    
# TODO: 之前的章节中多次使用了 pydantic 中的 BaseModel 来定义数据模型, 思考这里为什么要使用 dataclass 
@dataclass
class ContextPacket:
    """上下文信息包"""
    content: str  # 信息内容
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    token_count: int = 0  # token数 (自动计算)
    relevance_score: float = 0.0  # 0.0 - 1.0  # 相关性评分 TODO: 与什么的相关性? 是动态计算的吗?

    def __post_init__(self):
        """自动计算 token 数"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)

@dataclass
class ContextConfig:
    """上下文构建配置"""
    max_tokens: int = 8000  # 上下文最大 token 预算
    reserve_ratio: float = 0.15  # 生成余量 (10-20%)
    min_relevance: float = 0.3  # 最小相关性阈值
    enable_mmr: bool = True  # 启用最大边际相关性 (多样性)
    mmr_lambda: float = 0.7  # MMR平衡参数 (0=纯多样性, 1=纯相关性)
    system_prompt_template: str = ""  # 系统提示模板
    enable_compression: bool = True  # 启用压缩

    def get_available_tokens(self) -> int:
        """获取可用 token 预算 (扣除余量)"""
        return int(self.max_tokens * (1 - self.reserve_ratio))

# TODO: ContextBuilder 似乎没有被封装为 Tool, 是否可以进行扩展?
class ContextBuilder:
    """上下文构建器 - 实现 GSSC 流水线
    
    用法示例:
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )
    
    context = builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """

    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RAGTool] = None,
        config: Optional[ContextConfig] = None
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self._encoding = tiktoken.get_encoding("cl100k_base")  # tokenizer?

    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None
    ):
        """构建完整上下文
        
        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            additional_packets: 额外的上下文包
        
        Returns:
            结构化上下文字符串
        """

        # 1. Gather: 收集候选信息
        packets: List[ContextPacket] = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or []
        )

        # 2. Select: 筛选与排序
        selected_packets: List[ContextPacket] = self._select(packets, user_query)

        # 3. Structure: 组织成结构化上下文
        structured_context: str = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            system_instructions=system_instructions
        )

        # 4. Compress: 压缩与优化
        final_context: str = self._compress(structured_context)
        
        return final_context

    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """Gather: 收集候选信息"""
        packets = []

        # P0: 系统指令
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))
        
        # P1: 记忆检索结果
        if self.memory_tool:
            try:
                # 搜索任务状态相关记忆
                state_results = self.memory_tool.execute(  # TODO: execute -> run ?
                    "search",
                    query="(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    min_importance=0.7,
                    limit=5
                )
                if state_results and "未找到" not in state_results:
                    packets.append(ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))
                
                # 搜索查询相关记忆
                related_results = self.memory_tool.execute(  # TODO: execute -> run ?
                    "search",
                    query=user_query,
                    limit=5
                )
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))

            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")
    
        # P2: RAG检索结果
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
                    packets.append(ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            
            except Exception as e:
                print(f"⚠️ RAG检索失败: {e}")
        
        # P3: 对话历史 (辅助)
        if conversation_history:
            # 只保留最近 N 条
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            packets.append(ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))
        
        # 添加额外包
        packets.extend(additional_packets)
        
        return packets

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """Select: 基于分数与预算的筛选"""
        # 1 计算相关性 (词重叠度)
        query_tokens = set(user_query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())
            if len(query_tokens) > 0:
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            
        # 2 时间衰减
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600  # 1小时时间尺度, 可暴露到配置
            return math.exp(-delta / tau)
        
        # 3 综合性评分
        scored_packets: List[Tuple[float, ContextPacket]] = []
        for p in packets:
            rec = recency_score(p.timestamp)
            score = 0.7 * p.relevance_score + 0.3 * rec
            scored_packets.append((score, p))
        
        # 4 过滤系统指令, 其余按分数排序
        system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
        remaining = [p for (_, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
                     if p.metadata.get("type") != "instructions"]
        
        # 5 过滤最小相关性
        filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]

        # 6 按预算填充
        available_tokens = self.config.get_available_tokens()
        selected: List[ContextPacket] = []
        used_tokens = 0

        ## 6.1 先添加系统指令
        for p in system_packets:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count
        
        ## 6.2 再添加其他按分数排序的包
        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count
        
        return selected
    
    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str,
        system_instructions: Optional[str]
    ) -> str:
        """Structure: 组织成结构化上下文模板"""
        sections = []

        # [Role & Policies]: 系统指令
        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)
        
        # [Task]: 当前任务
        sections.append(f"[Task]\n用户问题: {user_query}")

        # [State]: 任务状态
        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\n关键进展与未决问题: \n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)
        
        # [Evidence]: 事实依据 (Memory/RAG/Tool_result)
        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\n事实与引用: \n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)
        
        # [Context]: 辅助上下文 (对话历史等)
        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]
        if p3_packets:
            context_section = "[Context]\n对话历史与背景: \n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)
        
        # [Output]: 输出约束
        output_section = """[Output]
请按以下格式回答:
1. 结论 (简洁明确)
2. 依据 (列出支撑证据及来源)
3. 风险与假设 (如有)
4. 下一步行动建议 (如适用)
"""
        sections.append(output_section)
        
        return "\n\n".join(sections)
    
    def _compress(self, context: str) -> str:
        """Compress: 压缩与规范化"""
        if not self.config.enable_compression:
            return context
        
        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()

        if current_tokens <= available_tokens:
            return context
        
        # 简单截断策略 (保留前 N 个 token)
        # TODO: 实际应用中可用 LLM 做高保真摘要
        print(f"⚠️ 上下文超预算 ({current_tokens} > {available_tokens}), 执行截断")

        # 按段落截断, 保留结构
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0

        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens
        
        return "\n".join(compressed_lines)
