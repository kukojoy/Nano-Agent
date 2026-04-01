from typing import Any, Callable, Optional
from .base import Tool


class ToolRegistry:
    """HelloAgents工具注册表
    
    提供工具的注册, 管理和执行功能.
    
    支持两种工具注册方式:
        1. Tool对象注册 (推荐)
        2. 函数注册 (简便, 适合简单工具)
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True):
        """注册 Tool 对象
        
        Args: 
            tool: 继承自 Tool 基类的工具实例对象
            auto_expand: 是否自动展开可扩展工具 (默认True)
        """        

        print("正在注册工具:", tool.name)  # 调试输出工具名称

        # 检查工具是否可展开
        if auto_expand and hasattr(tool, "get_expanded_tools"): # and tool.expandable:
            print(f"工具 '{tool.name}' 支持自动展开, 正在展开...")  # 调试输出展开信息
            expanded_tools = tool.get_expanded_tools()

            if not expanded_tools:
                print(f"⚠️ 警告: 工具 '{tool.name}' 展开子工具失败.")

            if expanded_tools:
                # 注册所有被展开的子工具
                for sub_tool in expanded_tools:
                    if sub_tool.name in self._tools:
                        print(f"⚠️ 警告: 工具 '{sub_tool.name}' 已存在, 将被覆盖.")
                    self._tools[sub_tool.name] = sub_tool
                print(f"✅ 工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立子工具.")
                return

        # 普通工具或不展开的工具
        if tool.name in self._tools:
            print(f"⚠️ 警告: 工具 '{tool.name}' 已存在, 将被覆盖.")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册.")  

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        直接注册函数作为工具(简便方式)

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数, 接受字符串参数, 返回字符串结果
        """
        if name in self._functions:
            print(f"⚠️ 警告: 工具 '{name}' 已存在, 将被覆盖.")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册.")
    
    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            print(f"🗑️ 工具 '{name}' 已注销.")
        elif name in self._functions:
            del self._functions[name]
            print(f"🗑️ 工具 '{name}' 已注销.")
        else:
            print(f"⚠️ 工具 '{name}' 不存在.")

    def get_tool(self, name: str) -> Optional[Tool]:
        """根据名称获取Tool对象"""
        return self._tools.get(name)
    
    def get_function(self, name: str) -> Optional[Callable]:
        """根据名称获取函数工具"""
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """执行工具
        
        Args:
            name: 工具名称
            input_text: 工具输入参数
        """
        # 优先查找Tool对象
        if name in self._tools:
            tool = self._tools[name]
            try:
                return tool.run({"input": input_text})
            except Exception as e:
                return f"❌ 工具 '{name}' 执行失败: {str(e)}"
        # 查找函数工具
        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"❌ 工具 '{name}' 执行失败: {str(e)}"
        # 未找到工具
        else:
            return f"❌ 工具 '{name}' 不存在."

    def get_tools_description(self) -> str:
        """获取所有可用工具的格式化描述字符串
        
        Returns:
            工具描述字符串, 每行一个工具, 包含工具名称和描述, 用于构建提示词
        """
        descriptions = []
    
        # Tool对象描述
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
    
        # 函数工具描述
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")
    
        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def list_tools(self) -> list[str]:
        """列出所有注册的工具名称"""
        return list(self._tools.keys()) + list(self._functions.keys())
    
    def get_all_tools(self) -> list[Tool]:
        """获取所有Tool对象"""
        return list(self._tools.values())
    
    def clear(self):
        """清空所有注册的工具"""
        self._tools.clear()
        self._functions.clear()
        print("🧹 工具注册表已清空.")

# 全局工具注册表
global_registry = ToolRegistry()