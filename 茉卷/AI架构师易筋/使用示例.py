from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Annotated

# 定义系统状态结构
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], "List of HumanMessage objects"]
    current_task: str
    sub_tasks: dict
    error_log: list

# 定义任务分解智能体
class TaskDecomposer:
    def __init__(self, state: AgentState):
        self.state = state

    def task(self) -> AgentState:
        # 任务分解逻辑（示例）
        user_query = self.state["messages"][-1].content
        sub_tasks = {
            "transport": f"规划{user_query}中的交通方案",
            "hotel": f"推荐{user_query}期间的住宿",
            "attraction": f"建议{user_query}的景点路线"
        }
        self.state["sub_tasks"] = sub_tasks
        return self.state

# 定义交通规划智能体
class TransportPlanner:
    def __init__(self, state: AgentState):
        self.state = state

    def task(self) -> AgentState:
        # 交通规划逻辑
        # task = self.state["sub_tasks"]["transport"]
        new_message = HumanMessage(content=f"交通方案：北京-上海高铁G12次，08:00-12:00。预算：553元")
        self.state["messages"].append(new_message)
        return self.state

# 定义住宿推荐智能体
class HotelRecommender:
    def __init__(self, state: AgentState):
        self.state = state

    def task(self) -> AgentState:
        # 住宿推荐逻辑
        new_message = HumanMessage(content="推荐酒店：上海中心大厦君悦\n• 外滩景观房 ￥1200/晚\n• 含双早+行政酒廊")
        self.state["messages"].append(new_message)
        return self.state

# 定义景点规划智能体
class AttractionAdvisor:
    def __init__(self, state: AgentState):
        self.state = state

    def task(self) -> AgentState:
        # 景点规划逻辑
        new_message = HumanMessage(content="推荐路线：\nDay1: 外滩-城隍庙\nDay2: 迪士尼乐园\nDay3: 上海博物馆")
        self.state["messages"].append(new_message)
        return self.state

# 定义错误处理智能体
class ErrorHandler:
    def __init__(self, state: AgentState):
        self.state = state

    def task(self) -> AgentState:
        # 错误处理逻辑
        error = self.state["error_log"][-1]
        new_message = HumanMessage(content=f"系统遇到问题：{error}，已启动备用方案...")
        self.state["messages"].append(new_message)
        return self.state



draft_state = {
    "messages": [HumanMessage(content="帮我规划清明节上海3日游")],
    "current_task": "",
    "sub_tasks": {},
    "error_log": []
    }

agents= {
    "task_decomposer": TaskDecomposer(),  
    "transport": TransportPlanner(),
    "hotel": HotelRecommender(),
    "attraction": AttractionAdvisor(),
    "error_handler": ErrorHandler()
    }

# 构建LangGraph工作流
# 假设 StateGraph 是一个有效的类，这里需要确保它已经被正确导入
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("task_decomposer", agents["task_decomposer"].task)
workflow.add_node("transport", agents["transport"].task) 
workflow.add_node("hotel", agents["hotel"].task)
workflow.add_node("attraction", agents["attraction"].task)
workflow.add_node("error_handler", agents["error_handler"].task)

# 设置边关系
workflow.add_edge("task_decomposer", "transport")
workflow.add_edge("transport", "hotel")
workflow.add_edge("hotel", "attraction")

# 配置错误处理路径
workflow.add_conditional_edges(
    "transport",
    lambda state: "error" if "error" in state["error_log"] else "continue",
    {"error": "error_handler", "continue": "hotel"}
)

workflow.set_entry_point("task_decomposer")
workflow.set_finish_point("attraction")

# 编译执行
app = workflow.compile()

# 运行示例
result = app.invoke(draft_state)

print(result)