from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Annotated

# 定义系统状态结构
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], "List of HumanMessage objects"]
    current_task: str
    sub_tasks: dict
    error_log: list

# 定义各领域智能体
def task_decomposer(state: AgentState) -> AgentState:
# 任务分解逻辑（示例）
    user_query = state["messages"][-1].content
    sub_tasks = {
    "transport": f"规划{user_query}中的交通方案",
    "hotel": f"推荐{user_query}期间的住宿",
    "attraction": f"建议{user_query}的景点路线"
        }
    state["sub_tasks"] = sub_tasks
    return state

def transport_planner(state: AgentState) -> AgentState:
    # 交通规划智能体（可接入实时API）
    task = state["sub_tasks"]["transport"]
    new_message = HumanMessage(content=f"交通方案：北京-上海高铁G12次，08:00-12:00。预算：553元")
    state["messages"].append(new_message)
    return state

def hotel_recommender(state: AgentState) -> AgentState:
    # 住宿推荐智能体（含价格筛选逻辑）
    new_message = HumanMessage(content="推荐酒店：上海中心大厦君悦\n• 外滩景观房 ￥1200/晚\n• 含双早+行政酒廊")
    state["messages"].append(new_message)
    return state

def attraction_advisor(state: AgentState) -> AgentState:
    # 景点规划智能体（含路线优化）
    new_message = HumanMessage(content="推荐路线：\nDay1: 外滩-城隍庙\nDay2: 迪士尼乐园\nDay3: 上海博物馆")
    state["messages"].append(new_message)
    return state

def error_handler(state: AgentState) -> AgentState:
    # 错误处理智能体
    error = state["error_log"][-1]
    new_message = HumanMessage(content=f"系统遇到问题：{error}，已启动备用方案...")
    state["messages"].append(new_message)
    return state

# 构建LangGraph工作流
# 假设 StateGraph 是一个有效的类，这里需要确保它已经被正确导入
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("task_decomposer", task_decomposer)
workflow.add_node("transport", transport_planner) 
workflow.add_node("hotel", hotel_recommender)
workflow.add_node("attraction", attraction_advisor)
workflow.add_node("error_handler", error_handler)

# 设置边关系
workflow.add_edge("task_decomposer", "transport")
workflow.add_edge("transport", "hotel)")