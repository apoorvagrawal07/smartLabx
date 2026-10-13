from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph.state import GraphState
from app.graph.nodes import (
    router_node,
    direct_node,
    rag_node,
    tool_node,
    route_decision,
    agent_llm_node,
    execute_tool_node,
    tool_decision,
)


def build_graph(
    model_gateway,
    rag_service,
    tool_executor
):

    graph = StateGraph(GraphState)

    graph.add_node(
        "router",
        router_node
    )

    graph.add_node(
        "llm",
        lambda state: agent_llm_node(
            state,
            model_gateway
        )
    )

    graph.add_node(
        "direct",
        lambda state: direct_node(
            state,
            model_gateway
        )
    )

    graph.add_node(
        "rag",
        lambda state: rag_node(
            state,
            rag_service
        )
    )

    graph.add_node(
        "tool",
        lambda state: execute_tool_node(
            state,
            tool_executor
        )
    )

    graph.add_edge(
        START,
        "router"
    )

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "llm": "llm",
            "rag": "rag",
            "direct": "direct",
        }
    )

    graph.add_conditional_edges(
        "llm",
        tool_decision,
        {
            "tool": "tool",
            "end": END,
        }
    )

    graph.add_edge(
        "tool",
        "llm"
    )

    graph.add_edge(
        "direct",
        END
    )

    graph.add_edge(
        "rag",
        END
    )
    
    checkpointer = MemorySaver()

    return graph.compile(
    checkpointer=checkpointer
)

