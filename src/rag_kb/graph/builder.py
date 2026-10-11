"""Assembly of the Corrective RAG graph.

START -> rewrite -> retrieve -> grade --(enough relevant)----------------> generate -> END
                       ^                 \\--(too few, retries left)--> broaden --+
                       +-----------------------------------------------------------+
(too few, retry budget spent) -> generate
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_kb.config import Settings
from rag_kb.graph.nodes import Nodes, SearchFn
from rag_kb.graph.state import RAGState


def build_graph(
    llm: BaseChatModel,
    search: SearchFn,
    settings: Settings,
    json_llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    nodes = Nodes(llm=llm, json_llm=json_llm or llm, search=search, settings=settings)

    graph = StateGraph(RAGState)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("broaden", nodes.broaden)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", nodes.route_after_grade, {"generate": "generate", "broaden": "broaden"}
    )
    graph.add_edge("broaden", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
