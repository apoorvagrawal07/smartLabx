
from app.graph.workflow import build_graph

from app.services.model_gateway import ModelGateway
from app.rag.rag_service import RAGService
from app.rag.retriever import Retriever
from app.tools.tool_executor import ToolExecutor


model_gateway = ModelGateway()


class LazyRetriever:

    def __init__(self):
        self._retriever = None

    def retrieve(self, query, limit=3):
        if self._retriever is None:
            self._retriever = Retriever()

        return self._retriever.retrieve(query, limit)


rag_service = RAGService(
    retriever=LazyRetriever(),
    model_gateway=model_gateway,
)

tool_executor = ToolExecutor()


workflow = build_graph(
    model_gateway=model_gateway,
    rag_service=rag_service,
    tool_executor=tool_executor,
)


def test(message):

    print("\n")
    print("=" * 70)
    print("INPUT:", message)

    result = workflow.invoke({
        "user_message": message
    })

    print("ROUTE:", result.get("route"))
    print("ANSWER:", result.get("answer"))

    return result


if __name__ == "__main__":
    test(
   "what is"
)