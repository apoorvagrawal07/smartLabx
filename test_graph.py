
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


def test(message, config):
    checkpoint = workflow.get_state(config)
    history = checkpoint.values.get("messages", [])

    result = workflow.invoke(
        {
            "messages": history + [
                {"role": "user", "content": message}
            ],
            "user_message": message,
            "iteration_count": 0,
        },
        config=config,
    )

    print("INPUT:", message)
    print("ANSWER:", result["answer"])
    print()

    return result


if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "apoorva-memory-test"
        }
    }

    test("My name is Apoorva.", config)
    test("What is my name?", config)

    checkpoint = workflow.get_state(config)

    print("SAVED MESSAGES:")
    for message in checkpoint.values["messages"]:
        print(f'{message["role"]}: {message.get("content", "")}')

    user_messages = [
        message["content"]
        for message in checkpoint.values["messages"]
        if message["role"] == "user"
    ]
    assert user_messages == [
        "My name is Apoorva.",
        "What is my name?",
    ]

    assert checkpoint.next == ()
    print("\nCheckpoint test passed.")