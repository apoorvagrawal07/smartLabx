from app.rag.retriever import Retriever


retriever = Retriever()

results = retriever.retrieve(
    "What happens when a Kafka consumer crashes?"
)

for source, content, distance in results:

    print("=" * 60)

    print("SOURCE:", source)
    print("DISTANCE:", distance)
    print("CONTENT:", content)