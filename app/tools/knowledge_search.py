def create_knowledge_search_tool(
    retriever
):

    def knowledge_search(
        query: str
    ):

        results = retriever.retrieve(
            query,
            limit=3
        )

        return [
            {
                "source": source,
                "content": content,
                "distance": distance
            }
            for source, content, distance
            in results
        ]

    return knowledge_search