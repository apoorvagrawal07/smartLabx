from test_graph import workflow

queries = [
    "what is 135*7?",
    "what time is it?",
    "calculate 125 * 42 and what time is it?",
]

for msg in queries:
    print("INPUT:", msg)
    result = workflow.invoke({"user_message": msg})
    print("ROUTE:", result.get("route"))
    print("ANSWER:", result.get("answer"))
    print("---")