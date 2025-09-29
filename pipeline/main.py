from pipeline.workflow import build_graph
from pipeline.main import call_agent

import time
from langchain_core.messages import HumanMessage

def call_agent(graph, user_input: str, chat_id: str):
    init_state = {
        "user_message": user_input
    }
    result = graph.invoke(
        input = init_state,
        config = {"configurable": {"thread_id": chat_id}}
    )

    print ("CHAT HISTORY:")
    for m in result["chat_history"]:
        m.pretty_print()
    print ('\n' * 5, "DATA:", sep = "")
    for data in result["retrieved_data"]:
        print (data.payload["metadata"], f"Line: {data.payload["position"] + 2}", sep = '\n', end = '\n' * 3)

def test_chatbot():
    call_agent(graph, """
What was Amazon's OperatingIncomeLoss for Q2 2025?
               """, "001")
#     call_agent(graph, """
# What is the NetIncomeLoss of CVS Health in Q1 2025?
#                """, "001")

if __name__ == "__main__":
    start_time = time.time()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = "gpt-4o-mini"

    global graph
    graph = build_graph(
        model_name = model_name, 
        embedding_model = embedding_model
    )

    # test_chatbot()
    print ("Total services/chatbot execution time:", time.time() - start_time)