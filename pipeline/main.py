from pipeline.workflow import build_graph
from langchain_core.messages import HumanMessage

def call_agent(user_input: str, chat_id: str):
    human_msg = HumanMessage(content = user_input)
    init_state = {
        "user_message": user_input
    }
    result = graph.invoke(
        input = init_state,
        config = {"configurable": {"thread_id": chat_id}}
    )
    for m in result["chat_history"]:
        m.pretty_print()
    print (result["data_position"])
    print (result["data_metadata"])

def test_chatbot():
    call_agent("""
What was Amazon's operating income loss for Q2 2025?
               """, "005")
    call_agent("""
NetIncomeLoss of CVS health in Q1 2025?
               """, "005")

if __name__ == "__main__":
    model_name = 'gpt-4o-mini'
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    global graph
    graph = build_graph(
        model_name = model_name, 
        embedding_model = embedding_model
    )
    test_chatbot()