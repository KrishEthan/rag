import chainlit as cl
from rag import EthanRAG
from model import PlanExecuteState
from nodes import planning_node, replanning_node, agent_action_node, should_end
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planning_node)
workflow.add_node("agent", agent_action_node)
workflow.add_node("replan", replanning_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)
 


app = workflow.compile()
config = {"recursion_limit": 50}


@cl.on_message
async def on_message(message: cl.Message):
    inputs = {"input": message.content}
    async with cl.Step("Planning....", type="llm") as step:
        cl.user_session.set("step", step)

    pdfs = [file for file in message.elements if "pdf" in file.mime]
    if pdfs:
        for pdf in pdfs:
            file_path = pdf.path
            try:
                rag = EthanRAG()
                result = rag.process_and_store_pdf(file_path, pdf.name)
                step.name = "Processing your PDF..."
                step.type = "embedding"
            except Exception as e:
                await cl.Message(content=f"An error occurred during PDF processing: {e}").send() 
    
        

    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            print(f"{k}: {v}: {type(k)}: {type(v)}")

