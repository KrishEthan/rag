import chainlit as cl

async def set_output_callback(output: str, name: str | None = None):
    step: cl.Step = cl.user_session.get("step")
    
    if name:
        step.name = name
        step.type = "llm"

    step.output = output
    await step.update()

async def message_callback(message: str):
    await cl.Message(message, type="assistant_message").send()

