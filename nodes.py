import logging
from gtts import gTTS
from model import PlanExecuteState, Plan, Act, Response
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END
from prompt import planner_prompt, replanner_prompt, agent_executor_prompt
from settings import env_settings
from tools import internet_search_tool, stock_price_retrieval, stock_financial_metrics_retrieval, calculator, retrieve_from_documents
from utils import set_output_callback, message_callback
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", 
    temperature=0, 
    api_key=env_settings.OPENAI_API_KEY)

plan_structured_output = llm.with_structured_output(Plan)
replanner_structured_output = llm.with_structured_output(Act)

planner = planner_prompt | plan_structured_output
replanner = replanner_prompt | replanner_structured_output
react_agent_executor = create_react_agent(llm, tools=[internet_search_tool, stock_price_retrieval, stock_financial_metrics_retrieval, calculator, retrieve_from_documents])


async def planning_node(state: PlanExecuteState) -> PlanExecuteState:
    """
    This function takes the current state of the planning process, invokes
    the planner to generate a plan, and returns the updated state.  It
    includes error handling with logging.

    Args:
        state (PlanExecuteState): The current state of the planning process.

    Returns:
        PlanExecuteState: The updated state, including the generated plan.
    """
    try:
        logger.info(f"Planning node invoked")
        response: Plan = await planner.ainvoke(
            {
                "messages": [("user", state["input"])]
            }
        )
        steps_list = [f"{idx + 1}. {step}" for idx, step in enumerate(response.steps)]
        await set_output_callback("\n".join(steps_list))
        
        return {
            "plan": response.steps,
            "ui_steps": response.ui_steps,
        }
    except Exception as e:
        logger.exception(f"Error in planning_node: {e}")
        raise e

async def replanning_node(state: PlanExecuteState) -> PlanExecuteState | Response:
    try:
        response: Act = await replanner.ainvoke(
            {
                "input": state["input"],
                "plan": "\n".join(state["plan"]),
                "past_steps": "\n".join([f"{t[0]}: {t[1]}" for t in state["past_steps"]]) if state["past_steps"] else "None yet"
            }
        )

        logger.info(f"Replanning response: {response}")

        if isinstance(response.action, Response):
            logger.info(f"Final response: {response.action.response}")
            podcast_summary = await generate_podcast_summary(response.action.response)
            await text_to_speech(podcast_summary)

            return {
                "response": response.action.response,
            }
        else:
            return {
                "plan": response.action.steps,
            }
            
    except Exception as e:
        logger.exception(f"Error in replanning_node: {e}")
        raise e
    
async def agent_action_node(state: PlanExecuteState) -> PlanExecuteState:
    """
    This function takes the current state of the planning process, invokes
    the agent to execute the plan, and returns the updated state. It
    includes error handling with logging.

    Args:
        state (PlanExecuteState): The current state of the planning process.

    Returns:
        PlanExecuteState: The updated state after executing the plan.
    """
    try:
        logger.info(f"Agent action node invoked")
        plan = state['plan']
        task = plan[0]
        input=state['input']
        ui_step = state['ui_steps'][0]

        plan="\n".join(f"{i+1}. {step}" for i, step in enumerate(plan)),
        past_steps="\n".join([f"{t[0]}: {t[1]}" for t in state["past_steps"]]) if state["past_steps"] else "None yet",
        
        executor_prompt = f"""
            Original user input: {input}
            
            Full plan:
            {plan}

            Past steps executed:
            {past_steps}
            
            You are tasked with executing step 1: {task}. Use the original input and past steps to maintain context.
        """

        response = await react_agent_executor.ainvoke(
            {
                "messages": [("user", executor_prompt)],
            }
        )
        content = response["messages"][-1].content
        await set_output_callback(content, name=ui_step)
        return {
            "plan": plan[1:],  
            "ui_steps": state["ui_steps"][1:] if state["ui_steps"] else [],
            "past_steps": state["past_steps"] + [(task, content)],
        }
    except Exception as e:
        logger.exception(f"Error in agent_action_node: {e}")
        raise e

async def generate_podcast_summary(final_response: str) -> str:
    """
    Generates a concise podcast-style summary from the final response.

    Args:
        final_response (str): The final response from the agent.

    Returns:
        str: A podcast-style summary text.
    """
    try:
        summary_prompt = f"""
        You are a podcast host summarizing a recent analysis. Create a concise, engaging summary (2-3 sentences) 
        of the following content for a podcast audience. Use a friendly and conversational tone, 
        and avoid technical jargon unless necessary:

        {final_response}

        Summary:
        """
        summary_response = await llm.ainvoke(summary_prompt)
        summary_text = summary_response.content
        logger.info(f"Generated podcast summary: {summary_text}")
        return summary_text
    except Exception as e:
        logger.exception(f"Error in generate_podcast_summary: {e}")
        raise e

async def text_to_speech(text: str, output_file: str = "podcast_summary.mp3"):
    """
    Converts the given text to speech using gTTS and saves it as an MP3 file.

    Args:
        text (str): The text to convert to speech.
        output_file (str): The name of the output MP3 file.
    """
    try:
        logger.info(f"Converting text to speech: {text[:50]}...")
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        logger.info(f"Podcast summary saved as {output_file}")
    except Exception as e:
        logger.exception(f"Error in text_to_speech: {e}")
        raise e

async def should_end(state: PlanExecuteState):
    logger.info(f"Checking if should end {state}")
    if "response" in state and state["response"]:
        logger.info(f"Final response: {state['response']}")
        await message_callback(state["response"])
        return END
    else:
        logger.info(f"Continuing execution")
        return "agent"
