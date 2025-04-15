from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
             """For the given objective, come up with a detailed step-by-step plan tailored for financial analysis and recommendations. Your plan should include:
                1. Understanding the user's question clearly.
                2. Identifying needed information, such as stock prices, financial metrics, market trends, or calculations.
                3. Selecting tools like `stock_price_retrieval`, `stock_financial_metrics_retrieval`, `tavily_search`, `calculator`and `retrieve_from_documents`.
                4. Planning tool usage sequence.
                5. Synthesizing information for response.
                6. Assessing risks, ensuring alignment with high risk tolerance and short-term horizon.
                This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
                7. Whenever user mention about documents or file make use of `retrieve_from_documents` tool.
                8. Whenever user mention about stock price or financial metrics make use of `stock_price_retrieval` and `stock_financial_metrics_retrieval` tools.
                9. Whenever user mention about calculation make use of `calculator` tool.
                10. Whenever user mention about search make use of `tavily_search` tool.
                11. Current year is 2025.
                12. And make sure the final response is properly formatted and easy to understand, context of everything discussed should be part of the final response.
            """
        ),
        ("placeholder", "{messages}")
    ]
)

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        Your objective was this:
        {input}

        Your original plan was this:
        {plan}

        You have currently done the follow steps:
        {past_steps}

        Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan. Also when you are done make sure the final answer is well formatted and summarized everything from the planning and past steps.
        Make sure the final response is properly formatted and easy to understand, context of everything discussed should be part of the final response.
    """
)

agent_executor_prompt = ChatPromptTemplate.from_template(
    """
        Original user input: {input}
        
        Full plan:
        {plan}

        Past steps executed:
        {past_steps}
        
        You are tasked with executing step 1: {task}. Use the original input and past steps to maintain context.
    """
)