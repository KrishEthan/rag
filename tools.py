import math
import requests
import numexpr
from typing import Optional, Dict, Union, Any
from settings import env_settings  
import logging
from langchain_core.tools import tool
from rag import EthanRAG
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)

internet_search_tool = TavilySearchResults(max_results=5, tavily_api_key=env_settings.TAVILY_API_KEY)

@tool
def calculator(expression: str) -> str:
    """
    Calculate expression using Python's numexpr library.  Handles common math
    functions and operators.

    Args:
        expression (str): A single-line mathematical expression.

    Returns:
        str: The result of the calculation, or an error message.

    Examples:
        "37593 * 67"
        "37593**(1/5)"
        "sin(pi/4)"
    """
    try:
        local_dict: Dict[str, float] = {"pi": math.pi, "e": math.e}
        result: Union[int, float] = numexpr.evaluate(
            expression.strip(),
            global_dict={},
            local_dict=local_dict,
        )
        return str(result)
    except Exception as e:
        logger.exception(f"Calculation failed: {e}") 
        raise

headers: Dict[str, str] = {"X-API-KEY": env_settings.FINANCIAL_DATASET_API_KEY}

@tool
def stock_price_retrieval(ticker: str) -> Union[str, Dict[str, str]]:
    """
    Retrieves the current stock price for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        Union[str, Dict[str, str]]: The stock price as a string, or an error
        dictionary if the retrieval fails.
    """
    try:
        url: str = (
            f'https://api.financialdatasets.ai/prices/snapshot'
            f'?ticker={ticker}'
        )
        response: requests.Response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data: Dict[str, Any] = response.json()
            price: Optional[float] = data.get("snapshot", {}).get("price")
            if price is not None:
                return str(price)
            else:
                logger.error("Stock price not found in response.")
                raise ValueError("Stock price not found in response.")
        else:
            logger.error(f"Failed to retrieve stock price. Status code: {response.status_code}")
            raise requests.exceptions.HTTPError(f"Failed to retrieve stock price. Status code: {response.status_code}")
    except requests.RequestException as e:
        logger.exception(f"Request error: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise

@tool
def stock_financial_metrics_retrieval(ticker: str) -> Union[Dict[str, Any], Dict[str, str]]:
    """
    Retrieves financial metrics for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        Union[Dict[str, Any], Dict[str, str]]: A dictionary containing the
        financial metrics, or an error dictionary if the retrieval fails.
    """
    try:
        url: str = (
            f'https://api.financialdatasets.ai/financial-metrics/snapshot'
            f'?ticker={ticker}'
        )
        response: requests.Response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data: Dict[str, Any] = response.json()
            snapshot: Optional[Dict[str, Any]] = data.get("snapshot")
            if snapshot:
                return snapshot
            else:
                logger.error("Financial metrics snapshot not found in response.")
                raise ValueError("Financial metrics snapshot not found in response.")
        else:
            logger.error(f"Failed to retrieve financial metrics. Status code: {response.status_code}")
            raise requests.exceptions.HTTPError(f"Failed to retrieve financial metrics. Status code: {response.status_code}")
    except requests.RequestException as e:
        logger.exception(f"Request error: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise

@tool
async def retrieve_from_documents(query: str):
    """
    Retrieves relevant documents based on the provided query.

    Args:
        query (str): The search query.

    Returns:
        str: The retrieved documents as a string.
    """
    try:
        ethan_rag = EthanRAG()
        result = ethan_rag.query(query)
        return result
    except Exception as e:
        logger.exception(f"Document retrieval failed: {e}")
        raise