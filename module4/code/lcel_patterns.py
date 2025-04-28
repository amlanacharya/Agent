"""
LCEL Patterns
------------
This module demonstrates common patterns and techniques using LangChain Expression Language (LCEL).
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import langchain
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with 'pip install langchain-core' for LCEL functionality.")

# Try to import the GroqClient for LLM integration
try:
    # Try module3 path first
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    from module3.code.groq_client import GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    try:
        # Try module2-llm path
        from module2_llm.code.groq_client import GroqClient
        GROQ_AVAILABLE = True
    except ImportError:
        GROQ_AVAILABLE = False
        logger.warning("GroqClient not available. Some features will be limited.")


class SimpleLLMClient:
    """A simple LLM client for testing."""

    def generate_text(self, prompt, **kwargs):
        """Generate text from a prompt."""
        # This is a very simple simulation
        return {
            "choices": [
                {
                    "message": {
                        "content": f"This is a simulated response to: {prompt[:50]}..."
                    }
                }
            ]
        }

    def extract_text_from_response(self, response):
        """Extract text from a response."""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return response


class LCELPatterns:
    """Demonstrates common LCEL patterns."""

    def __init__(self):
        """Initialize the LCEL patterns demo."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LCEL patterns. Install with 'pip install langchain'")

        # Initialize LLM client
        if GROQ_AVAILABLE:
            self.llm_client = GroqClient()
        else:
            self.llm_client = SimpleLLMClient()

    def pattern1_basic_chain(self):
        """
        Pattern 1: Basic Chain

        This pattern demonstrates the simplest LCEL chain that:
        1. Takes a question
        2. Formats it into a prompt
        3. Passes the prompt to the LLM
        4. Returns the response
        """
        # Create a prompt template
        prompt = PromptTemplate.from_template("""
        Please answer the following question:

        Question: {question}

        Answer:
        """)

        # Create the chain
        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | RunnableLambda(lambda x: self.llm_client.generate_text(str(x)))
            | RunnableLambda(lambda x: self.llm_client.extract_text_from_response(x))
        )

        return chain

    def pattern2_transformation_chain(self):
        """
        Pattern 2: Transformation Chain

        This pattern demonstrates how to transform inputs and outputs in a chain:
        1. Takes a question
        2. Transforms it to uppercase
        3. Formats it into a prompt
        4. Passes the prompt to the LLM
        5. Transforms the response to title case
        6. Returns the transformed response
        """
        # Create a prompt template
        prompt = PromptTemplate.from_template("""
        Please answer the following question:

        Question: {question}

        Answer:
        """)

        # Create input and output transformation functions
        def transform_input(question):
            return question.upper()

        def transform_output(response):
            return response.title()

        # Create the chain
        chain = (
            RunnableLambda(transform_input)
            | {"question": RunnablePassthrough()}
            | prompt
            | RunnableLambda(lambda x: self.llm_client.generate_text(str(x)))
            | RunnableLambda(lambda x: self.llm_client.extract_text_from_response(x))
            | RunnableLambda(transform_output)
        )

        return chain

    def pattern3_branching_chain(self):
        """
        Pattern 3: Branching Chain

        This pattern demonstrates how to create a chain with branching logic:
        1. Takes a question
        2. Classifies the question type
        3. Routes to the appropriate sub-chain
        4. Returns the response
        """
        # Create prompt templates for different question types
        factual_prompt = PromptTemplate.from_template("""
        Please provide a factual answer to the following question:

        Question: {question}

        Factual Answer:
        """)

        opinion_prompt = PromptTemplate.from_template("""
        Please provide your opinion on the following question:

        Question: {question}

        Opinion:
        """)

        # Create a function to classify the question type
        def classify_question(question):
            if any(word in question.lower() for word in ["think", "opinion", "feel", "believe"]):
                return "opinion"
            else:
                return "factual"

        # Create sub-chains for each question type
        def create_chain_for_type(prompt_template):
            return (
                {"question": RunnablePassthrough()}
                | prompt_template
                | RunnableLambda(lambda x: self.llm_client.generate_text(str(x)))
                | RunnableLambda(lambda x: self.llm_client.extract_text_from_response(x))
            )

        factual_chain = create_chain_for_type(factual_prompt)
        opinion_chain = create_chain_for_type(opinion_prompt)

        # Create a router function
        def route_question(question):
            question_type = classify_question(question)

            if question_type == "opinion":
                return opinion_chain.invoke(question)
            else:
                return factual_chain.invoke(question)

        # Create the main chain
        chain = RunnableLambda(route_question)

        return chain

    def pattern4_parallel_chain(self):
        """
        Pattern 4: Parallel Chain

        This pattern demonstrates how to process inputs in parallel:
        1. Takes a question
        2. Processes it through multiple parallel paths
        3. Combines the results
        4. Returns the combined response
        """
        # Create prompt templates for different perspectives
        scientific_prompt = PromptTemplate.from_template("""
        Please provide a scientific perspective on the following question:

        Question: {question}

        Scientific Perspective:
        """)

        historical_prompt = PromptTemplate.from_template("""
        Please provide a historical perspective on the following question:

        Question: {question}

        Historical Perspective:
        """)

        # Create sub-chains for each perspective
        def create_chain_for_perspective(prompt_template):
            return (
                {"question": RunnablePassthrough()}
                | prompt_template
                | RunnableLambda(lambda x: self.llm_client.generate_text(str(x)))
                | RunnableLambda(lambda x: self.llm_client.extract_text_from_response(x))
            )

        scientific_chain = create_chain_for_perspective(scientific_prompt)
        historical_chain = create_chain_for_perspective(historical_prompt)

        # Create a parallel chain
        parallel_chain = RunnableParallel(
            scientific=scientific_chain,
            historical=historical_chain
        )

        # Create a function to combine results
        def combine_perspectives(results):
            scientific = results["scientific"]
            historical = results["historical"]

            return f"""
            Multiple Perspectives:

            Scientific Perspective:
            {scientific}

            Historical Perspective:
            {historical}
            """

        # Create the main chain
        chain = parallel_chain | RunnableLambda(combine_perspectives)

        return chain

    def pattern5_memory_chain(self):
        """
        Pattern 5: Memory Chain

        This pattern demonstrates how to create a chain with memory:
        1. Takes a question and conversation history
        2. Formats the history and question into a prompt
        3. Passes the prompt to the LLM
        4. Returns the response
        """
        # Create a prompt template
        prompt = PromptTemplate.from_template("""
        You are having a conversation with a human. Respond to their latest question or message.

        Conversation History:
        {history}

        Human: {question}

        AI:
        """)

        # Create a function to format conversation history
        def format_history(history):
            formatted_history = []
            for message in history:
                role = message.get("role", "")
                content = message.get("content", "")

                if role.lower() == "human":
                    formatted_history.append(f"Human: {content}")
                elif role.lower() == "ai":
                    formatted_history.append(f"AI: {content}")

            return "\n".join(formatted_history)

        # Create the chain
        chain = (
            {
                "question": lambda x: x["question"],
                "history": lambda x: format_history(x["history"])
            }
            | prompt
            | RunnableLambda(lambda x: self.llm_client.generate_text(str(x)))
            | RunnableLambda(lambda x: self.llm_client.extract_text_from_response(x))
        )

        return chain


# Example usage
if __name__ == "__main__":
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not available. Install with 'pip install langchain' to run these examples.")
        sys.exit(1)

    # Create the LCEL patterns demo
    patterns = LCELPatterns()

    # Pattern 1: Basic Chain
    print("\n=== Pattern 1: Basic Chain ===")
    chain1 = patterns.pattern1_basic_chain()
    result1 = chain1.invoke("What is the capital of France?")
    print(f"Result: {result1}")

    # Pattern 2: Transformation Chain
    print("\n=== Pattern 2: Transformation Chain ===")
    chain2 = patterns.pattern2_transformation_chain()
    result2 = chain2.invoke("What is the capital of France?")
    print(f"Result: {result2}")

    # Pattern 3: Branching Chain
    print("\n=== Pattern 3: Branching Chain ===")
    chain3 = patterns.pattern3_branching_chain()

    factual_result = chain3.invoke("What is the capital of France?")
    print(f"Factual Result: {factual_result}")

    opinion_result = chain3.invoke("What do you think about Paris?")
    print(f"Opinion Result: {opinion_result}")

    # Pattern 4: Parallel Chain
    print("\n=== Pattern 4: Parallel Chain ===")
    chain4 = patterns.pattern4_parallel_chain()
    result4 = chain4.invoke("What is climate change?")
    print(f"Result: {result4}")

    # Pattern 5: Memory Chain
    print("\n=== Pattern 5: Memory Chain ===")
    chain5 = patterns.pattern5_memory_chain()

    history = [
        {"role": "human", "content": "My name is Alice."},
        {"role": "ai", "content": "Hello Alice, nice to meet you!"},
        {"role": "human", "content": "I'm interested in artificial intelligence."},
        {"role": "ai", "content": "That's great! AI is a fascinating field with many applications."}
    ]

    result5 = chain5.invoke({"question": "What was my name again?", "history": history})
    print(f"Result: {result5}")
