"""
RAG Evaluation Frameworks

This module implements various evaluation metrics and frameworks for assessing
the performance of RAG systems, including:
- Retrieval evaluation metrics
- Generation quality metrics
- End-to-end RAG evaluation
- Evaluation visualization

All implementations use LangChain Expression Language (LCEL) for improved
readability and composability.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Check if required packages are available
try:
    from langchain.evaluation import load_evaluator
    LANGCHAIN_EVAL_AVAILABLE = True
except ImportError:
    LANGCHAIN_EVAL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class RetrievalEvaluator:
    """Base class for evaluating retrieval performance."""

    def __init__(self, relevant_docs: Optional[List[Document]] = None):
        """
        Initialize the retrieval evaluator.

        Args:
            relevant_docs: Optional list of known relevant documents for ground truth
        """
        self.relevant_docs = relevant_docs

    def precision_at_k(self, retrieved_docs: List[Document], k: int = 5) -> float:
        """
        Calculate precision@k for retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents
            k: Number of top documents to consider

        Returns:
            Precision@k score (0-1)
        """
        if not self.relevant_docs or not retrieved_docs:
            return 0.0

        # Get top-k documents
        top_k_docs = retrieved_docs[:k]

        # Count relevant documents in top-k
        relevant_in_top_k = sum(1 for doc in top_k_docs if self._is_relevant(doc))

        # Calculate precision@k
        return relevant_in_top_k / k if k > 0 else 0.0

    def recall_at_k(self, retrieved_docs: List[Document], k: int = 5) -> float:
        """
        Calculate recall@k for retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents
            k: Number of top documents to consider

        Returns:
            Recall@k score (0-1)
        """
        if not self.relevant_docs or not retrieved_docs:
            return 0.0

        # Get top-k documents
        top_k_docs = retrieved_docs[:k]

        # Count relevant documents in top-k
        relevant_in_top_k = sum(1 for doc in top_k_docs if self._is_relevant(doc))

        # Calculate recall@k
        return relevant_in_top_k / len(self.relevant_docs) if self.relevant_docs else 0.0

    def mean_reciprocal_rank(self, retrieved_docs: List[Document]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) for retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            MRR score (0-1)
        """
        if not self.relevant_docs or not retrieved_docs:
            return 0.0

        # Find the rank of the first relevant document
        for i, doc in enumerate(retrieved_docs):
            if self._is_relevant(doc):
                return 1.0 / (i + 1)  # Reciprocal rank

        return 0.0  # No relevant documents found

    def _is_relevant(self, doc: Document) -> bool:
        """
        Check if a document is in the list of relevant documents.

        Args:
            doc: Document to check

        Returns:
            True if document is relevant, False otherwise
        """
        if not self.relevant_docs:
            return False

        # Check if document is in relevant_docs
        # This is a simple implementation - in practice, you might want to use
        # more sophisticated matching (e.g., semantic similarity)
        return any(self._docs_match(doc, rel_doc) for rel_doc in self.relevant_docs)

    def _docs_match(self, doc1: Document, doc2: Document) -> bool:
        """
        Check if two documents match.

        Args:
            doc1: First document
            doc2: Second document

        Returns:
            True if documents match, False otherwise
        """
        # Simple matching based on content
        # In practice, you might want to use more sophisticated matching
        return doc1.page_content.strip() == doc2.page_content.strip()


class GenerationEvaluator:
    """Base class for evaluating generation quality."""

    def __init__(self, llm: Any = None):
        """
        Initialize the generation evaluator.

        Args:
            llm: Language model for LLM-based evaluation
        """
        self.llm = llm

    def evaluate_relevance(self,
                           query: str,
                           response: str,
                           retrieved_docs: List[Document]) -> float:
        """
        Evaluate the relevance of the response to the query.

        Args:
            query: User query
            response: Generated response
            retrieved_docs: Retrieved documents used for generation

        Returns:
            Relevance score (0-1)
        """
        if not query or not response:
            return 0.0

        if LANGCHAIN_EVAL_AVAILABLE and self.llm:
            try:
                # Use LangChain's built-in relevance evaluator
                evaluator = load_evaluator("relevance", llm=self.llm)
                eval_result = evaluator.evaluate_strings(
                    prediction=response,
                    input=query
                )
                return float(eval_result.get("score", 0.0))
            except Exception as e:
                print(f"Error using LangChain evaluator: {e}")

        # Fallback to simple keyword matching
        query_keywords = set(query.lower().split())
        response_words = set(response.lower().split())

        # Calculate Jaccard similarity
        if not query_keywords or not response_words:
            return 0.0

        intersection = query_keywords.intersection(response_words)
        union = query_keywords.union(response_words)

        return len(intersection) / len(union)

    def evaluate_faithfulness(self,
                              response: str,
                              retrieved_docs: List[Document]) -> float:
        """
        Evaluate the faithfulness of the response to the retrieved documents.

        Args:
            response: Generated response
            retrieved_docs: Retrieved documents used for generation

        Returns:
            Faithfulness score (0-1)
        """
        if not response or not retrieved_docs:
            return 0.0

        if LANGCHAIN_EVAL_AVAILABLE and self.llm:
            try:
                # Use LangChain's built-in faithfulness evaluator
                evaluator = load_evaluator("faithfulness", llm=self.llm)

                # Combine retrieved documents into a single context
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                eval_result = evaluator.evaluate_strings(
                    prediction=response,
                    input=context
                )
                return float(eval_result.get("score", 0.0))
            except Exception as e:
                print(f"Error using LangChain evaluator: {e}")

        # Fallback to simple content overlap
        doc_content = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        response_words = set(response.lower().split())
        doc_words = set(doc_content.split())

        # Calculate word overlap
        if not response_words or not doc_words:
            return 0.0

        overlap_count = sum(1 for word in response_words if word in doc_words)

        return overlap_count / len(response_words) if response_words else 0.0

    def evaluate_coherence(self, response: str) -> float:
        """
        Evaluate the coherence of the response.

        Args:
            response: Generated response

        Returns:
            Coherence score (0-1)
        """
        if not response:
            return 0.0

        if LANGCHAIN_EVAL_AVAILABLE and self.llm:
            try:
                # Use LangChain's built-in coherence evaluator
                evaluator = load_evaluator("coherence", llm=self.llm)

                eval_result = evaluator.evaluate_strings(
                    prediction=response,
                    input=""  # No input needed for coherence
                )
                return float(eval_result.get("score", 0.0))
            except Exception as e:
                print(f"Error using LangChain evaluator: {e}")

        # Fallback to simple sentence count heuristic
        # More sentences generally indicate more coherent responses
        # This is a very simple heuristic and not very accurate
        sentences = response.split(".")
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        # Simple heuristic: more sentences = more coherent, up to a point
        sentence_count = len(valid_sentences)

        # Normalize to 0-1 range (assuming 5+ sentences is fully coherent)
        return min(sentence_count / 5.0, 1.0)


class RAGEvaluator:
    """End-to-end RAG system evaluator."""

    def __init__(self,
                 llm: Any = None,
                 relevant_docs: Optional[List[Document]] = None,
                 retrieval_weight: float = 0.5,
                 generation_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the RAG evaluator.

        Args:
            llm: Language model for LLM-based evaluation
            relevant_docs: Optional list of known relevant documents for ground truth
            retrieval_weight: Weight for retrieval metrics in the overall score (0-1)
            generation_weights: Weights for different generation metrics
        """
        self.retrieval_evaluator = RetrievalEvaluator(relevant_docs)
        self.generation_evaluator = GenerationEvaluator(llm)
        self.retrieval_weight = retrieval_weight
        self.generation_weights = generation_weights or {
            "relevance": 0.4,
            "faithfulness": 0.4,
            "coherence": 0.2
        }

    def evaluate(self,
                 query: str,
                 response: str,
                 retrieved_docs: List[Document],
                 k: int = 5) -> Dict[str, float]:
        """
        Evaluate the end-to-end RAG system.

        Args:
            query: User query
            response: Generated response
            retrieved_docs: Retrieved documents used for generation
            k: Number of top documents to consider for retrieval metrics

        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate retrieval
        precision = self.retrieval_evaluator.precision_at_k(retrieved_docs, k)
        recall = self.retrieval_evaluator.recall_at_k(retrieved_docs, k)
        mrr = self.retrieval_evaluator.mean_reciprocal_rank(retrieved_docs)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Combine retrieval metrics
        retrieval_score = (precision + recall + mrr + f1) / 4

        # Evaluate generation
        relevance = self.generation_evaluator.evaluate_relevance(query, response, retrieved_docs)
        faithfulness = self.generation_evaluator.evaluate_faithfulness(response, retrieved_docs)
        coherence = self.generation_evaluator.evaluate_coherence(response)

        # Combine generation metrics
        generation_score = (
            relevance * self.generation_weights["relevance"] +
            faithfulness * self.generation_weights["faithfulness"] +
            coherence * self.generation_weights["coherence"]
        )

        # Calculate overall score
        overall_score = (
            retrieval_score * self.retrieval_weight +
            generation_score * (1 - self.retrieval_weight)
        )

        # Return all metrics
        return {
            "precision@k": precision,
            "recall@k": recall,
            "mrr": mrr,
            "f1": f1,
            "retrieval_score": retrieval_score,
            "relevance": relevance,
            "faithfulness": faithfulness,
            "coherence": coherence,
            "generation_score": generation_score,
            "overall_score": overall_score
        }

    def as_lcel_chain(self):
        """Return the RAG evaluator as an LCEL chain."""
        def evaluate_rag(inputs):
            query = inputs["query"]
            response = inputs["response"]
            retrieved_docs = inputs["retrieved_docs"]
            k = inputs.get("k", 5)

            return self.evaluate(query, response, retrieved_docs, k)

        return RunnableLambda(evaluate_rag)


class EvaluationVisualizer:
    """Visualization tools for RAG evaluation results."""

    @staticmethod
    def plot_metrics(metrics: Dict[str, float], title: str = "RAG Evaluation Metrics"):
        """
        Plot evaluation metrics as a bar chart.

        Args:
            metrics: Dictionary of evaluation metrics
            title: Chart title
        """
        if not VISUALIZATION_AVAILABLE:
            print("Matplotlib and Seaborn are required for visualization.")
            return

        # Filter metrics for visualization
        plot_metrics = {
            k: v for k, v in metrics.items()
            if k in [
                "precision@k", "recall@k", "mrr", "f1",
                "relevance", "faithfulness", "coherence",
                "retrieval_score", "generation_score", "overall_score"
            ]
        }

        # Create figure
        plt.figure(figsize=(12, 6))

        # Create bar chart
        bars = plt.bar(
            plot_metrics.keys(),
            plot_metrics.values(),
            color=sns.color_palette("viridis", len(plot_metrics))
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        # Add labels and title
        plt.xlabel("Metrics")
        plt.ylabel("Score (0-1)")
        plt.title(title)
        plt.ylim(0, 1.1)  # Set y-axis limit
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels
        plt.tight_layout()

        # Show plot
        plt.show()

    @staticmethod
    def plot_comparison(
        metrics_list: List[Dict[str, float]],
        labels: List[str],
        title: str = "RAG System Comparison"
    ):
        """
        Plot comparison of multiple RAG systems.

        Args:
            metrics_list: List of metrics dictionaries for different systems
            labels: Labels for each system
            title: Chart title
        """
        if not VISUALIZATION_AVAILABLE:
            print("Matplotlib and Seaborn are required for visualization.")
            return

        if len(metrics_list) != len(labels):
            raise ValueError("Number of metrics dictionaries must match number of labels")

        # Select key metrics for comparison
        key_metrics = ["retrieval_score", "generation_score", "overall_score"]

        # Create data for plotting
        data = {
            label: [metrics.get(metric, 0) for metric in key_metrics]
            for label, metrics in zip(labels, metrics_list)
        }

        # Create figure
        plt.figure(figsize=(10, 6))

        # Set bar width and positions
        bar_width = 0.2
        r = np.arange(len(key_metrics))

        # Create bars
        for i, (label, values) in enumerate(data.items()):
            position = [x + bar_width * i for x in r]
            plt.bar(
                position,
                values,
                width=bar_width,
                label=label,
                color=sns.color_palette("viridis", len(data))[i]
            )

        # Add labels and title
        plt.xlabel("Metrics")
        plt.ylabel("Score (0-1)")
        plt.title(title)
        plt.ylim(0, 1.1)  # Set y-axis limit
        plt.xticks([r + bar_width * (len(data) - 1) / 2 for r in range(len(key_metrics))], key_metrics)
        plt.legend()
        plt.tight_layout()

        # Show plot
        plt.show()

    @staticmethod
    def create_radar_chart(
        metrics: Dict[str, float],
        title: str = "RAG System Performance"
    ):
        """
        Create a radar chart for RAG system performance.

        Args:
            metrics: Dictionary of evaluation metrics
            title: Chart title
        """
        if not VISUALIZATION_AVAILABLE:
            print("Matplotlib and Seaborn are required for visualization.")
            return

        # Select metrics for radar chart
        radar_metrics = [
            "precision@k", "recall@k", "mrr", "f1",
            "relevance", "faithfulness", "coherence"
        ]

        # Get values for selected metrics
        values = [metrics.get(metric, 0) for metric in radar_metrics]

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Number of variables
        N = len(radar_metrics)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Add values (and close the loop)
        values += values[:1]

        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid')

        # Fill area
        ax.fill(angles, values, alpha=0.25)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics)

        # Set y-axis limit
        ax.set_ylim(0, 1)

        # Add title
        plt.title(title)

        # Show plot
        plt.tight_layout()
        plt.show()


class RAGASEvaluator:
    """
    Evaluator using RAGAS metrics for RAG systems.

    RAGAS is a framework specifically designed for evaluating RAG systems.
    It provides metrics like:
    - Answer Relevance
    - Faithfulness
    - Context Relevance
    - Context Precision
    - Context Recall

    This class provides a wrapper around RAGAS for easy integration.
    """

    def __init__(self, llm: Any = None):
        """
        Initialize the RAGAS evaluator.

        Args:
            llm: Language model for evaluation
        """
        self.llm = llm
        self.ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        """
        Check if RAGAS is available.

        Returns:
            True if RAGAS is installed, False otherwise
        """
        try:
            # Just try to import, we don't need to use the module here
            import ragas  # noqa: F401
            return True
        except ImportError:
            return False

    def evaluate(self,
                 query: str,
                 response: str,
                 retrieved_docs: List[Document]) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics.

        Args:
            query: User query
            response: Generated response
            retrieved_docs: Retrieved documents used for generation

        Returns:
            Dictionary of RAGAS metrics
        """
        if not self.ragas_available:
            print("RAGAS is required for this evaluation. Install with 'pip install ragas'")
            return {
                "answer_relevance": 0.0,
                "faithfulness": 0.0,
                "context_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }

        try:
            # Import RAGAS components
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall
            )
            from ragas.metrics.critique import harmfulness
            import pandas as pd

            # Prepare data
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [[doc.page_content for doc in retrieved_docs]]
            }

            # Create DataFrame
            df = pd.DataFrame(data)

            # Calculate metrics
            results = {}

            # Answer relevancy
            relevancy_score = answer_relevancy.score(df, llm=self.llm)
            results["answer_relevance"] = float(relevancy_score.iloc[0])

            # Faithfulness
            faith_score = faithfulness.score(df, llm=self.llm)
            results["faithfulness"] = float(faith_score.iloc[0])

            # Context precision
            context_prec_score = context_precision.score(df, llm=self.llm)
            results["context_precision"] = float(context_prec_score.iloc[0])

            # Context recall
            context_rec_score = context_recall.score(df, llm=self.llm)
            results["context_recall"] = float(context_rec_score.iloc[0])

            # Harmfulness (optional)
            try:
                harm_score = harmfulness.score(df, llm=self.llm)
                results["harmfulness"] = 1.0 - float(harm_score.iloc[0])  # Invert so higher is better
            except:
                results["harmfulness"] = 1.0  # Default to 1.0 (not harmful)

            return results

        except Exception as e:
            print(f"Error using RAGAS: {e}")
            return {
                "answer_relevance": 0.0,
                "faithfulness": 0.0,
                "context_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }

    def as_lcel_chain(self):
        """Return the RAGAS evaluator as an LCEL chain."""
        def evaluate_ragas(inputs):
            query = inputs["query"]
            response = inputs["response"]
            retrieved_docs = inputs["retrieved_docs"]

            return self.evaluate(query, response, retrieved_docs)

        return RunnableLambda(evaluate_ragas)


# Example usage
def example_usage():
    """Example of how to use the evaluation framework."""
    # Import required packages
    from langchain.schema.document import Document
    from langchain.chat_models import ChatGroq

    # Create sample data
    query = "What is RAG?"
    response = """
    RAG (Retrieval-Augmented Generation) is a technique that enhances language models
    by retrieving relevant information from external sources before generating a response.
    This approach combines the strengths of retrieval-based and generation-based methods,
    allowing the model to access up-to-date information and provide more accurate answers.
    """

    retrieved_docs = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval and generation for improved LLM responses.",
            metadata={"source": "document1.txt", "page": 1}
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant information before generating responses.",
            metadata={"source": "document2.txt", "page": 5}
        ),
        Document(
            page_content="RAG systems use vector databases to store and retrieve relevant context for language model queries.",
            metadata={"source": "document3.txt", "page": 12}
        )
    ]

    # Create ground truth relevant documents
    relevant_docs = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval and generation for improved LLM responses.",
            metadata={"source": "document1.txt", "page": 1}
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant information before generating responses.",
            metadata={"source": "document2.txt", "page": 5}
        )
    ]

    # Initialize language model
    try:
        llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")
    except:
        # Fallback to a mock LLM
        class MockLLM:
            def invoke(self, prompt: str) -> str:
                """Mock LLM invoke method."""
                # We don't use the prompt parameter, but it's required for the interface
                return "This is a mock response."
        llm = MockLLM()

    # Create evaluator
    evaluator = RAGEvaluator(
        llm=llm,
        relevant_docs=relevant_docs,
        retrieval_weight=0.4,
        generation_weights={
            "relevance": 0.4,
            "faithfulness": 0.4,
            "coherence": 0.2
        }
    )

    # Evaluate RAG system
    metrics = evaluator.evaluate(query, response, retrieved_docs)

    # Print metrics
    print("RAG Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Visualize results (if matplotlib is available)
    try:
        EvaluationVisualizer.plot_metrics(metrics)
    except:
        pass

    # Try RAGAS evaluation if available
    try:
        ragas_evaluator = RAGASEvaluator(llm=llm)
        ragas_metrics = ragas_evaluator.evaluate(query, response, retrieved_docs)

        print("\nRAGAS Metrics:")
        for metric, value in ragas_metrics.items():
            print(f"  {metric}: {value:.4f}")
    except:
        pass


# Additional utility functions for RAG evaluation

def evaluate_retriever(
    retriever: Any,
    queries: List[str],
    relevant_docs_map: Dict[str, List[Document]],
    k: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a retriever on multiple queries.

    Args:
        retriever: Retriever to evaluate
        queries: List of queries to test
        relevant_docs_map: Dictionary mapping queries to relevant documents
        k: Number of documents to retrieve

    Returns:
        Dictionary of evaluation metrics for each query
    """
    results = {}

    for query in queries:
        # Get relevant documents for this query
        relevant_docs = relevant_docs_map.get(query, [])

        # Create evaluator
        evaluator = RetrievalEvaluator(relevant_docs)

        # Retrieve documents
        try:
            retrieved_docs = retriever.get_relevant_documents(query)[:k]
        except:
            # Try alternative retrieval method
            try:
                retrieved_docs = retriever.invoke(query)[:k]
            except:
                print(f"Failed to retrieve documents for query: {query}")
                retrieved_docs = []

        # Evaluate retrieval
        precision = evaluator.precision_at_k(retrieved_docs, k)
        recall = evaluator.recall_at_k(retrieved_docs, k)
        mrr = evaluator.mean_reciprocal_rank(retrieved_docs)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Store results
        results[query] = {
            "precision@k": precision,
            "recall@k": recall,
            "mrr": mrr,
            "f1": f1
        }

    return results


def evaluate_rag_system(
    rag_system: Any,
    queries: List[str],
    relevant_docs_map: Dict[str, List[Document]],
    llm: Any = None,
    k: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a complete RAG system on multiple queries.

    Args:
        rag_system: RAG system to evaluate
        queries: List of queries to test
        relevant_docs_map: Dictionary mapping queries to relevant documents
        llm: Language model for generation evaluation
        k: Number of documents to consider

    Returns:
        Dictionary of evaluation metrics for each query
    """
    results = {}

    for query in queries:
        # Get relevant documents for this query
        relevant_docs = relevant_docs_map.get(query, [])

        # Create evaluator
        evaluator = RAGEvaluator(llm, relevant_docs)

        # Generate response and get retrieved documents
        try:
            # Try different possible interfaces
            try:
                # Standard invoke with query
                result = rag_system.invoke(query)
                if isinstance(result, dict):
                    response = result.get("answer", result.get("response", ""))
                    retrieved_docs = result.get("source_documents", result.get("documents", []))
                else:
                    response = result
                    retrieved_docs = []
            except:
                try:
                    # Try with dictionary input
                    result = rag_system.invoke({"query": query})
                    if isinstance(result, dict):
                        response = result.get("answer", result.get("response", ""))
                        retrieved_docs = result.get("source_documents", result.get("documents", []))
                    else:
                        response = result
                        retrieved_docs = []
                except:
                    print(f"Failed to generate response for query: {query}")
                    response = ""
                    retrieved_docs = []
        except Exception as e:
            print(f"Error evaluating RAG system: {e}")
            response = ""
            retrieved_docs = []

        # Evaluate RAG system
        metrics = evaluator.evaluate(query, response, retrieved_docs, k)

        # Store results
        results[query] = metrics

    return results


def save_evaluation_results(
    results: Dict[str, Dict[str, float]],
    output_file: str = "rag_evaluation_results.json"
):
    """
    Save evaluation results to a file.

    Args:
        results: Evaluation results
        output_file: Output file path
    """
    import json

    # Convert to serializable format
    serializable_results = {}
    for query, metrics in results.items():
        serializable_results[query] = {k: float(v) for k, v in metrics.items()}

    # Save to file
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Evaluation results saved to {output_file}")


if __name__ == "__main__":
    # Run example usage
    example_usage()
