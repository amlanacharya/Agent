"""
Exercise 2: Chunking Strategies for Optimal Retrieval

This exercise focuses on implementing different text chunking strategies
for document retrieval systems.
"""

import re
from typing import List, Dict, Any, Optional, Callable, Union
import math

class ChunkingError(Exception):
    """Exception raised when chunking fails."""
    pass

# Exercise 1: Implement a Size-Based Chunker
class SizeBasedChunker:
    """
    Chunks text based on a fixed size with optional overlap.

    This chunker splits text into chunks of approximately equal size,
    with configurable overlap between chunks to maintain context.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 0,
                 respect_sentences: bool = False):
        """
        Initialize the size-based chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            respect_sentences: Whether to avoid breaking sentences when possible
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if overlap < 0:
            raise ValueError("Overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately equal size.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks

        Raises:
            ChunkingError: If chunking fails
        """
        if not text:
            return []

        try:
            chunks = []
            start = 0

            while start < len(text):
                # Calculate end position
                end = min(start + self.chunk_size, len(text))

                # Adjust end position if respecting sentences
                if self.respect_sentences and end < len(text):
                    # Look for sentence boundaries (., !, ?)
                    sentence_end = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )

                    if sentence_end > start:
                        end = sentence_end + 2  # Include the period and space

                # Extract chunk
                chunk = text[start:end]
                chunks.append(chunk)

                # Move start position for next chunk, accounting for overlap
                start = max(start + 1, end - self.overlap)  # Ensure we make progress

                # Break if we've reached the end of the text
                if start >= len(text):
                    break

            return chunks

        except Exception as e:
            raise ChunkingError(f"Error in size-based chunking: {str(e)}")

# Exercise 2: Implement a Delimiter-Based Chunker
class DelimiterBasedChunker:
    """
    Chunks text based on specified delimiters.

    This chunker splits text at delimiter boundaries, then combines
    the resulting sections into chunks that respect a maximum size.
    """

    def __init__(self, delimiter: str = "\n\n", max_chunk_size: int = 1000,
                 min_chunk_size: int = 100):
        """
        Initialize the delimiter-based chunker.

        Args:
            delimiter: String delimiter to split on
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size for a chunk to be considered complete
        """
        if max_chunk_size <= 0:
            raise ValueError("Max chunk size must be positive")
        if min_chunk_size <= 0:
            raise ValueError("Min chunk size must be positive")
        if min_chunk_size > max_chunk_size:
            raise ValueError("Min chunk size must be less than or equal to max chunk size")

        self.delimiter = delimiter
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text based on delimiters while respecting maximum chunk size.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks

        Raises:
            ChunkingError: If chunking fails
        """
        if not text:
            return []

        try:
            # Split text by delimiter
            sections = text.split(self.delimiter)
            chunks = []
            current_chunk = []
            current_size = 0

            for section in sections:
                section_size = len(section)

                # If adding this section would exceed max size and we already have content
                if current_size + section_size > self.max_chunk_size and current_chunk:
                    # Finalize current chunk
                    chunks.append(self.delimiter.join(current_chunk))
                    current_chunk = [section]
                    current_size = section_size
                else:
                    # Add section to current chunk
                    current_chunk.append(section)
                    current_size += section_size

            # Add the final chunk if there's anything left
            if current_chunk:
                chunks.append(self.delimiter.join(current_chunk))

            return chunks

        except Exception as e:
            raise ChunkingError(f"Error in delimiter-based chunking: {str(e)}")

# Exercise 3: Implement a Recursive Chunker
class RecursiveChunker:
    """
    Chunks text recursively using a hierarchy of delimiters.

    This chunker tries multiple delimiters in sequence, starting with larger
    semantic units and progressively moving to smaller ones.
    """

    def __init__(self, delimiters: List[str] = ["\n\n", "\n", ". ", " "],
                 max_chunk_size: int = 1000):
        """
        Initialize the recursive chunker.

        Args:
            delimiters: List of delimiters to try in order (from largest to smallest units)
            max_chunk_size: Maximum size of each chunk in characters
        """
        if max_chunk_size <= 0:
            raise ValueError("Max chunk size must be positive")
        if not delimiters:
            raise ValueError("At least one delimiter must be provided")

        self.delimiters = delimiters
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text recursively using a hierarchy of delimiters.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks

        Raises:
            ChunkingError: If chunking fails
        """
        if not text:
            return []

        try:
            return self._recursive_chunk(text, self.delimiters, self.max_chunk_size)
        except Exception as e:
            raise ChunkingError(f"Error in recursive chunking: {str(e)}")

    def _recursive_chunk(self, text: str, delimiters: List[str], max_size: int) -> List[str]:
        """
        Recursively chunk text using the given delimiters.

        Args:
            text: Text to chunk
            delimiters: List of delimiters to try
            max_size: Maximum chunk size

        Returns:
            List of chunks
        """
        # Base case: text is small enough or we've run out of delimiters
        if len(text) <= max_size or not delimiters:
            return [text]

        # Try splitting with the current delimiter
        delimiter = delimiters[0]
        if delimiter in text:
            sections = text.split(delimiter)
            chunks = []
            current_chunk = []
            current_size = 0

            for section in sections:
                # Account for delimiter size in calculations
                section_with_delimiter = section
                if current_chunk:  # Add delimiter except for the first section
                    section_with_delimiter = delimiter + section

                section_size = len(section_with_delimiter)

                # If adding this section would exceed max size and we already have content
                if current_size + section_size > max_size and current_chunk:
                    # Current chunk is full, process it recursively with next delimiter
                    combined_text = delimiter.join(current_chunk)
                    if len(combined_text) > max_size and len(delimiters) > 1:
                        chunks.extend(self._recursive_chunk(combined_text, delimiters[1:], max_size))
                    else:
                        chunks.append(combined_text)

                    # Start a new chunk with this section
                    current_chunk = [section]
                    current_size = len(section)
                else:
                    # Add section to current chunk
                    current_chunk.append(section)
                    current_size += section_size

            # Process the final chunk
            if current_chunk:
                combined_text = delimiter.join(current_chunk)
                if len(combined_text) > max_size and len(delimiters) > 1:
                    chunks.extend(self._recursive_chunk(combined_text, delimiters[1:], max_size))
                else:
                    chunks.append(combined_text)

            return chunks

        # If current delimiter not found, try the next one
        return self._recursive_chunk(text, delimiters[1:], max_size)

# Exercise 4: Implement a Semantic Chunker
class SemanticChunker:
    """
    Chunks text based on semantic units like paragraphs and sentences.

    This chunker attempts to preserve semantic meaning by respecting
    natural language boundaries.
    """

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """
        Initialize the semantic chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size for a chunk to be considered complete
        """
        if max_chunk_size <= 0:
            raise ValueError("Max chunk size must be positive")
        if min_chunk_size <= 0:
            raise ValueError("Min chunk size must be positive")
        if min_chunk_size > max_chunk_size:
            raise ValueError("Min chunk size must be less than or equal to max chunk size")

        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text based on semantic units like paragraphs and sentences.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks

        Raises:
            ChunkingError: If chunking fails
        """
        if not text:
            return []

        try:
            # Split into paragraphs
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = []
            current_size = 0

            for paragraph in paragraphs:
                paragraph_size = len(paragraph)

                # If paragraph fits in current chunk, add it
                if current_size + paragraph_size <= self.max_chunk_size:
                    current_chunk.append(paragraph)
                    current_size += paragraph_size
                else:
                    # If paragraph is too large, split into sentences
                    if paragraph_size > self.max_chunk_size:
                        # Split by sentence boundaries
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        for sentence in sentences:
                            sentence_size = len(sentence)

                            if current_size + sentence_size <= self.max_chunk_size:
                                current_chunk.append(sentence)
                                current_size += sentence_size
                            else:
                                # Finalize current chunk if it's not empty
                                if current_chunk:
                                    chunks.append("\n\n".join(current_chunk))

                                # Start new chunk with this sentence or split further if needed
                                if sentence_size > self.max_chunk_size:
                                    # If sentence is still too large, split arbitrarily
                                    for i in range(0, sentence_size, self.max_chunk_size):
                                        chunks.append(sentence[i:i + self.max_chunk_size])
                                else:
                                    current_chunk = [sentence]
                                    current_size = sentence_size
                    else:
                        # Finalize current chunk
                        if current_chunk:
                            chunks.append("\n\n".join(current_chunk))

                        # Start new chunk with this paragraph
                        current_chunk = [paragraph]
                        current_size = paragraph_size

            # Add the final chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            return chunks

        except Exception as e:
            raise ChunkingError(f"Error in semantic chunking: {str(e)}")

# Exercise 5: Implement a Token-Aware Chunker
class TokenAwareChunker:
    """
    Chunks text based on token count rather than character count.

    This chunker ensures chunks fit within LLM context windows by
    counting tokens instead of characters.
    """

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50,
                 tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize the token-aware chunker.

        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            tokenizer: Function that converts text to tokens (if None, uses a simple word splitter)
        """
        if max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("Overlap tokens must be non-negative")
        if overlap_tokens >= max_tokens:
            raise ValueError("Overlap tokens must be less than max tokens")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer or (lambda text: text.split())

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text based on token count rather than character count.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks

        Raises:
            ChunkingError: If chunking fails
        """
        if not text:
            return []

        try:
            # Tokenize the text
            tokens = self.tokenizer(text)
            if not tokens:
                return []

            chunks = []

            # Create chunks with overlap
            for i in range(0, len(tokens), self.max_tokens - self.overlap_tokens):
                # Get tokens for this chunk
                chunk_tokens = tokens[i:i + self.max_tokens]

                # Convert back to text
                if chunk_tokens:
                    if isinstance(chunk_tokens[0], str):
                        # Simple word tokenizer
                        chunk_text = " ".join(chunk_tokens)
                    else:
                        # Custom tokenizer that might return non-string tokens
                        # This is a placeholder - in a real implementation, you would
                        # use the tokenizer's detokenize method
                        chunk_text = " ".join(str(token) for token in chunk_tokens)

                    chunks.append(chunk_text)

            return chunks

        except Exception as e:
            raise ChunkingError(f"Error in token-aware chunking: {str(e)}")

# Exercise 6: Implement a Chunking Evaluator
class ChunkingEvaluator:
    """
    Evaluates the quality of different chunking strategies.

    This evaluator compares chunking strategies based on metrics like
    chunk size distribution, semantic coherence, and retrieval performance.
    """

    def __init__(self):
        """Initialize the chunking evaluator."""
        self.metrics = {}

    def evaluate_size_distribution(self, chunks: List[str]) -> Dict[str, float]:
        """
        Evaluate the size distribution of chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary of size distribution metrics
        """
        if not chunks:
            return {
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
                "std_dev": 0,
                "size_uniformity": 0
            }

        # Calculate size metrics
        sizes = [len(chunk) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)

        # Calculate standard deviation
        variance = sum((size - avg_size) ** 2 for size in sizes) / len(sizes)
        std_dev = math.sqrt(variance)

        # Calculate size uniformity (1 = perfectly uniform, 0 = highly variable)
        if max_size == min_size:
            size_uniformity = 1.0
        else:
            size_uniformity = 1.0 - (std_dev / avg_size)
            size_uniformity = max(0.0, min(1.0, size_uniformity))

        return {
            "avg_size": avg_size,
            "min_size": min_size,
            "max_size": max_size,
            "std_dev": std_dev,
            "size_uniformity": size_uniformity
        }

    def evaluate_semantic_coherence(self, chunks: List[str]) -> Dict[str, float]:
        """
        Evaluate the semantic coherence of chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary of semantic coherence metrics
        """
        if not chunks:
            return {
                "complete_sentences_ratio": 0,
                "paragraph_preservation": 0
            }

        # Count complete sentences
        complete_sentences = 0
        total_sentences = 0

        for chunk in chunks:
            # Simple heuristic: sentences end with ., !, or ?
            sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())

            # Filter out empty sentences
            sentences = [s for s in sentences if s.strip()]

            if not sentences:
                continue

            total_sentences += len(sentences)

            # Count sentences that start with capital letter and end with punctuation
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and re.match(r'^[A-Z].*[.!?]$', sentence):
                    complete_sentences += 1

        # Calculate paragraph preservation
        paragraph_breaks = sum(chunk.count("\n\n") for chunk in chunks)
        paragraph_preservation = paragraph_breaks / len(chunks) if len(chunks) > 1 else 0

        # Normalize paragraph preservation to 0-1 scale
        paragraph_preservation = min(1.0, paragraph_preservation / 5)

        return {
            "complete_sentences_ratio": complete_sentences / total_sentences if total_sentences else 0,
            "paragraph_preservation": paragraph_preservation
        }

    def compare_strategies(self, text: str, chunkers: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Compare different chunking strategies on the same text.

        Args:
            text: Input text to chunk
            chunkers: Dictionary of chunker instances by name

        Returns:
            Dictionary of evaluation results by chunker name
        """
        results = {}

        for name, chunker in chunkers.items():
            try:
                # Generate chunks
                chunks = chunker.chunk_text(text)

                # Evaluate chunks
                size_metrics = self.evaluate_size_distribution(chunks)
                semantic_metrics = self.evaluate_semantic_coherence(chunks)

                # Combine metrics
                results[name] = {
                    "chunk_count": len(chunks),
                    "size_metrics": size_metrics,
                    "semantic_metrics": semantic_metrics,
                    # Calculate an overall score (simple weighted average)
                    "overall_score": (
                        0.4 * size_metrics["size_uniformity"] +
                        0.6 * semantic_metrics["complete_sentences_ratio"]
                    )
                }

            except Exception as e:
                results[name] = {"error": str(e)}

        return results

# Example usage
if __name__ == "__main__":
    # Sample text for testing
    sample_text = '''
    # Document Chunking Strategies

    Effective document chunking is critical for retrieval-augmented generation systems.
    The way documents are split affects both retrieval accuracy and generation quality.

    ## Size-Based Chunking

    The simplest approach is to split documents based on a fixed size. This works well
    for homogeneous documents but can break semantic connections.

    ## Delimiter-Based Chunking

    This strategy splits text based on specific delimiters like paragraphs or sections.
    It preserves natural document boundaries but can result in uneven chunk sizes.

    ## Recursive Chunking

    This advanced strategy tries multiple delimiters in sequence, starting with larger
    semantic units and progressively moving to smaller ones.

    ## Semantic Chunking

    This approach uses semantic understanding to identify logical boundaries for chunking.
    It preserves meaning but requires more sophisticated processing.

    ## Token-Aware Chunking

    This strategy counts tokens rather than characters, ensuring chunks fit within LLM
    context windows. It's essential for direct integration with specific LLMs.
    '''

    # Create chunkers
    size_chunker = SizeBasedChunker(chunk_size=200, overlap=50)
    delimiter_chunker = DelimiterBasedChunker(delimiter="\n\n", max_chunk_size=200)
    recursive_chunker = RecursiveChunker(max_chunk_size=200)
    semantic_chunker = SemanticChunker(max_chunk_size=200)
    token_chunker = TokenAwareChunker(max_tokens=50, overlap_tokens=10)

    # Create evaluator
    evaluator = ChunkingEvaluator()

    # Compare strategies
    results = evaluator.compare_strategies(
        sample_text,
        {
            "size_based": size_chunker,
            "delimiter_based": delimiter_chunker,
            "recursive": recursive_chunker,
            "semantic": semantic_chunker,
            "token_aware": token_chunker
        }
    )

    # Print results
    print("Chunking Strategy Comparison:")
    for name, result in results.items():
        if "error" in result:
            print(f"{name}: Error - {result['error']}")
        else:
            print(f"{name}:")
            print(f"  Chunk count: {result['chunk_count']}")
            print(f"  Avg size: {result['size_metrics']['avg_size']:.1f} chars")
            print(f"  Size uniformity: {result['size_metrics']['size_uniformity']:.2f}")
            print(f"  Complete sentences: {result['semantic_metrics']['complete_sentences_ratio']:.2f}")
            print(f"  Overall score: {result['overall_score']:.2f}")
            print()
