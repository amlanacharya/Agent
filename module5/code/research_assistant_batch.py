"""
Research Literature Assistant

This module implements a specialized RAG system for academic papers and research literature,
including:
- Academic paper processing with section extraction
- Citation tracking and verification
- Research question analysis
- Literature review generation
- A complete Research Literature Assistant using LCEL

All implementations use LangChain Expression Language (LCEL) for improved
readability and composability.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Set
import re
import json
import os
import time
from collections import defaultdict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores import FAISS, Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available. Install with 'pip install tqdm' for progress bars.")


class AcademicPaperProcessor:
    """Process academic papers with specialized techniques."""

    def __init__(self):
        self.section_headers = [
            "abstract", "introduction", "background", "related work",
            "methodology", "methods", "experiments", "results",
            "discussion", "conclusion", "references"
        ]

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from academic paper.

        Args:
            text: Full text of the academic paper

        Returns:
            Dictionary mapping section names to section content
        """
        sections = {}
        current_section = "unknown"
        current_content = []

        # Split text into lines
        lines = text.split('\n')

        for line in lines:
            # Check if line is a section header
            line_lower = line.lower().strip()
            is_header = False

            # Check if line matches any section header pattern
            for header in self.section_headers:
                # Match exact header or header with number (e.g., "1. Introduction")
                if (line_lower == header or
                    re.match(r'^\d+\.?\s+' + header + r'$', line_lower) or
                    re.match(r'^' + header + r'\s*$', line_lower)):

                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)

                    # Start new section
                    current_section = header
                    current_content = []
                    is_header = True
                    break

            # If not a header, add to current section content
            if not is_header:
                current_content.append(line)

        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)

        return sections

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from academic paper.

        Args:
            text: Full text of the academic paper

        Returns:
            List of extracted citations with metadata
        """
        citations = []

        # Extract citations in format [Author, Year]
        author_year_pattern = r'\[([A-Za-z\s]+),\s*(\d{4})\]'
        for match in re.finditer(author_year_pattern, text):
            author = match.group(1).strip()
            year = match.group(2)

            citations.append({
                "type": "author_year",
                "author": author,
                "year": year,
                "position": match.span()
            })

        # Extract citations in format [1], [2], etc.
        numbered_pattern = r'\[(\d+)\]'
        for match in re.finditer(numbered_pattern, text):
            number = match.group(1)

            citations.append({
                "type": "numbered",
                "number": number,
                "position": match.span()
            })

        return citations

    def extract_figures_and_tables(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract figures and tables from academic paper.

        Args:
            text: Full text of the academic paper

        Returns:
            Dictionary with lists of figures and tables
        """
        figures = []
        tables = []

        # Extract figures (Figure X: Caption)
        figure_pattern = r'(Figure|Fig\.?)\s+(\d+)[\.:]?\s*(.*?)(?=\n\n|\n[A-Z]|$)'
        for match in re.finditer(figure_pattern, text, re.DOTALL):
            figure_num = match.group(2)
            caption = match.group(3).strip()

            figures.append({
                "number": figure_num,
                "caption": caption,
                "position": match.span()
            })

        # Extract tables (Table X: Caption)
        table_pattern = r'Table\s+(\d+)[\.:]?\s*(.*?)(?=\n\n|\n[A-Z]|$)'
        for match in re.finditer(table_pattern, text, re.DOTALL):
            table_num = match.group(1)
            caption = match.group(2).strip()

            tables.append({
                "number": table_num,
                "caption": caption,
                "position": match.span()
            })

        return {
            "figures": figures,
            "tables": tables
        }

    def process_paper(self, file_path: str) -> Dict[str, Any]:
        """Process academic paper from PDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with processed paper data
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Extract metadata
        metadata = self._extract_metadata(pages)

        # Extract full text
        full_text = "\n\n".join([page.page_content for page in pages])

        # Extract sections
        sections = self.extract_sections(full_text)

        # Extract citations
        citations = self.extract_citations(full_text)

        # Extract figures and tables
        figures_tables = self.extract_figures_and_tables(full_text)

        # Create specialized chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        for section_name, section_text in sections.items():
            section_chunks = splitter.create_documents(
                texts=[section_text],
                metadatas=[{
                    "source": file_path,
                    "section": section_name,
                    **metadata
                }]
            )
            chunks.extend(section_chunks)

        return {
            "chunks": chunks,
            "metadata": metadata,
            "citations": citations,
            "figures": figures_tables["figures"],
            "tables": figures_tables["tables"],
            "sections": sections
        }

    def _extract_metadata(self, pages: List[Document]) -> Dict[str, Any]:
        """Extract metadata from academic paper.

        Args:
            pages: List of document pages

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        # Get first page content for metadata extraction
        if not pages:
            return metadata

        first_page = pages[0].page_content

        # Extract title (usually the first line)
        lines = first_page.split('\n')
        if lines:
            metadata["title"] = lines[0].strip()

        # Extract authors (usually after title)
        author_line = None
        for i, line in enumerate(lines[1:5]):  # Check first few lines
            if line and not line.startswith('Abstract'):
                author_line = line
                break

        if author_line:
            # Split by common author separators
            authors = re.split(r',|\band\b|;', author_line)
            metadata["authors"] = [author.strip() for author in authors if author.strip()]

        # Extract publication year
        year_pattern = r'\b(19|20)\d{2}\b'
        year_match = re.search(year_pattern, first_page[:500])  # Check first part of document
        if year_match:
            metadata["year"] = year_match.group(0)

        # Extract DOI if present
        doi_pattern = r'doi:?\s*(10\.\d+/[^\s]+)'
        doi_match = re.search(doi_pattern, first_page, re.IGNORECASE)
        if doi_match:
            metadata["doi"] = doi_match.group(1)

        return metadata



class CitationTracker:
    """Track and verify citations across academic papers."""

    def __init__(self):
        self.citation_graph = {}  # Paper ID -> [Cited Paper IDs]
        self.papers = {}  # Paper ID -> Paper Metadata
        self.citing_papers = defaultdict(list)  # Paper ID -> [Papers that cite it]

    def add_paper(self, paper_id: str, metadata: Dict[str, Any], citations: List[Dict[str, Any]]):
        """Add a paper and its citations to the tracker.

        Args:
            paper_id: Unique identifier for the paper
            metadata: Paper metadata
            citations: List of citations in the paper
        """
        # Store paper metadata
        self.papers[paper_id] = metadata

        # Extract cited paper IDs from citations
        cited_paper_ids = []
        for citation in citations:
            # For author-year citations
            if citation["type"] == "author_year":
                # Create a citation ID from author and year
                cited_id = f"{citation['author']}_{citation['year']}"
                cited_paper_ids.append(cited_id)

            # For numbered citations, we need reference mapping
            # This would typically come from parsing the references section
            elif citation["type"] == "numbered" and "reference_id" in citation:
                cited_paper_ids.append(citation["reference_id"])

        # Store outgoing citations
        self.citation_graph[paper_id] = cited_paper_ids

        # Update incoming citations
        for cited_id in cited_paper_ids:
            self.citing_papers[cited_id].append(paper_id)

    def get_citing_papers(self, paper_id: str) -> List[str]:
        """Get papers that cite the given paper.

        Args:
            paper_id: ID of the paper

        Returns:
            List of paper IDs that cite the given paper
        """
        return self.citing_papers.get(paper_id, [])

    def get_cited_papers(self, paper_id: str) -> List[str]:
        """Get papers cited by the given paper.

        Args:
            paper_id: ID of the paper

        Returns:
            List of paper IDs cited by the given paper
        """
        return self.citation_graph.get(paper_id, [])

    def verify_citation(self, source_id: str, target_id: str, claim: str) -> Dict[str, Any]:
        """Verify if a citation supports a claim.

        Args:
            source_id: ID of the citing paper
            target_id: ID of the cited paper
            claim: The claim to verify

        Returns:
            Verification result with confidence score
        """
        # Check if the citation exists
        if target_id not in self.get_cited_papers(source_id):
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": "Citation not found"
            }

        # In a real implementation, this would involve:
        # 1. Retrieving the content of the cited paper
        # 2. Using an LLM to check if the claim is supported by the content
        # 3. Calculating a confidence score

        # For this implementation, we'll return a placeholder
        return {
            "verified": True,
            "confidence": 0.8,
            "reason": "Citation exists, but content verification requires LLM integration"
        }

    def generate_citation_path(self, start_id: str, end_id: str) -> List[str]:
        """Find citation path between papers using breadth-first search.

        Args:
            start_id: ID of the starting paper
            end_id: ID of the target paper

        Returns:
            List of paper IDs forming a path from start to end, or empty list if no path exists
        """
        # Check if papers exist
        if start_id not in self.papers or end_id not in self.papers:
            return []

        # Breadth-first search
        queue = [(start_id, [start_id])]
        visited = set([start_id])

        while queue:
            (node, path) = queue.pop(0)

            # Check outgoing citations
            for next_node in self.get_cited_papers(node):
                if next_node == end_id:
                    # Found the target
                    return path + [next_node]
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))

        # No path found
        return []

    def get_citation_network(self, paper_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get the citation network around a paper.

        Args:
            paper_id: ID of the central paper
            depth: How many levels of citations to include

        Returns:
            Dictionary with citation network data
        """
        if paper_id not in self.papers:
            return {"nodes": [], "links": []}

        # Initialize with the central paper
        nodes = {paper_id: self.papers.get(paper_id, {})}
        links = []

        # Process outgoing citations (papers cited by this paper)
        def process_outgoing(node_id, current_depth):
            if current_depth <= 0:
                return

            for cited_id in self.get_cited_papers(node_id):
                # Add node if not already added
                if cited_id not in nodes:
                    nodes[cited_id] = self.papers.get(cited_id, {})

                # Add link
                links.append({
                    "source": node_id,
                    "target": cited_id,
                    "type": "cites"
                })

                # Process next level
                process_outgoing(cited_id, current_depth - 1)

        # Process incoming citations (papers that cite this paper)
        def process_incoming(node_id, current_depth):
            if current_depth <= 0:
                return

            for citing_id in self.get_citing_papers(node_id):
                # Add node if not already added
                if citing_id not in nodes:
                    nodes[citing_id] = self.papers.get(citing_id, {})

                # Add link
                links.append({
                    "source": citing_id,
                    "target": node_id,
                    "type": "cites"
                })

                # Process next level
                process_incoming(citing_id, current_depth - 1)

        # Start processing
        process_outgoing(paper_id, depth)
        process_incoming(paper_id, depth)

        # Convert nodes dictionary to list
        node_list = [{"id": k, **v} for k, v in nodes.items()]

        return {
            "nodes": node_list,
            "links": links
        }


class ResearchQuestionAnalyzer:
    """Analyze and decompose research questions."""

    def __init__(self, llm):
        """Initialize the research question analyzer.

        Args:
            llm: Language model for analysis
        """
        self.llm = llm

        # Define question types
        self.question_types = {
            "factual": "Seeking specific facts or information",
            "conceptual": "Seeking explanation of concepts or theories",
            "comparative": "Comparing multiple concepts, methods, or studies",
            "causal": "Seeking cause-effect relationships",
            "methodological": "Asking about research methods or techniques",
            "gap": "Identifying research gaps or future directions",
            "synthesis": "Requiring synthesis of multiple sources"
        }

        # Create analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze the following research question in detail:

        Question: {question}

        Provide a structured analysis in JSON format with the following:
        1. Question type (one of: factual, conceptual, comparative, causal, methodological, gap, synthesis)
        2. Key research concepts (list of main concepts in the question)
        3. Required background knowledge (what domain knowledge is needed)
        4. Potential search queries (3-5 different queries to find relevant information)
        5. Complexity (simple, moderate, complex)

        JSON Response:
        """)

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze a research question to identify key components.

        Args:
            question: The research question to analyze

        Returns:
            Dictionary with question analysis
        """
        # Invoke the LLM
        response = self.llm.invoke(
            self.analysis_prompt.format(question=question)
        )

        # Try to parse as JSON
        try:
            analysis = json.loads(response.content.strip())
        except json.JSONDecodeError:
            # Fallback to simple analysis
            analysis = self._simple_analysis(question)

        return analysis

    def identify_question_type(self, question: str) -> str:
        """Identify the type of research question.

        Args:
            question: The research question

        Returns:
            Question type
        """
        question_lower = question.lower()

        # Simple rule-based classification
        if any(term in question_lower for term in ["what is", "define", "describe"]):
            return "conceptual"
        elif any(term in question_lower for term in ["compare", "difference between", "versus", "vs"]):
            return "comparative"
        elif any(term in question_lower for term in ["why", "cause", "effect", "impact", "influence"]):
            return "causal"
        elif any(term in question_lower for term in ["how to", "method", "technique", "approach"]):
            return "methodological"
        elif any(term in question_lower for term in ["gap", "future", "limitation", "unexplored"]):
            return "gap"
        elif any(term in question_lower for term in ["synthesize", "integrate", "combine"]):
            return "synthesis"
        else:
            return "factual"

    def extract_research_concepts(self, question: str) -> List[str]:
        """Extract key research concepts from the question.

        Args:
            question: The research question

        Returns:
            List of key concepts
        """
        # In a real implementation, this would use NLP techniques
        # For this implementation, we'll use a simple approach

        # Remove common question words and stopwords
        stopwords = ["what", "is", "are", "how", "why", "when", "where", "which",
                     "the", "a", "an", "in", "on", "at", "to", "for", "with", "by",
                     "of", "and", "or", "research", "study", "paper", "papers"]

        # Tokenize and filter
        words = question.lower().replace("?", "").replace(".", "").split()
        filtered_words = [word for word in words if word not in stopwords]

        # Extract potential noun phrases (simplified)
        concepts = []
        current_concept = []

        for word in filtered_words:
            current_concept.append(word)

            # If we hit certain words, end the current concept
            if word in ["and", "or", "but", "because", "however"]:
                if len(current_concept) > 1:
                    # Remove the conjunction
                    concept = " ".join(current_concept[:-1])
                    concepts.append(concept)
                current_concept = []

        # Add the last concept if it exists
        if current_concept:
            concept = " ".join(current_concept)
            concepts.append(concept)

        return concepts

    def generate_search_queries(self, question: str) -> List[str]:
        """Generate multiple search queries for the research question.

        Args:
            question: The research question

        Returns:
            List of search queries
        """
        # Extract concepts
        concepts = self.extract_research_concepts(question)

        # Generate base query
        base_query = " ".join(concepts[:3])  # Use top 3 concepts

        # Generate variations
        queries = [
            base_query,
            f"recent research on {base_query}",
            f"{base_query} review paper",
            f"{base_query} methodology"
        ]

        # If it's a comparative question, try to extract the compared entities
        if self.identify_question_type(question) == "comparative":
            if "between" in question.lower():
                parts = question.lower().split("between")[1].split("and")
                if len(parts) >= 2:
                    entity1 = parts[0].strip()
                    entity2 = parts[1].strip().split()[0]  # Take first word after "and"
                    queries.append(f"comparison of {entity1} and {entity2}")

        return queries

    def _simple_analysis(self, question: str) -> Dict[str, Any]:
        """Perform simple rule-based analysis when LLM fails.

        Args:
            question: The research question

        Returns:
            Dictionary with question analysis
        """
        question_type = self.identify_question_type(question)
        concepts = self.extract_research_concepts(question)
        queries = self.generate_search_queries(question)

        # Determine complexity based on length and concepts
        if len(question.split()) > 20 or len(concepts) > 4:
            complexity = "complex"
        elif len(question.split()) > 10 or len(concepts) > 2:
            complexity = "moderate"
        else:
            complexity = "simple"

        return {
            "question_type": question_type,
            "key_concepts": concepts,
            "required_background": [f"{concept} fundamentals" for concept in concepts[:2]],
            "search_queries": queries,
            "complexity": complexity
        }


class LiteratureReviewGenerator:
    """Generate literature reviews from multiple papers."""

    def __init__(self, llm):
        """Initialize the literature review generator.

        Args:
            llm: Language model for review generation
        """
        self.llm = llm

        # Create review generation prompt
        self.review_prompt = ChatPromptTemplate.from_template("""
        Generate a comprehensive literature review on the following research question:

        Question: {question}

        Based on the following paper excerpts:

        {paper_excerpts}

        Paper metadata:
        {paper_metadata}

        Your literature review should:
        1. Provide an overview of the current state of research
        2. Compare and contrast different approaches
        3. Identify key findings and consensus
        4. Highlight research gaps
        5. Include proper citations to the source papers

        Literature Review:
        """)

        # Create paper summarization prompt
        self.summary_prompt = ChatPromptTemplate.from_template("""
        Summarize the following paper excerpts into a concise summary:

        Title: {title}
        Authors: {authors}
        Year: {year}

        Excerpts:
        {excerpts}

        Provide a 3-5 sentence summary that captures:
        1. The main research question/objective
        2. The methodology used
        3. The key findings
        4. The significance of the work

        Summary:
        """)

        # Create methodology comparison prompt
        self.methodology_prompt = ChatPromptTemplate.from_template("""
        Compare the methodologies used in the following papers:

        {paper_methodologies}

        Your comparison should:
        1. Identify similarities and differences in approaches
        2. Evaluate strengths and limitations of each methodology
        3. Discuss which methodologies are most appropriate for which research questions

        Methodology Comparison:
        """)

        # Create research gap prompt
        self.gap_prompt = ChatPromptTemplate.from_template("""
        Based on the following paper excerpts and the research question:

        Question: {question}

        Paper excerpts:
        {paper_excerpts}

        Identify research gaps and future directions by considering:
        1. What aspects of the question remain unexplored?
        2. What limitations exist in current methodologies?
        3. What contradictions or inconsistencies exist in the literature?
        4. What new research directions are suggested by the current findings?

        Research Gaps:
        """)

        # Create citation formatting prompt
        self.citation_prompt = ChatPromptTemplate.from_template("""
        Format the following review text with proper citations:

        Review text:
        {review_text}

        Available papers for citation:
        {citations}

        Format each citation as [Author, Year] inline and add a References section at the end.

        Formatted Review:
        """)

    def generate_review(self, question: str, relevant_chunks: List[Document], papers_metadata: List[Dict[str, Any]]) -> str:
        """Generate a literature review for a research question.

        Args:
            question: The research question
            relevant_chunks: List of relevant document chunks
            papers_metadata: Metadata for the papers

        Returns:
            Generated literature review
        """
        # Format paper metadata
        paper_metadata_str = self._format_metadata_for_prompt(papers_metadata)

        # Process chunks in batches to respect token limits
        review_content = self._process_review_in_batches(question, relevant_chunks, paper_metadata_str)

        # Format with citations
        citations_str = self._format_citations_for_prompt(papers_metadata)
        formatted_review = self.format_with_citations(review_content, citations_str)

        return formatted_review

    def _process_review_in_batches(self, question: str, chunks: List[Document], paper_metadata_str: str) -> str:
        """Process literature review generation in batches to respect token limits.

        Args:
            question: The research question
            chunks: List of document chunks
            paper_metadata_str: Formatted paper metadata

        Returns:
            Generated review content
        """
        # For Groq free tier, limit each batch to ~5000 tokens
        MAX_BATCH_CHARS = 20000

        # Group chunks by source document
        source_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            if source not in source_chunks:
                source_chunks[source] = []
            source_chunks[source].append(chunk)

        # Create batches based on document sources
        batches = []
        current_batch = []
        current_batch_size = 0

        # First, try to keep chunks from the same document together
        for source, source_chunk_list in source_chunks.items():
            source_text = self._format_chunks_for_prompt(source_chunk_list)
            source_size = len(source_text)

            # If this source's chunks would exceed batch size, split it
            if source_size > MAX_BATCH_CHARS:
                # Split large sources into smaller batches
                sub_batches = []
                sub_batch = []
                sub_batch_size = 0

                for chunk in source_chunk_list:
                    chunk_text = f"Excerpt (from {chunk.metadata.get('source', 'Unknown Source')}, section: {chunk.metadata.get('section', 'Unknown Section')}):\n{chunk.page_content}\n"
                    chunk_size = len(chunk_text)

                    if sub_batch_size + chunk_size > MAX_BATCH_CHARS and sub_batch:
                        sub_batches.append(sub_batch)
                        sub_batch = []
                        sub_batch_size = 0

                    sub_batch.append(chunk)
                    sub_batch_size += chunk_size

                if sub_batch:
                    sub_batches.append(sub_batch)

                # Add sub-batches to main batches
                for sub_batch in sub_batches:
                    batches.append(sub_batch)

            # If adding this source would exceed the batch size, start a new batch
            elif current_batch_size + source_size > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = source_chunk_list
                current_batch_size = source_size
            else:
                # Add to current batch
                current_batch.extend(source_chunk_list)
                current_batch_size += source_size

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # If we have no batches, return a simple message
        if not batches:
            return f"Unable to generate literature review for question: {question}"

        # If we have only one batch, process it directly
        if len(batches) == 1:
            paper_excerpts = self._format_chunks_for_prompt(batches[0])
            response = self.llm.invoke(
                self.review_prompt.format(
                    question=question,
                    paper_excerpts=paper_excerpts,
                    paper_metadata=paper_metadata_str
                )
            )
            return response.content

        # Process each batch to get partial reviews
        partial_reviews = []

        # Create iterator with progress bar if available
        if TQDM_AVAILABLE:
            batch_iterator = tqdm(enumerate(batches), total=len(batches), desc="Processing literature review batches")
        else:
            batch_iterator = enumerate(batches)
            print(f"Processing {len(batches)} literature review batches...")

        for i, batch in batch_iterator:
            paper_excerpts = self._format_chunks_for_prompt(batch)

            # Create a batch-specific prompt
            batch_prompt = ChatPromptTemplate.from_template("""
            Generate a partial literature review on the following research question:

            Question: {question}

            Based on the following paper excerpts (batch {batch_num} of {total_batches}):

            {paper_excerpts}

            Paper metadata:
            {paper_metadata}

            Your partial review should:
            1. Summarize the key findings from these specific excerpts
            2. Identify methodologies used in these excerpts
            3. Note any consensus or contradictions in these excerpts

            Partial Literature Review:
            """)

            start_time = time.time()
            try:
                response = self.llm.invoke(
                    batch_prompt.format(
                        question=question,
                        batch_num=i+1,
                        total_batches=len(batches),
                        paper_excerpts=paper_excerpts,
                        paper_metadata=paper_metadata_str
                    )
                )
                partial_reviews.append(response.content)

                # Show processing time if tqdm not available
                if not TQDM_AVAILABLE:
                    elapsed = time.time() - start_time
                    print(f"  Batch {i+1}/{len(batches)} processed in {elapsed:.2f}s")

            except Exception as e:
                # If a batch fails, note the error but continue with other batches
                error_msg = f"[Error processing batch {i+1}: {str(e)}]"
                partial_reviews.append(error_msg)

                # Show error if tqdm not available
                if not TQDM_AVAILABLE:
                    print(f"  Error in batch {i+1}: {str(e)}")

        # Combine partial reviews into a final review
        if partial_reviews:
            # Create a final review prompt
            final_prompt = ChatPromptTemplate.from_template("""
            Generate a comprehensive literature review by synthesizing these partial reviews:

            Question: {question}

            Partial reviews:
            {reviews}

            Paper metadata:
            {paper_metadata}

            Your final literature review should:
            1. Provide an overview of the current state of research
            2. Compare and contrast different approaches
            3. Identify key findings and consensus
            4. Highlight research gaps
            5. Include proper citations to the source papers

            Final Literature Review:
            """)

            try:
                response = self.llm.invoke(
                    final_prompt.format(
                        question=question,
                        reviews="\n\n".join(partial_reviews),
                        paper_metadata=paper_metadata_str
                    )
                )
                return response.content
            except Exception as e:
                # If final review fails, return the concatenated partial reviews
                return f"Literature Review for: {question}\n\n" + "\n\n".join(partial_reviews)

        # Fallback if something went wrong
        return f"Unable to generate a complete literature review for question: {question}"

    def summarize_paper(self, paper_chunks: List[Document], metadata: Dict[str, Any]) -> str:
        """Summarize a single paper.

        Args:
            paper_chunks: Chunks from the paper
            metadata: Paper metadata

        Returns:
            Paper summary
        """
        # Extract metadata
        title = metadata.get("title", "Unknown Title")
        authors = ", ".join(metadata.get("authors", ["Unknown Author"]))
        year = metadata.get("year", "Unknown Year")

        # Process paper in batches to respect token limits
        return self._process_paper_in_batches(paper_chunks, title, authors, year)

    def _process_paper_in_batches(self, paper_chunks: List[Document], title: str, authors: str, year: str) -> str:
        """Process a paper in batches to respect token limits.

        Args:
            paper_chunks: Chunks from the paper
            title: Paper title
            authors: Paper authors
            year: Publication year

        Returns:
            Paper summary
        """
        # For Groq free tier, limit each batch to ~5000 tokens (approx. 4000 words or ~20,000 chars)
        MAX_BATCH_CHARS = 20000

        # Prioritize important sections first
        prioritized_chunks = []

        # First, identify and prioritize abstract, intro, and conclusion
        for chunk in paper_chunks:
            section = chunk.metadata.get("section", "").lower()
            # Give priority score (lower is higher priority)
            if section == "abstract":
                chunk.metadata["priority"] = 1
            elif section == "introduction":
                chunk.metadata["priority"] = 2
            elif section in ["conclusion", "discussion"]:
                chunk.metadata["priority"] = 3
            else:
                chunk.metadata["priority"] = 4
            prioritized_chunks.append(chunk)

        # Sort chunks by priority
        prioritized_chunks.sort(key=lambda x: x.metadata["priority"])

        # Create batches
        batches = []
        current_batch = []
        current_batch_size = 0

        for chunk in prioritized_chunks:
            chunk_size = len(chunk.page_content)

            # If adding this chunk would exceed the batch size, start a new batch
            if current_batch_size + chunk_size > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0

            current_batch.append(chunk)
            current_batch_size += chunk_size

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # If we have no batches (unlikely), return a simple message
        if not batches:
            return f"Unable to summarize paper: {title} by {authors} ({year})"

        # If we have only one batch, process it directly
        if len(batches) == 1:
            excerpts = "\n\n".join([chunk.page_content for chunk in batches[0]])
            response = self.llm.invoke(
                self.summary_prompt.format(
                    title=title,
                    authors=authors,
                    year=year,
                    excerpts=excerpts
                )
            )
            return response.content

        # Process each batch to get partial summaries
        partial_summaries = []

        # Create iterator with progress bar if available
        if TQDM_AVAILABLE:
            batch_iterator = tqdm(enumerate(batches), total=len(batches), desc="Processing paper summary batches")
        else:
            batch_iterator = enumerate(batches)
            print(f"Processing {len(batches)} paper summary batches...")

        for i, batch in batch_iterator:
            excerpts = "\n\n".join([chunk.page_content for chunk in batch])

            # Create a batch-specific prompt
            batch_prompt = ChatPromptTemplate.from_template("""
            Summarize the following section of a paper:

            Title: {title}
            Authors: {authors}
            Year: {year}
            Section: Batch {batch_num} of {total_batches}

            Excerpts:
            {excerpts}

            Provide a concise summary of this section of the paper.
            Focus on key points, findings, and contributions.

            Summary:
            """)

            start_time = time.time()
            try:
                response = self.llm.invoke(
                    batch_prompt.format(
                        title=title,
                        authors=authors,
                        year=year,
                        batch_num=i+1,
                        total_batches=len(batches),
                        excerpts=excerpts
                    )
                )
                partial_summaries.append(response.content)

                # Show processing time if tqdm not available
                if not TQDM_AVAILABLE:
                    elapsed = time.time() - start_time
                    print(f"  Batch {i+1}/{len(batches)} processed in {elapsed:.2f}s")

            except Exception as e:
                # If a batch fails, note the error but continue with other batches
                error_msg = f"[Error processing batch {i+1}: {str(e)}]"
                partial_summaries.append(error_msg)

                # Show error if tqdm not available
                if not TQDM_AVAILABLE:
                    print(f"  Error in batch {i+1}: {str(e)}")

        # Combine partial summaries into a final summary
        if partial_summaries:
            # Create a final summary prompt
            final_prompt = ChatPromptTemplate.from_template("""
            Create a comprehensive summary of the following paper based on these partial summaries:

            Title: {title}
            Authors: {authors}
            Year: {year}

            Partial summaries:
            {summaries}

            Provide a 3-5 sentence summary that captures:
            1. The main research question/objective
            2. The methodology used
            3. The key findings
            4. The significance of the work

            Final Summary:
            """)

            try:
                response = self.llm.invoke(
                    final_prompt.format(
                        title=title,
                        authors=authors,
                        year=year,
                        summaries="\n\n".join(partial_summaries)
                    )
                )
                return response.content
            except Exception as e:
                # If final summary fails, return the concatenated partial summaries
                return f"Paper: {title} by {authors} ({year})\n\n" + "\n\n".join(partial_summaries)

        # Fallback if something went wrong
        return f"Unable to generate a complete summary for {title} by {authors} ({year})"

    def compare_methodologies(self, papers: List[Dict[str, Any]]) -> str:
        """Compare methodologies across papers.

        Args:
            papers: List of papers with methodology sections

        Returns:
            Methodology comparison
        """
        # Format methodologies
        paper_methodologies = []

        for paper in papers:
            title = paper.get("metadata", {}).get("title", "Unknown Title")
            authors = ", ".join(paper.get("metadata", {}).get("authors", ["Unknown Author"]))
            year = paper.get("metadata", {}).get("year", "Unknown Year")

            # Extract methodology section if available
            methodology = ""
            for section_name, section_content in paper.get("sections", {}).items():
                if section_name in ["methodology", "methods", "experiments"]:
                    methodology = section_content
                    break

            if methodology:
                paper_methodologies.append(f"Paper: {title} ({authors}, {year})\n\nMethodology:\n{methodology}\n")

        # If no methodologies found, return message
        if not paper_methodologies:
            return "No methodology sections found in the provided papers."

        # Format for prompt
        paper_methodologies_str = "\n---\n".join(paper_methodologies)

        # Generate comparison
        response = self.llm.invoke(
            self.methodology_prompt.format(
                paper_methodologies=paper_methodologies_str
            )
        )

        return response.content

    def identify_research_gaps(self, papers: List[Dict[str, Any]], question: str) -> str:
        """Identify research gaps related to the question.

        Args:
            papers: List of papers
            question: Research question

        Returns:
            Research gaps analysis
        """
        # Extract relevant excerpts
        paper_excerpts = []

        for paper in papers:
            title = paper.get("metadata", {}).get("title", "Unknown Title")
            authors = ", ".join(paper.get("metadata", {}).get("authors", ["Unknown Author"]))
            year = paper.get("metadata", {}).get("year", "Unknown Year")

            # Extract conclusion and discussion sections if available
            relevant_content = ""
            for section_name, section_content in paper.get("sections", {}).items():
                if section_name in ["conclusion", "discussion", "limitations", "future work"]:
                    relevant_content += section_content + "\n\n"

            if relevant_content:
                paper_excerpts.append({
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "content": relevant_content,
                    "priority": 1  # Higher priority for conclusion/discussion sections
                })
            else:
                # If no relevant sections found, use abstract and introduction
                for section_name, section_content in paper.get("sections", {}).items():
                    if section_name in ["abstract", "introduction"]:
                        if not relevant_content:  # Only add if we don't have content yet
                            relevant_content = section_content

                if relevant_content:
                    paper_excerpts.append({
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "content": relevant_content,
                        "priority": 2  # Medium priority for abstract/intro
                    })
                else:
                    # If still no content, use a sample of all sections
                    all_content = "\n\n".join([section[:500] for section in paper.get("sections", {}).values()])

                    if all_content:
                        paper_excerpts.append({
                            "title": title,
                            "authors": authors,
                            "year": year,
                            "content": all_content,
                            "priority": 3  # Lower priority for general content
                        })

        # Sort by priority
        paper_excerpts.sort(key=lambda x: x["priority"])

        # Process in batches to respect token limits
        return self._process_gaps_in_batches(paper_excerpts, question)

    def _process_gaps_in_batches(self, paper_excerpts: List[Dict[str, Any]], question: str) -> str:
        """Process research gaps analysis in batches to respect token limits.

        Args:
            paper_excerpts: List of paper excerpts with metadata
            question: Research question

        Returns:
            Research gaps analysis
        """
        # For Groq free tier, limit each batch to ~5000 tokens
        MAX_BATCH_CHARS = 20000

        # Create batches
        batches = []
        current_batch = []
        current_batch_size = 0

        for excerpt in paper_excerpts:
            excerpt_text = f"Paper: {excerpt['title']} ({excerpt['authors']}, {excerpt['year']})\n\nExcerpts:\n{excerpt['content']}\n"
            excerpt_size = len(excerpt_text)

            # If adding this excerpt would exceed the batch size, start a new batch
            if current_batch_size + excerpt_size > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0

            current_batch.append(excerpt_text)
            current_batch_size += excerpt_size

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # If we have no batches, return a simple message
        if not batches:
            return f"Unable to identify research gaps for question: {question}"

        # If we have only one batch, process it directly
        if len(batches) == 1:
            paper_excerpts_str = "\n---\n".join(batches[0])
            response = self.llm.invoke(
                self.gap_prompt.format(
                    question=question,
                    paper_excerpts=paper_excerpts_str
                )
            )
            return response.content

        # Process each batch to get partial analyses
        partial_analyses = []

        # Create iterator with progress bar if available
        if TQDM_AVAILABLE:
            batch_iterator = tqdm(enumerate(batches), total=len(batches), desc="Processing research gaps batches")
        else:
            batch_iterator = enumerate(batches)
            print(f"Processing {len(batches)} research gaps batches...")

        for i, batch in batch_iterator:
            batch_excerpts = "\n---\n".join(batch)

            # Create a batch-specific prompt
            batch_prompt = ChatPromptTemplate.from_template("""
            Based on the following paper excerpts and the research question:

            Question: {question}

            Paper excerpts (batch {batch_num} of {total_batches}):
            {paper_excerpts}

            Identify potential research gaps and future directions by considering:
            1. What aspects of the question remain unexplored in these papers?
            2. What limitations exist in the methodologies described?
            3. What contradictions or inconsistencies exist in these findings?
            4. What new research directions are suggested by these papers?

            Partial Research Gaps Analysis:
            """)

            start_time = time.time()
            try:
                response = self.llm.invoke(
                    batch_prompt.format(
                        question=question,
                        batch_num=i+1,
                        total_batches=len(batches),
                        paper_excerpts=batch_excerpts
                    )
                )
                partial_analyses.append(response.content)

                # Show processing time if tqdm not available
                if not TQDM_AVAILABLE:
                    elapsed = time.time() - start_time
                    print(f"  Batch {i+1}/{len(batches)} processed in {elapsed:.2f}s")

            except Exception as e:
                # If a batch fails, note the error but continue with other batches
                error_msg = f"[Error processing batch {i+1}: {str(e)}]"
                partial_analyses.append(error_msg)

                # Show error if tqdm not available
                if not TQDM_AVAILABLE:
                    print(f"  Error in batch {i+1}: {str(e)}")

        # Combine partial analyses into a final analysis
        if partial_analyses:
            # Create a final analysis prompt
            final_prompt = ChatPromptTemplate.from_template("""
            Create a comprehensive research gaps analysis based on these partial analyses:

            Research Question: {question}

            Partial analyses:
            {analyses}

            Provide a final research gaps analysis that:
            1. Identifies the most significant gaps in the literature
            2. Highlights promising future research directions
            3. Notes methodological limitations in current research
            4. Suggests potential approaches to address these gaps

            Final Research Gaps Analysis:
            """)

            try:
                response = self.llm.invoke(
                    final_prompt.format(
                        question=question,
                        analyses="\n\n".join(partial_analyses)
                    )
                )
                return response.content
            except Exception as e:
                # If final analysis fails, return the concatenated partial analyses
                return f"Research Gaps for: {question}\n\n" + "\n\n".join(partial_analyses)

        # Fallback if something went wrong
        return f"Unable to generate a complete research gaps analysis for question: {question}"

    def format_with_citations(self, review_text: str, citations: str) -> str:
        """Format review text with proper citations.

        Args:
            review_text: The review text
            citations: Available citations

        Returns:
            Formatted review with citations
        """
        # Generate formatted review
        response = self.llm.invoke(
            self.citation_prompt.format(
                review_text=review_text,
                citations=citations
            )
        )

        return response.content

    def _format_chunks_for_prompt(self, chunks: List[Document]) -> str:
        """Format document chunks for the prompt.

        Args:
            chunks: List of document chunks

        Returns:
            Formatted string
        """
        formatted_chunks = []

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            source = metadata.get("source", "Unknown Source")
            section = metadata.get("section", "Unknown Section")

            formatted_chunks.append(f"Excerpt {i+1} (from {source}, section: {section}):\n{chunk.page_content}\n")

        return "\n".join(formatted_chunks)

    def _format_metadata_for_prompt(self, papers_metadata: List[Dict[str, Any]]) -> str:
        """Format paper metadata for the prompt.

        Args:
            papers_metadata: List of paper metadata

        Returns:
            Formatted string
        """
        formatted_metadata = []

        for i, metadata in enumerate(papers_metadata):
            title = metadata.get("title", "Unknown Title")
            authors = ", ".join(metadata.get("authors", ["Unknown Author"]))
            year = metadata.get("year", "Unknown Year")
            doi = metadata.get("doi", "No DOI")

            formatted_metadata.append(f"Paper {i+1}:\nTitle: {title}\nAuthors: {authors}\nYear: {year}\nDOI: {doi}\n")

        return "\n".join(formatted_metadata)

    def _format_citations_for_prompt(self, papers_metadata: List[Dict[str, Any]]) -> str:
        """Format citations for the prompt.

        Args:
            papers_metadata: List of paper metadata

        Returns:
            Formatted string
        """
        formatted_citations = []

        for metadata in papers_metadata:
            title = metadata.get("title", "Unknown Title")
            authors = ", ".join(metadata.get("authors", ["Unknown Author"]))
            year = metadata.get("year", "Unknown Year")
            doi = metadata.get("doi", "")

            citation = f"{authors} ({year}). {title}"
            if doi:
                citation += f". DOI: {doi}"

            formatted_citations.append(citation)

        return "\n".join(formatted_citations)

class ResearchLiteratureAssistant:
    """Complete Research Literature Assistant using LCEL."""

    def __init__(self, llm, embedding_model=None):
        """Initialize the research literature assistant.

        Args:
            llm: Language model for various tasks
            embedding_model: Model to generate embeddings (optional)
        """
        self.llm = llm
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.paper_processor = AcademicPaperProcessor()
        self.citation_tracker = CitationTracker()
        self.question_analyzer = ResearchQuestionAnalyzer(llm)
        self.review_generator = LiteratureReviewGenerator(llm)
        self.vectorstore = None
        self.papers = {}

    def add_paper(self, file_path: str) -> Dict[str, Any]:
        """Process and add a paper to the assistant.

        Args:
            file_path: Path to the PDF file

        Returns:
            Processed paper data
        """
        # Process the paper
        paper_data = self.paper_processor.process_paper(file_path)

        # Generate a paper ID (using filename as ID)
        paper_id = os.path.basename(file_path).replace(".pdf", "")

        # Add to citation tracker
        self.citation_tracker.add_paper(
            paper_id,
            paper_data["metadata"],
            paper_data["citations"]
        )

        # Store the paper data
        self.papers[paper_id] = {
            "metadata": paper_data["metadata"],
            "sections": paper_data["sections"],
            "figures": paper_data["figures"],
            "tables": paper_data["tables"],
            "citations": paper_data["citations"],
            "chunks": paper_data["chunks"]
        }

        # If vectorstore exists, add the chunks
        if self.vectorstore:
            self.vectorstore.add_documents(paper_data["chunks"])

        return paper_data

    def build_index(self):
        """Build vector index from processed papers.

        Returns:
            The built vectorstore
        """
        # Collect all chunks
        all_chunks = []
        for paper_id, paper_data in self.papers.items():
            all_chunks.extend(paper_data["chunks"])

        # Create vectorstore
        if all_chunks:
            self.vectorstore = FAISS.from_documents(
                all_chunks,
                self.embedding_model
            )
        else:
            # Create empty vectorstore
            self.vectorstore = FAISS.from_texts(
                ["placeholder"],
                self.embedding_model
            )

        return self.vectorstore

    def create_chain(self):
        """Create LCEL chain for the assistant.

        Returns:
            The LCEL chain
        """
        # Ensure vectorstore is built
        if not self.vectorstore:
            self.build_index()

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        # Define chain components

        # 1. Question analysis
        def analyze_question(query):
            analysis = self.question_analyzer.analyze_question(query)
            return {
                "query": query,
                "analysis": analysis
            }

        # 2. Generate search queries
        def generate_search_queries(inputs):
            query = inputs["query"]
            analysis = inputs["analysis"]

            # Use search queries from analysis if available
            if "search_queries" in analysis and analysis["search_queries"]:
                search_queries = analysis["search_queries"]
            else:
                # Generate search queries
                search_queries = self.question_analyzer.generate_search_queries(query)

            return {
                **inputs,
                "search_queries": search_queries
            }

        # 3. Retrieve documents
        def retrieve_documents(inputs):
            search_queries = inputs["search_queries"]

            # Retrieve documents for each query
            all_docs = []
            for query in search_queries:
                docs = retriever.get_relevant_documents(query)
                all_docs.extend(docs)

            # Remove duplicates
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)

            return {
                **inputs,
                "documents": unique_docs[:20]  # Limit to top 20 unique documents
            }

        # 4. Extract paper metadata
        def extract_paper_metadata(inputs):
            documents = inputs["documents"]

            # Extract unique paper sources
            paper_sources = set()
            for doc in documents:
                source = doc.metadata.get("source")
                if source:
                    paper_sources.add(source)

            # Get metadata for each paper
            papers_metadata = []
            for source in paper_sources:
                paper_id = os.path.basename(source).replace(".pdf", "")
                if paper_id in self.papers:
                    papers_metadata.append(self.papers[paper_id]["metadata"])

            return {
                **inputs,
                "papers_metadata": papers_metadata
            }

        # 5. Generate literature review
        def generate_review(inputs):
            query = inputs["query"]
            documents = inputs["documents"]
            papers_metadata = inputs["papers_metadata"]

            # Generate review
            review = self.review_generator.generate_review(
                query,
                documents,
                papers_metadata
            )

            return {
                **inputs,
                "review": review
            }

        # Create the chain
        chain = (
            RunnableLambda(analyze_question)
            | RunnableLambda(generate_search_queries)
            | RunnableLambda(retrieve_documents)
            | RunnableLambda(extract_paper_metadata)
            | RunnableLambda(generate_review)
        )

        return chain

    def answer_research_question(self, question: str) -> Dict[str, Any]:
        """Answer a research question using the assistant.

        Args:
            question: The research question

        Returns:
            Dictionary with question analysis, retrieved documents, and generated review
        """
        # Create chain if not already created
        chain = self.create_chain()

        # Run the chain
        result = chain.invoke(question)

        return result

    def generate_literature_review(self, topic: str) -> str:
        """Generate a literature review on a topic.

        Args:
            topic: The research topic

        Returns:
            Generated literature review
        """
        # Format as a research question if it's not already
        if not topic.endswith("?"):
            question = f"What is the current state of research on {topic}?"
        else:
            question = topic

        # Use the answer_research_question method
        result = self.answer_research_question(question)

        # Return just the review
        return result.get("review", "No review generated. Please add papers first.")

    def get_citation_network(self, paper_id: str = None, depth: int = 1) -> Dict[str, Any]:
        """Get the citation network for visualization.

        Args:
            paper_id: ID of the central paper (optional)
            depth: How many levels of citations to include

        Returns:
            Citation network data
        """
        # If no paper_id provided, use the first paper
        if paper_id is None and self.papers:
            paper_id = next(iter(self.papers.keys()))

        # If no papers, return empty network
        if not paper_id or not self.papers:
            return {"nodes": [], "links": []}

        # Get citation network
        return self.citation_tracker.get_citation_network(paper_id, depth)

    def summarize_papers(self) -> Dict[str, str]:
        """Generate summaries for all papers.

        Returns:
            Dictionary mapping paper IDs to summaries
        """
        summaries = {}

        for paper_id, paper_data in self.papers.items():
            # Get chunks for this paper
            chunks = paper_data["chunks"]
            metadata = paper_data["metadata"]

            # Generate summary
            summary = self.review_generator.summarize_paper(chunks, metadata)
            summaries[paper_id] = summary

        return summaries

    def compare_paper_methodologies(self) -> str:
        """Compare methodologies across all papers.

        Returns:
            Methodology comparison
        """
        # Convert papers to format expected by compare_methodologies
        papers_for_comparison = []
        for paper_id, paper_data in self.papers.items():
            papers_for_comparison.append({
                "metadata": paper_data["metadata"],
                "sections": paper_data["sections"]
            })

        # Generate comparison
        return self.review_generator.compare_methodologies(papers_for_comparison)

    def identify_gaps(self, question: str) -> str:
        """Identify research gaps related to a question.

        Args:
            question: Research question

        Returns:
            Research gaps analysis
        """
        # Convert papers to format expected by identify_research_gaps
        papers_for_analysis = []
        for paper_id, paper_data in self.papers.items():
            papers_for_analysis.append({
                "metadata": paper_data["metadata"],
                "sections": paper_data["sections"]
            })

        # Generate gaps analysis
        return self.review_generator.identify_research_gaps(papers_for_analysis, question)