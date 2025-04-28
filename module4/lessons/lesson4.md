# 📄 Module 4: Document Processing & RAG Foundations - Lesson 4: Metadata Extraction & Management 🏷️

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🏷️ Understand the importance of metadata in RAG systems
- 🔍 Master techniques for automatic metadata extraction
- 🧠 Implement content-based metadata generation
- 🏗️ Design structured metadata schemas for consistent organization
- 🔄 Develop metadata filtering mechanisms for retrieval refinement
- 📊 Learn best practices for metadata management in production systems

---

## 🏷️ Why Metadata Matters in RAG Systems

<img src="https://media.giphy.com/media/l0HlHFRbmaZtBRhXG/giphy.gif" width="50%" height="50%"/>

### The Power of Metadata

Metadata is the "data about data" that provides context, structure, and additional information about your documents. In RAG systems, metadata serves several critical functions:

1. **Enhanced Retrieval**: Enables filtering and faceted search beyond pure semantic similarity
2. **Context Preservation**: Maintains document relationships and hierarchies
3. **Source Attribution**: Provides provenance information for generated responses
4. **Quality Control**: Helps identify and filter out low-quality or irrelevant content
5. **User Experience**: Enables more intuitive and targeted information access

### Types of Metadata in RAG Systems

| Metadata Type | Description | Examples |
|---------------|-------------|----------|
| Document Properties | Information about the document itself | Author, creation date, file size, format |
| Content-based | Extracted from document content | Topics, entities, keywords, sentiment |
| Structural | Information about document organization | Headings, sections, paragraph positions |
| Custom | User-defined attributes | Categories, tags, importance ratings |
| Relational | Connections between documents | References, citations, prerequisites |

---

## 🔍 Automatic Metadata Extraction Techniques

### 1. Document Property Extraction

The most basic form of metadata comes from the document's own properties:

```python
def extract_file_metadata(file_path):
    """Extract basic metadata from file properties."""
    file_stats = os.stat(file_path)
    
    return {
        'file_name': os.path.basename(file_path),
        'file_path': file_path,
        'file_size': file_stats.st_size,
        'created_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
        'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        'extension': os.path.splitext(file_path)[1].lower()[1:]
    }
```

For specific file formats, you can extract additional metadata:

```python
def extract_pdf_metadata(pdf_path):
    """Extract metadata from PDF documents."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        info = reader.metadata
        
        return {
            'title': info.get('/Title', ''),
            'author': info.get('/Author', ''),
            'creator': info.get('/Creator', ''),
            'producer': info.get('/Producer', ''),
            'subject': info.get('/Subject', ''),
            'creation_date': info.get('/CreationDate', ''),
            'modification_date': info.get('/ModDate', ''),
            'page_count': len(reader.pages)
        }
```

### 2. Structural Metadata Extraction

Structural metadata captures the organization and hierarchy of a document:

```python
def extract_document_structure(document):
    """Extract structural metadata from a document."""
    structure = {
        'sections': [],
        'has_table_of_contents': False,
        'has_appendix': False,
        'has_references': False
    }
    
    # Extract headings and create section hierarchy
    headings = []
    for item in document['content']:
        if item.get('style', '').startswith('Heading'):
            level = int(item['style'].replace('Heading', ''))
            headings.append({
                'text': item['text'],
                'level': level,
                'position': item.get('position', 0)
            })
    
    # Organize headings into nested structure
    current_section = None
    for heading in headings:
        if heading['level'] == 1:
            # Top-level section
            section = {
                'title': heading['text'],
                'level': 1,
                'subsections': []
            }
            structure['sections'].append(section)
            current_section = section
        elif current_section and heading['level'] > current_section['level']:
            # Subsection
            subsection = {
                'title': heading['text'],
                'level': heading['level']
            }
            current_section['subsections'].append(subsection)
    
    # Check for special sections
    section_titles = [s['title'].lower() for s in structure['sections']]
    structure['has_table_of_contents'] = any('content' in title for title in section_titles)
    structure['has_appendix'] = any('appendix' in title for title in section_titles)
    structure['has_references'] = any(title in ['references', 'bibliography'] for title in section_titles)
    
    return structure
```

---

## 🧠 Content-Based Metadata Generation

Content-based metadata is extracted from the actual content of the document, providing semantic insights.

### 1. Topic Extraction

```python
class TopicExtractor:
    """Extract topics from document content."""
    
    def __init__(self, num_topics=5, min_topic_frequency=2):
        self.num_topics = num_topics
        self.min_topic_frequency = min_topic_frequency
        self.stop_words = set(stopwords.words('english'))
        
    def extract_topics(self, text):
        """Extract main topics from text."""
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words and len(t) > 3]
        
        # Count word frequencies
        word_freq = Counter(tokens)
        
        # Get most common words as topics
        topics = [word for word, count in word_freq.most_common(self.num_topics) 
                 if count >= self.min_topic_frequency]
        
        return topics
```

### 2. Entity Extraction

Entity extraction identifies specific types of information like people, organizations, locations, and dates:

```python
class EntityExtractor:
    """Extract named entities from text."""
    
    def __init__(self):
        # In a real implementation, you might use spaCy or another NLP library
        pass
        
    def extract_entities(self, text):
        """Extract named entities from text."""
        entities = {
            'person': [],
            'organization': [],
            'location': [],
            'date': [],
            'misc': []
        }
        
        # Example implementation using regex patterns
        # Person names (simplified)
        person_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        for match in re.finditer(person_pattern, text):
            entities['person'].append({
                'text': match.group(0),
                'position': match.span()
            })
            
        # Dates (simplified)
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
        for match in re.finditer(date_pattern, text):
            entities['date'].append({
                'text': match.group(0),
                'position': match.span()
            })
            
        # Organizations (simplified)
        org_indicators = ['Inc', 'Corp', 'LLC', 'Ltd', 'Company', 'Organization']
        for indicator in org_indicators:
            pattern = rf'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+{indicator})\b'
            for match in re.finditer(pattern, text):
                entities['organization'].append({
                    'text': match.group(0),
                    'position': match.span()
                })
                
        return entities
```

### 3. Keyword Extraction

```python
class KeywordExtractor:
    """Extract keywords from document content."""
    
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.stop_words = set(stopwords.words('english'))
        
    def extract_keywords(self, text):
        """Extract keywords using TF-IDF approach."""
        # Tokenize and clean
        sentences = sent_tokenize(text)
        word_frequencies = {}
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word.isalpha() and word not in self.stop_words and len(word) > 3:
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        # Normalize frequencies
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        normalized_frequencies = {
            word: frequency/max_frequency 
            for word, frequency in word_frequencies.items()
        }
        
        # Sort by frequency and return top N
        sorted_keywords = sorted(
            normalized_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [word for word, _ in sorted_keywords[:self.top_n]]
```

### 4. Sentiment Analysis

```python
class SentimentAnalyzer:
    """Analyze sentiment of document content."""
    
    def analyze_sentiment(self, text):
        """
        Perform basic sentiment analysis.
        Returns a score between -1 (negative) and 1 (positive).
        """
        # Simple lexicon-based approach
        positive_words = {'good', 'great', 'excellent', 'positive', 'wonderful', 
                         'best', 'love', 'perfect', 'better', 'impressive'}
        negative_words = {'bad', 'terrible', 'awful', 'negative', 'worst',
                         'hate', 'poor', 'horrible', 'disappointing', 'worse'}
        
        # Tokenize and count
        tokens = word_tokenize(text.lower())
        positive_count = sum(1 for word in tokens if word in positive_words)
        negative_count = sum(1 for word in tokens if word in negative_words)
        
        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            return 0  # Neutral
            
        return (positive_count - negative_count) / total
```

---

## 🏗️ Structured Metadata Schemas

Creating consistent metadata schemas ensures that your RAG system can reliably use metadata for filtering and retrieval.

### 1. Base Metadata Schema

```python
class DocumentMetadata(BaseModel):
    """Base metadata schema for documents."""
    
    # Document identity
    doc_id: str
    title: Optional[str] = None
    
    # Source information
    source: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    
    # Temporal information
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    ingested_date: datetime = Field(default_factory=datetime.now)
    
    # Document properties
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    
    # Content properties
    language: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    
    # Quality indicators
    quality_score: Optional[float] = None
    confidence: Optional[float] = 1.0
    
    # Custom fields
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 2. Chunk Metadata Schema

```python
class ChunkMetadata(BaseModel):
    """Metadata schema for document chunks."""
    
    # Chunk identity
    chunk_id: str
    doc_id: str  # Parent document ID
    
    # Chunk position
    chunk_index: int
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None
    
    # Structural information
    section: Optional[str] = None
    section_level: Optional[int] = None
    is_table: bool = False
    is_list: bool = False
    is_title: bool = False
    
    # Content properties
    word_count: int
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    
    # Semantic properties
    embedding_model: Optional[str] = None
    
    # Inherited document metadata
    doc_title: Optional[str] = None
    doc_source: Optional[str] = None
    doc_author: Optional[str] = None
    doc_created_date: Optional[datetime] = None
    
    # Custom fields
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

## 🔄 Metadata Filtering for Retrieval

Metadata enables powerful filtering capabilities that complement semantic search.

### 1. Basic Metadata Filtering

```python
def filter_by_metadata(chunks, filters):
    """
    Filter chunks based on metadata criteria.
    
    Args:
        chunks: List of chunks with metadata
        filters: Dictionary of metadata filters
        
    Returns:
        Filtered list of chunks
    """
    filtered_chunks = []
    
    for chunk in chunks:
        matches_all = True
        
        for key, value in filters.items():
            # Handle nested keys (e.g., 'metadata.author')
            if '.' in key:
                parts = key.split('.')
                chunk_value = chunk
                for part in parts:
                    if isinstance(chunk_value, dict) and part in chunk_value:
                        chunk_value = chunk_value[part]
                    else:
                        chunk_value = None
                        break
            else:
                chunk_value = chunk.get(key)
            
            # Check if value matches
            if chunk_value is None:
                matches_all = False
                break
                
            # Handle different filter types
            if isinstance(value, list):
                # List membership
                if chunk_value not in value:
                    matches_all = False
                    break
            elif isinstance(value, dict) and 'operator' in value:
                # Comparison operators
                op = value['operator']
                compare_value = value['value']
                
                if op == 'eq' and chunk_value != compare_value:
                    matches_all = False
                elif op == 'neq' and chunk_value == compare_value:
                    matches_all = False
                elif op == 'gt' and not (chunk_value > compare_value):
                    matches_all = False
                elif op == 'gte' and not (chunk_value >= compare_value):
                    matches_all = False
                elif op == 'lt' and not (chunk_value < compare_value):
                    matches_all = False
                elif op == 'lte' and not (chunk_value <= compare_value):
                    matches_all = False
                elif op == 'contains' and compare_value not in chunk_value:
                    matches_all = False
            else:
                # Direct equality
                if chunk_value != value:
                    matches_all = False
                    break
        
        if matches_all:
            filtered_chunks.append(chunk)
    
    return filtered_chunks
```

### 2. Combining Semantic Search with Metadata Filtering

```python
def hybrid_search(query, chunks, embedding_model, filters=None, top_k=5):
    """
    Perform hybrid search combining semantic similarity with metadata filtering.
    
    Args:
        query: Search query
        chunks: List of chunks with content and metadata
        embedding_model: Model to generate embeddings
        filters: Optional metadata filters
        top_k: Number of results to return
        
    Returns:
        List of most relevant chunks
    """
    # Apply metadata filtering first if filters provided
    if filters:
        filtered_chunks = filter_by_metadata(chunks, filters)
    else:
        filtered_chunks = chunks
    
    if not filtered_chunks:
        return []
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query)
    
    # Calculate similarity scores
    results = []
    for chunk in filtered_chunks:
        # Get or generate chunk embedding
        if 'embedding' in chunk:
            chunk_embedding = chunk['embedding']
        else:
            chunk_embedding = embedding_model.encode(chunk['content'])
            
        # Calculate cosine similarity
        similarity = cosine_similarity(
            [query_embedding], 
            [chunk_embedding]
        )[0][0]
        
        results.append({
            'chunk': chunk,
            'similarity': similarity
        })
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]
```

---

## 📊 Metadata Management in Production Systems

### 1. Metadata Indexing

For efficient filtering, metadata should be properly indexed:

```python
class MetadataIndex:
    """Index for efficient metadata-based retrieval."""
    
    def __init__(self):
        self.indices = {}  # Field name -> value -> document IDs
        
    def add_document(self, doc_id, metadata):
        """Add document metadata to the index."""
        for field, value in metadata.items():
            if field not in self.indices:
                self.indices[field] = {}
                
            # Handle different value types
            if isinstance(value, list):
                # For list values, index each item
                for item in value:
                    self._add_to_index(field, item, doc_id)
            else:
                self._add_to_index(field, value, doc_id)
    
    def _add_to_index(self, field, value, doc_id):
        """Add a single value to the index."""
        if value not in self.indices[field]:
            self.indices[field][value] = set()
        self.indices[field][value].add(doc_id)
    
    def search(self, filters):
        """Search for documents matching all filters."""
        matching_ids = None
        
        for field, value in filters.items():
            if field not in self.indices:
                return set()  # No documents match this field
                
            if isinstance(value, list):
                # Union of all documents matching any value in the list
                field_matches = set()
                for v in value:
                    if v in self.indices[field]:
                        field_matches.update(self.indices[field][v])
            else:
                # Documents matching the specific value
                field_matches = self.indices[field].get(value, set())
            
            # Intersect with previous matches
            if matching_ids is None:
                matching_ids = field_matches
            else:
                matching_ids &= field_matches
                
            if not matching_ids:
                break  # No need to continue if no matches
                
        return matching_ids or set()
```

### 2. Metadata Validation and Cleaning

```python
class MetadataProcessor:
    """Process and validate document metadata."""
    
    def __init__(self, schema=None):
        self.schema = schema
        
    def process_metadata(self, metadata):
        """Clean and validate metadata."""
        # Make a copy to avoid modifying the original
        processed = metadata.copy()
        
        # Basic cleaning
        for key, value in processed.items():
            # Convert empty strings to None
            if value == '':
                processed[key] = None
                
            # Strip whitespace from string values
            elif isinstance(value, str):
                processed[key] = value.strip()
                
            # Ensure dates are in ISO format
            elif key.endswith('_date') and value and not isinstance(value, datetime):
                try:
                    processed[key] = parse_date(value)
                except:
                    processed[key] = None
        
        # Schema validation if schema provided
        if self.schema:
            try:
                processed = self.schema(**processed).dict()
            except ValidationError as e:
                # Handle validation errors
                print(f"Metadata validation error: {e}")
                # Fall back to original with minimal cleaning
        
        return processed
```

### 3. Metadata Update Strategies

```python
class MetadataManager:
    """Manage metadata throughout document lifecycle."""
    
    def __init__(self, db_client):
        self.db = db_client
        
    def update_document_metadata(self, doc_id, metadata_updates):
        """Update document metadata."""
        # Get current metadata
        current = self.db.get_document_metadata(doc_id)
        if not current:
            raise ValueError(f"Document {doc_id} not found")
            
        # Apply updates
        updated = current.copy()
        for key, value in metadata_updates.items():
            if value is None and key in updated:
                # Remove the field if value is None
                del updated[key]
            else:
                updated[key] = value
                
        # Update timestamp
        updated['last_modified'] = datetime.now()
        
        # Save to database
        self.db.update_document_metadata(doc_id, updated)
        
        # Update any dependent chunks
        self._propagate_to_chunks(doc_id, metadata_updates)
        
        return updated
        
    def _propagate_to_chunks(self, doc_id, metadata_updates):
        """Propagate relevant metadata updates to chunks."""
        # Fields that should be propagated to chunks
        propagate_fields = {
            'title': 'doc_title',
            'author': 'doc_author',
            'source': 'doc_source',
            'created_date': 'doc_created_date'
        }
        
        chunk_updates = {}
        for doc_field, chunk_field in propagate_fields.items():
            if doc_field in metadata_updates:
                chunk_updates[chunk_field] = metadata_updates[doc_field]
                
        if chunk_updates:
            self.db.update_chunk_metadata_by_doc_id(doc_id, chunk_updates)
```

---

## 💪 Practice Exercises

1. **Implement a Custom Metadata Extractor**: Create a metadata extractor for a specific domain (e.g., scientific papers, legal documents, or technical documentation) that extracts domain-specific metadata.

2. **Build a Metadata-Enhanced Retrieval System**: Extend a basic semantic search system to incorporate metadata filtering for more precise retrieval.

3. **Create a Metadata Visualization Tool**: Develop a tool that visualizes the metadata distribution across a document collection to identify patterns and gaps.

4. **Implement Automatic Quality Scoring**: Create a system that automatically assigns quality scores to documents based on metadata analysis.

5. **Design a Metadata Schema for a Specific Use Case**: Develop a comprehensive metadata schema for a specific application (e.g., educational content, product documentation, or research papers).

---

## 🔍 Key Takeaways

1. **Metadata Enhances RAG**: Metadata significantly improves retrieval precision and enables more sophisticated filtering beyond semantic search.

2. **Automatic Extraction is Powerful**: Combining document properties with content-based extraction provides rich metadata with minimal manual effort.

3. **Structured Schemas Matter**: Consistent metadata schemas ensure reliable filtering and organization of document collections.

4. **Hybrid Search is Superior**: Combining semantic search with metadata filtering provides the best retrieval performance.

5. **Metadata Management is Ongoing**: Effective metadata systems require continuous validation, updating, and optimization.

---

## 📚 Resources

- [Langchain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Pydantic Data Validation](https://docs.pydantic.dev/latest/)
- [NLTK for Text Processing](https://www.nltk.org/)
- [spaCy for Entity Extraction](https://spacy.io/usage/linguistic-features#named-entities)
- [Pinecone Hybrid Search](https://www.pinecone.io/learn/hybrid-search/)
- [Dublin Core Metadata Initiative](https://www.dublincore.org/)

---

## 🚀 Next Steps

In the next lesson, we'll explore **Building a Document Q&A System**, where we'll combine all the components we've learned so far—document processing, chunking, embeddings, and metadata—to create a complete RAG system that can answer questions from multiple documents with source attribution and confidence scoring.

---
