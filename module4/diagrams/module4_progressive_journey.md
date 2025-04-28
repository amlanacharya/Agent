# Module 4: Progressive Journey Diagrams

This document presents a progressive journey through the key components of Module 4, showing how document processing and RAG systems are built step by step.

## 1. Document Processing Pipeline

### Stage 1: Basic Document Loading
```mermaid
flowchart LR
    A[Raw Document] --> B[Document Loader]
    B --> C[Text Content]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 2: Multi-Format Document Processing
```mermaid
flowchart LR
    A1[PDF Document] --> B[Document Loader]
    A2[Text Document] --> B
    A3[DOCX Document] --> B
    A4[CSV Document] --> B
    B --> C[Normalized Text Content]
    B --> D[Basic Metadata]
    style A1 fill:#f9f9f9,stroke:#333,stroke-width:1px
    style A2 fill:#f9f9f9,stroke:#333,stroke-width:1px
    style A3 fill:#f9f9f9,stroke:#333,stroke-width:1px
    style A4 fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#e1f7d5,stroke:#333,stroke-width:1px
    style D fill:#ffe6cc,stroke:#333,stroke-width:1px
```

### Stage 3: Document Processing with Error Handling
```mermaid
flowchart TD
    A[Raw Document] --> B[Document Loader]
    B --> C{Format Supported?}
    C -->|Yes| D[Extract Content]
    C -->|No| E[Fallback Processing]
    D --> F[Normalize Text]
    E --> F
    F --> G[Extract Metadata]
    G --> H[Document Object]
    H --> I[Content]
    H --> J[Metadata]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#ffe6cc,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#ffcccc,stroke:#333,stroke-width:1px
    style F fill:#d4f1f9,stroke:#333,stroke-width:1px
    style G fill:#d4f1f9,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
    style I fill:#e1f7d5,stroke:#333,stroke-width:1px
    style J fill:#ffe6cc,stroke:#333,stroke-width:1px
```

### Stage 4: Complete Document Processing Pipeline
```mermaid
flowchart TD
    A[Raw Document] --> B[Document Loader]
    B --> C{Format Supported?}
    C -->|Yes| D[Extract Content]
    C -->|No| E[Fallback Processing]
    D --> F[Normalize Text]
    E --> F
    F --> G[Extract Metadata]
    G --> H[Document Object]
    H --> I[Text Splitter]
    I --> J[Document Chunks]
    J --> K[Embedding Model]
    K --> L[Vector Embeddings]
    J --> M[Metadata Enrichment]
    M --> N[Enhanced Metadata]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#ffe6cc,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#ffcccc,stroke:#333,stroke-width:1px
    style F fill:#d4f1f9,stroke:#333,stroke-width:1px
    style G fill:#d4f1f9,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
    style I fill:#d4f1f9,stroke:#333,stroke-width:1px
    style J fill:#e1f7d5,stroke:#333,stroke-width:1px
    style K fill:#d4f1f9,stroke:#333,stroke-width:1px
    style L fill:#e1f7d5,stroke:#333,stroke-width:1px
    style M fill:#d4f1f9,stroke:#333,stroke-width:1px
    style N fill:#ffe6cc,stroke:#333,stroke-width:1px
```

## 2. Chunking Strategies

### Stage 1: Simple Size-Based Chunking
```mermaid
flowchart LR
    A[Document Text] --> B[Size-Based Splitter]
    B --> C[Chunk 1]
    B --> D[Chunk 2]
    B --> E[Chunk 3]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#e1f7d5,stroke:#333,stroke-width:1px
    style D fill:#e1f7d5,stroke:#333,stroke-width:1px
    style E fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 2: Overlap-Based Chunking
```mermaid
flowchart LR
    A[Document Text] --> B[Overlap Splitter]
    B --> C[Chunk 1]
    B --> D[Chunk 2]
    B --> E[Chunk 3]
    C -.-> D
    D -.-> E
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#e1f7d5,stroke:#333,stroke-width:1px
    style D fill:#e1f7d5,stroke:#333,stroke-width:1px
    style E fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 3: Semantic Chunking
```mermaid
flowchart TD
    A[Document Text] --> B[Semantic Splitter]
    B --> C{Semantic Boundary?}
    C -->|Yes| D[Create Chunk]
    C -->|No| E[Continue Processing]
    E --> C
    D --> F[Chunk 1]
    D --> G[Chunk 2]
    D --> H[Chunk 3]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#ffe6cc,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#e1f7d5,stroke:#333,stroke-width:1px
    style G fill:#e1f7d5,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 4: Recursive Chunking
```mermaid
flowchart TD
    A[Document Text] --> B[Recursive Splitter]
    B --> C[Split by Sections]
    C --> D[Section 1]
    C --> E[Section 2]
    D --> F[Split by Paragraphs]
    E --> F
    F --> G[Paragraph 1]
    F --> H[Paragraph 2]
    F --> I[Paragraph 3]
    G --> J[Split by Sentences]
    H --> J
    I --> J
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#e1f7d5,stroke:#333,stroke-width:1px
    style E fill:#e1f7d5,stroke:#333,stroke-width:1px
    style F fill:#d4f1f9,stroke:#333,stroke-width:1px
    style G fill:#e1f7d5,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
    style I fill:#e1f7d5,stroke:#333,stroke-width:1px
    style J fill:#d4f1f9,stroke:#333,stroke-width:1px
```

### Stage 5: Token-Aware Chunking
```mermaid
flowchart TD
    A[Document Text] --> B[Token Counter]
    B --> C[Token-Aware Splitter]
    C --> D{Token Limit Reached?}
    D -->|Yes| E[Create Chunk]
    D -->|No| F[Add More Text]
    F --> D
    E --> G[Chunk 1]
    E --> H[Chunk 2]
    E --> I[Chunk 3]
    G --> J[Metadata]
    H --> J
    I --> J
    J --> K[Token Count]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#ffe6cc,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#d4f1f9,stroke:#333,stroke-width:1px
    style G fill:#e1f7d5,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
    style I fill:#e1f7d5,stroke:#333,stroke-width:1px
    style J fill:#ffe6cc,stroke:#333,stroke-width:1px
    style K fill:#ffe6cc,stroke:#333,stroke-width:1px
```

## 3. Embedding Generation

### Stage 1: Basic Embedding Pipeline
```mermaid
flowchart LR
    A[Text Chunk] --> B[Embedding Model]
    B --> C[Vector Embedding]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 2: Embedding Pipeline with Preprocessing
```mermaid
flowchart LR
    A[Text Chunk] --> B[Text Preprocessor]
    B --> C[Normalize]
    C --> D[Remove Stopwords]
    D --> E[Embedding Model]
    E --> F[Vector Embedding]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 3: Advanced Embedding Pipeline
```mermaid
flowchart TD
    A[Text Chunks] --> B[Batch Processor]
    B --> C[Embedding Cache]
    C --> D{Cache Hit?}
    D -->|Yes| E[Return Cached Embedding]
    D -->|No| F[Text Preprocessor]
    F --> G[Embedding Model]
    G --> H[Vector Embeddings]
    H --> I[Cache Result]
    I --> J[Return Embeddings]
    E --> J
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#ffe6cc,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#d4f1f9,stroke:#333,stroke-width:1px
    style G fill:#d4f1f9,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
    style I fill:#d4f1f9,stroke:#333,stroke-width:1px
    style J fill:#e1f7d5,stroke:#333,stroke-width:1px
```

## 4. RAG System Architecture

### Stage 1: Basic RAG System
```mermaid
flowchart LR
    A[User Question] --> B[Retriever]
    B --> C[Vector Database]
    C --> D[Relevant Documents]
    D --> E[Generator]
    E --> F[Answer]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#ffe6cc,stroke:#333,stroke-width:1px
    style D fill:#e1f7d5,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 2: RAG with Query Processing
```mermaid
flowchart LR
    A[User Question] --> B[Query Analyzer]
    B --> C[Query Expansion]
    C --> D[Retriever]
    D --> E[Vector Database]
    E --> F[Relevant Documents]
    F --> G[Generator]
    G --> H[Answer]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#ffe6cc,stroke:#333,stroke-width:1px
    style F fill:#e1f7d5,stroke:#333,stroke-width:1px
    style G fill:#d4f1f9,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 3: Hybrid RAG System
```mermaid
flowchart TD
    A[User Question] --> B[Query Analyzer]
    B --> C[Query Expansion]
    C --> D[Semantic Retriever]
    C --> E[Keyword Retriever]
    D --> F[Vector Database]
    E --> G[Inverted Index]
    F --> H[Semantic Results]
    G --> I[Keyword Results]
    H --> J[Result Merger]
    I --> J
    J --> K[Ranked Results]
    K --> L[Generator]
    L --> M[Answer]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#ffe6cc,stroke:#333,stroke-width:1px
    style G fill:#ffe6cc,stroke:#333,stroke-width:1px
    style H fill:#e1f7d5,stroke:#333,stroke-width:1px
    style I fill:#e1f7d5,stroke:#333,stroke-width:1px
    style J fill:#d4f1f9,stroke:#333,stroke-width:1px
    style K fill:#e1f7d5,stroke:#333,stroke-width:1px
    style L fill:#d4f1f9,stroke:#333,stroke-width:1px
    style M fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 4: RAG with Confidence Assessment
```mermaid
flowchart TD
    A[User Question] --> B[Query Analyzer]
    B --> C[Retriever]
    C --> D[Vector Database]
    D --> E[Relevant Documents]
    E --> F[Confidence Scorer]
    F --> G{Confidence Level}
    G -->|High| H[Direct Answer Generator]
    G -->|Medium| I[Hedged Answer Generator]
    G -->|Low| J[Uncertainty Generator]
    H --> K[Answer with Citations]
    I --> L[Answer with Caveats]
    J --> M[Low Confidence Response]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#ffe6cc,stroke:#333,stroke-width:1px
    style E fill:#e1f7d5,stroke:#333,stroke-width:1px
    style F fill:#d4f1f9,stroke:#333,stroke-width:1px
    style G fill:#ffe6cc,stroke:#333,stroke-width:1px
    style H fill:#d4f1f9,stroke:#333,stroke-width:1px
    style I fill:#d4f1f9,stroke:#333,stroke-width:1px
    style J fill:#d4f1f9,stroke:#333,stroke-width:1px
    style K fill:#e1f7d5,stroke:#333,stroke-width:1px
    style L fill:#e1f7d5,stroke:#333,stroke-width:1px
    style M fill:#e1f7d5,stroke:#333,stroke-width:1px
```

### Stage 5: Complete Document Q&A System
```mermaid
flowchart TD
    A[User Question] --> B[Question Analyzer]
    B --> C{Question Type}
    C -->|Metadata Query| D[Metadata Retriever]
    C -->|Content Query| E[Query Processor]
    D --> F[Metadata Store]
    F --> G[Metadata Results]
    E --> H[Query Expansion]
    H --> I[Hybrid Retriever]
    I --> J[Vector Database]
    J --> K[Relevant Documents]
    K --> L[Confidence Scorer]
    L --> M{Confidence Level}
    M -->|High| N[Direct Answer Generator]
    M -->|Medium| O[Synthesis Engine]
    M -->|Low| P[Uncertainty Handler]
    G --> Q[Metadata Answer Generator]
    N --> R[Answer with Citations]
    O --> S[Synthesized Answer]
    P --> T[Low Confidence Response]
    Q --> U[Metadata Response]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#ffe6cc,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#ffe6cc,stroke:#333,stroke-width:1px
    style G fill:#e1f7d5,stroke:#333,stroke-width:1px
    style H fill:#d4f1f9,stroke:#333,stroke-width:1px
    style I fill:#d4f1f9,stroke:#333,stroke-width:1px
    style J fill:#ffe6cc,stroke:#333,stroke-width:1px
    style K fill:#e1f7d5,stroke:#333,stroke-width:1px
    style L fill:#d4f1f9,stroke:#333,stroke-width:1px
    style M fill:#ffe6cc,stroke:#333,stroke-width:1px
    style N fill:#d4f1f9,stroke:#333,stroke-width:1px
    style O fill:#d4f1f9,stroke:#333,stroke-width:1px
    style P fill:#d4f1f9,stroke:#333,stroke-width:1px
    style Q fill:#d4f1f9,stroke:#333,stroke-width:1px
    style R fill:#e1f7d5,stroke:#333,stroke-width:1px
    style S fill:#e1f7d5,stroke:#333,stroke-width:1px
    style T fill:#e1f7d5,stroke:#333,stroke-width:1px
    style U fill:#e1f7d5,stroke:#333,stroke-width:1px
```

## 5. Streamlit App Architecture

```mermaid
flowchart TD
    A[User Interface] --> B[Document Upload]
    A --> C[Question Input]
    A --> D[Configuration]
    B --> E[Document Processor]
    E --> F[Document Chunks]
    F --> G[Embedding Generator]
    G --> H[Vector Database]
    C --> I[Question Processor]
    I --> J[RAG System]
    D --> J
    H --> J
    J --> K[Answer Generator]
    K --> L[Answer Display]
    L --> M[Source Attribution]
    L --> N[Confidence Score]
    style A fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B fill:#d4f1f9,stroke:#333,stroke-width:1px
    style C fill:#d4f1f9,stroke:#333,stroke-width:1px
    style D fill:#d4f1f9,stroke:#333,stroke-width:1px
    style E fill:#d4f1f9,stroke:#333,stroke-width:1px
    style F fill:#e1f7d5,stroke:#333,stroke-width:1px
    style G fill:#d4f1f9,stroke:#333,stroke-width:1px
    style H fill:#ffe6cc,stroke:#333,stroke-width:1px
    style I fill:#d4f1f9,stroke:#333,stroke-width:1px
    style J fill:#d4f1f9,stroke:#333,stroke-width:1px
    style K fill:#d4f1f9,stroke:#333,stroke-width:1px
    style L fill:#e1f7d5,stroke:#333,stroke-width:1px
    style M fill:#e1f7d5,stroke:#333,stroke-width:1px
    style N fill:#e1f7d5,stroke:#333,stroke-width:1px
```
