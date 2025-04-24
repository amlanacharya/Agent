# 📊 Module 2: Memory Systems - Explanatory Diagrams

Visual explanations of the key concepts in Module 2 through sequence diagrams, flowcharts, and class diagrams.

## 🧠 Memory System Architecture

The following class diagram illustrates the different memory types and their relationships:

```mermaid
classDiagram
    class WorkingMemory {
        +String current_context
        +add(item)
        +get()
        +clear()
    }
    
    class ShortTermMemory {
        +List items
        +int capacity
        +add(item)
        +get_recent(n)
        +get_all()
        +clear()
    }
    
    class LongTermMemory {
        +Dict knowledge_store
        +String storage_path
        +add(key, value)
        +get(key)
        +search(query)
        +save()
        +load()
    }
    
    class EpisodicMemory {
        +List episodes
        +add_episode(episode)
        +get_episodes(filter)
        +get_episode_by_id(id)
        +summarize_episodes(filter)
    }
    
    class AgentMemorySystem {
        +WorkingMemory working_memory
        +ShortTermMemory short_term
        +LongTermMemory long_term
        +EpisodicMemory episodic
        +add_interaction(interaction)
        +get_context(query)
        +store_knowledge(key, value)
        +retrieve_knowledge(query)
    }
    
    AgentMemorySystem --> WorkingMemory : contains
    AgentMemorySystem --> ShortTermMemory : contains
    AgentMemorySystem --> LongTermMemory : contains
    AgentMemorySystem --> EpisodicMemory : contains
```

## 🔍 Vector Database System

This diagram shows the components of the vector database system:

```mermaid
classDiagram
    class VectorStore {
        +Dict vectors
        +Dict metadata
        +String storage_path
        +add(id, vector, metadata)
        +get(id)
        +search(query_vector, top_k)
        +delete(id)
        +save()
        +load()
    }
    
    class DocumentIndexer {
        +VectorStore vector_store
        +Function text_to_vector
        +add_document(doc_id, content, metadata)
        +search_documents(query, top_k)
        +update_document(doc_id, content, metadata)
        +delete_document(doc_id)
    }
    
    class SimpleVectorizer {
        +vectorize_text(text)
        -preprocess_text(text)
        -compute_vector(tokens)
    }
    
    class SimilaritySearch {
        +cosine_similarity(vec1, vec2)
        +euclidean_distance(vec1, vec2)
        +dot_product(vec1, vec2)
        +find_nearest(query_vec, vectors, top_k)
    }
    
    DocumentIndexer --> VectorStore : uses
    DocumentIndexer --> SimpleVectorizer : uses
    VectorStore --> SimilaritySearch : uses
```

## 🔄 Retrieval Process Flow

The following sequence diagram illustrates the retrieval process:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Memory
    participant VectorDB
    participant Retriever
    
    User->>Agent: Ask question
    Agent->>Memory: Get conversation context
    Memory-->>Agent: Return recent interactions
    
    Agent->>Retriever: Query with context
    Retriever->>Retriever: Process query
    Retriever->>Retriever: Expand query terms
    
    Retriever->>VectorDB: Search for relevant documents
    VectorDB->>VectorDB: Convert query to vector
    VectorDB->>VectorDB: Compute similarity scores
    VectorDB-->>Retriever: Return top matches
    
    Retriever->>Retriever: Rank results
    Retriever->>Retriever: Filter irrelevant matches
    Retriever-->>Agent: Return relevant information
    
    Agent->>Agent: Generate response using retrieved info
    Agent->>Memory: Update with new interaction
    Agent->>User: Return response with citations
```

## 📚 Knowledge Base Architecture

This flowchart shows how the knowledge base system works:

```mermaid
flowchart TD
    subgraph Input["Input Processing"]
        A1[User Question] --> A2[Extract Query Intent]
        A2 --> A3[Identify Key Entities]
        A3 --> A4[Determine Context]
    end
    
    subgraph KnowledgeBase["Knowledge Base"]
        B1[Vector Database]
        B2[Metadata Store]
        B3[Citation Manager]
        B4[Uncertainty Handler]
    end
    
    subgraph Retrieval["Retrieval System"]
        C1[Query Processing]
        C2[Context Integration]
        C3[Similarity Search]
        C4[Result Ranking]
    end
    
    subgraph Response["Response Generation"]
        D1[Answer Formulation]
        D2[Citation Addition]
        D3[Confidence Assessment]
        D4[Explanation Generation]
    end
    
    A4 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> B1
    B1 --> C4
    B2 --> C4
    C4 --> D1
    D1 --> D2
    B3 --> D2
    D2 --> D3
    B4 --> D3
    D3 --> D4
    D4 --> E[Final Response]
```

## 🧠 Knowledge Base Assistant Interaction

This sequence diagram shows how the knowledge base assistant handles a user query:

```mermaid
sequenceDiagram
    participant User
    participant KBAgent as Knowledge Base Assistant
    participant KB as Knowledge Base
    participant Citations as Citation Manager
    participant Uncertainty as Uncertainty Handler
    
    User->>KBAgent: Ask question
    KBAgent->>KBAgent: Process question
    KBAgent->>KB: Search for relevant knowledge
    KB-->>KBAgent: Return knowledge results
    
    alt Knowledge found with high confidence
        KBAgent->>Citations: Generate citations
        Citations-->>KBAgent: Return formatted citations
        KBAgent->>KBAgent: Generate answer with citations
    else Knowledge found with low confidence
        KBAgent->>Uncertainty: Handle uncertain response
        Uncertainty-->>KBAgent: Return uncertainty indication
        KBAgent->>KBAgent: Generate tentative answer
    else No relevant knowledge
        KBAgent->>KBAgent: Generate "I don't know" response
        KBAgent->>KBAgent: Offer to learn new information
    end
    
    KBAgent->>User: Return response
    
    alt User provides new information
        User->>KBAgent: Provide correction/new information
        KBAgent->>KB: Learn from user input
        KB-->>KBAgent: Confirm knowledge added
        KBAgent->>User: Acknowledge learning
    end
```

