# 📊 Module 2-LLM: Memory Systems with Groq API - Explanatory Diagrams

This document provides visual explanations of the key concepts in Module 2-LLM through sequence diagrams, flowcharts, and class diagrams, highlighting the integration with real LLMs via the Groq API.

## 🧠 LLM-Enhanced Memory System Architecture

The following class diagram illustrates the different memory types with LLM integration:

```mermaid
classDiagram
    class GroqClient {
        +String api_key
        +String model
        +generate_text(prompt, max_tokens)
        +generate_embedding(text)
        +extract_text_from_response(response)
    }
    
    class WorkingMemory {
        +String current_context
        +GroqClient groq_client
        +add(item)
        +get()
        +clear()
        +summarize_context()
    }
    
    class ShortTermMemory {
        +List items
        +int capacity
        +GroqClient groq_client
        +add(item)
        +get_recent(n)
        +get_all()
        +clear()
        +extract_key_information(query)
        +summarize_recent(n)
    }
    
    class LongTermMemory {
        +Dict knowledge_store
        +String storage_path
        +GroqClient groq_client
        +add(key, value, metadata)
        +get(key)
        +search(query)
        +semantic_search(query, top_k)
        +save()
        +load()
        +extract_facts(text)
    }
    
    class EpisodicMemory {
        +List episodes
        +GroqClient groq_client
        +add_episode(episode)
        +get_episodes(filter)
        +get_episode_by_id(id)
        +summarize_episodes(filter)
        +extract_patterns(episodes)
    }
    
    class AgentMemorySystem {
        +WorkingMemory working_memory
        +ShortTermMemory short_term
        +LongTermMemory long_term
        +EpisodicMemory episodic
        +GroqClient groq_client
        +add_interaction(interaction)
        +get_context(query)
        +store_knowledge(key, value)
        +retrieve_knowledge(query)
        +integrate_context(query)
        +extract_learning_opportunities(interaction)
    }
    
    AgentMemorySystem --> WorkingMemory : contains
    AgentMemorySystem --> ShortTermMemory : contains
    AgentMemorySystem --> LongTermMemory : contains
    AgentMemorySystem --> EpisodicMemory : contains
    AgentMemorySystem --> GroqClient : uses
    WorkingMemory --> GroqClient : uses
    ShortTermMemory --> GroqClient : uses
    LongTermMemory --> GroqClient : uses
    EpisodicMemory --> GroqClient : uses
```

## 🔍 LLM-Enhanced Vector Database System

This diagram shows the components of the vector database system with real embeddings:

```mermaid
classDiagram
    class GroqClient {
        +String api_key
        +String model
        +generate_text(prompt, max_tokens)
        +generate_embedding(text)
    }
    
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
        +GroqClient groq_client
        +add_document(doc_id, content, metadata)
        +search_documents(query, top_k)
        +update_document(doc_id, content, metadata)
        +delete_document(doc_id)
        +extract_document_entities(content)
    }
    
    class QueryExpansion {
        +GroqClient groq_client
        +expand_query(query)
        +generate_related_terms(query)
        +extract_entities(query)
    }
    
    class SimilaritySearch {
        +cosine_similarity(vec1, vec2)
        +euclidean_distance(vec1, vec2)
        +dot_product(vec1, vec2)
        +find_nearest(query_vec, vectors, top_k)
    }
    
    DocumentIndexer --> VectorStore : uses
    DocumentIndexer --> GroqClient : uses for embeddings
    QueryExpansion --> GroqClient : uses for expansion
    VectorStore --> SimilaritySearch : uses
```

## 🔄 LLM-Enhanced Retrieval Process Flow

The following sequence diagram illustrates the retrieval process with LLM enhancement:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Memory
    participant LLM as Groq LLM
    participant VectorDB
    participant Retriever
    
    User->>Agent: Ask question
    Agent->>Memory: Get conversation context
    Memory-->>Agent: Return recent interactions
    
    Agent->>LLM: Generate context summary
    LLM-->>Agent: Return summarized context
    
    Agent->>LLM: Enhance query with context
    LLM-->>Agent: Return enhanced query
    
    Agent->>Retriever: Query with enhanced query
    Retriever->>LLM: Expand query terms
    LLM-->>Retriever: Return expanded query
    
    Retriever->>VectorDB: Search for relevant documents
    VectorDB->>LLM: Generate embeddings for query
    LLM-->>VectorDB: Return query embeddings
    VectorDB->>VectorDB: Compute similarity scores
    VectorDB-->>Retriever: Return top matches
    
    Retriever->>LLM: Rank and filter results
    LLM-->>Retriever: Return ranked information
    Retriever-->>Agent: Return relevant information
    
    Agent->>LLM: Generate response using retrieved info
    LLM-->>Agent: Return generated response
    Agent->>Memory: Update with new interaction
    Agent->>User: Return response with citations
```

## 📚 LLM-Enhanced Knowledge Base Architecture

This flowchart shows how the knowledge base system works with LLM integration:

```mermaid
flowchart TD
    subgraph Input["Input Processing"]
        A1[User Question] --> A2[Extract Query Intent]
        A2 --> A3[Identify Key Entities]
        A3 --> A4[Determine Context]
    end
    
    subgraph LLM["Groq LLM"]
        L1[Query Enhancement]
        L2[Context Integration]
        L3[Response Generation]
        L4[Fact Extraction]
        L5[Uncertainty Assessment]
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
    
    A4 --> L1
    L1 --> C1
    C1 --> L2
    L2 --> C2
    C2 --> C3
    C3 --> B1
    B1 --> C4
    B2 --> C4
    C4 --> L3
    L3 --> D1
    D1 --> D2
    B3 --> D2
    D2 --> L5
    L5 --> D3
    B4 --> D3
    D3 --> D4
    D4 --> E[Final Response]
    
    %% Learning flow
    E --> L4
    L4 --> B1
```

## 🧠 LLM-Enhanced Knowledge Base Assistant Interaction

This sequence diagram shows how the knowledge base assistant handles a user query with LLM integration:

```mermaid
sequenceDiagram
    participant User
    participant KBAgent as Knowledge Base Assistant
    participant LLM as Groq LLM
    participant KB as Knowledge Base
    participant Citations as Citation Manager
    participant Uncertainty as Uncertainty Handler
    
    User->>KBAgent: Ask question
    KBAgent->>LLM: Process and enhance question
    LLM-->>KBAgent: Return enhanced question
    
    KBAgent->>KB: Search for relevant knowledge
    KB->>LLM: Generate embeddings for search
    LLM-->>KB: Return embeddings
    KB-->>KBAgent: Return knowledge results
    
    KBAgent->>LLM: Assess confidence in results
    LLM-->>KBAgent: Return confidence assessment
    
    alt Knowledge found with high confidence
        KBAgent->>Citations: Generate citations
        Citations-->>KBAgent: Return formatted citations
        KBAgent->>LLM: Generate answer with citations
        LLM-->>KBAgent: Return formatted answer
    else Knowledge found with low confidence
        KBAgent->>Uncertainty: Handle uncertain response
        Uncertainty->>LLM: Generate uncertainty language
        LLM-->>Uncertainty: Return uncertainty phrasing
        Uncertainty-->>KBAgent: Return uncertainty indication
        KBAgent->>LLM: Generate tentative answer
        LLM-->>KBAgent: Return tentative answer
    else No relevant knowledge
        KBAgent->>LLM: Generate "I don't know" response
        LLM-->>KBAgent: Return "I don't know" response
        KBAgent->>LLM: Generate learning offer
        LLM-->>KBAgent: Return learning offer
    end
    
    KBAgent->>User: Return response
    
    alt User provides new information
        User->>KBAgent: Provide correction/new information
        KBAgent->>LLM: Extract facts from user input
        LLM-->>KBAgent: Return extracted facts
        KBAgent->>KB: Learn from extracted facts
        KB-->>KBAgent: Confirm knowledge added
        KBAgent->>LLM: Generate acknowledgment
        LLM-->>KBAgent: Return acknowledgment
        KBAgent->>User: Acknowledge learning
    end
```

## 🔄 Comparing Simulated vs. LLM-Enhanced Approaches

This diagram illustrates the key differences between the simulated approach and the LLM-enhanced approach:

```mermaid
flowchart TD
    subgraph Simulated["Simulated Approach (Module 2)"]
        S1[Rule-Based Processing]
        S2[Template Responses]
        S3[Simple Vector Representations]
        S4[Basic Similarity Search]
        S5[Predefined Uncertainty Rules]
    end
    
    subgraph LLMEnhanced["LLM-Enhanced Approach (Module 2-LLM)"]
        L1[Natural Language Understanding]
        L2[Dynamic Response Generation]
        L3[Semantic Embeddings]
        L4[Context-Aware Retrieval]
        L5[Nuanced Uncertainty Handling]
    end
    
    S1 --> |"Enhanced by"| L1
    S2 --> |"Enhanced by"| L2
    S3 --> |"Enhanced by"| L3
    S4 --> |"Enhanced by"| L4
    S5 --> |"Enhanced by"| L5
    
    L1 --> |"Enables"| B1[Intent Recognition]
    L2 --> |"Enables"| B2[Personalized Responses]
    L3 --> |"Enables"| B3[Semantic Understanding]
    L4 --> |"Enables"| B4[Relevant Information Retrieval]
    L5 --> |"Enables"| B5[Responsible AI Behavior]
```

These diagrams provide visual explanations of the key concepts and architectures in Module 2-LLM, highlighting how real LLM integration enhances the memory systems and knowledge base assistant compared to the simulated approach in the standard Module 2.
