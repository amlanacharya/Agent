# 📊 Module 5: Advanced RAG Systems - Diagrams

This document contains diagrams illustrating the key concepts and architectures in Module 5.

## 🔍 Advanced Retrieval Strategies

### Hybrid Search Architecture

```mermaid
graph TD
    Query[User Query] --> QueryProcessor[Query Processor]
    QueryProcessor --> SemanticSearch[Semantic Search]
    QueryProcessor --> KeywordSearch[Keyword Search]

    SemanticSearch --> Embeddings[Embedding Model]
    Embeddings --> VectorDB[Vector Database]
    VectorDB --> SemanticResults[Semantic Results]

    KeywordSearch --> BM25[BM25 Algorithm]
    BM25 --> KeywordResults[Keyword Results]

    SemanticResults --> ResultFusion[Result Fusion]
    KeywordResults --> ResultFusion

    ResultFusion --> FinalResults[Final Results]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style FinalResults fill:#bbf,stroke:#333,stroke-width:2px
    style SemanticSearch fill:#dfd,stroke:#333,stroke-width:1px
    style KeywordSearch fill:#dfd,stroke:#333,stroke-width:1px
    style ResultFusion fill:#fdd,stroke:#333,stroke-width:1px
```

### Multi-Index Retrieval

```mermaid
graph TD
    Query[User Query] --> QueryRouter[Query Router]

    QueryRouter -->|Technical Query| TechnicalIndex[Technical Index]
    QueryRouter -->|General Query| GeneralIndex[General Index]
    QueryRouter -->|Medical Query| MedicalIndex[Medical Index]

    TechnicalIndex --> TechnicalEmbeddings[Technical Embeddings]
    GeneralIndex --> GeneralEmbeddings[General Embeddings]
    MedicalIndex --> MedicalEmbeddings[Medical Embeddings]

    TechnicalEmbeddings --> TechnicalResults[Technical Results]
    GeneralEmbeddings --> GeneralResults[General Results]
    MedicalEmbeddings --> MedicalResults[Medical Results]

    TechnicalResults --> ResultMerger[Result Merger]
    GeneralResults --> ResultMerger
    MedicalResults --> ResultMerger

    ResultMerger --> FinalResults[Final Results]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style FinalResults fill:#bbf,stroke:#333,stroke-width:2px
    style QueryRouter fill:#fdd,stroke:#333,stroke-width:1px
    style ResultMerger fill:#fdd,stroke:#333,stroke-width:1px
```

### Parent Document Retrieval

```mermaid
graph TD
    Document[Original Document] --> ParentSplitter[Parent Splitter]
    ParentSplitter --> ParentChunks[Parent Chunks]

    ParentChunks --> ChildSplitter[Child Splitter]
    ChildSplitter --> ChildChunks[Child Chunks]

    ChildChunks --> VectorStore[Vector Store]
    ParentChunks --> DocStore[Document Store]

    Query[User Query] --> VectorStore
    VectorStore --> ChildMatches[Child Matches]

    ChildMatches --> ParentLookup[Parent Lookup]
    DocStore --> ParentLookup

    ParentLookup --> ParentDocuments[Parent Documents]

    style Document fill:#f9f,stroke:#333,stroke-width:2px
    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style ParentDocuments fill:#bbf,stroke:#333,stroke-width:2px
    style ParentLookup fill:#fdd,stroke:#333,stroke-width:1px
```

### Contextual Compression

```mermaid
graph TD
    Query[User Query] --> BaseRetriever[Base Retriever]
    BaseRetriever --> RetrievedDocs[Retrieved Documents]

    RetrievedDocs --> Compressor[Document Compressor]
    Query --> Compressor

    Compressor --> LLM[LLM]
    LLM --> RelevanceFilter[Relevance Filter]

    RelevanceFilter --> CompressedDocs[Compressed Documents]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style CompressedDocs fill:#bbf,stroke:#333,stroke-width:2px
    style Compressor fill:#fdd,stroke:#333,stroke-width:1px
    style LLM fill:#dfd,stroke:#333,stroke-width:1px
```

## 🔄 Query Transformation Techniques

### Query Expansion Architecture

```mermaid
graph TD
    OriginalQuery[Original Query] --> QueryAnalyzer[Query Analyzer]

    QueryAnalyzer --> KeywordExtractor[Keyword Extractor]
    QueryAnalyzer --> EntityExtractor[Entity Extractor]
    QueryAnalyzer --> IntentClassifier[Intent Classifier]

    KeywordExtractor --> SynonymExpander[Synonym Expander]
    EntityExtractor --> RelatedEntities[Related Entities]
    IntentClassifier --> QueryReformulator[Query Reformulator]

    SynonymExpander --> ExpandedQuery[Expanded Query]
    RelatedEntities --> ExpandedQuery
    QueryReformulator --> ExpandedQuery

    ExpandedQuery --> Retriever[Retriever]

    style OriginalQuery fill:#f9f,stroke:#333,stroke-width:2px
    style ExpandedQuery fill:#bbf,stroke:#333,stroke-width:2px
    style QueryAnalyzer fill:#fdd,stroke:#333,stroke-width:1px
    style Retriever fill:#dfd,stroke:#333,stroke-width:1px
```

### Multi-Query Retrieval

```mermaid
graph TD
    OriginalQuery[Original Query] --> QueryGenerator[Query Generator]

    QueryGenerator --> LLM[LLM]
    LLM --> QueryVariations[Query Variations]

    QueryVariations --> Query1[Query 1]
    QueryVariations --> Query2[Query 2]
    QueryVariations --> Query3[Query 3]

    Query1 --> Retriever1[Retriever]
    Query2 --> Retriever2[Retriever]
    Query3 --> Retriever3[Retriever]

    Retriever1 --> Results1[Results 1]
    Retriever2 --> Results2[Results 2]
    Retriever3 --> Results3[Results 3]

    Results1 --> ResultMerger[Result Merger]
    Results2 --> ResultMerger
    Results3 --> ResultMerger

    ResultMerger --> FinalResults[Final Results]

    style OriginalQuery fill:#f9f,stroke:#333,stroke-width:2px
    style FinalResults fill:#bbf,stroke:#333,stroke-width:2px
    style QueryGenerator fill:#fdd,stroke:#333,stroke-width:1px
    style LLM fill:#dfd,stroke:#333,stroke-width:1px
    style ResultMerger fill:#fdd,stroke:#333,stroke-width:1px
```

### HyDE (Hypothetical Document Embeddings)

```mermaid
graph TD
    Query[User Query] --> LLM[LLM]
    LLM --> HypotheticalDoc[Hypothetical Document]

    HypotheticalDoc --> EmbeddingModel[Embedding Model]
    EmbeddingModel --> DocEmbedding[Document Embedding]

    DocEmbedding --> VectorStore[Vector Store]
    VectorStore --> SimilarDocs[Similar Documents]

    SimilarDocs --> FinalResults[Final Results]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style FinalResults fill:#bbf,stroke:#333,stroke-width:2px
    style LLM fill:#dfd,stroke:#333,stroke-width:1px
    style EmbeddingModel fill:#fdd,stroke:#333,stroke-width:1px
```

## 📊 Reranking and Result Optimization

### Cross-Encoder Reranking

```mermaid
graph TD
    Query[User Query] --> Retriever[Initial Retriever]
    Retriever --> InitialResults[Initial Results]

    Query --> CrossEncoder[Cross-Encoder]
    InitialResults --> CrossEncoder

    CrossEncoder --> ScoredPairs[Scored Query-Document Pairs]
    ScoredPairs --> Reranker[Reranker]

    Reranker --> RerankedResults[Reranked Results]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style RerankedResults fill:#bbf,stroke:#333,stroke-width:2px
    style CrossEncoder fill:#dfd,stroke:#333,stroke-width:1px
    style Reranker fill:#fdd,stroke:#333,stroke-width:1px
```

### Reciprocal Rank Fusion

```mermaid
graph TD
    Query[User Query] --> Retriever1[Retriever 1]
    Query --> Retriever2[Retriever 2]
    Query --> Retriever3[Retriever 3]

    Retriever1 --> Ranking1[Ranking 1]
    Retriever2 --> Ranking2[Ranking 2]
    Retriever3 --> Ranking3[Ranking 3]

    Ranking1 --> RRFCalculator[RRF Calculator]
    Ranking2 --> RRFCalculator
    Ranking3 --> RRFCalculator

    RRFCalculator --> FusedRanking[Fused Ranking]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style FusedRanking fill:#bbf,stroke:#333,stroke-width:2px
    style RRFCalculator fill:#fdd,stroke:#333,stroke-width:1px
```

### Maximal Marginal Relevance

```mermaid
graph TD
    Query[User Query] --> Retriever[Initial Retriever]
    Retriever --> InitialResults[Initial Results]

    InitialResults --> MMRSelector[MMR Selector]
    Query --> MMRSelector

    MMRSelector --> DiverseResults[Diverse Results]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style DiverseResults fill:#bbf,stroke:#333,stroke-width:2px
    style MMRSelector fill:#fdd,stroke:#333,stroke-width:1px
```

## 🧠 Self-Querying and Adaptive RAG

### Self-Querying Retrieval

```mermaid
graph TD
    Query[User Query] --> LLM[LLM]

    LLM --> QueryExtractor[Query Extractor]
    LLM --> FilterExtractor[Filter Extractor]

    QueryExtractor --> SemanticQuery[Semantic Query]
    FilterExtractor --> MetadataFilters[Metadata Filters]

    SemanticQuery --> VectorStore[Vector Store]
    MetadataFilters --> VectorStore

    VectorStore --> FilteredResults[Filtered Results]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style FilteredResults fill:#bbf,stroke:#333,stroke-width:2px
    style LLM fill:#dfd,stroke:#333,stroke-width:1px
    style VectorStore fill:#fdd,stroke:#333,stroke-width:1px
```

### Query Router

```mermaid
graph TD
    Query[User Query] --> QueryClassifier[Query Classifier]

    QueryClassifier --> FactualQuery[Factual Query]
    QueryClassifier --> ConceptualQuery[Conceptual Query]
    QueryClassifier --> ProceduralQuery[Procedural Query]

    FactualQuery --> FactRetriever[Fact Retriever]
    ConceptualQuery --> ConceptRetriever[Concept Retriever]
    ProceduralQuery --> ProcedureRetriever[Procedure Retriever]

    FactRetriever --> ResultMerger[Result Merger]
    ConceptRetriever --> ResultMerger
    ProcedureRetriever --> ResultMerger

    ResultMerger --> FinalResults[Final Results]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style FinalResults fill:#bbf,stroke:#333,stroke-width:2px
    style QueryClassifier fill:#dfd,stroke:#333,stroke-width:1px
    style ResultMerger fill:#fdd,stroke:#333,stroke-width:1px
```

### Multi-Hop Reasoning

```mermaid
graph TD
    Query[Initial Query] --> Retriever1[Retriever]
    Retriever1 --> Context1[Context 1]

    Context1 --> LLM1[LLM]
    LLM1 --> IntermediateAnswer[Intermediate Answer]

    IntermediateAnswer --> FollowupGenerator[Followup Generator]
    FollowupGenerator --> FollowupQuery[Followup Query]

    FollowupQuery --> Retriever2[Retriever]
    Retriever2 --> Context2[Context 2]

    Context2 --> LLM2[LLM]
    IntermediateAnswer --> LLM2

    LLM2 --> FinalAnswer[Final Answer]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style FinalAnswer fill:#bbf,stroke:#333,stroke-width:2px
    style LLM1 fill:#dfd,stroke:#333,stroke-width:1px
    style LLM2 fill:#dfd,stroke:#333,stroke-width:1px
    style FollowupGenerator fill:#fdd,stroke:#333,stroke-width:1px
```

## 📚 Research Literature Assistant

### Complete Architecture

```mermaid
graph TD
    Papers[Research Papers] --> DocumentProcessor[Document Processor]
    DocumentProcessor --> Chunker[Chunker]
    DocumentProcessor --> MetadataExtractor[Metadata Extractor]

    Chunker --> EmbeddingModel[Embedding Model]
    EmbeddingModel --> VectorStore[Vector Store]

    MetadataExtractor --> CitationTracker[Citation Tracker]
    MetadataExtractor --> MetadataStore[Metadata Store]

    Query[Research Query] --> QueryAnalyzer[Query Analyzer]
    QueryAnalyzer --> QueryReformulator[Query Reformulator]

    QueryReformulator --> AdvancedRetriever[Advanced Retriever]
    VectorStore --> AdvancedRetriever
    MetadataStore --> AdvancedRetriever

    AdvancedRetriever --> Reranker[Reranker]
    Reranker --> RelevantChunks[Relevant Chunks]

    RelevantChunks --> Synthesizer[Synthesizer]
    CitationTracker --> Synthesizer

    Synthesizer --> LLM[LLM]
    LLM --> Answer[Research Answer]

    style Papers fill:#f9f,stroke:#333,stroke-width:2px
    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style Answer fill:#bbf,stroke:#333,stroke-width:2px
    style AdvancedRetriever fill:#fdd,stroke:#333,stroke-width:1px
    style Synthesizer fill:#fdd,stroke:#333,stroke-width:1px
    style LLM fill:#dfd,stroke:#333,stroke-width:1px
```

### Step-Back Prompting

```mermaid
graph TD
    Query[Complex Query] --> StepBackLLM[Step-Back LLM]

    StepBackLLM --> HighLevelQuestion[High-Level Question]
    HighLevelQuestion --> HighLevelRetriever[High-Level Retriever]
    HighLevelRetriever --> ConceptualContext[Conceptual Context]

    Query --> DetailRetriever[Detail Retriever]
    DetailRetriever --> SpecificContext[Specific Context]

    ConceptualContext --> SynthesisLLM[Synthesis LLM]
    SpecificContext --> SynthesisLLM
    Query --> SynthesisLLM

    SynthesisLLM --> ComprehensiveAnswer[Comprehensive Answer]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style ComprehensiveAnswer fill:#bbf,stroke:#333,stroke-width:2px
    style StepBackLLM fill:#dfd,stroke:#333,stroke-width:1px
    style SynthesisLLM fill:#dfd,stroke:#333,stroke-width:1px
    style HighLevelRetriever fill:#fdd,stroke:#333,stroke-width:1px
```

### Controlled RAG (C-RAG)

```mermaid
graph TD
    Query[User Query] --> Retriever[Retriever]
    Retriever --> RetrievedDocs[Retrieved Documents]

    RetrievedDocs --> FactChecker[Fact Checker]
    Query --> FactChecker

    FactChecker --> VerifiedFacts[Verified Facts]
    FactChecker --> UncertainFacts[Uncertain Facts]

    VerifiedFacts --> ControlledLLM[Controlled LLM]
    UncertainFacts --> UncertaintyHandler[Uncertainty Handler]

    UncertaintyHandler --> ControlledLLM
    ControlledLLM --> ConstrainedResponse[Constrained Response]

    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style ConstrainedResponse fill:#bbf,stroke:#333,stroke-width:2px
    style FactChecker fill:#fdd,stroke:#333,stroke-width:1px
    style ControlledLLM fill:#dfd,stroke:#333,stroke-width:1px
    style UncertaintyHandler fill:#fdd,stroke:#333,stroke-width:1px
```

## 📊 RAG Evaluation Framework

### Comprehensive Evaluation Pipeline

```mermaid
graph TD
    RAGSystem[RAG System] --> EvalDataset[Evaluation Dataset]

    EvalDataset --> RetrievalMetrics[Retrieval Metrics]
    EvalDataset --> GenerationMetrics[Generation Metrics]
    EvalDataset --> EndToEndMetrics[End-to-End Metrics]

    RetrievalMetrics --> Precision[Precision]
    RetrievalMetrics --> Recall[Recall]
    RetrievalMetrics --> MRR[Mean Reciprocal Rank]

    GenerationMetrics --> Faithfulness[Faithfulness]
    GenerationMetrics --> Relevance[Relevance]
    GenerationMetrics --> Coherence[Coherence]

    EndToEndMetrics --> AnswerCorrectness[Answer Correctness]
    EndToEndMetrics --> ContextUtilization[Context Utilization]

    Precision --> MetricsDashboard[Metrics Dashboard]
    Recall --> MetricsDashboard
    MRR --> MetricsDashboard
    Faithfulness --> MetricsDashboard
    Relevance --> MetricsDashboard
    Coherence --> MetricsDashboard
    AnswerCorrectness --> MetricsDashboard
    ContextUtilization --> MetricsDashboard

    MetricsDashboard --> SystemOptimization[System Optimization]

    style RAGSystem fill:#f9f,stroke:#333,stroke-width:2px
    style SystemOptimization fill:#bbf,stroke:#333,stroke-width:2px
    style MetricsDashboard fill:#fdd,stroke:#333,stroke-width:1px
    style EvalDataset fill:#dfd,stroke:#333,stroke-width:1px
```

### RAGAS Evaluation Framework

```mermaid
graph TD
    RAGOutput[RAG System Output] --> FaithfulnessEval[Faithfulness Evaluator]
    RAGOutput --> AnswerRelevanceEval[Answer Relevance Evaluator]
    RAGOutput --> ContextRelevanceEval[Context Relevance Evaluator]
    RAGOutput --> ContextRecallEval[Context Recall Evaluator]

    Query[User Query] --> AnswerRelevanceEval
    Query --> ContextRelevanceEval
    Query --> ContextRecallEval

    GoldStandard[Gold Standard Answers] --> ContextRecallEval

    FaithfulnessEval --> FaithfulnessScore[Faithfulness Score]
    AnswerRelevanceEval --> RelevanceScore[Relevance Score]
    ContextRelevanceEval --> ContextScore[Context Score]
    ContextRecallEval --> RecallScore[Recall Score]

    FaithfulnessScore --> AggregateScore[Aggregate RAGAS Score]
    RelevanceScore --> AggregateScore
    ContextScore --> AggregateScore
    RecallScore --> AggregateScore

    style RAGOutput fill:#f9f,stroke:#333,stroke-width:2px
    style Query fill:#f9f,stroke:#333,stroke-width:2px
    style AggregateScore fill:#bbf,stroke:#333,stroke-width:2px
    style FaithfulnessEval fill:#fdd,stroke:#333,stroke-width:1px
    style AnswerRelevanceEval fill:#fdd,stroke:#333,stroke-width:1px
```

## 🔗 LCEL Implementation

### LCEL Chain Architecture

```mermaid
graph TD
    Input[Input] --> InputProcessor[Input Processor]

    InputProcessor --> QueryPath[Query Path]
    InputProcessor --> ContextPath[Context Path]

    QueryPath --> QueryTransformer[Query Transformer]
    QueryTransformer --> Retriever[Retriever]

    Retriever --> DocFormatter[Document Formatter]
    DocFormatter --> ContextPath

    QueryPath --> PromptTemplate[Prompt Template]
    ContextPath --> PromptTemplate

    PromptTemplate --> LLM[LLM]
    LLM --> OutputParser[Output Parser]

    OutputParser --> FinalOutput[Final Output]

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style FinalOutput fill:#bbf,stroke:#333,stroke-width:2px
    style InputProcessor fill:#fdd,stroke:#333,stroke-width:1px
    style PromptTemplate fill:#dfd,stroke:#333,stroke-width:1px
    style LLM fill:#dfd,stroke:#333,stroke-width:1px
```

### LCEL Branching Logic

```mermaid
graph TD
    Input[Input] --> Router[Router]

    Router -->|Factual Query| FactualChain[Factual Chain]
    Router -->|Conceptual Query| ConceptualChain[Conceptual Chain]
    Router -->|Procedural Query| ProceduralChain[Procedural Chain]

    FactualChain --> FactualRetriever[Factual Retriever]
    ConceptualChain --> ConceptualRetriever[Conceptual Retriever]
    ProceduralChain --> ProceduralRetriever[Procedural Retriever]

    FactualRetriever --> FactualLLM[Factual LLM]
    ConceptualRetriever --> ConceptualLLM[Conceptual LLM]
    ProceduralRetriever --> ProceduralLLM[Procedural LLM]

    FactualLLM --> OutputMerger[Output Merger]
    ConceptualLLM --> OutputMerger
    ProceduralLLM --> OutputMerger

    OutputMerger --> FinalOutput[Final Output]

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style FinalOutput fill:#bbf,stroke:#333,stroke-width:2px
    style Router fill:#fdd,stroke:#333,stroke-width:1px
    style OutputMerger fill:#fdd,stroke:#333,stroke-width:1px
```

## 🔌 From RAG to Tools: The Evolution

### The Information-Action Gap

```mermaid
graph TD
    UserQuery[User Query] --> QueryAnalyzer[Query Analyzer]

    QueryAnalyzer -->|Information Need| RAGSystem[RAG System]
    QueryAnalyzer -->|Action Need| ActionNeeded[Action Needed]

    RAGSystem --> InformationResponse[Information Response]
    ActionNeeded --> LimitedCapability[Limited Capability]

    InformationResponse --> UserSatisfied[User Satisfied]
    LimitedCapability --> UserFrustrated[User Frustrated]

    style UserQuery fill:#f9f,stroke:#333,stroke-width:2px
    style InformationResponse fill:#9f9,stroke:#333,stroke-width:2px
    style LimitedCapability fill:#f99,stroke:#333,stroke-width:2px
    style UserSatisfied fill:#9f9,stroke:#333,stroke-width:2px
    style UserFrustrated fill:#f99,stroke:#333,stroke-width:2px
```

### Tool Integration Architecture

```mermaid
graph TD
    UserRequest[User Request] --> RequestAnalyzer[Request Analyzer]

    RequestAnalyzer -->|Information Need| RAGSystem[RAG System]
    RequestAnalyzer -->|Action Need| ToolSelector[Tool Selector]

    RAGSystem --> RetrievedInfo[Retrieved Information]
    ToolSelector --> ToolRegistry[Tool Registry]

    ToolRegistry --> SelectedTool[Selected Tool]
    SelectedTool --> ParameterExtractor[Parameter Extractor]

    ParameterExtractor --> ToolExecution[Tool Execution]
    ToolExecution --> ToolResult[Tool Result]

    RetrievedInfo --> ResponseGenerator[Response Generator]
    ToolResult --> ResponseGenerator

    ResponseGenerator --> FinalResponse[Final Response]

    style UserRequest fill:#f9f,stroke:#333,stroke-width:2px
    style FinalResponse fill:#bbf,stroke:#333,stroke-width:2px
    style ToolSelector fill:#fdd,stroke:#333,stroke-width:1px
    style ToolExecution fill:#fdd,stroke:#333,stroke-width:1px
    style RAGSystem fill:#dfd,stroke:#333,stroke-width:1px
    style ResponseGenerator fill:#dfd,stroke:#333,stroke-width:1px
```

### Function Calling Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant ToolRegistry
    participant Tool

    User->>Agent: Request requiring tool use
    Agent->>LLM: Process request
    LLM->>ToolRegistry: Query available tools
    ToolRegistry->>LLM: Return tool schemas
    LLM->>Agent: Generate function call
    Agent->>Tool: Execute with parameters
    Tool->>Agent: Return result
    Agent->>LLM: Generate response with tool result
    LLM->>Agent: Final response
    Agent->>User: Deliver response
```

### Tool Types and Applications

```mermaid
graph TD
    Tools[Tool Categories] --> InfoTools[Information Access Tools]
    Tools --> CompTools[Computational Tools]
    Tools --> ContentTools[Content Creation Tools]
    Tools --> SystemTools[System Interaction Tools]
    Tools --> DomainTools[Domain-Specific Tools]

    InfoTools --> SearchTool[Search Engine]
    InfoTools --> APITool[API Client]
    InfoTools --> DBTool[Database Query]

    CompTools --> CalcTool[Calculator]
    CompTools --> SpreadsheetTool[Spreadsheet]
    CompTools --> StatsTool[Statistical Analysis]

    ContentTools --> ImageTool[Image Generator]
    ContentTools --> CodeTool[Code Generator]
    ContentTools --> DocTool[Document Creator]

    SystemTools --> FileTool[File System Operations]
    SystemTools --> AppTool[Application Controller]
    SystemTools --> DeviceTool[Device Controller]

    DomainTools --> FinanceTool[Financial Calculator]
    DomainTools --> LegalTool[Legal Document Analyzer]
    DomainTools --> MedicalTool[Medical Diagnostic Assistant]

    style Tools fill:#f9f,stroke:#333,stroke-width:2px
    style InfoTools fill:#dfd,stroke:#333,stroke-width:1px
    style CompTools fill:#dfd,stroke:#333,stroke-width:1px
    style ContentTools fill:#dfd,stroke:#333,stroke-width:1px
    style SystemTools fill:#dfd,stroke:#333,stroke-width:1px
    style DomainTools fill:#dfd,stroke:#333,stroke-width:1px
```

### RAG-Tool Hybrid System

```mermaid
graph TD
    UserQuery[User Query] --> QueryRouter[Query Router]

    QueryRouter -->|Information Need| RAGPipeline[RAG Pipeline]
    QueryRouter -->|Action Need| ToolPipeline[Tool Pipeline]
    QueryRouter -->|Hybrid Need| HybridPipeline[Hybrid Pipeline]

    RAGPipeline --> Retriever[Retriever]
    Retriever --> RetrievedDocs[Retrieved Documents]
    RetrievedDocs --> RAGProcessor[RAG Processor]

    ToolPipeline --> ToolSelector[Tool Selector]
    ToolSelector --> SelectedTool[Selected Tool]
    SelectedTool --> ToolExecutor[Tool Executor]
    ToolExecutor --> ToolResult[Tool Result]

    HybridPipeline --> Retriever
    HybridPipeline --> ToolSelector
    RetrievedDocs --> HybridProcessor[Hybrid Processor]
    ToolResult --> HybridProcessor

    RAGProcessor --> InfoResponse[Information Response]
    ToolExecutor --> ActionResponse[Action Response]
    HybridProcessor --> EnhancedResponse[Enhanced Response]

    InfoResponse --> ResponseAggregator[Response Aggregator]
    ActionResponse --> ResponseAggregator
    EnhancedResponse --> ResponseAggregator

    ResponseAggregator --> FinalResponse[Final Response]

    style UserQuery fill:#f9f,stroke:#333,stroke-width:2px
    style FinalResponse fill:#bbf,stroke:#333,stroke-width:2px
    style QueryRouter fill:#fdd,stroke:#333,stroke-width:1px
    style RAGProcessor fill:#dfd,stroke:#333,stroke-width:1px
    style ToolExecutor fill:#dfd,stroke:#333,stroke-width:1px
    style HybridProcessor fill:#dfd,stroke:#333,stroke-width:1px
```

### Progressive Agent Capabilities

```mermaid
graph LR
    Basic[Basic LLM] --> Memory[Memory Systems]
    Memory --> Validation[Structured Validation]
    Validation --> RAG[RAG Systems]
    RAG --> AdvancedRAG[Advanced RAG]
    AdvancedRAG --> Tools[Tool Integration]
    Tools --> Planning[Planning & Reasoning]
    Planning --> MultiAgent[Multi-Agent Systems]

    style Basic fill:#f9f,stroke:#333,stroke-width:1px
    style Memory fill:#f9f,stroke:#333,stroke-width:1px
    style Validation fill:#f9f,stroke:#333,stroke-width:1px
    style RAG fill:#f9f,stroke:#333,stroke-width:1px
    style AdvancedRAG fill:#f9f,stroke:#333,stroke-width:1px
    style Tools fill:#bbf,stroke:#333,stroke-width:2px
    style Planning fill:#dfd,stroke:#333,stroke-width:1px
    style MultiAgent fill:#dfd,stroke:#333,stroke-width:1px
```
