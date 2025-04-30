# Multi-Tool Agent

This document illustrates the architecture and functionality of the multi-tool agent in Module 6.

## Multi-Tool Agent Architecture

```mermaid
classDiagram
    class MultiToolAgent {
        -tool_registry: ToolRegistry
        -llm_client: LLMClient
        -memory: ConversationMemory
        -config: AgentConfig
        +process_query(query: str): str
        +select_tools(query: str): List[Dict]
        +execute_tools(tool_selections: List[Dict]): List[ToolResponse]
        +generate_response(query: str, tool_results: List[ToolResponse]): str
        +explain_reasoning(): str
        +handle_errors(error: Exception): str
    }

    class ToolRegistry {
        -tools: Dict[str, BaseTool]
        -categories: Dict[str, List[str]]
        +register_tool(tool: BaseTool)
        +get_tool(tool_name: str): BaseTool
        +list_tools(): List[str]
        +get_all_schemas(): List[Dict]
    }

    class LLMClient {
        -provider: str
        -model: str
        -api_key: str
        +generate_text(prompt: str): str
        +generate_chat_response(messages: List[Dict]): str
        +function_calling(messages: List[Dict], functions: List[Dict]): Dict
    }

    class ConversationMemory {
        -messages: List[Dict]
        -max_messages: int
        +add_message(role: str, content: str)
        +get_messages(): List[Dict]
        +clear()
        +get_context(): str
    }

    class AgentConfig {
        +max_tool_calls: int
        +explain_reasoning: bool
        +verbose: bool
        +timeout: int
        +retry_attempts: int
    }

    class BaseTool {
        +name: str
        +description: str
        +metadata: Dict
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
    }

    MultiToolAgent o-- ToolRegistry : uses
    MultiToolAgent o-- LLMClient : uses
    MultiToolAgent o-- ConversationMemory : uses
    MultiToolAgent o-- AgentConfig : configured by
    ToolRegistry o-- BaseTool : contains
```

## Multi-Tool Agent Query Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as MultiToolAgent
    participant LLM
    participant Registry as ToolRegistry
    participant Tools

    User->>Agent: Send query

    Agent->>Registry: Get available tools
    Registry-->>Agent: Return tool schemas

    Agent->>LLM: Send query + tool schemas
    LLM-->>Agent: Tool selection(s)

    loop For each selected tool
        Agent->>Registry: Get tool
        Registry-->>Agent: Return tool
        Agent->>Tools: Execute tool with parameters
        Tools-->>Agent: Return result
    end

    Agent->>LLM: Send query + tool results
    LLM-->>Agent: Generate final response

    Agent->>User: Return response
```

## Tool Selection Logic

```mermaid
flowchart TD
    A[User Query] --> B[MultiToolAgent]
    B --> C[LLM for Tool Selection]

    C --> D{Query Analysis}

    D -->|Weather Question| E[Weather Tool]
    D -->|Search Question| F[Search Tool]
    D -->|Financial Question| G[Finance Tool]
    D -->|General Question| H[LLM Tool]
    D -->|Complex Question| I[Multiple Tools]

    I -->|Scenario 1| E
    I -->|Scenario 1| F
    I -->|Scenario 2| F
    I -->|Scenario 2| G
    I -->|Scenario 3| E
    I -->|Scenario 3| G
    I -->|Scenario 4| E
    I -->|Scenario 4| F
    I -->|Scenario 4| G

    E --> J[Tool Execution]
    F --> J
    G --> J
    H --> J
    J --> K[Result Processing]
    K --> L[Response Generation]
    L --> M[User Response]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#dfd,stroke:#333,stroke-width:1px
    style D fill:#fdd,stroke:#333,stroke-width:2px
    style E,F,G,H fill:#dfd,stroke:#333,stroke-width:1px
    style I fill:#ffd,stroke:#333,stroke-width:1px
    style J,K,L fill:#ffd,stroke:#333,stroke-width:1px
    style M fill:#f9f,stroke:#333,stroke-width:2px
```

## Error Handling Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as MultiToolAgent
    participant LLM
    participant PrimaryTool
    participant FallbackTool

    User->>Agent: "What's the weather in Atlantis?"

    Agent->>LLM: Process query for tool selection
    LLM-->>Agent: Select weather tool

    Agent->>PrimaryTool: Get weather for "Atlantis"
    PrimaryTool-->>Agent: Error: Location not found

    Agent->>Agent: Error handling logic

    Agent->>LLM: Report error and request alternative approach
    LLM-->>Agent: Suggest using search tool instead

    Agent->>FallbackTool: Search for "Atlantis weather information"
    FallbackTool-->>Agent: Search results about fictional Atlantis

    Agent->>LLM: Generate response with search results
    LLM-->>Agent: Final response explaining the situation

    Agent->>User: "I couldn't find weather for Atlantis as it's a fictional location..."
```

## Multi-Step Reasoning

```mermaid
sequenceDiagram
    participant User
    participant Agent as MultiToolAgent
    participant LLM
    participant Tool1 as SearchTool
    participant Tool2 as WeatherTool
    participant Tool3 as FinanceTool

    User->>Agent: "How might today's weather in California affect wine stocks?"

    Agent->>LLM: Process query
    LLM-->>Agent: Need to break this into steps

    Agent->>LLM: Step 1: What information do we need?
    LLM-->>Agent: Need California weather and wine stock information

    Agent->>Tool2: Get weather for "California"
    Tool2-->>Agent: California weather data (hot and dry)

    Agent->>Tool1: Search "major wine companies stocks"
    Tool1-->>Agent: List of wine company stocks

    Agent->>Tool3: Get stock info for "WINE, BF.B, STZ"
    Tool3-->>Agent: Current wine stock prices and trends

    Agent->>Tool1: Search "how weather affects wine production"
    Tool1-->>Agent: Information about weather impact on vineyards

    Agent->>LLM: Analyze all collected information
    LLM-->>Agent: Generated analysis connecting weather to potential stock impact

    Agent->>User: "Today's hot, dry weather in California could stress vineyards, potentially affecting wine production and stocks like..."
```

## Tool Chain Execution

```mermaid
flowchart TD
    A[User Query] --> B[Query Analysis]
    B --> C{Tool Selection}

    C --> D[Tool Chain Executor]

    D --> E[Step 1: Search Tool]
    E --> F[Step 2: Weather Tool]
    F --> G[Step 3: Finance Tool]
    G --> H[Step 4: LLM Analysis]

    H --> I[Response Generation]
    I --> J[User Response]

    D --> K[Execution Monitoring]
    K --> L{Error Detected?}

    L -->|Yes| M[Error Recovery]
    M --> N[Alternative Tool]
    N --> D

    L -->|No| H

    classDef userNode fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decisionNode fill:#fdd,stroke:#333,stroke-width:2px;
    classDef coreNode fill:#bbf,stroke:#333,stroke-width:2px;
    classDef toolNode fill:#dfd,stroke:#333,stroke-width:1px;
    classDef processNode fill:#ffd,stroke:#333,stroke-width:1px;

    class A,J userNode;
    class B,C,K,L decisionNode;
    class D coreNode;
    class E,F,G,H toolNode;
    class I,M,N processNode;
```
