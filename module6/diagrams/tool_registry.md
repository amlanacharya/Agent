# Tool Registry System

This document illustrates the design and functionality of the tool registry system in Module 6.

## Tool Registry Architecture

```mermaid
classDiagram
    class ToolRegistry {
        -tools: Dict[str, BaseTool]
        -categories: Dict[str, List[str]]
        +register_tool(tool: BaseTool)
        +unregister_tool(tool_name: str)
        +get_tool(tool_name: str): BaseTool
        +list_tools(): List[str]
        +get_tools_by_category(category: str): List[BaseTool]
        +add_tool_to_category(tool_name: str, category: str)
        +get_all_schemas(): List[Dict]
        +to_langchain_tools(): List[LangChainTool]
    }

    class BaseTool {
        +name: str
        +description: str
        +metadata: Dict
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
    }

    class ToolManager {
        -registry: ToolRegistry
        -llm_client: LLMClient
        +select_tool(query: str): Dict
        +execute_tool(tool_name: str, **kwargs): ToolResponse
        +execute_with_llm(query: str): str
        +get_available_tools(): List[Dict]
        +add_tool(tool: BaseTool)
        +remove_tool(tool_name: str)
    }

    ToolRegistry o-- BaseTool : contains
    ToolManager o-- ToolRegistry : uses
```

## Tool Registration Flow

```mermaid
sequenceDiagram
    participant Developer
    participant ToolRegistry
    participant Tool

    Developer->>Tool: Create new tool
    Developer->>ToolRegistry: register_tool(tool)
    ToolRegistry->>Tool: Get name and schema
    Tool-->>ToolRegistry: Return name and schema
    ToolRegistry->>ToolRegistry: Store tool reference
    ToolRegistry-->>Developer: Confirmation

    Note over Developer,ToolRegistry: Tool is now available in the registry
```

## Tool Discovery and Selection

```mermaid
flowchart TD
    A[User Query] --> B[Tool Manager]
    B --> C{Select Tool}

    C -->|Weather Query| D[Weather Tool]
    C -->|Search Query| E[Search Tool]
    C -->|Finance Query| F[Finance Tool]
    C -->|General Query| G[LLM Tool]

    B --> H[Tool Registry]
    H --> I[Available Tools]
    I --> C

    D --> J[Execute Tool]
    E --> J
    F --> J
    G --> J
    J --> K[Process Result]
    K --> L[Generate Response]
    L --> M[User Response]

    classDef userNode fill:#f9f,stroke:#333,stroke-width:2px;
    classDef managerNode fill:#bbf,stroke:#333,stroke-width:2px;
    classDef decisionNode fill:#fdd,stroke:#333,stroke-width:2px;
    classDef toolNode fill:#dfd,stroke:#333,stroke-width:1px;
    classDef processNode fill:#ffd,stroke:#333,stroke-width:1px;

    class A,M userNode;
    class B,H managerNode;
    class C decisionNode;
    class D,E,F,G,I toolNode;
    class J,K,L processNode;
```

## Dynamic Tool Registration

```mermaid
sequenceDiagram
    participant Agent
    participant ToolRegistry
    participant PluginManager
    participant ExternalSource

    Agent->>PluginManager: Discover new tools
    PluginManager->>ExternalSource: Request available tools
    ExternalSource-->>PluginManager: Tool definitions

    loop For each tool definition
        PluginManager->>PluginManager: Validate tool definition
        PluginManager->>PluginManager: Create tool instance
        PluginManager->>ToolRegistry: Register new tool
        ToolRegistry-->>PluginManager: Confirmation
    end

    PluginManager-->>Agent: Report newly registered tools

    Note over Agent,ToolRegistry: New tools are now available to the agent
```

## Tool Categories and Filtering

```mermaid
flowchart TD
    A[Tool Registry] --> B[Categories]

    B --> C[Information Retrieval]
    B --> D[Content Generation]
    B --> E[Data Analysis]
    B --> F[External APIs]

    C --> C1[Search Tool]
    C --> C2[Knowledge Base Tool]

    D --> D1[OpenAI Tool]
    D --> D2[Groq Tool]

    E --> E1[Calculator Tool]
    E --> E2[Chart Generator Tool]

    F --> F1[Weather Tool]
    F --> F2[Finance Tool]

    G[Agent] --> A
    G --> H{Query Type}

    H -->|Information Needed| C
    H -->|Content Creation| D
    H -->|Calculation Needed| E
    H -->|External Data Needed| F

    classDef registryNode fill:#bbf,stroke:#333,stroke-width:2px;
    classDef categoryNode fill:#fdd,stroke:#333,stroke-width:1px;
    classDef toolTypeNode fill:#dfd,stroke:#333,stroke-width:1px;
    classDef toolNode fill:#ffd,stroke:#333,stroke-width:1px;
    classDef queryNode fill:#f9f,stroke:#333,stroke-width:2px;

    class A,G registryNode;
    class B categoryNode;
    class C,D,E,F toolTypeNode;
    class C1,C2,D1,D2,E1,E2,F1,F2 toolNode;
    class H queryNode;
```
