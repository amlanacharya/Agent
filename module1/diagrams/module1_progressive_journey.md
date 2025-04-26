# The Progressive Journey Through Module 1: Agent Fundamentals

This document presents a stage-by-stage breakdown of the agent architecture in Module 1, gradually building up to the complete picture.

## 1. Agent Architecture: Stage-by-Stage Breakdown

### Stage 1: The Basic Sense-Think-Act Loop

Let's start with the core loop that forms the foundation of all agents:

```mermaid
flowchart TD
    Input[User Input] --> Sense
    Sense --> Think
    Think --> Act
    Act --> Response[Agent Response]
    
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef io fill:#dfd,stroke:#333,stroke-width:1px;
    
    class Sense,Think,Act process;
    class Input,Response io;
```

### Stage 2: The SimpleAgent Implementation

Now let's look at how the SimpleAgent implements this loop:

```mermaid
flowchart TD
    Input[User Input] --> Sense
    
    subgraph SimpleAgent["SimpleAgent"]
        Sense["sense(user_input)\nProcess raw input"] --> ProcessedInput[Processed Input]
        ProcessedInput --> Think["think(processed_input)\nDecide how to respond"]
        Think --> ActionPlan[Action Plan]
        ActionPlan --> Act["act(action_plan)\nExecute the response"]
        Act --> UpdateState["Update internal state"]
    end
    
    Act --> Response[Agent Response]
    
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef data fill:#ffd,stroke:#333,stroke-width:1px;
    classDef io fill:#dfd,stroke:#333,stroke-width:1px;
    
    class Sense,Think,Act,UpdateState process;
    class ProcessedInput,ActionPlan data;
    class Input,Response io;
```

### Stage 3: Adding Prompt Templates

Let's add prompt templates to create the PromptDrivenAgent:

```mermaid
flowchart TD
    Input[User Input] --> Sense
    
    subgraph PromptDrivenAgent["PromptDrivenAgent"]
        Sense["sense(user_input)\nProcess with templates"] --> ProcessedInput[Processed Input]
        ProcessedInput --> Think["think(processed_input)\nUse prompt templates"]
        Think --> ActionPlan[Action Plan]
        ActionPlan --> Act["act(action_plan)\nExecute the response"]
        Act --> UpdateState["Update conversation history"]
        
        PromptLibrary["Prompt Library"] --> |Select template| Think
    end
    
    Act --> Response[Agent Response]
    
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef data fill:#ffd,stroke:#333,stroke-width:1px;
    classDef io fill:#dfd,stroke:#333,stroke-width:1px;
    classDef component fill:#f9f,stroke:#333,stroke-width:1px;
    
    class Sense,Think,Act,UpdateState process;
    class ProcessedInput,ActionPlan data;
    class Input,Response io;
    class PromptLibrary component;
```

### Stage 4: Adding State Management

Now let's add state management to create the StatefulAgent:

```mermaid
flowchart TD
    Input[User Input] --> Sense
    
    subgraph StatefulAgent["StatefulAgent"]
        Sense["sense(user_input)\nProcess with context"] --> ProcessedInput[Processed Input]
        StateManager["State Manager"] --> |Provide context| Think
        ProcessedInput --> Think["think(processed_input, context)\nUse context for decisions"]
        Think --> ActionPlan[Action Plan]
        ActionPlan --> Act["act(action_plan)\nExecute the response"]
        Act --> UpdateState["update_state(interaction)"]
        UpdateState --> StateManager
        
        PromptLibrary["Prompt Library"] --> |Select template| Think
    end
    
    Act --> Response[Agent Response]
    
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef data fill:#ffd,stroke:#333,stroke-width:1px;
    classDef io fill:#dfd,stroke:#333,stroke-width:1px;
    classDef component fill:#f9f,stroke:#333,stroke-width:1px;
    
    class Sense,Think,Act,UpdateState process;
    class ProcessedInput,ActionPlan data;
    class Input,Response io;
    class PromptLibrary,StateManager component;
```

### Stage 5: Complete Agent Architecture

Finally, let's add task management to create the TaskManagerAgent:

```mermaid
flowchart TD
    Input[User Input] --> Sense
    
    subgraph TaskManagerAgent["TaskManagerAgent"]
        Sense["sense(user_input)\nProcess with intent detection"] --> ProcessedInput[Processed Input with Intent]
        StateManager["State Manager"] --> |Provide context| Think
        ProcessedInput --> Think["think(processed_input, context)\nUse context for decisions"]
        Think --> ActionPlan[Action Plan]
        ActionPlan --> Act["act(action_plan)\nExecute the response"]
        Act --> UpdateState["update_state(interaction)"]
        UpdateState --> StateManager
        
        PromptLibrary["Prompt Library"] --> |Select template| Think
        TaskManager["Task Manager"] --> |Task operations| Think
        Think --> |Task updates| TaskManager
    end
    
    Act --> Response[Agent Response]
    
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef data fill:#ffd,stroke:#333,stroke-width:1px;
    classDef io fill:#dfd,stroke:#333,stroke-width:1px;
    classDef component fill:#f9f,stroke:#333,stroke-width:1px;
    
    class Sense,Think,Act,UpdateState process;
    class ProcessedInput,ActionPlan data;
    class Input,Response io;
    class PromptLibrary,StateManager,TaskManager component;
```

## 2. Prompt Engineering Flow: Stage-by-Stage Breakdown

### Stage 1: Basic Prompt Template

Let's start with a basic prompt template:

```mermaid
flowchart TD
    Template["Prompt Template"] --> |Fill variables| FilledTemplate["Filled Template"]
    FilledTemplate --> Response["Generated Response"]
    
    classDef template fill:#f9f,stroke:#333,stroke-width:1px;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef output fill:#dfd,stroke:#333,stroke-width:1px;
    
    class Template template;
    class FilledTemplate process;
    class Response output;
```

### Stage 2: Adding Context and Variables

Now let's add context and variables:

```mermaid
flowchart TD
    Template["Prompt Template"] --> |Fill variables| FilledTemplate["Filled Template"]
    Variables["Template Variables"] --> FilledTemplate
    Context["Context Information"] --> Variables
    FilledTemplate --> Response["Generated Response"]
    
    classDef template fill:#f9f,stroke:#333,stroke-width:1px;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef data fill:#ffd,stroke:#333,stroke-width:1px;
    classDef output fill:#dfd,stroke:#333,stroke-width:1px;
    
    class Template template;
    class FilledTemplate process;
    class Variables,Context data;
    class Response output;
```

### Stage 3: Complete Prompt Engineering Flow

Finally, let's add the complete prompt engineering flow:

```mermaid
flowchart TD
    subgraph Input["Input Processing"]
        A1[User Input] --> A2[Extract Intent]
        A2 --> A3[Identify Entities]
        A3 --> A4[Determine Context]
    end
    
    subgraph Prompt["Prompt System"]
        B1[Select Template]
        B2[Fill Template Variables]
        B3[Apply Role Context]
        B4[Add Examples]
    end
    
    subgraph Response["Response Generation"]
        C1[Generate Response]
        C2[Format Output]
        C3[Apply Constraints]
    end
    
    A4 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D[Final Response]
    
    classDef input fill:#f9f,stroke:#333,stroke-width:1px;
    classDef prompt fill:#bbf,stroke:#333,stroke-width:1px;
    classDef response fill:#dfd,stroke:#333,stroke-width:1px;
    classDef output fill:#dff,stroke:#333,stroke-width:1px;
    
    class A1,A2,A3,A4 input;
    class B1,B2,B3,B4 prompt;
    class C1,C2,C3 response;
    class D output;
```

## 3. Class Hierarchy: Stage-by-Stage Breakdown

### Stage 1: SimpleAgent

Let's start with the SimpleAgent class:

```mermaid
classDiagram
    class SimpleAgent {
        +dict state
        +sense(user_input)
        +think(processed_input)
        +act(action_plan)
        +agent_loop(user_input)
    }
```

### Stage 2: Adding PromptDrivenAgent

Now let's add the PromptDrivenAgent:

```mermaid
classDiagram
    class SimpleAgent {
        +dict state
        +sense(user_input)
        +think(processed_input)
        +act(action_plan)
        +agent_loop(user_input)
    }
    
    class PromptTemplate {
        +str template
        +fill(variables)
    }
    
    class PromptLibrary {
        +dict templates
        +get_template(name)
        +add_template(name, template)
    }
    
    class PromptDrivenAgent {
        +PromptLibrary prompts
        +sense(user_input)
        +think(processed_input)
        +act(action_plan)
        +get_prompt(prompt_name)
    }
    
    SimpleAgent <|-- PromptDrivenAgent : extends
    PromptDrivenAgent --> PromptLibrary : uses
    PromptLibrary --> PromptTemplate : contains
```

### Stage 3: Adding StatefulAgent

Let's add the StatefulAgent:

```mermaid
classDiagram
    class SimpleAgent {
        +dict state
        +sense(user_input)
        +think(processed_input)
        +act(action_plan)
        +agent_loop(user_input)
    }
    
    class PromptDrivenAgent {
        +PromptLibrary prompts
        +sense(user_input)
        +think(processed_input)
        +act(action_plan)
        +get_prompt(prompt_name)
    }
    
    class AgentStateManager {
        +ShortTermMemory short_term
        +LongTermMemory long_term
        +EpisodicMemory episodic
        +update_conversation(role, content)
        +get_conversation_history()
        +save_state()
        +load_state()
    }
    
    class StatefulAgent {
        +AgentStateManager state_manager
        +sense(user_input)
        +think(processed_input, context)
        +act(action_plan)
        +get_context()
        +update_state(interaction)
    }
    
    SimpleAgent <|-- PromptDrivenAgent : extends
    PromptDrivenAgent <|-- StatefulAgent : extends
    StatefulAgent --> AgentStateManager : uses
```

### Stage 4: Complete Class Hierarchy

Finally, let's add the TaskManagerAgent to complete the hierarchy:

```mermaid
classDiagram
    class SimpleAgent {
        +dict state
        +sense(user_input)
        +think(processed_input)
        +act(action_plan)
        +agent_loop(user_input)
    }
    
    class PromptDrivenAgent {
        +PromptLibrary prompts
        +sense(user_input)
        +think(processed_input)
        +act(action_plan)
        +get_prompt(prompt_name)
    }
    
    class StatefulAgent {
        +AgentStateManager state_manager
        +sense(user_input)
        +think(processed_input, context)
        +act(action_plan)
        +get_context()
        +update_state(interaction)
    }
    
    class TaskManagerAgent {
        +TaskManager task_manager
        +sense(user_input)
        +think(processed_input, context)
        +act(action_plan)
        +add_task(task)
        +update_task(task_id, updates)
        +delete_task(task_id)
        +list_tasks(filter)
    }
    
    class TaskManager {
        +List tasks
        +add_task(task)
        +get_task(task_id)
        +update_task(task_id, updates)
        +delete_task(task_id)
        +list_tasks(filter)
    }
    
    SimpleAgent <|-- PromptDrivenAgent : extends
    PromptDrivenAgent <|-- StatefulAgent : extends
    StatefulAgent <|-- TaskManagerAgent : extends
    TaskManagerAgent --> TaskManager : uses
```
