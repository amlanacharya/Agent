# 📊 Module 1: Agent Fundamentals - Explanatory Diagrams

Visual explanations of the key concepts in Module 1 through sequence diagrams, flowcharts, and class diagrams.

## 🔄 Sense-Think-Act Loop

The following sequence diagram illustrates the core agent loop that forms the foundation of all agents in this module:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Memory
    
    User->>Agent: Input (text, command, query)
    
    %% Sense Phase
    Agent->>Agent: sense(input)
    Note over Agent: Process and normalize input
    Note over Agent: Extract intent and entities
    
    %% Think Phase
    Agent->>Memory: Retrieve context
    Memory-->>Agent: Return relevant context
    Agent->>Agent: think(processed_input, context)
    Note over Agent: Determine appropriate response
    Note over Agent: Plan actions based on intent
    
    %% Act Phase
    Agent->>Agent: act(response_plan)
    Note over Agent: Execute planned actions
    Agent->>Memory: Update memory with interaction
    Agent->>User: Return response
    
    %% Feedback Loop
    User->>Agent: Next input (with implicit feedback)
    Note over Agent: Loop continues with feedback
```

## 🧩 Agent Architecture

The following class diagram shows the relationship between the different agent types in Module 1:

```mermaid
classDiagram
    class SimpleAgent {
        +sense(input)
        +think(processed_input)
        +act(response_plan)
        +process(input)
    }
    
    class PromptDrivenAgent {
        +PromptLibrary prompts
        +sense(input)
        +think(processed_input)
        +act(response_plan)
        +process(input)
        +get_prompt(prompt_name)
    }
    
    class StatefulAgent {
        +AgentStateManager state_manager
        +sense(input)
        +think(processed_input, context)
        +act(response_plan)
        +process(input)
        +get_context()
        +update_state(interaction)
    }
    
    class TaskManagerAgent {
        +TaskManager task_manager
        +sense(input)
        +think(processed_input, context)
        +act(response_plan)
        +process(input)
        +add_task(task)
        +update_task(task_id, updates)
        +delete_task(task_id)
        +list_tasks(filter)
    }
    
    SimpleAgent <|-- PromptDrivenAgent : extends
    PromptDrivenAgent <|-- StatefulAgent : extends
    StatefulAgent <|-- TaskManagerAgent : extends
```

## 📝 Prompt Engineering Flow

This flowchart illustrates how prompts are processed in the PromptDrivenAgent:

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
```

## 🧠 State Management Architecture

This diagram shows how different memory types interact in the StatefulAgent:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant ShortTerm as Short-Term Memory
    participant LongTerm as Long-Term Memory
    participant Episodic as Episodic Memory
    
    User->>Agent: Input
    Agent->>ShortTerm: Add interaction
    Agent->>ShortTerm: Retrieve recent context
    ShortTerm-->>Agent: Return recent interactions
    
    Agent->>LongTerm: Query for relevant knowledge
    LongTerm-->>Agent: Return persistent knowledge
    
    Agent->>Agent: Generate response using context
    
    Agent->>ShortTerm: Update with new interaction
    Agent->>Episodic: Record complete interaction
    
    alt Contains important information
        Agent->>LongTerm: Store new knowledge
    end
    
    Agent->>User: Return response
```

## 🗃️ Task Manager Data Flow

This flowchart illustrates how data flows through the TaskManagerAgent:

```mermaid
flowchart TD
    subgraph Commands["Command Processing"]
        A1[Add Task Command]
        A2[Update Task Command]
        A3[Delete Task Command]
        A4[List Tasks Command]
        A5[Query Task Command]
    end
    
    subgraph TaskManager["Task Manager"]
        B1[Task Storage]
        B2[Task Validation]
        B3[Task Filtering]
        B4[Task Sorting]
    end
    
    subgraph StateManager["State Manager"]
        C1[Short-Term Memory]
        C2[Long-Term Memory]
        C3[User Preferences]
    end
    
    A1 --> B2
    B2 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B3
    B3 --> B4
    A5 --> B1
    
    B1 --> C2
    C3 --> B4
    C1 --> A1
    C1 --> A2
```

