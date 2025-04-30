# Function Calling Patterns

This document illustrates different patterns for function calling with language models in Module 6.

## Basic Function Calling Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant Tool
    
    User->>Agent: Send query
    Agent->>LLM: Process query
    LLM-->>Agent: Identify needed function
    Agent->>Tool: Call function with parameters
    Tool-->>Agent: Return result
    Agent->>LLM: Generate response with result
    LLM-->>Agent: Final response
    Agent->>User: Return response
    
    Note over Agent,LLM: Function selection phase
    Note over Agent,Tool: Function execution phase
    Note over Agent,LLM: Response generation phase
```

## OpenAI Function Calling

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant OpenAI API
    participant Tool Registry
    participant Tool
    
    User->>Agent: "What's the weather in New York?"
    Agent->>Tool Registry: Get available tools
    Tool Registry-->>Agent: Return tool schemas
    
    Agent->>OpenAI API: Send query + tool schemas
    
    Note right of OpenAI API: Model decides to call<br/>the weather tool
    
    OpenAI API-->>Agent: {<br/>  "function_call": {<br/>    "name": "weather",<br/>    "arguments": {<br/>      "location": "New York"<br/>    }<br/>  }<br/>}
    
    Agent->>Tool Registry: Get weather tool
    Tool Registry-->>Agent: Return weather tool
    
    Agent->>Tool: Call with {"location": "New York"}
    Tool-->>Agent: Return weather data
    
    Agent->>OpenAI API: Send query + tool call result
    OpenAI API-->>Agent: Generate final response
    
    Agent->>User: "The weather in New York is 72°F and sunny."
```

## Multi-Tool Function Calling

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant Tool Registry
    participant Tool 1
    participant Tool 2
    
    User->>Agent: "Compare the weather in NYC and LA"
    Agent->>LLM: Process query with available tools
    
    LLM-->>Agent: Need to call weather tool for NYC
    Agent->>Tool Registry: Get weather tool
    Tool Registry-->>Agent: Return weather tool
    Agent->>Tool 1: Get weather for NYC
    Tool 1-->>Agent: NYC weather data
    
    LLM-->>Agent: Need to call weather tool for LA
    Agent->>Tool Registry: Get weather tool
    Tool Registry-->>Agent: Return weather tool
    Agent->>Tool 1: Get weather for LA
    Tool 1-->>Agent: LA weather data
    
    Agent->>LLM: Generate comparison with both results
    LLM-->>Agent: Final comparison response
    
    Agent->>User: "NYC is 72°F and sunny, while LA is 78°F and clear..."
```

## Tool Chain Pattern

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant Search Tool
    participant Weather Tool
    participant OpenAI Tool
    
    User->>Agent: "How does the weather in Paris compare to historical averages?"
    
    Agent->>LLM: Process query
    LLM-->>Agent: Need to search for historical weather data
    
    Agent->>Search Tool: Search for "Paris historical weather averages"
    Search Tool-->>Agent: Search results about Paris weather history
    
    Agent->>LLM: Process search results
    LLM-->>Agent: Need current weather in Paris
    
    Agent->>Weather Tool: Get weather for "Paris"
    Weather Tool-->>Agent: Current Paris weather data
    
    Agent->>LLM: Need to compare current and historical data
    LLM-->>Agent: Need to generate detailed comparison
    
    Agent->>OpenAI Tool: Generate comparison of current vs. historical
    OpenAI Tool-->>Agent: Detailed comparison text
    
    Agent->>User: "Currently Paris is 68°F which is 5°F above the historical average..."
```

## Parallel Function Calling

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant Weather Tool
    participant Finance Tool
    participant News Tool
    
    User->>Agent: "How might today's weather affect AAPL stock?"
    
    Agent->>LLM: Process query
    
    par Get Multiple Data Points
        Agent->>Weather Tool: Get current weather
        Weather Tool-->>Agent: Weather data
    and
        Agent->>Finance Tool: Get AAPL stock info
        Finance Tool-->>Agent: AAPL stock data
    and
        Agent->>News Tool: Get recent AAPL news
        News Tool-->>Agent: Recent news articles
    end
    
    Agent->>LLM: Analyze all collected data
    LLM-->>Agent: Generated analysis
    
    Agent->>User: "Today's sunny weather in Cupertino could positively impact AAPL..."
```

## Error Handling in Function Calling

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant Primary Tool
    participant Fallback Tool
    
    User->>Agent: "What's the weather in Atlantis?"
    
    Agent->>LLM: Process query
    LLM-->>Agent: Need to call weather tool
    
    Agent->>Primary Tool: Get weather for "Atlantis"
    Primary Tool-->>Agent: Error: Location not found
    
    Agent->>LLM: Report error and ask for alternative
    LLM-->>Agent: Suggest searching for information instead
    
    Agent->>Fallback Tool: Search for "Is Atlantis a real place weather"
    Fallback Tool-->>Agent: Search results about fictional Atlantis
    
    Agent->>LLM: Generate response based on search results
    LLM-->>Agent: Final response explaining Atlantis is fictional
    
    Agent->>User: "Atlantis is a fictional city, so I can't provide weather information..."
```
