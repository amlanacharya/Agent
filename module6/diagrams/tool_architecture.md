# Tool Architecture

This diagram illustrates the architecture of the tool system in Module 6, showing the relationships between the base tool interface and the specific tool implementations.

## Base Tool Interface

```mermaid
classDiagram
    class BaseTool {
        +name: str
        +description: str
        +metadata: Dict
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
        +to_dict(): Dict
        +to_langchain_tool(): LangChainTool
    }
    
    class ToolResponse {
        +success: bool
        +result: Any
        +error: Optional[str]
        +metadata: Dict
    }
    
    BaseTool ..> ToolResponse : returns
```

## Tool Implementations

```mermaid
classDiagram
    class BaseTool {
        +name: str
        +description: str
        +metadata: Dict
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
        +to_dict(): Dict
        +to_langchain_tool(): LangChainTool
    }
    
    class OpenAITool {
        +api_key: str
        +model: str
        +max_tokens: int
        +temperature: float
        +max_retries: int
        +retry_delay: int
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
        +chat(messages): str
        +complete(prompt): str
    }
    
    class GroqTool {
        +api_key: str
        +model: str
        +max_tokens: int
        +temperature: float
        +max_retries: int
        +retry_delay: int
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
        +chat(messages): str
        +complete(prompt): str
        +generate_json(prompt_or_messages, schema): Dict
    }
    
    class SearchTool {
        +serper_api_key: str
        +max_results: int
        +max_retries: int
        +retry_delay: int
        +use_fallback: bool
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
        +search(query): List[SearchResult]
        +_search_with_serper(query): List[SearchResult]
        +_search_with_duckduckgo(query): List[SearchResult]
    }
    
    class WeatherTool {
        +api_key: str
        +base_url: str
        +max_retries: int
        +retry_delay: int
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
        +get_weather_by_location(location): WeatherResult
        +get_weather_by_coords(lat, lon): WeatherResult
        +get_current_weather(location_or_coords): Dict
    }
    
    class AlphaVantageTool {
        +api_key: str
        +base_url: str
        +max_retries: int
        +retry_delay: int
        +execute(**kwargs): ToolResponse
        +get_schema(): Dict
        +get_stock_quote(symbol): StockQuote
        +get_exchange_rate(from_currency, to_currency): ExchangeRate
        +search_symbols(keywords): List[Dict]
    }
    
    BaseTool <|-- OpenAITool
    BaseTool <|-- GroqTool
    BaseTool <|-- SearchTool
    BaseTool <|-- WeatherTool
    BaseTool <|-- AlphaVantageTool
    
    class SearchResult {
        +title: str
        +link: str
        +snippet: Optional[str]
        +position: Optional[int]
        +source: str
    }
    
    class WeatherResult {
        +location: str
        +temperature: float
        +feels_like: float
        +humidity: int
        +pressure: int
        +wind_speed: float
        +wind_direction: str
        +description: str
        +icon: str
        +timestamp: int
    }
    
    class StockQuote {
        +symbol: str
        +price: float
        +change: float
        +change_percent: str
        +volume: int
        +market_cap: Optional[float]
        +pe_ratio: Optional[float]
    }
    
    class ExchangeRate {
        +from_currency: str
        +to_currency: str
        +rate: float
        +timestamp: int
    }
    
    SearchTool ..> SearchResult : returns
    WeatherTool ..> WeatherResult : returns
    AlphaVantageTool ..> StockQuote : returns
    AlphaVantageTool ..> ExchangeRate : returns
```

## Tool Integration with LangChain

```mermaid
flowchart TD
    A[BaseTool] -->|to_langchain_tool| B[LangChain Tool]
    B -->|used by| C[LangChain Agent]
    
    D[OpenAITool] -->|to_langchain_tool| B
    E[GroqTool] -->|to_langchain_tool| B
    F[SearchTool] -->|to_langchain_tool| B
    G[WeatherTool] -->|to_langchain_tool| B
    H[AlphaVantageTool] -->|to_langchain_tool| B
    
    C -->|executes| I[Tool Chain]
    I -->|includes| B
    
    style A fill:#bbf,stroke:#333,stroke-width:2px
    style B fill:#dfd,stroke:#333,stroke-width:2px
    style C fill:#ffd,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:1px
    style E fill:#bbf,stroke:#333,stroke-width:1px
    style F fill:#bbf,stroke:#333,stroke-width:1px
    style G fill:#bbf,stroke:#333,stroke-width:1px
    style H fill:#bbf,stroke:#333,stroke-width:1px
    style I fill:#ffd,stroke:#333,stroke-width:1px
```
