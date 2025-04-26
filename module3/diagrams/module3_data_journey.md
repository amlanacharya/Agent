# The Journey of Data Through Module 3

This diagram illustrates the complete journey of data through Module 3's validation and processing pipeline, showing how all components interact to process, validate, and transform data.

```mermaid
flowchart TD
    %% Main data sources
    UserInput[User Input\n📝] --> InputValidation
    LLMOutput[LLM Output\n🤖] --> OutputParsing
    
    %% Main processing stages
    subgraph InputValidation["Input Validation 🛡️"]
        PydanticModels["Pydantic Models\nBasic type & constraint validation"]
        ValidationPatterns["Validation Patterns\nComplex business rules"]
        
        PydanticModels --> ValidationPatterns
    end
    
    subgraph OutputParsing["Output Parsing 🔄"]
        BasicParsing["Basic Parsing\nExtract JSON from text"]
        RetryMechanisms["Retry Mechanisms\nHandle parsing failures"]
        TwoStageParsing["Two-Stage Parsing\nExtraction & refinement"]
        
        BasicParsing --> RetryMechanisms
        RetryMechanisms --> TwoStageParsing
    end
    
    subgraph QualityAssessment["Quality Assessment 📊"]
        ClarityEval["Clarity Evaluator\nReadability & comprehension"]
        ConcisenessEval["Conciseness Evaluator\nBrevity & focus"]
        HelpfulnessEval["Helpfulness Evaluator\nUtility & relevance"]
        CoherenceEval["Coherence Evaluator\nLogical flow & organization"]
        EngagementEval["Engagement Evaluator\nTone & style"]
        
        QualityMetrics["Quality Metrics\nScores & issues"]
        
        ClarityEval & ConcisenessEval & HelpfulnessEval & CoherenceEval & EngagementEval --> QualityMetrics
    end
    
    subgraph DataTransformation["Data Transformation 🔄"]
        ModelAdapters["Model Adapters\nConvert between data models"]
        FieldMapping["Field Mapping\nMap fields between models"]
        Transformers["Transformers\nApply data transformations"]
        
        ModelAdapters --> FieldMapping
        ModelAdapters --> Transformers
    end
    
    %% Connections between main components
    InputValidation --> |Valid data| DataTransformation
    OutputParsing --> |Structured data| InputValidation
    OutputParsing --> |Text content| QualityAssessment
    
    %% Data destinations
    DataTransformation --> DBModels[Database Models\n💾]
    DataTransformation --> APIResponses[API Responses\n📤]
    QualityAssessment --> QualityReport[Quality Report\n📋]
    
    %% Special flows
    LLMOutput --> |Form data| FormExtractor[Form Extractor\nExtract structured form data]
    FormExtractor --> InputValidation
    
    %% Error handling
    InputValidation --> |Validation errors| ValidationErrors[Validation Errors\n❌]
    OutputParsing --> |Parsing failures| ParsingErrors[Parsing Errors\n❌]
    
    %% Utility components
    TextAnalysisUtils[Text Analysis Utilities\n🔍] --> ClarityEval & ConcisenessEval & HelpfulnessEval & CoherenceEval & EngagementEval
    
    %% Real-world examples
    subgraph Examples["Real-World Examples"]
        UserProfile["User Profile\nname, email, age, skills"]
        PaymentInfo["Payment Processing\npayment method, amount, card info"]
        ContentModeration["Content Moderation\nsafety checks, bias detection"]
        FormFilling["Form Filling Assistant\nextract & validate form data"]
    end
    
    InputValidation -.-> UserProfile & PaymentInfo
    QualityAssessment -.-> ContentModeration
    OutputParsing & InputValidation -.-> FormFilling
    
    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef output fill:#dfd,stroke:#333,stroke-width:1px;
    classDef error fill:#fdd,stroke:#333,stroke-width:1px;
    classDef utility fill:#dff,stroke:#333,stroke-width:1px;
    classDef example fill:#ffd,stroke:#333,stroke-width:1px;
    
    class UserInput,LLMOutput input;
    class InputValidation,OutputParsing,QualityAssessment,DataTransformation process;
    class DBModels,APIResponses,QualityReport output;
    class ValidationErrors,ParsingErrors error;
    class TextAnalysisUtils utility;
    class UserProfile,PaymentInfo,ContentModeration,FormFilling example;
```

## Component Interactions and Data Flow

This sequence diagram shows how the different components interact when processing a form submission from an LLM:

```mermaid
sequenceDiagram
    actor User
    participant LLM as LLM Service
    participant Parser as Output Parser
    participant Validator as Pydantic Validator
    participant Quality as Quality Validator
    participant Adapter as Model Adapter
    participant DB as Database
    
    User->>LLM: Submit form request
    LLM->>Parser: Generate form data (JSON in text)
    
    %% Parsing phase
    Parser->>Parser: Extract JSON from text
    
    alt Parsing succeeds
        Parser->>Validator: Pass extracted data
    else Parsing fails
        Parser->>LLM: Request correction
        LLM->>Parser: Provide corrected output
        Parser->>Parser: Try two-stage parsing
        Parser->>Validator: Pass extracted data
    end
    
    %% Validation phase
    Validator->>Validator: Validate against model schema
    
    alt Validation succeeds
        Validator->>Quality: Check text quality
        Quality->>Quality: Evaluate across dimensions
        Quality->>User: Provide quality report
        
        Validator->>Adapter: Pass validated data
    else Validation fails
        Validator->>User: Return validation errors
    end
    
    %% Transformation phase
    Adapter->>Adapter: Transform to DB model
    Adapter->>DB: Save data
    DB->>Adapter: Return saved data
    Adapter->>Adapter: Transform to response model
    Adapter->>User: Return success response
    
    %% Notes explaining key processes
    Note over Parser: Handles messy LLM outputs<br>with retry mechanisms
    Note over Validator: Applies both basic and<br>complex validation rules
    Note over Quality: Evaluates clarity, conciseness,<br>helpfulness, coherence, engagement
    Note over Adapter: Transforms between request,<br>database, and response models
```

## Data Models and Their Relationships

This class diagram shows the key data models and their relationships:

```mermaid
classDiagram
    %% Core validation components
    class BaseModel {
        +model_json_schema()
        +model_validate()
        +model_validate_json()
        +model_dump()
    }
    
    class QualityValidator {
        -evaluators: Dict
        -config: QualityValidatorConfig
        +validate(text, context)
        +validate_with_suggestions(text, context)
    }
    
    class OutputParser {
        -model_class: Type
        +parse(text)
        +get_format_instructions()
    }
    
    class ModelAdapter {
        -source_model: Type
        -target_model: Type
        -field_mapping: Dict
        -transformers: Dict
        +adapt(source)
        +adapt_many(sources)
    }
    
    %% Data models
    class UserProfile {
        +name: str
        +email: EmailStr
        +age: int
        +skills: List[str]
    }
    
    class PaymentInfo {
        +amount: float
        +currency: str
        +payment_method: PaymentMethod
        +credit_card_info: Optional[CreditCardInfo]
    }
    
    class QualityMetrics {
        +clarity_score: float
        +conciseness_score: float
        +helpfulness_score: float
        +coherence_score: float
        +engagement_score: float
        +overall_quality_level: QualityLevel
        +issues: List[QualityIssue]
        +overall_score()
    }
    
    class FormData {
        +fields: Dict[str, Any]
        +is_complete: bool
        +validate()
    }
    
    %% Relationships
    BaseModel <|-- UserProfile
    BaseModel <|-- PaymentInfo
    BaseModel <|-- QualityMetrics
    BaseModel <|-- FormData
    
    OutputParser --> BaseModel : parses into
    QualityValidator --> QualityMetrics : produces
    ModelAdapter --> BaseModel : transforms between
    
    %% Utility classes
    class TextAnalysisUtils {
        +calculate_flesch_kincaid_grade(text)
        +count_syllables(text)
        +get_passive_voice_ratio(text)
        +has_logical_connectors(text)
    }
    
    QualityValidator --> TextAnalysisUtils : uses
```
