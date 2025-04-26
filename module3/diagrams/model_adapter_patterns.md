# Model Adapter Patterns

This file contains diagrams illustrating the Model Adapter pattern and its applications in API flows.

## Common API Flows

The following sequence diagram illustrates three common API flows using the Model Adapter pattern:
1. Creating a user
2. Getting detailed user information
3. Updating a user

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Adapter as ModelAdapter/Registry
    participant DB as Database

    %% Create User Flow
    Client->>API: POST /users (CreateUserRequest)
    API->>Adapter: adapt(createRequest, UserDB)
    Note over Adapter: Transform password to password_hash
    Adapter->>DB: Save UserDB object
    DB->>Adapter: Return saved UserDB
    Adapter->>API: adapt(userDB, UserResponse)
    Note over Adapter: Exclude password_hash
    API->>Client: Return UserResponse

    %% Get User Detail Flow
    Client->>API: GET /users/{id}/detail
    API->>DB: Get UserDB
    API->>DB: Get UserProfileDB
    DB->>API: Return UserDB
    DB->>API: Return UserProfileDB
    API->>Adapter: adapt with both models
    Note over Adapter: Combine data from both sources
    Adapter->>API: Return UserDetailResponse
    API->>Client: Return UserDetailResponse

    %% Update User Flow
    Client->>API: PATCH /users/{id} (UpdateUserRequest)
    API->>DB: Get current UserDB
    DB->>API: Return UserDB
    API->>Adapter: Apply updates to UserDB
    Note over Adapter: Hash password if provided
    Adapter->>DB: Save updated UserDB
    DB->>Adapter: Return updated UserDB
    Adapter->>API: adapt(userDB, UserResponse)
    API->>Client: Return UserResponse
```

## Class Diagram: Model Adapter System

The following class diagram illustrates the structure of the Model Adapter pattern and the relationships between different components:

```mermaid
classDiagram
    class ModelAdapter {
        +Type source_model
        +Type target_model
        +Dict field_mapping
        +Dict transformers
        +List exclude_fields
        +bool include_unmapped
        +adapt(source)
        +adapt_many(sources)
    }

    class AdapterRegistry {
        +Dict adapters
        +register(source_model, target_model, adapter)
        +get_adapter(source_model, target_model)
        +adapt(source, target_model)
        +adapt_many(sources, target_model)
    }

    class CreateUserRequest {
        +str username
        +str email
        +str password
        +str full_name
    }

    class UpdateUserRequest {
        +str email
        +str password
        +str full_name
        +bool is_active
    }

    class UserDB {
        +str id
        +str username
        +str email
        +str password_hash
        +str full_name
        +bool is_active
        +datetime created_at
        +datetime updated_at
    }

    class UserProfileDB {
        +str user_id
        +str bio
        +str avatar_url
        +str location
        +str website
        +Dict social_links
        +Dict preferences
    }

    class UserResponse {
        +str id
        +str username
        +str email
        +str full_name
        +bool is_active
        +datetime created_at
    }

    class UserDetailResponse {
        +str id
        +str username
        +str email
        +str full_name
        +bool is_active
        +datetime created_at
        +str bio
        +str avatar_url
        +str location
        +str website
        +Dict social_links
    }

    AdapterRegistry --> ModelAdapter : contains
    CreateUserRequest ..> UserDB : adapts to
    UserDB ..> UserResponse : adapts to
    UpdateUserRequest ..> UserDB : updates
    UserDB ..> UserDetailResponse : combines with profile
    UserProfileDB ..> UserDetailResponse : combines with user
```

## Data Flow Diagram

The following flowchart illustrates how data flows through the different layers of the application when using the Model Adapter pattern:

```mermaid
flowchart TD
    subgraph API["API Layer"]
        A1[CreateUserRequest] --> |validated data|B1
        A2[UpdateUserRequest] --> |validated updates|B1
        A3[UserResponse] --> |returned to client|Client
        A4[UserDetailResponse] --> |returned to client|Client
        A5[UserListResponse] --> |returned to client|Client
    end

    subgraph Adapter["Adapter System"]
        B1[ModelAdapter]
        B2[AdapterRegistry] --> |provides|B1
        B3["Transformers"] --> |apply data transformations|B1
    end

    subgraph DB["Database Layer"]
        C1[UserDB]
        C2[UserProfileDB]
    end

    Client[Client] --> |sends requests|API

    B1 --> |transforms request to DB model|C1
    C1 --> |retrieved data|B1
    C2 --> |profile data|B1

    B1 --> |transforms DB model to response|A3
    B1 --> |combines user & profile data|A4

    C1 --> |multiple records|B1
    B1 --> |transforms collection|A5
```
