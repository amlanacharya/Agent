# When and Where to Modify the Adapter System

When using the Model Adapter pattern, understanding what to change and when is critical.

## Common Change Scenarios

### 1. Adding a New Field to an Existing Entity

**Example:** Adding a `loyalty_points` field to track customer rewards

**What to modify:**
- Add the field to the appropriate models (request, database, response)
- Update or create transformers if the field needs special handling
- No need to modify the adapter class itself

```python
# Add to models
class UserDB(BaseModel):
    # Existing fields...
    loyalty_points: int = 0

class UserResponse(BaseModel):
    # Existing fields...  
    loyalty_points: int = 0

# Update existing adapter with transformer if needed
user_response_adapter = ModelAdapter(
    UserDB,
    UserResponse,
    transformers={
        # Existing transformers...
        "loyalty_points": lambda points: max(0, points)  # Ensure non-negative
    }
)
```

### 2. Adding Validation to an Existing Field

**Example:** Adding stricter validation to email formats

**What to modify:**
- Update the validator in the request model only
- No changes needed to adapters or other models

```python
class CreateUserRequest(BaseModel):
    # Existing fields...
    email: str
    
    @field_validator('email')
    def validate_email(cls, v):
        # More sophisticated validation
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', v):
            raise ValueError("Invalid email format")
        # Check for disposable email domains
        domain = v.split('@')[1]
        if domain in DISPOSABLE_EMAIL_DOMAINS:
            raise ValueError("Disposable email addresses not allowed")
        return v
```

### 3. Adding a Completely New Entity

**Example:** Adding product categories to an e-commerce system

**What to modify:**
- Create new models (request, database, response)
- Create new adapters for the new entity
- Register adapters in the registry

```python
# Create new models
class CreateCategoryRequest(BaseModel):
    name: str
    parent_category_id: Optional[str] = None
    
class CategoryDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    parent_category_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
class CategoryResponse(BaseModel):
    id: str
    name: str
    parent_category_id: Optional[str] = None

# Create adapters
category_db_adapter = ModelAdapter(
    CreateCategoryRequest, 
    CategoryDB
)

category_response_adapter = ModelAdapter(
    CategoryDB,
    CategoryResponse
)

# Register adapters
registry.register(CreateCategoryRequest, CategoryDB, category_db_adapter)
registry.register(CategoryDB, CategoryResponse, category_response_adapter)
```

### 4. Adding a Relationship Between Entities

**Example:** Linking users to their favorite products

**What to modify:**
- Add fields to the appropriate models
- Create new adapters for combined data if needed
- Add transformers to handle the relationship

```python
# Add to UserDB
class UserDB(BaseModel):
    # Existing fields...
    favorite_product_ids: List[str] = Field(default_factory=list)

# Create new response model
class UserWithFavoritesResponse(BaseModel):
    # User fields...
    favorite_products: List[ProductResponse] = Field(default_factory=list)

# Create a custom adapter function
async def create_user_with_favorites_response(user_db):
    # Get base user response
    user_response = registry.adapt(user_db, UserResponse)
    
    # Fetch favorite products
    product_dbs = await product_repository.get_many(user_db.favorite_product_ids)
    product_responses = registry.adapt_many(product_dbs, ProductResponse)
    
    # Create combined response
    return UserWithFavoritesResponse(
        **user_response.model_dump(),
        favorite_products=product_responses
    )
```

### 5. Changing How Data is Transformed

**Example:** Modifying how prices are calculated with new tax rules

**What to modify:**
- Update the transformer functions
- No need to modify models or adapter classes

```python
# Update existing transformer
def calculate_order_totals(products: List[Dict[str, Any]], shipping_address: Dict[str, str]) -> Dict[str, float]:
    subtotal = sum(item["price"] * item["quantity"] for item in products)
    
    # New tax calculation based on shipping location
    if shipping_address["country"] == "US":
        state_tax_rates = {"CA": 0.0925, "NY": 0.0845, "TX": 0.0625}
        tax_rate = state_tax_rates.get(shipping_address["state"], 0.05)
    else:
        tax_rate = 0.10  # International tax rate
    
    tax = subtotal * tax_rate
    
    # Updated shipping logic
    shipping = calculate_shipping_cost(products, shipping_address)
    
    return {
        "subtotal": subtotal,
        "tax": tax,
        "tax_rate": tax_rate,
        "shipping_cost": shipping,
        "total_price": subtotal + tax + shipping
    }

# Register updated transformer in adapter
order_adapter = ModelAdapter(
    OrderRequestModel,
    OrderDBModel,
    transformers={
        "totals": lambda req: calculate_order_totals(req.products, req.shipping_address)
    }
)
```

### 6. Adding a New Representation of Existing Data

**Example:** Adding a simplified order representation for mobile apps

**What to modify:**
- Create a new response model
- Create a new adapter for the new model
- Register the adapter

```python
# Create new model
class MobileOrderResponse(BaseModel):
    id: str
    status: str
    date: str  # Formatted date string instead of datetime
    total: float
    item_count: int

# Create new adapter with transformers
mobile_order_adapter = ModelAdapter(
    OrderDB,
    MobileOrderResponse,
    transformers={
        "date": lambda dt: dt.strftime("%b %d, %Y"),
        "total": lambda order: order.total_price,
        "item_count": lambda order: sum(item["quantity"] for item in order.products)
    }
)

# Register adapter
registry.register(OrderDB, MobileOrderResponse, mobile_order_adapter)
```

## Decision Guide: What to Change

When implementing a change, follow this decision tree:

1. **Are you adding/modifying what data is stored?**
   - Update the database models

2. **Are you changing how users provide data?**
   - Update the request models

3. **Are you changing how data is presented to users?**
   - Update the response models

4. **Are you changing validation rules?**
   - Update validators in request models

5. **Are you changing how data transforms between layers?**
   - Update transformers in adapters

6. **Are you adding a new way to convert between existing models?**
   - Create a new adapter and register it

7. **Are you combining data from multiple sources?**
   - Create a specialized adapter function that fetches and combines data

## Benefits of This Approach

This approach to managing changes offers several advantages:

1. **Minimal code changes** - Only update what needs to change
2. **Reduced risk** - Changes are isolated and don't affect other parts
3. **Incremental updates** - Easy to add new features without breaking existing ones
4. **Clear locations** - Always know where to make changes
5. **Testable** - Each change can be tested in isolation

