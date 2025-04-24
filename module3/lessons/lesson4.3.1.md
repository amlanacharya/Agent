# E-Commerce System Using the Model Adapter Pattern

ShopSmart is an e-commerce platform that connects merchants with customers. The platform manages products, orders, user accounts, and payment processing.

## Core Models in the System

### API Request Models
```python
# What customers send when placing an order
class CreateOrderRequest(BaseModel):
    product_ids: List[str]
    shipping_address_id: str
    payment_method_id: str
    coupon_code: Optional[str] = None
    special_instructions: Optional[str] = None
    
    @field_validator('product_ids')
    def validate_products(cls, v):
        if not v:
            raise ValueError("Order must contain at least one product")
        return v

# What merchants send when creating a product
class CreateProductRequest(BaseModel):
    name: str
    description: str
    price: float
    category_id: str
    stock_quantity: int
    sku: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("Price must be greater than zero")
        return v
    
    @field_validator('stock_quantity')
    def validate_stock(cls, v):
        if v < 0:
            raise ValueError("Stock quantity cannot be negative")
        return v
```

### Database Models
```python
class OrderDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    status: str = "pending"  # pending, processing, shipped, delivered, cancelled
    products: List[Dict[str, Any]]  # Includes product_id, quantity, price_at_order
    shipping_address: Dict[str, str]
    payment_details: Dict[str, Any]
    subtotal: float
    tax: float
    shipping_cost: float
    total_price: float
    discount_applied: float = 0
    coupon_code: Optional[str] = None
    special_instructions: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class ProductDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    merchant_id: str
    name: str
    description: str
    price: float
    sale_price: Optional[float] = None
    category_id: str
    stock_quantity: int
    sku: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    is_featured: bool = False
    average_rating: float = 0
    review_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
```

### API Response Models
```python
class OrderSummaryResponse(BaseModel):
    id: str
    status: str
    total_price: float
    created_at: datetime
    product_count: int

class OrderDetailResponse(BaseModel):
    id: str
    status: str
    products: List[Dict[str, Any]]
    shipping_address: Dict[str, str]
    subtotal: float
    tax: float
    shipping_cost: float
    total_price: float
    discount_applied: float
    coupon_code: Optional[str]
    special_instructions: Optional[str]
    created_at: datetime
    tracking_number: Optional[str] = None

class ProductListResponse(BaseModel):
    id: str
    name: str
    price: float
    sale_price: Optional[float]
    average_rating: float
    is_featured: bool
    thumbnail_url: str  # Generated from product ID

class ProductDetailResponse(BaseModel):
    id: str
    name: str
    description: str
    price: float
    sale_price: Optional[float]
    category_id: str
    category_name: str  # Joined from categories
    stock_quantity: int
    attributes: Dict[str, Any]
    is_active: bool
    average_rating: float
    review_count: int
    merchant: Dict[str, Any]  # Simplified merchant info
    related_products: List[ProductListResponse]
    image_urls: List[str]  # Generated from product ID
```

## Setting Up Adapters

```python
# Initialize the registry
registry = AdapterRegistry()

# Calculate order totals transformation
def calculate_order_totals(products: List[Dict[str, Any]]) -> Dict[str, float]:
    subtotal = sum(item["price"] * item["quantity"] for item in products)
    tax = subtotal * 0.08  # 8% tax rate
    shipping = 5.99 if subtotal < 50 else 0  # Free shipping over $50
    
    return {
        "subtotal": subtotal,
        "tax": tax,
        "shipping_cost": shipping,
        "total_price": subtotal + tax + shipping
    }

# CreateOrderRequest -> OrderDB adapter
create_order_adapter = ModelAdapter(
    CreateOrderRequest,
    OrderDB,
    transformers={
        "products": lambda product_ids: [{"product_id": pid, "quantity": 1} for pid in product_ids],
        # More transformers would be here in a real implementation
    }
)
registry.register(CreateOrderRequest, OrderDB, create_order_adapter)

# OrderDB -> OrderSummaryResponse adapter
order_summary_adapter = ModelAdapter(
    OrderDB, 
    OrderSummaryResponse,
    transformers={
        "product_count": lambda products: sum(item["quantity"] for item in products)
    }
)
registry.register(OrderDB, OrderSummaryResponse, order_summary_adapter)

# OrderDB -> OrderDetailResponse adapter
order_detail_adapter = ModelAdapter(
    OrderDB,
    OrderDetailResponse
)
registry.register(OrderDB, OrderDetailResponse, order_detail_adapter)

# CreateProductRequest -> ProductDB adapter
create_product_adapter = ModelAdapter(
    CreateProductRequest,
    ProductDB
)
registry.register(CreateProductRequest, ProductDB, create_product_adapter)

# ProductDB -> ProductListResponse adapter
def generate_thumbnail_url(product_id: str) -> str:
    return f"https://shopsmart.example.com/images/{product_id}/thumbnail.jpg"

product_list_adapter = ModelAdapter(
    ProductDB,
    ProductListResponse,
    transformers={
        "thumbnail_url": lambda product_id: generate_thumbnail_url(product_id)
    }
)
registry.register(ProductDB, ProductListResponse, product_list_adapter)
```

## Using the Adapters in API Endpoints

```python
# Example FastAPI endpoints

@app.post("/orders/", response_model=OrderSummaryResponse)
async def create_order(order_request: CreateOrderRequest, user_id: str = Depends(get_current_user_id)):
    # Convert request to database model
    order_db = registry.adapt(order_request, OrderDB, user_id=user_id)
    
    # Fetch product information to calculate prices
    products_with_prices = await fetch_products_with_prices(order_request.product_ids)
    order_db.products = products_with_prices
    
    # Calculate totals
    totals = calculate_order_totals(products_with_prices)
    for key, value in totals.items():
        setattr(order_db, key, value)
    
    # Apply any coupon code
    if order_db.coupon_code:
        discount = await apply_coupon(order_db.coupon_code, order_db.subtotal)
        order_db.discount_applied = discount
        order_db.total_price -= discount
    
    # Save to database
    saved_order = await order_repository.create(order_db)
    
    # Convert to response model
    return registry.adapt(saved_order, OrderSummaryResponse)

@app.get("/orders/{order_id}", response_model=OrderDetailResponse)
async def get_order(order_id: str, user_id: str = Depends(get_current_user_id)):
    # Fetch from database
    order_db = await order_repository.get_by_id(order_id)
    
    # Verify ownership
    if order_db.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this order")
    
    # Add tracking info if available
    tracking_info = await shipping_service.get_tracking(order_id)
    
    # Convert to response model
    return registry.adapt(order_db, OrderDetailResponse, tracking_number=tracking_info.get("tracking_number"))

@app.post("/products/", response_model=ProductDetailResponse)
async def create_product(product_request: CreateProductRequest, merchant_id: str = Depends(get_merchant_id)):
    # Convert request to database model
    product_db = registry.adapt(product_request, ProductDB, merchant_id=merchant_id)
    
    # Save to database
    saved_product = await product_repository.create(product_db)
    
    # Fetch category info and merchant details
    category = await category_repository.get_by_id(saved_product.category_id)
    merchant = await merchant_repository.get_by_id(merchant_id)
    
    # Generate image URLs
    image_urls = [
        f"https://shopsmart.example.com/images/{saved_product.id}/main.jpg",
        f"https://shopsmart.example.com/images/{saved_product.id}/angle1.jpg",
        f"https://shopsmart.example.com/images/{saved_product.id}/angle2.jpg"
    ]
    
    # Find related products
    related_products_db = await product_repository.find_related(saved_product.id, limit=3)
    related_products = registry.adapt_many(related_products_db, ProductListResponse)
    
    # Create a detailed response
    return ProductDetailResponse(
        **saved_product.model_dump(),
        category_name=category.name,
        merchant={
            "id": merchant.id,
            "name": merchant.name,
            "rating": merchant.rating
        },
        related_products=related_products,
        image_urls=image_urls
    )
```

## Scenario: Adding New Features

Let's say ShopSmart needs to implement new features. Here's what changes:

### Scenario 1: Add Subscription Orders

**New Requirement:** Customers can now subscribe to receive products on a regular basis.

**What Changes:**

1. **Add to API Request Model:**
```python
# Add to CreateOrderRequest
subscription_frequency: Optional[str] = None  # "weekly", "monthly", etc.
subscription_duration: Optional[int] = None  # Number of deliveries
```

2. **Add to Database Model:**
```python
# Add to OrderDB
is_subscription: bool = False
subscription_frequency: Optional[str] = None
subscription_duration: Optional[int] = None
subscription_next_date: Optional[datetime] = None
```

3. **Update API Response Models:**
```python
# Add to both OrderSummaryResponse and OrderDetailResponse
is_subscription: bool = False
subscription_details: Optional[Dict[str, Any]] = None
```

4. **Update the Adapter:**
```python
# Modify create_order_adapter by adding a transformer
create_order_adapter = ModelAdapter(
    CreateOrderRequest,
    OrderDB,
    transformers={
        # Existing transformers...
        "is_subscription": lambda req: bool(req.subscription_frequency),
        "subscription_next_date": lambda req: (
            datetime.now() + timedelta(days=7) if req.subscription_frequency == "weekly"
            else datetime.now() + timedelta(days=30) if req.subscription_frequency == "monthly"
            else None
        )
    }
)
```

5. **Update Response Adapters:**
```python
# Add a transformer for subscription_details
def format_subscription_details(order_db):
    if not order_db.is_subscription:
        return None
    return {
        "frequency": order_db.subscription_frequency,
        "remaining_deliveries": order_db.subscription_duration,
        "next_delivery_date": order_db.subscription_next_date
    }

order_summary_adapter = ModelAdapter(
    OrderDB, 
    OrderSummaryResponse,
    transformers={
        # Existing transformers...
        "subscription_details": format_subscription_details
    }
)

# Do the same for order_detail_adapter
```

### Scenario 2: Adding Product Bundling

**New Requirement:** Products can now be bundled together with special pricing.

**What Changes:**

1. **Create New Database Model:**
```python
class ProductBundleDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    product_ids: List[str]
    bundle_price: float
    discount_percentage: float
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
```

2. **Create New API Request/Response Models:**
```python
class CreateBundleRequest(BaseModel):
    name: str
    description: str
    product_ids: List[str]
    discount_percentage: float
    
    @field_validator('discount_percentage')
    def validate_discount(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Discount must be between 0 and 100 percent")
        return v

class BundleResponse(BaseModel):
    id: str
    name: str
    description: str
    products: List[ProductListResponse]
    original_price: float
    bundle_price: float
    savings_amount: float
    savings_percentage: float
```

3. **Create New Adapters:**
```python
# CreateBundleRequest -> ProductBundleDB
async def calculate_bundle_price(product_ids, discount_percentage):
    products = await product_repository.get_many(product_ids)
    original_price = sum(p.price for p in products)
    bundle_price = original_price * (1 - discount_percentage / 100)
    return bundle_price

bundle_adapter = ModelAdapter(
    CreateBundleRequest,
    ProductBundleDB,
    transformers={
        "bundle_price": lambda req: calculate_bundle_price(req.product_ids, req.discount_percentage)
    }
)
registry.register(CreateBundleRequest, ProductBundleDB, bundle_adapter)

# For the response, we need to combine data from multiple sources
async def create_bundle_response(bundle_db):
    products_db = await product_repository.get_many(bundle_db.product_ids)
    product_responses = registry.adapt_many(products_db, ProductListResponse)
    
    original_price = sum(p.price for p in products_db)
    savings = original_price - bundle_db.bundle_price
    
    return BundleResponse(
        **bundle_db.model_dump(),
        products=product_responses,
        original_price=original_price,
        savings_amount=savings,
        savings_percentage=(savings / original_price) * 100 if original_price > 0 else 0
    )
```

4. **Add to Order System:**
```python
# Update CreateOrderRequest
bundle_ids: List[str] = Field(default_factory=list)

# Update the order creation pipeline to handle bundles
async def process_order_with_bundles(order_request, user_id):
    order_db = registry.adapt(order_request, OrderDB, user_id=user_id)
    
    # Process regular products
    products_with_prices = await fetch_products_with_prices(order_request.product_ids)
    
    # Process bundles and add their products
    bundle_products = []
    for bundle_id in order_request.bundle_ids:
        bundle = await bundle_repository.get_by_id(bundle_id)
        bundle_items = await product_repository.get_many(bundle.product_ids)
        
        # Add each product from the bundle with the special price
        discount_factor = bundle.bundle_price / sum(p.price for p in bundle_items)
        for product in bundle_items:
            bundle_products.append({
                "product_id": product.id,
                "quantity": 1,
                "price": product.price * discount_factor,
                "bundle_id": bundle_id,
                "is_bundle_item": True
            })
    
    # Combine regular products and bundle products
    order_db.products = products_with_prices + bundle_products
    
    # Continue with existing order processing...
    return order_db
```

## Key Takeaways on Adapting to Changes

When new requirements arise, you typically need to:

1. **Add new fields to models** where appropriate
2. **Create new models** if you're introducing entire new concepts
3. **Update or create adapters** to handle the conversion between these models
4. **Add transformers** to perform any calculations or data manipulations needed

The beauty of the adapter pattern is that:

- **Changes are localized** - You modify only the models and adapters that need to change
- **Existing code is protected** - Old functionality continues to work
- **Common transformations are reused** - Transformers can be shared between adapters
- **Type safety is maintained** - New fields and models are properly typed

This architectural approach makes it much easier to evolve your system over time without introducing bugs or creating technical debt.
