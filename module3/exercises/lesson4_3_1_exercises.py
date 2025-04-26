"""
Lesson 4.3.1: User Hierarchy with Inheritance

This exercise demonstrates creating a model hierarchy for different types of users
(Guest, RegisteredUser, AdminUser) with appropriate inheritance relationships.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re
import uuid


class BaseUser(BaseModel):
    """Base user model with common fields for all user types."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ip_address: str
    user_agent: str
    last_active: datetime = Field(default_factory=datetime.now)
    
    @field_validator('ip_address')
    def validate_ip_address(cls, v):
        """Validate IP address format."""
        # Simple IPv4 validation
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(ip_pattern, v):
            raise ValueError("Invalid IP address format")
        
        # Check each octet is between 0-255
        octets = v.split('.')
        for octet in octets:
            num = int(octet)
            if num < 0 or num > 255:
                raise ValueError("IP address octets must be between 0-255")
        
        return v


class GuestUser(BaseUser):
    """Guest user model for unauthenticated visitors."""
    session_id: str
    visit_count: int = 1
    referrer: Optional[str] = None
    
    def increment_visit(self):
        """Increment the visit count."""
        self.visit_count += 1
        self.last_active = datetime.now()


class RegisteredUser(BaseUser):
    """Registered user model for authenticated regular users."""
    username: str
    email: str
    display_name: Optional[str] = None
    is_active: bool = True
    joined_date: datetime = Field(default_factory=datetime.now)
    last_login: datetime = Field(default_factory=datetime.now)
    
    @field_validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', v):
            raise ValueError("Username must be 3-20 characters and contain only letters, numbers, and underscores")
        return v
    
    @field_validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', v):
            raise ValueError("Invalid email format")
        return v
    
    def update_login(self):
        """Update the last login time."""
        self.last_login = datetime.now()
        self.last_active = datetime.now()


class AdminUser(RegisteredUser):
    """Admin user model with additional permissions."""
    admin_level: int = 1  # 1: basic admin, 2: super admin, 3: system admin
    permissions: List[str] = []
    managed_sections: List[str] = []
    is_super_admin: bool = False
    
    @field_validator('admin_level')
    def validate_admin_level(cls, v):
        """Validate admin level."""
        if v < 1 or v > 3:
            raise ValueError("Admin level must be between 1 and 3")
        return v
    
    def grant_permission(self, permission: str):
        """Grant a new permission to the admin."""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def revoke_permission(self, permission: str):
        """Revoke a permission from the admin."""
        if permission in self.permissions:
            self.permissions.remove(permission)
    
    def promote(self):
        """Promote the admin to the next level."""
        if self.admin_level < 3:
            self.admin_level += 1
            if self.admin_level == 3:
                self.is_super_admin = True
    
    def demote(self):
        """Demote the admin to the previous level."""
        if self.admin_level > 1:
            self.admin_level -= 1
            if self.admin_level < 3:
                self.is_super_admin = False


# Example usage
def main():
    """Demonstrate the user hierarchy."""
    # Create a guest user
    guest = GuestUser(
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        session_id="sess_12345"
    )
    print(f"Guest User: {guest.model_dump_json(indent=2)}")
    
    # Create a registered user
    registered = RegisteredUser(
        ip_address="192.168.1.2",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        username="johndoe",
        email="john@example.com",
        display_name="John Doe"
    )
    print(f"\nRegistered User: {registered.model_dump_json(indent=2)}")
    
    # Create an admin user
    admin = AdminUser(
        ip_address="192.168.1.3",
        user_agent="Mozilla/5.0 (X11; Linux x86_64)",
        username="admin_jane",
        email="jane@example.com",
        display_name="Jane Smith",
        admin_level=2,
        permissions=["manage_users", "edit_content"],
        managed_sections=["users", "content"]
    )
    print(f"\nAdmin User: {admin.model_dump_json(indent=2)}")
    
    # Demonstrate methods
    print("\nDemonstrating methods:")
    
    guest.increment_visit()
    print(f"Guest visit count after increment: {guest.visit_count}")
    
    registered.update_login()
    print(f"Registered user last login updated: {registered.last_login}")
    
    admin.grant_permission("delete_users")
    print(f"Admin permissions after grant: {admin.permissions}")
    
    admin.promote()
    print(f"Admin level after promotion: {admin.admin_level}")
    print(f"Is super admin: {admin.is_super_admin}")


if __name__ == "__main__":
    main()
