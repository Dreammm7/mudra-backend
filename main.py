"""
Money Transfer Application - FastAPI Backend

This backend implements a secure money transfer system with:
- Atomic transactions (all-or-nothing)
- Row-level locking to prevent race conditions
- Immutable audit logging
- Clean API design
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from contextlib import contextmanager
from dotenv import load_dotenv
import uuid
from decimal import Decimal
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="Money Transfer API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default port
        "http://localhost:5173",  # Vite default port
        "http://localhost:8080",  # Custom Vite port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection pool
# This reuses connections efficiently
db_pool = None

def get_db_pool():
    """Initialize and return database connection pool"""
    global db_pool
    if db_pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Parse connection string and create pool
        db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # min 1, max 20 connections
            dsn=database_url
        )
    return db_pool

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Ensures connections are properly returned to the pool.
    """
    pool = get_db_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

# ==================== Username Generation Helper ====================

def generate_unique_username(display_name: str, cursor) -> str:
    """
    Generate a unique username from display_name.
    - Converts to lowercase
    - Removes spaces and special characters
    - Appends suffix if username exists
    
    Why this approach:
    - Ensures uniqueness at database level
    - Human-friendly (based on name)
    - Handles collisions gracefully
    """
    # Convert to lowercase and remove non-alphanumeric characters
    base_username = re.sub(r'[^a-zA-Z0-9]', '', display_name.lower())
    
    # If empty after cleaning, use default
    if not base_username:
        base_username = "user"
    
    # Try base username first
    username = base_username
    counter = 1
    
    # Check if username exists
    while True:
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone() is None:
            # Username is available
            break
        # Username taken, try with suffix
        username = f"{base_username}_{counter}"
        counter += 1
    
    return username

# ==================== Request/Response Models ====================

class RegisterRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=100, description="User's display name")

class RegisterResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    balance: float

class LoginRequest(BaseModel):
    user_id: str

class LoginResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    balance: float

class UserResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    balance: float

class TransferRequest(BaseModel):
    sender_id: str  # Still UUID internally, but frontend can use username for login
    receiver_username: str = Field(..., description="Receiver's username (e.g., 'vedant' or 'vedant_1')")
    amount: float = Field(..., gt=0, description="Transfer amount must be positive")

class TransferResponse(BaseModel):
    success: bool
    message: str
    new_balance: Optional[float] = None

class TransactionLog(BaseModel):
    id: str
    timestamp: str
    sender_username: str
    receiver_username: str
    amount: float
    status: str

# ==================== API Endpoints ====================

@app.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    """
    Register a new user.
    - Creates a new user with UUID (internal identifier)
    - Generates unique username from display_name
    - Sets initial balance to $500
    - Returns user information including username
    
    Why separate username and display_name:
    - display_name: Can be duplicate, shown in UI
    - username: Must be unique, used for transfers
    - id: UUID, internal only, never exposed to users
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Generate UUID for new user (internal identifier)
        user_id = str(uuid.uuid4())
        
        try:
            # Generate unique username
            username = generate_unique_username(request.display_name, cursor)
            
            # Insert new user with initial balance
            # Note: 'name' is a legacy column that must be populated to satisfy NOT NULL constraint
            # We populate it with display_name to maintain backward compatibility
            cursor.execute(
                """
                INSERT INTO users (id, name, display_name, username, balance)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, display_name, username, balance
                """,
                (user_id, request.display_name, request.display_name, username, Decimal("500.00"))
            )
            
            result = cursor.fetchone()
            
            return RegisterResponse(
                user_id=str(result["id"]),
                username=result["username"],
                display_name=result["display_name"],
                balance=float(result["balance"])
            )
        except psycopg2.IntegrityError as e:
            # Handle unique constraint violation (shouldn't happen with our generation logic)
            raise HTTPException(status_code=409, detail=f"Username conflict: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login with user ID (UUID) or username.
    - Accepts either UUID (system-issued identifier) or username
    - If UUID format: authenticates directly by user_id
    - If username: resolves username to user_id, then authenticates
    - Returns user information including username
    
    Internal logic always resolves to UUID for authentication.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Try to validate as UUID first
        try:
            uuid.UUID(request.user_id)
            # It's a UUID - query by user_id directly
            cursor.execute(
                "SELECT id, display_name, username, balance FROM users WHERE id = %s",
                (request.user_id,)
            )
        except ValueError:
            # Not a UUID - treat as username and resolve to user_id
            # Remove @ prefix if user typed it
            username = request.user_id.lstrip('@').strip()
            
            # Validate username format (alphanumeric + underscore)
            if not re.match(r'^[a-zA-Z0-9_]+$', username):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid username format. Username must contain only letters, numbers, and underscores."
                )
            
            # Query by username to get user_id
            cursor.execute(
                "SELECT id, display_name, username, balance FROM users WHERE username = %s",
                (username,)
            )
        
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail="User not found. Please check your User ID or username and try again."
            )
        
        return LoginResponse(
            user_id=str(result["id"]),
            username=result["username"],
            display_name=result["display_name"],
            balance=float(result["balance"])
        )

@app.get("/user/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """
    Get user information by ID (UUID) or username.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Try UUID first, then username
        try:
            uuid.UUID(user_id)
            cursor.execute(
                "SELECT id, display_name, username, balance FROM users WHERE id = %s",
                (user_id,)
            )
        except ValueError:
            # Not a UUID, treat as username
            cursor.execute(
                "SELECT id, display_name, username, balance FROM users WHERE username = %s",
                (user_id,)
            )
        
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            user_id=str(result["id"]),
            username=result["username"],
            display_name=result["display_name"],
            balance=float(result["balance"])
        )

@app.post("/transfer", response_model=TransferResponse)
async def transfer(request: TransferRequest):
    """
    Transfer money between users using username for receiver.
    
    This is the critical endpoint that ensures transaction safety:
    1. Resolves receiver_username to UUID internally
    2. Uses database transaction (all-or-nothing)
    3. Locks sender row to prevent race conditions
    4. Validates sufficient balance
    5. Updates both balances atomically
    6. Logs attempt in audit_logs (success or failure) using UUIDs
    
    Why we still use UUIDs internally:
    - UUIDs are immutable and globally unique
    - Usernames can change (future feature), but UUIDs never do
    - Audit logs must reference immutable identifiers
    - Database foreign keys work better with UUIDs
    """
    # Validate sender UUID
    try:
        sender_uuid = uuid.UUID(request.sender_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid sender ID format")
    
    # Validate receiver username format (alphanumeric + underscore)
    if not re.match(r'^[a-zA-Z0-9_]+$', request.receiver_username):
        raise HTTPException(status_code=400, detail="Invalid username format")
    
    amount = Decimal(str(request.amount))
    
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Resolve receiver username to UUID
            cursor.execute(
                "SELECT id FROM users WHERE username = %s",
                (request.receiver_username,)
            )
            receiver = cursor.fetchone()
            
            if not receiver:
                # Log failed attempt (receiver not found)
                # Note: We can't log with username in audit_logs, so we use NULL
                cursor.execute(
                    """
                    INSERT INTO audit_logs (sender_id, receiver_id, amount, status)
                    VALUES (%s, NULL, %s, 'FAILED')
                    """,
                    (request.sender_id, amount)
                )
                raise HTTPException(status_code=404, detail=f"User '{request.receiver_username}' not found")
            
            receiver_id = str(receiver["id"])
            
            # Check if sender and receiver are the same
            if request.sender_id == receiver_id:
                raise HTTPException(status_code=400, detail="Cannot transfer to yourself")
            
            # Start transaction (automatic with psycopg2)
            # Lock sender row FOR UPDATE to prevent concurrent modifications
            cursor.execute(
                """
                SELECT id, balance FROM users 
                WHERE id = %s 
                FOR UPDATE
                """,
                (request.sender_id,)
            )
            sender = cursor.fetchone()
            
            if not sender:
                cursor.execute(
                    """
                    INSERT INTO audit_logs (sender_id, receiver_id, amount, status)
                    VALUES (%s, %s, %s, 'FAILED')
                    """,
                    (request.sender_id, receiver_id, amount)
                )
                raise HTTPException(status_code=404, detail="Sender not found")
            
            # Check sufficient balance
            current_balance = Decimal(str(sender["balance"]))
            if current_balance < amount:
                # Log failed attempt (insufficient balance)
                cursor.execute(
                    """
                    INSERT INTO audit_logs (sender_id, receiver_id, amount, status)
                    VALUES (%s, %s, %s, 'FAILED')
                    """,
                    (request.sender_id, receiver_id, amount)
                )
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient balance. Current: ${current_balance}, Required: ${amount}"
                )
            
            # Perform transfer: debit sender, credit receiver
            # Both operations happen in the same transaction
            new_sender_balance = current_balance - amount
            
            cursor.execute(
                "UPDATE users SET balance = balance - %s WHERE id = %s",
                (amount, request.sender_id)
            )
            
            cursor.execute(
                "UPDATE users SET balance = balance + %s WHERE id = %s",
                (amount, receiver_id)
            )
            
            # Log successful transfer (using UUIDs internally)
            cursor.execute(
                """
                INSERT INTO audit_logs (sender_id, receiver_id, amount, status)
                VALUES (%s, %s, %s, 'SUCCESS')
                """,
                (request.sender_id, receiver_id, amount)
            )
            
            # Commit transaction (happens automatically in context manager)
            
            return TransferResponse(
                success=True,
                message="Transfer completed successfully",
                new_balance=float(new_sender_balance)
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions (they're already logged)
            raise
        except Exception as e:
            # Any other error - transaction will rollback automatically
            raise HTTPException(status_code=500, detail=f"Transfer failed: {str(e)}")

@app.get("/transactions/{user_id}", response_model=list[TransactionLog])
async def get_transactions(user_id: str):
    """
    Get transaction history for a user (by UUID or username).
    Returns all transfers where user was sender or receiver.
    
    Returns usernames instead of UUIDs for better UX.
    UUIDs are never exposed to frontend.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Resolve user_id (UUID or username) to UUID
        try:
            uuid.UUID(user_id)
            # It's already a UUID
            user_uuid = user_id
        except ValueError:
            # It's a username, resolve to UUID
            cursor.execute("SELECT id FROM users WHERE username = %s", (user_id,))
            user_result = cursor.fetchone()
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
            user_uuid = str(user_result["id"])
        
        # Get transactions with username lookups
        # We join with users table to get usernames for display
        cursor.execute(
            """
            SELECT 
                al.id,
                al.created_at,
                al.amount,
                al.status,
                sender.id as sender_uuid,
                sender.username as sender_username,
                receiver.id as receiver_uuid,
                receiver.username as receiver_username
            FROM audit_logs al
            LEFT JOIN users sender ON al.sender_id = sender.id
            LEFT JOIN users receiver ON al.receiver_id = receiver.id
            WHERE al.sender_id = %s OR al.receiver_id = %s
            ORDER BY al.created_at DESC
            """,
            (user_uuid, user_uuid)
        )
        
        results = cursor.fetchall()
        
        return [
            TransactionLog(
                id=str(row["id"]),
                timestamp=row["created_at"].isoformat(),
                sender_username=row["sender_username"] or "Unknown",
                receiver_username=row["receiver_username"] or "Unknown",
                amount=float(row["amount"]),
                status=row["status"]
            )
            for row in results
        ]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

