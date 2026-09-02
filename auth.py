"""
auth.py
--------
OPTIONAL JWT role-based authentication. NOT applied to any existing
endpoint by default - your /check, /history, /score, etc. all keep
working exactly as before, with zero login required, until you
deliberately add the `Depends(require_role(...))` guard shown below.

Roles: "admin", "inspector", "viewer"

HOW TO ADOPT (when ready):
  1. `pip install python-jose[cryptography] passlib[bcrypt]`
  2. Set env var JWT_SECRET_KEY to a real random secret in production
     (a default is provided below ONLY so local dev doesn't crash -
     do not ship that default).
  3. In main.py, import what you need:
         from auth import require_role, create_access_token, router as auth_router
         app.include_router(auth_router)
  4. Protect a specific endpoint by adding a Depends, e.g. to restrict
     dashboard stats to inspectors and admins only:
         @app.get("/dashboard/stats")
         async def get_dashboard_stats(user=Depends(require_role("inspector", "admin"))):
             ...
     Every endpoint you DON'T add this to stays open, exactly as it is
     in the current working app - so you can roll this out one route
     at a time instead of all-or-nothing.
"""

import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-only-change-me-before-deployment")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours - long enough for a demo/inspection shift

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

router = APIRouter(prefix="/auth", tags=["auth"])


# --- Demo user store -------------------------------------------------
# Replace with real lookups against the `users` table (see
# migrations/001_initial_schema.sql) once db/database.py is adopted.
# Passwords below are bcrypt hashes of: admin123 / inspect123 / view123
_DEMO_USERS = {
    "admin":     {"role": "admin",     "hashed_password": pwd_context.hash("admin123")},
    "inspector": {"role": "inspector", "hashed_password": pwd_context.hash("inspect123")},
    "viewer":    {"role": "viewer",    "hashed_password": pwd_context.hash("view123")},
}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


def create_access_token(username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = _DEMO_USERS.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")
    token = create_access_token(form_data.username, user["role"])
    return TokenResponse(access_token=token, role=user["role"])


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username, role = payload.get("sub"), payload.get("role")
        if username is None:
            raise JWTError()
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return {"username": username, "role": role}


def require_role(*allowed_roles: str):
    """Use as a dependency: Depends(require_role('admin', 'inspector'))"""

    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{user['role']}' is not permitted - requires one of {allowed_roles}",
            )
        return user

    return checker
