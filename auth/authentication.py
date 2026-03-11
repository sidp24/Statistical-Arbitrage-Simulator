import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple
import bcrypt
import jwt
from functools import wraps

from config import SECRET_KEY, AUTH_ENABLED
from database.models import User, get_db, Session


class AuthenticationError(Exception):
    pass


class PasswordHasher:
    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


class TokenManager:
    def __init__(self, secret_key: str = SECRET_KEY, expiry_hours: int = 24):
        self.secret_key = secret_key
        self.expiry_hours = expiry_hours
        self.algorithm = "HS256"
    
    def create_token(self, user_id: int, email: str) -> str:
        payload = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.utcnow() + timedelta(hours=self.expiry_hours),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        payload = self.verify_token(token)
        if payload:
            return self.create_token(payload["user_id"], payload["email"])
        return None


class AuthService:
    def __init__(self):
        self.hasher = PasswordHasher()
        self.token_manager = TokenManager()
    
    def register_user(
        self,
        db: Session,
        email: str,
        username: str,
        password: str
    ) -> Tuple[User, str]:
        # Check if user exists
        existing = db.query(User).filter(
            (User.email == email) | (User.username == username)
        ).first()
        
        if existing:
            if existing.email == email:
                raise AuthenticationError("Email already registered")
            raise AuthenticationError("Username already taken")
        
        # Create user
        hashed_password = self.hasher.hash_password(password)
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
        )
        db.add(user)
        db.flush()
        
        # Generate token
        token = self.token_manager.create_token(user.id, user.email)
        
        return user, token
    
    def login(
        self,
        db: Session,
        email: str,
        password: str
    ) -> Tuple[User, str]:
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        if not self.hasher.verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")
        
        if not user.is_active:
            raise AuthenticationError("Account is deactivated")
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        # Generate token
        token = self.token_manager.create_token(user.id, user.email)
        
        return user, token
    
    def get_user_from_token(self, db: Session, token: str) -> Optional[User]:
        payload = self.token_manager.verify_token(token)
        if not payload:
            return None
        
        return db.query(User).filter(User.id == payload["user_id"]).first()
    
    def change_password(
        self,
        db: Session,
        user_id: int,
        old_password: str,
        new_password: str
    ) -> bool:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise AuthenticationError("User not found")
        
        if not self.hasher.verify_password(old_password, user.hashed_password):
            raise AuthenticationError("Current password is incorrect")
        
        user.hashed_password = self.hasher.hash_password(new_password)
        return True
    
    def reset_password_token(self, db: Session, email: str) -> Optional[str]:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return None
        
        # Generate a short-lived token (1 hour)
        token_manager = TokenManager(expiry_hours=1)
        return token_manager.create_token(user.id, user.email)


# Streamlit session helpers
def init_session_state():
    import streamlit as st
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'token' not in st.session_state:
        st.session_state.token = None


def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import streamlit as st
        
        if not AUTH_ENABLED:
            return func(*args, **kwargs)
        
        init_session_state()
        
        if not st.session_state.authenticated:
            st.warning("Please log in to access this feature.")
            show_login_form()
            return None
        
        return func(*args, **kwargs)
    
    return wrapper


def show_login_form():
    import streamlit as st
    
    auth_service = AuthService()
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                try:
                    with get_db() as db:
                        user, token = auth_service.login(db, email, password)
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.token = token
                        st.success("Logged in successfully!")
                        st.rerun()
                except AuthenticationError as e:
                    st.error(str(e))
    
    with tab2:
        with st.form("register_form"):
            email = st.text_input("Email", key="reg_email")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password", key="reg_pass")
            password_confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if password != password_confirm:
                    st.error("Passwords do not match")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        with get_db() as db:
                            user, token = auth_service.register_user(db, email, username, password)
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.session_state.token = token
                            st.success("Account created successfully!")
                            st.rerun()
                    except AuthenticationError as e:
                        st.error(str(e))


def logout():
    import streamlit as st
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.token = None


# Singleton auth service
auth_service = AuthService()
