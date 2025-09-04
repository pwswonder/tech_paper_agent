from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# 환경변수에서 DB 접속 URL 불러오기
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not POSTGRES_URL:
    raise ValueError("⚠️ POSTGRES_URL 환경변수가 설정되지 않았습니다.")

# SQLAlchemy 엔진 생성
engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# ✅ get_db 함수 정의
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
