from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from backend import models, schemas, crud
from backend.database import SessionLocal, engine, Base
from typing import List
from backend.routes import qa, document, user

from dotenv import load_dotenv
load_dotenv() 

# 테이블 생성
models.Base.metadata.create_all(bind=engine)

app = FastAPI()
# app.include_router(qa.router)  # ← 추가
app.include_router(document.router, prefix="")
app.include_router(qa.router, prefix="")
app.include_router(user.router)


# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 사용자 등록
@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db, user)



# 문서별 QA 내역 조회
@app.get("/qa/{document_id}", response_model=List[schemas.QAHistory])
def get_qa_list(document_id: int, db: Session = Depends(get_db)):
    return crud.get_qa_by_document(db, document_id)
