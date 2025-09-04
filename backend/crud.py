from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime

from backend import models, schemas


# -------------------------------------------------------------------
# 내부 유틸: SQLAlchemy 모델 컬럼 키만 남기는 필터
# - 스키마가 더 많은 필드를 가지더라도 모델에 존재하는 컬럼만 insert/update
# - 예: DocumentCreate에 title/meta가 있어도, 모델에 없으면 자동 제외
# -------------------------------------------------------------------
def _only_model_fields(model_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    model_cols = set(model_cls.__table__.columns.keys())
    return {k: v for k, v in payload.items() if k in model_cols}


# =========================
# 사용자
# =========================
def create_user(db: Session, user: schemas.UserCreate):
    # pydantic -> dict 변환 시 None/미설정 필드 제외
    data = user.dict(exclude_unset=True, exclude_none=True)
    db_user = models.User(**_only_model_fields(models.User, data))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# =========================
# 문서
# =========================
def create_document(db: Session, doc: schemas.DocumentCreate):
    """
    Document 모델의 컬럼 예시(가정):
      id, user_id, filename, file_path, summary, domain, uploaded_at, (선택)title
    - schemas.DocumentCreate는 title/meta 같은 확장 필드를 가질 수 있으므로,
      모델에 없는 키는 자동으로 필터링해서 주입한다.
    """
    data = doc.dict(exclude_unset=True, exclude_none=True)
    # uploaded_at이 모델에서 default가 없으면 여기서 채워줌
    if "uploaded_at" in models.Document.__table__.columns.keys() and "uploaded_at" not in data:
        data["uploaded_at"] = datetime.utcnow()
    db_doc = models.Document(**_only_model_fields(models.Document, data))
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


# =========================
# QA 히스토리
# =========================
def save_qa_history(db: Session, qa: schemas.QACreate):
    """
    QACreate는 question/answer를 정규 필드로, 과거 호환 alias(user_input/ai_answer)도 허용.
    - dict(by_alias=False)로 뽑아도 되고, populate_by_name=True 덕분에 question/answer 사용 가능.
    - 모델 컬럼 예시: id, document_id, question, answer, created_at
    """
    # by_alias=False: 정규 키(question/answer) 기준
    data = qa.dict(by_alias=False, exclude_unset=True, exclude_none=True)
    if "created_at" in models.QAHistory.__table__.columns.keys() and "created_at" not in data:
        data["created_at"] = datetime.utcnow()
    db_qa = models.QAHistory(**_only_model_fields(models.QAHistory, data))
    db.add(db_qa)
    db.commit()
    db.refresh(db_qa)
    return db_qa


def get_qa_by_document(db: Session, document_id: int):
    return (
        db.query(models.QAHistory)
        .filter(models.QAHistory.document_id == document_id)
        .order_by(models.QAHistory.created_at.asc())
        .all()
    )


def get_document_by_id(db: Session, document_id: int):
    return db.query(models.Document).filter(models.Document.id == document_id).first()
