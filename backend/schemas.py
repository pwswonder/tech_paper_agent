from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any


# =========================
# 사용자 스키마
# =========================
class UserBase(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# =========================
# 문서 스키마
# =========================
# 👉 실제 운영 컬럼과 맞춤: filename/summary/domain/file_path/uploaded_at
#    title은 메타 목적(optional)로 유지 (file_reader에서 meta.title 공급 가능)
class DocumentBase(BaseModel):
    user_id: int
    filename: str  # 실제 파일명 (예: paper.pdf)
    file_path: str  # 서버 저장 경로 (예: uploaded_docs/paper.pdf)
    title: Optional[str] = None  # 메타/표시용 (없어도 무방)
    summary: Optional[str] = None  # 분석 결과
    domain: Optional[str] = None  # 분석 결과(기술 도메인)
    meta: Optional[Dict[str, Any]] = None  # 확장 메타 (checksum/page_count 등)


class DocumentCreate(DocumentBase):
    """
    POST /documents/analyze_only 에서 DB에 저장할 때 사용.
    - 분석 결과(summary/domain)와 파일 메타를 포함할 수 있도록 확장.
    """

    pass


class Document(DocumentBase):
    id: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


# 👉 GET /documents 용 가벼운 응답 스키마 (app.py가 기대하는 필드만)
class DocumentListItem(BaseModel):
    id: int
    filename: str
    domain: Optional[str] = None
    summary: Optional[str] = None
    uploaded_at: datetime

    # [ADDED] app.py의 레거시 코드 표시 호환 필드
    base_code: Optional[str] = None

    # [ADDED] app.py의 Base code 아티팩트 섹션과 정합을 맞추기 위한 선택 필드들
    basecode_py_path: Optional[str] = None
    basecode_source: Optional[str] = None
    basecode_summary: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    spec_warnings: Optional[List[str]] = None
    basecode_error: Optional[str] = None

    class Config:
        from_attributes = True


# =========================
# QA 스키마
# =========================
# 👉 정규 필드는 question/answer로 통일.
#    (과거 라우터가 user_input/ai_answer로 보낼 수도 있으므로 alias 허용)
class QACreate(BaseModel):
    document_id: int
    question: str = Field(..., alias="user_input")
    answer: str = Field(..., alias="ai_answer")

    class Config:
        from_attributes = True
        populate_by_name = (
            True  # question/answer로도, user_input/ai_answer로도 받아줌(호환)
        )


class QAHistory(BaseModel):
    id: int
    document_id: int
    question: str
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True


# 👉 app.py의 히스토리 렌더는 question/answer/created_at만 사용
class QAHistoryOut(BaseModel):
    question: str
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True


# =========================
# 요청/응답 스키마
# =========================
class ExistingDocQARequest(BaseModel):
    document_id: int
    question: str


# [ADDED] /documents/analyze_only 응답 스키마
# - app.py 업로드 성공 후 result에서 사용하는 필드(document_id, base_code, 및 신규 아티팩트)와 정합
class AnalyzeOnlyResponse(BaseModel):  # [ADDED]
    document_id: int
    # 레거시 호환
    base_code: Optional[str] = None
    # 신규 아티팩트 (있으면 채워서 내려줌)
    basecode_py_path: Optional[str] = None
    basecode_source: Optional[str] = None
    basecode_summary: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    spec_warnings: Optional[List[str]] = None
    basecode_error: Optional[str] = None
