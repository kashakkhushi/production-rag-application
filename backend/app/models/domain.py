from datetime import datetime
import uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime

class Base(DeclarativeBase):
    pass

class DocumentDB(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="processing")  # processing, indexed, failed
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chunk_count: Mapped[int] = mapped_column(default=0)
    error_message: Mapped[str] = mapped_column(String, nullable=True)
