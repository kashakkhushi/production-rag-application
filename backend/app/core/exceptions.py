from fastapi import Request
from fastapi.responses import JSONResponse

class RAGException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

class DocumentProcessingError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, status_code=422)

class RetrievalError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message}
    )
