from fastapi import FastAPI, HTTPException
import httpx
import os
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import logging
import hashlib
import time
import asyncpg
import json
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gateway Service with Cache")

# Конфигурация
CALCULUS_SERVICE_URL = os.getenv("CALCULUS_SERVICE_URL", "http://calculus:8001")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/expressions_db")

#Подключение к БД
async def get_db_connection():
    """Создает подключение к БД"""
    return await asyncpg.connect(DATABASE_URL)

# Утилиты
# Хэш выражения
def generate_expression_hash(expression: str) -> str:
    return hashlib.sha256(expression.strip().encode()).hexdigest()

async def get_from_cache(expression: str):
    try:
        expr_hash = generate_expression_hash(expression)
        
        conn = await get_db_connection()
        try:
            # Ищем по хэшу
            cached = await conn.fetchrow(
                """
                SELECT expression, result, calculation_time_ms, created_at 
                FROM expressions_cache 
                WHERE expression_hash = $1
                """,
                expr_hash
            )
            
            if cached:
                # Обновляем информацию о доступе
                await conn.execute(
                    """
                    UPDATE expressions_cache 
                    SET last_accessed = $1, access_count = access_count + 1
                    WHERE expression_hash = $2
                    """,
                    datetime.utcnow(), expr_hash
                )
                
                logger.info(f"Cache HIT for expression: {expression}")
                return {
                    "result": cached["result"],
                    "expression": cached["expression"],
                    "calculation_time_ms": cached["calculation_time_ms"],
                    "cached_at": cached["created_at"]
                }
            
            logger.info(f"Cache MISS for expression: {expression}")
            return None
            
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Error accessing cache: {e}")
        return None

async def save_to_cache(expression: str, result: float, calc_time_ms: int):
    try:
        expr_hash = generate_expression_hash(expression)
        
        conn = await get_db_connection()
        try:
            await conn.execute(
                """
                INSERT INTO expressions_cache 
                (expression_hash, expression, result, calculation_time_ms)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (expression_hash) DO NOTHING
                """,
                expr_hash, expression, result, calc_time_ms
            )
            
            logger.info(f"Saved to cache: {expression} = {result}")
            
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}")

async def log_request(expression: str, from_cache: bool, 
                     response_time_ms: int, status_code: int):
    try:
        conn = await get_db_connection()
        try:
            await conn.execute(
                """
                INSERT INTO request_logs 
                (expression, from_cache, response_time_ms, status_code)
                VALUES ($1, $2, $3, $4)
                """,
                expression, from_cache, response_time_ms, status_code
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Failed to log request: {e}")

# Модели данных
class ExpressionRequest(BaseModel):
    expression: str

class PolynomialRequest(BaseModel):
    coefficients: str
    precision: Optional[int] = 6

# Результаты
class EvaluationResponse(BaseModel):
    result: float
    expression: str
    error: Optional[str] = None
    from_cache: bool = False
    calculation_time_ms: Optional[int] = None
    cache_hit: bool = False
    
    model_config = ConfigDict(extra='ignore')

# Используем dict вместо complex для корней
class PolynomialResponse(BaseModel):
    roots: List[Dict[str, float]]  # [{"real": 1.0, "imag": 0.0}, ...]
    coefficients: List[float]
    polynomial_string: str
    degree: int
    real_roots: List[float]
    complex_roots: List[Dict[str, float]]
    error: Optional[str] = None
    from_cache: bool = False
    calculation_time_ms: Optional[int] = None
    cache_hit: bool = False
    
    model_config = ConfigDict(extra='ignore')

# Обновляем таблицу БД для многочленов
async def init_polynomial_tables():
    conn = await get_db_connection()
    try:
        # Таблица для кэша многочленов
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS polynomials_cache (
                id SERIAL PRIMARY KEY,
                coefficients_hash VARCHAR(64) UNIQUE NOT NULL,
                coefficients TEXT NOT NULL,
                roots_json TEXT NOT NULL,
                degree INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                calculation_time_ms INTEGER
            )
        """)
        
        # Индексы
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_poly_hash 
            ON polynomials_cache(coefficients_hash)
        """)
        
        logger.info("Polynomial tables initialized")
    except Exception as e:
        logger.error(f"Failed to init polynomial tables: {e}")
    finally:
        await conn.close()

@app.on_event("startup")
async def startup_event():
    await init_polynomial_tables()
    logger.info("Gateway with polynomial support started")

# ========== ПОЛИНОМЫ ==========

def generate_polynomial_hash(coefficients: str) -> str:
    return hashlib.sha256(coefficients.strip().encode()).hexdigest()

async def get_polynomial_from_cache(coefficients: str):
    try:
        poly_hash = generate_polynomial_hash(coefficients)
        
        conn = await get_db_connection()
        try:
            cached = await conn.fetchrow(
                """
                SELECT coefficients, roots_json, calculation_time_ms, created_at 
                FROM polynomials_cache 
                WHERE coefficients_hash = $1
                """,
                poly_hash
            )
            
            if cached:
                await conn.execute(
                    """
                    UPDATE polynomials_cache 
                    SET last_accessed = $1, access_count = access_count + 1
                    WHERE coefficients_hash = $2
                    """,
                    datetime.utcnow(), poly_hash
                )
                
                logger.info(f"Polynomial cache HIT for coefficients: {coefficients}")
                
                roots_data = json.loads(cached["roots_json"])
                
                result = {
                    "coefficients": roots_data["coefficients"],
                    "roots": roots_data["roots"],
                    "polynomial_string": roots_data["polynomial_string"],
                    "degree": roots_data["degree"],
                    "real_roots": roots_data["real_roots"],
                    "complex_roots": roots_data["complex_roots"],
                    "from_cache": True,
                    "cache_hit": True,
                    "calculation_time_ms": cached["calculation_time_ms"]
                }
                
                return result
            
            logger.info(f"Polynomial cache MISS for coefficients: {coefficients}")
            return None
            
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Error accessing polynomial cache: {e}")
        return None
        
async def save_polynomial_to_cache(coefficients: str, result_data: dict, calc_time_ms: int):
    try:
        poly_hash = generate_polynomial_hash(coefficients)
        
        roots_for_json = {
            "roots": result_data.get("roots", []),
            "coefficients": result_data.get("coefficients", []),
            "polynomial_string": result_data.get("polynomial_string", ""),
            "degree": result_data.get("degree", 0),
            "real_roots": result_data.get("real_roots", []),
            "complex_roots": result_data.get("complex_roots", [])
        }
        
        conn = await get_db_connection()
        try:
            await conn.execute(
                """
                INSERT INTO polynomials_cache 
                (coefficients_hash, coefficients, roots_json, degree, calculation_time_ms)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (coefficients_hash) DO NOTHING
                """,
                poly_hash,
                json.dumps(result_data["coefficients"]),
                json.dumps(roots_for_json),
                result_data["degree"],
                calc_time_ms
            )
            
            logger.info(f"Saved polynomial to cache: {result_data['polynomial_string']}")
            
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Failed to save polynomial to cache: {e}")

# Новый endpoint для многочленов
@app.post("/polynomial/roots", response_model=PolynomialResponse)
async def solve_polynomial(request: PolynomialRequest):
    """
    Находит корни многочлена с кэшированием
    Пример: {"coefficients": "1|0|8", "precision": 6}
    """
    start_time = time.time()
    
    try:
        coefficients = request.coefficients.strip()
        logger.info(f"Gateway: получены коэффициенты: '{coefficients}'")
        
        if not coefficients:
            raise HTTPException(
                status_code=400,
                detail="Coefficients cannot be empty"
            )
        
        # Шаг 1: Проверяем кэш
        cached_result = await get_polynomial_from_cache(coefficients)
        
        if cached_result:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            await log_request(f"POLY:{coefficients}", True, response_time_ms, 200)
            
            return PolynomialResponse(**cached_result)
        
        # Шаг 2: Если нет в кэше - вычисляем через Calculus
        logger.info(f"Gateway: многочлен не найден в кэше, вычисляем...")
        calc_start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CALCULUS_SERVICE_URL}/polynomial/roots",
                json={
                    "coefficients": coefficients,
                    "precision": request.precision
                }
            )
            
            calc_time_ms = int((time.time() - calc_start_time) * 1000)
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Шаг 3: Сохраняем результат в кэш
                await save_polynomial_to_cache(coefficients, result_data, calc_time_ms)
                
                response_time_ms = int((time.time() - start_time) * 1000)
                
                await log_request(f"POLY:{coefficients}", False, response_time_ms, 200)
                
                # Создаем финальный ответ, добавляя информацию о кэше
                final_result = {
                    **result_data,
                    "from_cache": False,
                    "cache_hit": False,
                    "calculation_time_ms": calc_time_ms
                }
                
                return PolynomialResponse(**final_result)
            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", str(error_data))
                except:
                    error_message = response.text
                
                logger.error(f"Gateway: ошибка от calculus service: {error_message}")
                
                response_time_ms = int((time.time() - start_time) * 1000)
                await log_request(f"POLY:{coefficients}", False, response_time_ms, response.status_code)
                
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_message
                )
                
    except httpx.RequestError as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_request(f"POLY:{request.coefficients}", False, response_time_ms, 503)
        
        logger.error(f"Gateway: ошибка соединения с calculus service: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to calculus service: {str(e)}"
        )
    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_request(f"POLY:{request.coefficients}", False, response_time_ms, 500)
        
        logger.error(f"Gateway: внутренняя ошибка: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Основной endpoint для выражений
@app.post("/calculate", response_model=EvaluationResponse)
async def calculate_expression(request: ExpressionRequest):
    start_time = time.time()
    
    try:
        expression = request.expression.strip()
        logger.info(f"Gateway: получено выражение: '{expression}'")
        
        if not expression:
            raise HTTPException(
                status_code=400,
                detail="Expression cannot be empty"
            )
        
        # Шаг 1: Проверяем кэш
        cached_result = await get_from_cache(expression)
        
        if cached_result:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            await log_request(expression, True, response_time_ms, 200)
            
            return EvaluationResponse(
                result=cached_result["result"],
                expression=cached_result["expression"],
                from_cache=True,
                cache_hit=True,
                calculation_time_ms=cached_result["calculation_time_ms"]
            )
        
        # Шаг 2: Если нет в кэше - вычисляем через Calculus
        logger.info(f"Gateway: выражение '{expression}' не найдено в кэше, вычисляем...")
        calc_start_time = time.time()
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{CALCULUS_SERVICE_URL}/evaluate",
                json={"expression": expression}
            )
            
            calc_time_ms = int((time.time() - calc_start_time) * 1000)
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Шаг 3: Сохраняем результат в кэш
                await save_to_cache(expression, result_data["result"], calc_time_ms)
                
                response_time_ms = int((time.time() - start_time) * 1000)
                
                await log_request(expression, False, response_time_ms, 200)
                
                return EvaluationResponse(
                    result=result_data["result"],
                    expression=result_data["expression"],
                    from_cache=False,
                    cache_hit=False,
                    calculation_time_ms=calc_time_ms
                )
            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", str(error_data))
                except:
                    error_message = response.text
                
                logger.error(f"Gateway: ошибка от calculus service: {error_message}")
                
                response_time_ms = int((time.time() - start_time) * 1000)
                await log_request(expression, False, response_time_ms, response.status_code)
                
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_message
                )
                
    except httpx.RequestError as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_request(request.expression, False, response_time_ms, 503)
        
        logger.error(f"Gateway: ошибка соединения с calculus service: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to calculus service: {str(e)}"
        )

# Дополнительные endpoints
@app.get("/cache/stats")
async def get_cache_stats():
    try:
        conn = await get_db_connection()
        try:
            # Статистика выражений
            expr_stats = await conn.fetchrow(
                "SELECT COUNT(*) as count, SUM(access_count) as accesses FROM expressions_cache"
            )
            
            # Статистика многочленов
            poly_stats = await conn.fetchrow(
                "SELECT COUNT(*) as count, SUM(access_count) as accesses FROM polynomials_cache"
            )
            
            # Популярные выражения
            popular_expr = await conn.fetch(
                """
                SELECT expression, access_count 
                FROM expressions_cache 
                ORDER BY access_count DESC 
                LIMIT 5
                """
            )
            
            # Популярные многочлены
            popular_poly = await conn.fetch(
                """
                SELECT coefficients, access_count 
                FROM polynomials_cache 
                ORDER BY access_count DESC 
                LIMIT 5
                """
            )
            
            return {
                "expressions_cache": {
                    "total_entries": expr_stats["count"] or 0,
                    "total_accesses": expr_stats["accesses"] or 0
                },
                "polynomials_cache": {
                    "total_entries": poly_stats["count"] or 0,
                    "total_accesses": poly_stats["accesses"] or 0
                },
                "most_popular_expressions": [
                    {"expression": p["expression"], "access_count": p["access_count"]}
                    for p in popular_expr
                ],
                "most_popular_polynomials": [
                    {"coefficients": json.loads(p["coefficients"]), "access_count": p["access_count"]}
                    for p in popular_poly
                ]
            }
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        # Проверяем доступность calculus
        async with httpx.AsyncClient(timeout=2.0) as client:
            calculus_response = await client.get(f"{CALCULUS_SERVICE_URL}/health")
            calculus_health = calculus_response.json() if calculus_response.status_code == 200 else {"status": "unreachable"}
        
        # Проверяем доступность БД
        conn = await get_db_connection()
        await conn.close()
        db_status = "healthy"
        
        return {
            "status": "healthy",
            "service": "gateway",
            "database": db_status,
            "calculus_service": calculus_health
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "gateway",
            "error": str(e)
        }

@app.get("/")
async def root():
    return {
        "service": "Gateway API with Polynomial Support",
        "description": "Принимает выражения и многочлены, проверяет кэш в БД",
        "endpoints": {
            "POST /calculate": "Вычислить математическое выражение",
            "POST /polynomial/roots": "Найти корни многочлена",
            "GET /cache/stats": "Статистика кэша",
            "GET /health": "Проверка здоровья"
        },
        "polynomial_format": "коэффициенты через '|', например: '1|0|8' для x^2 + 8",
        "max_polynomial_degree": 25
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)