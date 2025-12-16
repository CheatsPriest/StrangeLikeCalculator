from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import logging
import numpy as np
from numpy.polynomial import Polynomial
import sympy as sp
import json
import ast
import operator
import math

coef_limit: int = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Calculus Service")

#Модели данных
class ExpressionRequest(BaseModel):
    expression: str

class PolynomialRequest(BaseModel):
    coefficients: str  #Формат: "a0|a1|a2|...|an" для a0 + a1*x + a2*x^2 + ... + an*x^n
    precision: Optional[int] = 6  #Точность вычисления корней

class EvaluationResult(BaseModel):
    result: float
    expression: str
    error: Optional[str] = None
    
    model_config = ConfigDict(extra='ignore')

#Используем dict вместо complex для корней
class PolynomialResult(BaseModel):
    roots: List[Dict[str, float]]  #[{"real": 1.0, "imag": 0.0}, ...]
    coefficients: List[float]
    polynomial_string: str
    degree: int
    real_roots: List[float]
    complex_roots: List[Dict[str, float]]
    error: Optional[str] = None
    
    model_config = ConfigDict(extra='ignore')

# ========== МАТЕМАТИЧЕСКИЕ ВЫРАЖЕНИЯ ==========
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

class SafeEval(ast.NodeTransformer):
    def visit_Name(self, node):
        allowed_names = {'pi', 'e'}
        if node.id in allowed_names:
            return node
        raise ValueError(f"Использование переменных запрещено: {node.id}")
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            allowed_functions = {
                'abs', 'round', 
                'sin', 'cos', 'tan',
                'asin', 'acos', 'atan',
                'sqrt', 'exp', 'log', 'log10', 'log2',
                'ceil', 'floor', 'trunc',
                'degrees', 'radians',
                'sinh', 'cosh', 'tanh',
            }
            if node.func.id not in allowed_functions:
                raise ValueError(f"Функция '{node.func.id}' не разрешена")
            
            for arg in node.args:
                self.visit(arg)
            return node
        raise ValueError(f"Неподдерживаемый вызов функции: {ast.dump(node)}")

def safe_eval(expr: str) -> float:
    try:
        # Убираем пробелы
        expr = expr.strip()
        
        if not expr:
            raise ValueError("Пустое выражение")
        
        tree = ast.parse(expr, mode='eval')
        
        # Проверяем безопасность
        SafeEval().visit(tree)
        
        # Компилируем
        code = compile(tree, '<string>', 'eval')
        
        # Словарь с безопасными функциями
        safe_globals = {
            '__builtins__': {},
            # Операторы уже встроены в eval через ast
            # Константы
            'pi': math.pi,
            'e': math.e,
            # Математические функции
            'abs': abs,
            'round': round,
            # Тригонометрия
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            # Гиперболические
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            # Экспоненты и логарифмы
            'exp': math.exp,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            # Округление
            'ceil': math.ceil,
            'floor': math.floor,
            'trunc': math.trunc,
            # Преобразование углов
            'degrees': math.degrees,
            'radians': math.radians,
        }
        
        result = eval(code, safe_globals, {})
        
        if not isinstance(result, (int, float)):
            raise ValueError(f"Результат должен быть числом, получено: {type(result)}")
        
        if math.isnan(result):
            raise ValueError("Результат - NaN (не число)")
        if math.isinf(result):
            raise ValueError("Результат - бесконечность")
        
        return float(result)
        
    except SyntaxError as e:
        raise ValueError(f"Синтаксическая ошибка в выражении: {e}")
    except ZeroDivisionError:  # ← ДОБАВЬТЕ ЭТУ СТРОЧКУ!
        raise ValueError("Деление на ноль")  # ← И ЭТУ!
    except MemoryError:
        raise ValueError("Выражение слишком сложное")
    except Exception as e:
        raise ValueError(f"Ошибка вычисления: {str(e)}")

# ========== МНОГОЧЛЕНЫ ==========

def parse_coefficients(coeff_str: str) -> List[float]:
    try:
        coeffs = [float(c.strip()) for c in coeff_str.split('|') if c.strip()]
        
        #while len(coeffs) > 1 and abs(coeffs[-1]) < 1e-12:
            #coeffs.pop()
        
        if not coeffs:
            raise ValueError("Нет коэффициентов")
            
        if len(coeffs) - 1 > coef_limit:
            raise ValueError(f"Степень многочлена ({len(coeffs)-1}) превышает максимальную ({coef_limit})")
            
        return coeffs
        
    except ValueError as e:
        raise ValueError(f"Ошибка парсинга коэффициентов: {e}")

def complex_to_dict(z: complex) -> Dict[str, float]:
    return {"real": float(z.real), "imag": float(z.imag)}

def find_polynomial_roots(coefficients: List[float], precision: int = 6) -> Dict[str, Any]:
    try:
        p = Polynomial(coefficients[::-1])  # numpy использует обратный порядок
        
        degree = len(coefficients) - 1
        
        # Находим корни
        all_roots = p.roots()
        
        if len(coefficients) > 0 and abs(coefficients[-1]) < 1e-10:
            all_roots = np.append(all_roots, 0.0)
        
        all_roots = [complex(round(root.real, precision), round(root.imag, precision)) 
                    for root in all_roots]
        unique_roots = []
        for root in all_roots:
            is_duplicate = False
            for unique_root in unique_roots:
                if (abs(root.real - unique_root.real) < 1e-8 and 
                    abs(root.imag - unique_root.imag) < 1e-8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_roots.append(root)
        
        all_roots = unique_roots
        
        real_roots = [float(root.real) for root in all_roots if abs(root.imag) < 1e-10]
        complex_roots = [root for root in all_roots if abs(root.imag) >= 1e-10]
        
        # Строим строковое представление
        terms = []
        for i, coeff in enumerate(coefficients):
            if abs(coeff) > 1e-12:
                coeff_str = str(coeff)
                if i == 0:
                    terms.append(coeff_str)
                elif i == 1:
                    terms.append(f"{coeff_str}x")
                else:
                    terms.append(f"{coeff_str}x^{i}")
        
        poly_str = " + ".join(terms).replace("+ -", "- ")
        
        roots_dict = [complex_to_dict(root) for root in all_roots]
        complex_roots_dict = [complex_to_dict(root) for root in complex_roots]
        
        return {
            "roots": roots_dict,
            "coefficients": coefficients,
            "polynomial_string": poly_str,
            "degree": degree,
            "real_roots": real_roots,
            "complex_roots": complex_roots_dict
        }
        
    except Exception as e:
        raise ValueError(f"Ошибка вычисления корней: {e}")

# ========== ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "service": "Calculus Service",
        "description": "Вычисляет математические выражения и корни многочленов",
        "endpoints": {
            "POST /evaluate": "Вычислить математическое выражение",
            "POST /polynomial/roots": "Найти корни многочлена",
            "GET /test/{expression}": "Тест выражения",
            "GET /polynomial/test/{coefficients}": "Тест многочлена"
        },
        "polynomial_format": "коэффициенты через '|', например: '8|0|1' для x^2 + 8",
        "max_degree": coef_limit
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "calculus",
        "supported_features": ["expressions", "polynomial_roots", f"max_degree_{coef_limit}"]
    }

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_expression(request: ExpressionRequest):
    try:
        logger.info(f"Calculus: вычисление выражения: {request.expression}")
        
        if not request.expression or not request.expression.strip():
            raise ValueError("Выражение не может быть пустым")
        
        result = safe_eval(request.expression)
        
        logger.info(f"Calculus: результат: {request.expression} = {result}")
        
        return EvaluationResult(
            result=float(result),
            expression=str(request.expression),
            error=None
        )
        
    except ValueError as e:
        logger.warning(f"Calculus: ошибка в выражении '{request.expression}': {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e) 
        )
    except HTTPException:
        raise  
    except Exception as e:
        error_msg = str(e)
        if not error_msg:
            error_msg = f"Неизвестная ошибка: {type(e).__name__}"
        
        logger.error(f"Calculus: неожиданная ошибка: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )

@app.post("/polynomial/roots", response_model=PolynomialResult)
async def solve_polynomial(request: PolynomialRequest):
    try:
        logger.info(f"Calculus: получены коэффициенты: '{request.coefficients}'")
        
        coeffs = parse_coefficients(request.coefficients)
        
        logger.info(f"Calculus: коэффициенты: {coeffs}, степень: {len(coeffs)-1}")
        
        result = find_polynomial_roots(coeffs, request.precision)
        
        logger.info(f"Calculus: найдено {len(result['roots'])} корней")
        
        return PolynomialResult(**result)
        
    except ValueError as e:
        logger.warning(f"Calculus: ошибка в полиноме: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Calculus: неожиданная ошибка: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Тестовые endpoints
@app.get("/test/{expression}")
async def test_evaluation(expression: str):
    try:
        result = safe_eval(expression)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }

@app.get("/polynomial/test/{coefficients}")
async def test_polynomial(coefficients: str):
    try:
        coeffs = parse_coefficients(coefficients)
        result = find_polynomial_roots(coeffs)
        
        return {
            "coefficients": coeffs,
            "degree": len(coeffs) - 1,
            "roots_count": len(result["roots"]),
            "real_roots": result["real_roots"],
            "complex_roots_count": len(result["complex_roots"]),
            "success": True
        }
    except Exception as e:
        return {
            "coefficients": coefficients,
            "error": str(e),
            "success": False
        }

@app.get("/test-division")
async def test_division():
    test_cases = [
        ("10/0", "Должен вернуть 'Деление на ноль'"),
        ("424/0", "Должен вернуть 'Деление на ноль'"),
        ("(2+3)*(4+5)*55/0", "Должен вернуть 'Деление на ноль'"),
        ("10/2", "Должен вернуть 5.0"),
    ]
    results = []
    for expr, expected in test_cases:
        try:
            result = safe_eval(expr)
            results.append(f"{expr} -> {result} ({expected})")
        except ValueError as e:
            results.append(f"{expr} -> ValueError: {str(e)} ({expected})")
        except Exception as e:
            results.append(f"{expr} -> {type(e).__name__}: {str(e)} ({expected})")
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)