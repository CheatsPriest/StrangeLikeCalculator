from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
import logging
import os
import re
from typing import Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Math Calculator Web Interface")
templates = Jinja2Templates(directory="app/templates")

# Конфигурация
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")

def parse_polynomial(poly_str: str) -> Optional[str]:
    """
    Парсит строку многочлена
    Возвращает коэффициенты в формате 'an|...|a1|a0' (от младшей к старшей)
    """
    try:
        poly_str = poly_str.strip().replace(' ', '').replace('*', '').lower()
        
        if not poly_str:
            return None
        
        if '|' in poly_str:
            return poly_str
        
        if poly_str in ['0', '0.0']:
            return '0'
        
        if poly_str[0] not in '+-':
            poly_str = '+' + poly_str
        
        # Разбиваем на члены
        terms = []
        current = ''
        for i, char in enumerate(poly_str):
            if char in '+-' and i > 0:
                if current:
                    terms.append(current)
                current = char
            else:
                current += char
        if current:
            terms.append(current)
        
        # Собираем коэффициенты
        coeff_dict = {}
        for term in terms:
            if not term or term == '+':
                continue
                
            sign = 1 if term[0] == '+' else -1
            term = term[1:]
            
            if not term or term == '0':
                continue
            
            # Определяем степень
            if 'x' in term:
                if '^' in term:
                    # ax^b
                    if 'x^' in term:
                        coeff_part, degree_part = term.split('x^')
                    else:
                        # Может быть формата x2 или x3
                        x_pos = term.find('x')
                        coeff_part = term[:x_pos]
                        degree_part = term[x_pos+1:]
                    
                    coeff = float(coeff_part) if coeff_part not in ['', '+', '-'] else 1.0
                    degree = int(degree_part)
                else:
                    # ax
                    coeff_part = term.replace('x', '')
                    coeff = float(coeff_part) if coeff_part not in ['', '+', '-'] else 1.0
                    degree = 1
            else:
                # Константа
                coeff = float(term)
                degree = 0
            
            coeff_dict[degree] = coeff_dict.get(degree, 0.0) + sign * coeff
        
        if not coeff_dict:
            return '0'
        
        # Создаем массив от старшей к младшей
        max_degree = max(coeff_dict.keys())
        coeff_array = []
        for degree in range(max_degree, -1, -1):
            coeff = coeff_dict.get(degree, 0.0)
            coeff_array.append(coeff)
        
        # Убираем ведущие нули
        while len(coeff_array) > 1 and abs(coeff_array[0]) < 1e-12:
            coeff_array.pop(0)
        
        # Форматируем
        formatted = []
        for c in coeff_array:
            if abs(c - int(c)) < 1e-12:
                formatted.append(str(int(c)))
            else:
                formatted.append(str(round(c, 10)).rstrip('0').rstrip('.'))
        
        return '|'.join(formatted)
        
    except Exception as e:
        logger.error(f"Ошибка парсинга: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate")
async def calculate(
    request: Request,
    calculation_type: str = Form(...),
    expression: str = Form(...)
):
    try:
        logger.info(f"Web: получен запрос type={calculation_type}, expr={expression}")
        
        if calculation_type == "expression":
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{GATEWAY_URL}/calculate",
                    json={"expression": expression}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "type": "expression",
                        "result": result["result"],
                        "expression": result["expression"],
                        "from_cache": result.get("from_cache", False),
                        "calculation_time_ms": result.get("calculation_time_ms", 0)
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Ошибка от gateway: {response.text}",
                        "status_code": response.status_code
                    }
                    
        elif calculation_type == "polynomial":
            try:
                coefficients = parse_polynomial(expression)
                if not coefficients:
                    return {
                        "success": False,
                        "error": "Неверный формат многочлена"
                    }
                
                logger.info(f"Web: распарсенные коэффициенты: {coefficients}")
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{GATEWAY_URL}/polynomial/roots",
                        json={"coefficients": coefficients, "precision": 6}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Форматируем корни для отображения
                        formatted_roots = []
                        for i, root in enumerate(result["roots"], 1):
                            real = root["real"]
                            imag = root["imag"]
                            
                            if abs(imag) < 1e-10:
                                formatted_roots.append(f"x{i} = {real:.6f}")
                            else:
                                if imag >= 0:
                                    formatted_roots.append(f"x{i} = {real:.6f} + {imag:.6f}i")
                                else:
                                    formatted_roots.append(f"x{i} = {real:.6f} - {abs(imag):.6f}i")
                        
                        return {
                            "success": True,
                            "type": "polynomial",
                            "polynomial": result["polynomial_string"],
                            "degree": result["degree"],
                            "roots": formatted_roots,
                            "real_roots": result["real_roots"],
                            "complex_roots": len(result["complex_roots"]),
                            "from_cache": result.get("from_cache", False),
                            "calculation_time_ms": result.get("calculation_time_ms", 0)
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Ошибка от gateway: {response.text}",
                            "status_code": response.status_code
                        }
                        
            except ValueError as e:
                return {
                    "success": False,
                    "error": str(e)
                }
                
        else:
            return {
                "success": False,
                "error": f"Неизвестный тип вычисления: {calculation_type}"
            }
            
    except httpx.RequestError as e:
        logger.error(f"Web: ошибка соединения с gateway: {e}")
        return {
            "success": False,
            "error": f"Ошибка соединения с сервером: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Web: внутренняя ошибка: {e}")
        return {
            "success": False,
            "error": f"Внутренняя ошибка сервера: {str(e)}"
        }

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            gateway_response = await client.get(f"{GATEWAY_URL}/health")
            gateway_health = gateway_response.json() if gateway_response.status_code == 200 else {"status": "unreachable"}
        
        return {
            "status": "healthy",
            "service": "web-ui",
            "gateway": gateway_health
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "web-ui",
            "error": str(e)
        }

@app.get("/api/examples")
async def get_examples():
    return {
        "expressions": [
            "2+2",
            "3+5*6+(55-6)",
            "(2+3)*(4+5)",
            "10/2",
            "2**3 + 4",
            "17 % 5",
            "sin(10)+log10(e)+exp(pi)",
            "sin(tan(sqrt(abs(-100))))"
        ],
        "polynomials": [
            "x^2 + 8",
            "x^2 - 4",
            "3x^3 - 2x^2 + x - 1",
            "x^4 - 1",
            "2x^5 + 3x^3 - x",
            "x^2 + x + 1",
            "x^2-2x+1"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)