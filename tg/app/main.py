import asyncio
import logging
import os
import hashlib
import requests
import re
from typing import Optional
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import asyncio

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не установлен в переменных окружения")

GATEWAY_URL = "http://gateway:8000"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

class CalcStates(StatesGroup):
    waiting_for_type = State()
    waiting_for_expression = State()

def parse_polynomial(poly_str: str) -> Optional[str]:
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

def create_type_keyboard():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Математическое выражение", callback_data="type_expression")],
        [InlineKeyboardButton(text="Корни многочлена", callback_data="type_polynomial")],
        [InlineKeyboardButton(text="ℹ️ Help", callback_data="show_help")]
    ])
    return keyboard

@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Привет! Я бот для вычисления математических выражений и поиска корней многочленов.\n\n"
        "Выбери тип вычисления:",
        reply_markup=create_type_keyboard()
    )
    await state.set_state(CalcStates.waiting_for_type)

@dp.callback_query(F.data.startswith("type_"))
async def process_type_selection(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("✅ Тип выбран! Отправь выражение:")
    await state.update_data(calc_type=callback.data.split("_")[1])
    await state.set_state(CalcStates.waiting_for_expression)
    await callback.answer()

@dp.message(CalcStates.waiting_for_expression)
async def process_expression(message: Message, state: FSMContext):
    data = await state.get_data()
    calc_type = data.get("calc_type")
    expression = message.text.strip()
    
    await message.answer("🔄 Вычисляю...")
    
    try:
        if calc_type == "expression":
            response = requests.post(
                f"{GATEWAY_URL}/calculate",
                json={"expression": expression},
                timeout=10
            )

            try:
                data = response.json()
            except ValueError:
                data = {}

            if response.status_code == 200:
                result = data.get("result")
                if result is not None:
                    cache_status = "из кэша" if data.get("from_cache") else "сначала"
                    await message.answer(
                        f"Выражение: `{expression}`\n"
                        f"Результат: `{result}`\n"
                        f"{cache_status} ({data.get('calculation_time_ms', 0)} мс)",
                        parse_mode="Markdown"
                    )
                else:
                    await message.answer("❌ Не удалось вычислить выражение.")
            else:
                error_msg = (
                    data.get("error")
                    or data.get("detail")
                    or f"Ошибка {response.status_code}"
                )
                await message.answer(f"❌ {error_msg}")
                
        elif calc_type == "polynomial":
            coefficients = parse_polynomial(expression)
            if not coefficients:
                await message.answer("❌ Неверный формат многочлена.\nПримеры: `x^2+8`, `3x^3-2x^2+x-1`")
                return
            
            response = requests.post(
                f"{GATEWAY_URL}/polynomial/roots",
                json={"coefficients": coefficients, "precision": 6},
                timeout=30
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                roots_text = []
                for i, root in enumerate(result_data["roots"], 1):
                    real = root["real"]
                    imag = root["imag"]
                    
                    if abs(imag) < 1e-10:
                        roots_text.append(f"x{i} = {real:.6f}")
                    else:
                        if imag >= 0:
                            roots_text.append(f"x{i} = {real:.6f} + {imag:.6f}i")
                        else:
                            roots_text.append(f"x{i} = {real:.6f} - {abs(imag):.6f}i")
                
                cache_status = "из кэша" if result_data.get("from_cache") else "сначала"
                response_text = (
                    f"Многочлен степени {result_data['degree']}:\n"
                    #f"`{result_data['polynomial_string']}`\n\n"
                    f"**Корни:**\n" + "\n".join(roots_text) + "\n\n"
                    f"{cache_status} ({result_data.get('calculation_time_ms', 0)} мс)"
                )
                
                await message.answer(response_text, parse_mode="Markdown")
            else:
                error_data = response.json() if response.content else {"error": "Неизвестная ошибка"}
                error_msg = error_data.get("error", f"Ошибка {response.status_code}")
                await message.answer(f"❌ {error_msg}")
            
    except requests.exceptions.Timeout:
        logger.error("Timeout при запросе к gateway")
        await message.answer("Таймаут. Сервис перегружен, попробуй позже.")
    except requests.exceptions.ConnectionError:
        logger.error("Ошибка соединения с gateway")
        await message.answer("❌ Ошибка связи с вычислителем. Проверь, запущен ли gateway.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка HTTP запроса к gateway: {e}")
        await message.answer("❌ Ошибка сети. Попробуй позже.")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        await message.answer("❌ Произошла внутренняя ошибка.")
    
    await state.clear()
    await message.answer("Выбери тип следующего вычисления:", reply_markup=create_type_keyboard())


@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = (
        "Инструкция:\n\n"
        "Математические выражения:\n"
        "• Операторы: +, -, *, /, ** (степень), % (остаток)\n"
        "• Функции: sin(), cos(), tan(), log(), exp(), sqrt(), abs()\n"
        "• Примеры: 2+2, (2+3)*4, sin(10)+log10(e)\n\n"
        "Многочлены:\n"
        "• Формат: 3x^3+2x^2-x+1, x^2+8, x^4-1\n"
        "• Поддержка до 100 степени\n\n"
        "Нажми /start для начала!"
    )
    await message.answer(help_text)   

    
@dp.callback_query(F.data == "show_help")
async def help_callback(callback: CallbackQuery):
    help_text = (
        "Инструкция:\n\n"
        "Математические выражения:\n"
        "• Операторы: +, -, *, /, ** (степень), % (остаток)\n"
        "• Функции: sin(), cos(), tan(), log(), exp(), sqrt(), abs()\n"
        "• Примеры: 2+2, (2+3)*4, sin(10)+log10(e)\n\n"
        "Многочлены:\n"
        "• Формат: 3x^3+2x^2-x+1, x^2+8, x^4-1\n"
        "• Поддержка до 100 степени\n\n"
        "Нажми кнопку выше, чтобы выбрать тип вычисления."
    )
    await callback.message.answer(help_text)   
    await callback.answer()



async def main():
    logger.info("Запуск Telegram бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
