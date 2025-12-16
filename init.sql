-- Таблица для кэша выражений (существующая)
CREATE TABLE IF NOT EXISTS expressions_cache (
    id SERIAL PRIMARY KEY,
    expression_hash VARCHAR(64) UNIQUE NOT NULL,
    expression TEXT NOT NULL,
    result FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    calculation_time_ms INTEGER
);

-- Таблица для кэша многочленов (новая)
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
);

-- Таблица для логов запросов
CREATE TABLE IF NOT EXISTS request_logs (
    id SERIAL PRIMARY KEY,
    expression TEXT NOT NULL,
    from_cache BOOLEAN DEFAULT FALSE,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status_code INTEGER
);

-- Индексы
CREATE INDEX IF NOT EXISTS idx_expression_hash ON expressions_cache(expression_hash);
CREATE INDEX IF NOT EXISTS idx_expression_text ON expressions_cache(expression);
CREATE INDEX IF NOT EXISTS idx_poly_hash ON polynomials_cache(coefficients_hash);
CREATE INDEX IF NOT EXISTS idx_last_accessed ON expressions_cache(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_poly_last_accessed ON polynomials_cache(last_accessed DESC);