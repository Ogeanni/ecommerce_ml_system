CREATE TABLE IF NOT EXISTS services(
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    services_name TEXT NOT NULL,
    price REAL NOT NULL,
    billing_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contacts(
    department TEXT,
    email TEXT,
    availability_hours TEXT
);

CREATE TABLE IF NOT EXISTS policies(
    policy_name TEXT,
    refund_days INTEGER
);