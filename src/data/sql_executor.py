import sqlite3


def build_db(table: dict) -> sqlite3.Connection:
    """Create an in-memory SQLite database from a WikiSQL table."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    headers = table["header"]
    types = table["types"]

    type_map = {"text": "TEXT", "number": "REAL", "real": "REAL"}
    col_defs = ", ".join(
        f'`{col}` {type_map.get(t, "TEXT")}'
        for col, t in zip(headers, types)
    )
    cursor.execute(f"CREATE TABLE data ({col_defs})")

    # Insert rows, converting numeric values to float explicitly
    for row in table["rows"]:
        converted = []
        for val, col_type in zip(row, types):
            if col_type in ("real", "number"):
                try:
                    converted.append(float(val))
                except (ValueError, TypeError):
                    converted.append(val)
            else:
                converted.append(val)
        placeholders = ", ".join(["?"] * len(converted))
        cursor.execute(f"INSERT INTO data VALUES ({placeholders})", converted)

    conn.commit()
    return conn


def execute_sql(table: dict, sql: str) -> tuple[list, str | None]:
    """
    Execute a SQL query against an in-memory SQLite DB.

    Returns:
        (results, error): results is a list of rows, error is None if successful
    """
    # Replace "FROM table" with "FROM data" since our table is named "data"
    sql = sql.replace("FROM table", "FROM data")

    try:
        conn = build_db(table)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return [], str(e)