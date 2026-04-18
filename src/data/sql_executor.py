import sqlite3


def build_db(table: dict, case_insensitive: bool = False) -> sqlite3.Connection:
    """
    Create an in-memory SQLite database from a WikiSQL table.

    Args:
        table: WikiSQL table dict with 'header', 'types', 'rows'
        case_insensitive: if True, text columns are created with
            `COLLATE NOCASE` so that `=` comparisons ignore letter case.
            Used for the V1.5 post-processing ablation.
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    headers = table["header"]
    types = table["types"]

    text_type = "TEXT COLLATE NOCASE" if case_insensitive else "TEXT"
    type_map = {"text": text_type, "number": "REAL", "real": "REAL"}
    col_defs = ", ".join(
        f'`{col}` {type_map.get(t, text_type)}'
        for col, t in zip(headers, types)
    )
    cursor.execute(f"CREATE TABLE data ({col_defs})")

    # Insert rows, converting numeric values to float explicitly
    for row in table["rows"]:
        converted = []
        for val, col_type in zip(row, types):
            if col_type in ("real", "number"):
                try:
                    converted.append(float(str(val).replace(",", "")))
                except (ValueError, TypeError):
                    converted.append(val)
            else:
                converted.append(val)
        placeholders = ", ".join(["?"] * len(converted))
        cursor.execute(f"INSERT INTO data VALUES ({placeholders})", converted)

    conn.commit()
    return conn


def execute_sql(
    table: dict,
    sql: str,
    case_insensitive: bool = False,
) -> tuple[list, str | None]:
    """
    Execute a SQL query against an in-memory SQLite DB.

    Args:
        table: WikiSQL table dict
        sql: SQL query string (uses `FROM table`; will be rewritten to `FROM data`)
        case_insensitive: if True, text columns use `COLLATE NOCASE` so that
            `=` comparisons ignore letter case. Default False preserves the
            original case-sensitive behavior used in all pre-V1.5 results.

    Returns:
        (results, error): results is a list of rows, error is None if successful
    """
    # Replace "FROM table" with "FROM data" since our table is named "data"
    sql = sql.replace("FROM table", "FROM data")

    try:
        conn = build_db(table, case_insensitive=case_insensitive)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return [], str(e)