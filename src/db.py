import psycopg2


class PostgreSQLDatabase:
    def __init__(self, host, port, database, user, password, entity_id):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.entity_id = entity_id
        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        self.cursor = self.connection.cursor()

    def get_data(self, row: int, table: str):
        query = (
            f"SELECT ts, dbl_v FROM {table} WHERE (key = 56 AND entity_id = '{self.entity_id}') ORDER "
            f"BY ts DESC LIMIT {row} ;"
        )
        self.cursor.execute(query)
        if row > 1:
            results = [row[1] for row in self.cursor.fetchall()]
            return results
        elif row == 1:
            result = self.cursor.fetchone()[1]
            return result

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
