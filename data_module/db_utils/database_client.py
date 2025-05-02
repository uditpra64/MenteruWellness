import logging
import os

import pandas as pd
import psycopg2
from psycopg2 import errors, sql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseClient:
    def __init__(
        self,
        db_configs: dict,
        start_date: str,
        end_date: str,
        datetime_column: str = "time",
        output_path: str = "00_Data",
    ):
        """
        This class is designed to handle connections to one or multiple PostgreSQL databases,
        specified by a dictionary of configurations. It allows querying data within a
        specified date range and saving the results to a designated output path.
        The datetime column used for filtering the data can also be specified.

        Args:
            db_configs: A dictionary mapping host keys to their corresponding database configurations.
            start_date: The start date for the query.
            end_date: The end date for the query.
            datetime_column: The datetime column for the query.
            output_path: The path to save the data.
        """
        self.db_configs = db_configs
        self.start_date = start_date
        self.end_date = end_date
        self.datetime_column = datetime_column
        self.output_path = output_path

    def connect(self, host: str):
        """Establish and return a connection to the PostgreSQL database for the given host.

        Args:
            host: The key identifying which database configuration to use.

        Returns:
            A psycopg2 connection object.
        """
        db_config = self.db_configs.get(host)
        if not db_config:
            logger.error(f"No database configuration found for host '{host}'")
            raise
        try:
            connection = psycopg2.connect(**db_config)
            return connection
        except psycopg2.Error as error:
            logger.error(f"Error connecting to host '{host}': {error}")
            raise

    def fetch_table_data(
        self, cursor: psycopg2.extensions.cursor, table: str, host: str
    ) -> pd.DataFrame:
        """
        Fetch data from a single table on the specified host.

        Args:
            cursor: The psycopg2 cursor object.
            table: The name of the table to fetch data from.
            host: The host key for which the configuration should be used.

        Returns:
            A pandas DataFrame containing the fetched data.
        """
        try:
            if not all([table, self.datetime_column, self.start_date, self.end_date]):
                logger.error("Missing required parameters for query")
                raise

            # Validate that end_date is after start_date
            if self.end_date < self.start_date:
                logger.error("End date must be after start date")
                raise

            query = sql.SQL(
                """
                SELECT * FROM {} 
                WHERE SUBSTRING({}, 1, 8)::DATE 
                    BETWEEN {}::DATE AND {}::DATE
            """
            ).format(
                sql.Identifier(table),
                sql.Identifier(self.datetime_column),
                sql.Literal(self.start_date),
                sql.Literal(self.end_date),
            )
            cursor.execute(query)
            column_names = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=column_names)
            return df
        except errors.UndefinedTable as table_error:
            logger.error(
                f"Table '{table}' does not exist on host '{host}': {table_error}"
            )
            cursor.connection.rollback()  # Roll back the failed transaction
            return pd.DataFrame()  # Return empty DataFrame for missing tables

    def fetch_multiple_tables_data(self, tables: list, host: str) -> dict:
        """Fetch data from multiple tables on the specified host.

        It attempts to connect to the host using the provided configuration.
        Once connected, it iterates over the table list and queries each one.

        Args:
            table_names: A list of table names to fetch data from.
            host: The host key for which the configuration should be used.

        Returns:
            A dictionary mapping each table name to its corresponding DataFrame.
            Tables that do not exist are skipped.
        """
        # time the process

        data = {}
        connection = self.connect(host)

        for table in tables:
            logger.info(f"Fetching data for table {table}")
            try:
                with connection.cursor() as cursor:
                    data[table] = self.fetch_table_data(cursor, table, host)
                    logger.info(f"Data fetched for table {table}")
                    connection.commit()  # Commit successful transaction
            except Exception as e:
                connection.rollback()  # Ensure transaction is rolled back
                logger.error(f"Failed to fetch data for table {table}: {e}")
                continue  # Continue with next table even if this one fails

        connection.close()

        return data

    def save_data_to_csv(self, data: dict, host: str):
        """Save the fetched data to a CSV file.

        Args:
            data: A dictionary mapping table names to their corresponding DataFrames.
            host: The host key for which the data should be saved.
        """
        for table, df in data.items():
            # if folder does not exist, create it
            if not os.path.exists(f"{self.output_path}"):
                os.makedirs(f"{self.output_path}")
            df.to_csv(f"{self.output_path}/{table}.csv", index=False, encoding="cp932")
            logger.info(f"Data saved to {self.output_path}/{table}.csv")
