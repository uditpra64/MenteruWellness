import logging
import os
import sys
import time

from data_module.config.utils import get_path_from_config_for_outside_base_folder
from data_module.db_utils.database_client import DatabaseClient
from data_module.db_utils.db_host_tables import HOST_TABLES
from data_module.db_utils.private_info import DB_CONFIGS
from data_module.preprocess_utils.data_preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessRunner:
    """
    Data Preprocessing Runner

    Run data preprocessing pipeline

    1. Connect to database & get data
        - Print data size & memory usage (Optional)
    2. Preprocess data
    3. Save preprocessed data
    """

    def __init__(
        self,
        start_date_preprocess: str,
        end_date_preprocess: str,
        datetime_column: str = "time",
        raw_data_output_path: str = "rawData_output_DB_folder_name",
        preprocessed_output_path: str = "preprocessed_folder_path",
    ):
        self.raw_data_output_path = get_path_from_config_for_outside_base_folder(
            raw_data_output_path
        )
        self.preprocessed_output_path = get_path_from_config_for_outside_base_folder(
            preprocessed_output_path
        )
        self.start_date_preprocess = start_date_preprocess
        self.end_date_preprocess = end_date_preprocess
        self.datetime_column = datetime_column

        self.client = DatabaseClient(
            DB_CONFIGS,
            start_date=self.start_date_preprocess,
            end_date=self.end_date_preprocess,
            datetime_column=self.datetime_column,
            output_path=self.raw_data_output_path,
        )
        self.data_host1 = None
        self.data_host2 = None
        self.preprocessor = None
        self.preprocessed_df = None

    def run(self) -> None:
        """Run the full data preprocessing pipeline"""
        self.connect_and_get_data()
        self.print_data_size()
        self.preprocess_data()
        self.save_preprocessed_data()

    def connect_and_get_data(self) -> None:
        """Connect to database and fetch data"""
        logging.info("Connecting to database and fetching data")
        try:
            self.data_host1 = self.client.fetch_multiple_tables_data(
                HOST_TABLES["host1"], "host1"
            )
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise e
        try:
            self.data_host2 = self.client.fetch_multiple_tables_data(
                HOST_TABLES["host2"], "host2"
            )
            logging.info("Data fetched successfully from host2")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise e

    def print_data_size(self) -> None:
        """Print size of data and save raw CSV files"""
        data_host1_size_mb = sys.getsizeof(self.data_host1) / (1024 * 1024)
        data_host2_size_mb = sys.getsizeof(self.data_host2) / (1024 * 1024)
        print(
            f"Size of data_host1: {data_host1_size_mb:.4f} MB",
            f"Size of data_host2: {data_host2_size_mb:.4f} MB",
        )

        # TODO: DECIDE LATER TO REMOVE OR USE THIS
        # self.client.save_data_to_csv(self.data_host1, "host1")
        # self.client.save_data_to_csv(self.data_host2, "host2")

    def preprocess_data(self) -> None:
        """Combine raw data and apply preprocessing"""
        logging.info("Preprocessing data")

        # Copy data_host1 and update with data_host2
        combined_data_dict = self.data_host1.copy()
        combined_data_dict.update(self.data_host2)

        self.preprocessor = DataPreprocessor(combined_data_dict=combined_data_dict)
        self.preprocessed_df = self.preprocessor.run(
            start_date=self.start_date_preprocess,
            end_date=self.end_date_preprocess,
        )

    def save_preprocessed_data(self) -> None:
        """Save preprocessed data to file"""
        logging.info("Saving preprocessed data")

        self.preprocessor.save_preprocessed_data(
            dataframe=self.preprocessed_df,
            save_path=self.preprocessed_output_path,
            overwrite=True,
        )


if __name__ == "__main__":
    # ------------------------------------------------------------
    #  Run data preprocessing データ前処理パイプラインを実行
    # ------------------------------------------------------------

    start_time = time.time()
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    preprocess_runner = DataPreprocessRunner()
    preprocess_runner.run()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
