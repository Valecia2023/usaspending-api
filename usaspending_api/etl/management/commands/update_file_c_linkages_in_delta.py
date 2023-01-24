from django.core.management import call_command
from django.core.management.base import BaseCommand
from pyspark.sql import SparkSession

from usaspending_api.awards.delta_models import c_to_d_linkage_view_sql_strings, c_to_d_linkage_drop_view_sql_strings
from usaspending_api.common.helpers.spark_helpers import (
    configure_spark_session,
    get_active_spark_session,
    get_jvm_logger,
)
from usaspending_api.config import CONFIG


class Command(BaseCommand):

    help = (
        "This command links File C (financial_accounts_by_awards) records to File D (awards) records.",
        "It creates a new copy of the `financial_accounts_by_awards` table in the `int` schema based",
        "on the copy in the `raw` schema and applies the updates there. These records are linked to",
        "the `awards` table in the `int` schema",
    )

    spark: SparkSession

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-clone",
            action="store_true",
            required=False,
            help="Whether to create a full new `int` table instead of a clone.",
        )

        parser.add_argument(
            "--spark-s3-bucket",
            type=str,
            required=False,
            default=CONFIG.SPARK_S3_BUCKET,
            help="The destination bucket in S3 to write the data",
        )

    def get_unlinked_count(self, schema):
        unlinked_sql = f"""
        SELECT
            count(*)
        FROM
            {schema}.financial_accounts_by_awards
        WHERE
            award_id IS NULL;
        """

        results = self.spark.sql(unlinked_sql)
        return results.collect()[0][0]

    def handle(self, *args, **options):

        extra_conf = {
            # Config for Delta Lake tables and SQL. Need these to keep Dela table metadata in the metastore
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
            "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            # See comment below about old date and time values cannot parsed without these
            "spark.sql.legacy.parquet.datetimeRebaseModeInWrite": "LEGACY",  # for dates at/before 1900
            "spark.sql.legacy.parquet.int96RebaseModeInWrite": "LEGACY",  # for timestamps at/before 1900
            "spark.sql.jsonGenerator.ignoreNullFields": "false",  # keep nulls in our json
        }

        no_clone = options["no_clone"]
        spark_s3_bucket = options["spark_s3_bucket"]

        self.spark = get_active_spark_session()
        spark_created_by_command = False
        if not self.spark:
            spark_created_by_command = True
            self.spark = configure_spark_session(**extra_conf, spark_context=self.spark)  # type: SparkSession

        # Setup Logger
        logger = get_jvm_logger(self.spark, __name__)

        # Setup int table. Creates a shallow clone of the `raw` FABA table in the `int` schema.
        # If the --no-clone option is provided a full table is created instead.
        if no_clone:
            self.spark.sql("DROP TABLE IF EXISTS int.financial_accounts_by_awards")
            call_command(
                "create_delta_table",
                "--destination-table",
                "financial_accounts_by_awards",
                "--spark-s3-bucket",
                spark_s3_bucket,
                "--alt-db",
                "int",
            )
            self.spark.sql(
                "INSERT INTO int.financial_accounts_by_awards SELECT * FROM raw.financial_accounts_by_awards;"
            )
        else:
            self.spark.sql(
                """
            CREATE OR REPLACE TABLE int.financial_accounts_by_awards
            SHALLOW CLONE raw.financial_accounts_by_awards;
            """
            )

        # Log initial linkage count
        unlinked_count = self.get_unlinked_count("raw")
        logger.info(f"Count of unlinked records before updates: {unlinked_count:,}")

        # Run Linkage Queries
        linkage_queries = c_to_d_linkage_view_sql_strings
        for index, query in enumerate(linkage_queries):
            logger.info(f"Running query number: {index + 1}\nPreview of query: {query[:100]}")
            self.spark.sql(query)

        # Final MERGE of `union_all_priority` into `int.financial_accounts_by_awards`;
        # This merge will set the `award_id` field to the `id` of the linked `int.awards` record;
        # If any previously linked Awards have been deleted, the `award_id` will be set to NULL
        merge_sql = """
            MERGE INTO int.financial_accounts_by_awards AS faba
            USING union_all_priority AS updates
            ON faba.financial_accounts_by_awards_id = updates.financial_accounts_by_awards_id
            WHEN MATCHED
                THEN UPDATE SET faba.award_id = updates.award_id;
        """

        # Log the number of FABA records updated in the merge and final linkage count
        results = self.spark.sql(merge_sql)
        update_count = results.collect()
        logger.info(update_count)
        # logger.info(f"{update_count:,} int.financial_accounts_by_awards records were unlinked or linked to int.awards")
        unlinked_count = self.get_unlinked_count("int")
        logger.info(f"Count of unlinked records after updates: {unlinked_count:,}")

        # Run Linkage Queries
        drop_view_queries = c_to_d_linkage_drop_view_sql_strings
        for index, query in enumerate(drop_view_queries):
            logger.info(f"Running query number: {index + 1}\nPreview of query: {query[:100]}")
            self.spark.sql(query)

        if spark_created_by_command:
            self.spark.stop()
