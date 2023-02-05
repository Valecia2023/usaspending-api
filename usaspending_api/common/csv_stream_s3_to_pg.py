"""
WARNING WARNING WARNING
!!!!!!!!!!!!!!!!!!!!!!!
This module must be managed very carefully and stay lean.

The main functions: copy_csv_from_s3_to_pg and copy_csvs_from_s3_to_pg
are used in distributed/parallel/multiprocess execution (by Spark) and is
pickled via cloudpickle. As such it must not have any presumed setup code that would have run (like Django setup,
logging configuration, etc.) and must encapsulate all of those dependencies (like logging config) on its own.

Adding new imports to this module may inadvertently introduce a dependency that can't be pickled.

As it stands, even if new imports are added to the modules it already imports, it could lead to a problem.
"""
import boto3
import time
import psycopg2
import codecs
import csv
import gzip
import logging

import pandas as pd
import numpy as np

from contextlib import closing
from io import StringIO
from typing import Iterable, List, Dict, Union

from botocore.client import BaseClient
from pandas.io.sql import SQLTable
from pyspark.sql.types import StructField
from pyspark.pandas._typing import Dtype
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine import Connection

from usaspending_api.common.logging import AbbrevNamespaceUTCFormatter, ensure_logging
from usaspending_api.config import CONFIG
from usaspending_api.settings import LOGGING

logger = logging.getLogger("script")


def _get_boto3_s3_client() -> BaseClient:
    if not CONFIG.USE_AWS:
        boto3_session = boto3.session.Session(
            region_name=CONFIG.AWS_REGION,
            aws_access_key_id=CONFIG.AWS_ACCESS_KEY.get_secret_value(),
            aws_secret_access_key=CONFIG.AWS_SECRET_KEY.get_secret_value(),
        )
        s3_client = boto3_session.client(
            service_name="s3",
            region_name=CONFIG.AWS_REGION,
            endpoint_url=f"http://{CONFIG.AWS_S3_ENDPOINT}",
        )
    else:
        s3_client = boto3.client(
            service_name="s3",
            region_name=CONFIG.AWS_REGION,
            endpoint_url=f"https://{CONFIG.AWS_S3_ENDPOINT}",
        )
    return s3_client


def _stream_and_copy(
    configured_logger: logging.Logger,
    cursor: psycopg2._psycopg.cursor,
    s3_client: BaseClient,
    s3_bucket_name: str,
    s3_obj_key: str,
    target_pg_table: str,
    ordered_col_names: List[str],
    gzipped: bool,
    partition_prefix: str = "",
):
    start = time.time()
    configured_logger.info(f"{partition_prefix}Starting write of {s3_obj_key}")
    try:
        s3_obj = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_obj_key)
        # Getting Body gives a botocore.response.StreamingBody object back to allow "streaming" its contents
        s3_obj_body = s3_obj["Body"]
        with closing(s3_obj_body):  # make sure to close the stream when done
            if gzipped:
                with gzip.open(s3_obj_body, "rb") as csv_binary:
                    cursor.copy_expert(
                        sql=f"COPY {target_pg_table} ({','.join(ordered_col_names)}) FROM STDIN (FORMAT CSV)",
                        file=csv_binary,
                    )
            else:
                with codecs.getreader("utf-8")(s3_obj_body) as csv_stream_reader:
                    cursor.copy_expert(
                        sql=f"COPY {target_pg_table} ({','.join(ordered_col_names)}) FROM STDIN (FORMAT CSV)",
                        file=csv_stream_reader,
                    )
        elapsed = time.time() - start
        rows_copied = cursor.rowcount
        configured_logger.info(
            f"{partition_prefix}Finished writing {rows_copied} row(s) in {elapsed:.3f}s for {s3_obj_key}"
        )
        yield rows_copied
    except Exception as exc:
        configured_logger.error(f"{partition_prefix}ERROR writing {s3_obj_key}")
        configured_logger.exception(exc)
        raise exc


def copy_csv_from_s3_to_pg(
    s3_bucket_name: str,
    s3_obj_key: str,
    db_dsn: str,
    target_pg_table: str,
    ordered_col_names: List[str],
    gzipped: bool = True,
    work_mem_override: int = None,
):
    """Stream a CSV file from S3 into a Postgres table using the SQL bulk COPY command

    WARNING: See note above in module docstring about this function being pickle-able, and maintaining a lean set of
    outward dependencies on other modules/code that require setup/config.
    """
    ensure_logging(logging_config_dict=LOGGING, formatter_class=AbbrevNamespaceUTCFormatter, logger_to_use=logger)
    try:
        with psycopg2.connect(dsn=db_dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                if work_mem_override:
                    cursor.execute("SET work_mem TO %s", (work_mem_override,))
                s3_client = _get_boto3_s3_client()
                results_generator = _stream_and_copy(
                    configured_logger=logger,
                    cursor=cursor,
                    s3_client=s3_client,
                    s3_bucket_name=s3_bucket_name,
                    s3_obj_key=s3_obj_key,
                    target_pg_table=target_pg_table,
                    gzipped=gzipped,
                    ordered_col_names=ordered_col_names,
                )
                return list(results_generator)[0]
    except Exception as exc:
        logger.error(f"ERROR writing {s3_obj_key}")
        logger.exception(exc)
        raise exc


def copy_csvs_from_s3_to_pg(
    batch_num: int,
    s3_bucket_name: str,
    s3_obj_keys: Iterable[str],
    db_dsn: str,
    target_pg_table: str,
    ordered_col_names: List[str],
    gzipped: bool = True,
    work_mem_override: int = None,
):
    """An optimized form of ``copy_csv_from_s3_to_pg`` that can save on runtime by instantiating the psycopg2 DB
    connection and s3_client only once per partition, where a partition could represent processing several files
    """
    ensure_logging(logging_config_dict=LOGGING, formatter_class=AbbrevNamespaceUTCFormatter, logger_to_use=logger)
    s3_obj_keys = list(s3_obj_keys)  # convert from Iterator (generator) to a concrete List
    batch_size = len(s3_obj_keys)
    batch_start = time.time()
    partition_prefix = f"Partition#{batch_num}: "
    logger.info(f"{partition_prefix}Starting write of a batch of {batch_size} on partition {batch_num}")
    try:
        with psycopg2.connect(dsn=db_dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                if work_mem_override:
                    cursor.execute("SET work_mem TO %s", (work_mem_override,))
                s3_client = _get_boto3_s3_client()
                for s3_obj_key in s3_obj_keys:
                    yield from _stream_and_copy(
                        configured_logger=logger,
                        cursor=cursor,
                        s3_client=s3_client,
                        s3_bucket_name=s3_bucket_name,
                        s3_obj_key=s3_obj_key,
                        target_pg_table=target_pg_table,
                        ordered_col_names=ordered_col_names,
                        gzipped=gzipped,
                        partition_prefix=partition_prefix,
                    )
                batch_elapsed = time.time() - batch_start
                logger.info(
                    f"{partition_prefix}Finished writing batch of {batch_size} "
                    f"{'file' if batch_size == 1 else 'files'} on partition {batch_num} in {batch_elapsed:.3f}s"
                )
    except Exception as exc_batch:
        logger.error(f"{partition_prefix}ERROR writing batch/partition number {batch_num}")
        logger.exception(exc_batch)
        raise exc_batch


def copy_data_as_csv_to_pg(
    partition_idx: int,
    partition_data: Iterable[str],
    db_dsn: str,
    target_pg_table: str,
    ordered_col_names: List[str],
    template_pandas_dataframe: pd.DataFrame,
):
    """Process a partition of data records, converting them to in-memory CSV format and using SQL COPY to
    insert them into Postgres. Instantiate the psycopg2 DB connection only once per partition.
    """
    ensure_logging(logging_config_dict=LOGGING, formatter_class=AbbrevNamespaceUTCFormatter, logger_to_use=logger)
    # Consume Iterator (generator) by Pandas DataFrame WITHOUT iterating it (into memory) yet
    pdf = pd.DataFrame(data=partition_data, columns=template_pandas_dataframe.columns).astype(
        template_pandas_dataframe.dtypes
    )
    batch_start = time.time()
    partition_prefix = f"Partition#{partition_idx}: "
    logger.info(f"{partition_prefix}Starting write of a batch on partition {partition_idx}")
    try:
        psycopg2_dialect_conn_string = db_dsn.replace("postgres://", "postgresql+psycopg2://")
        sqlalchemy_engine = create_engine(psycopg2_dialect_conn_string, isolation_level="AUTOCOMMIT")
        rowcount = insert_pandas_dataframe(df=pdf, table=target_pg_table, engine=sqlalchemy_engine, method="copy")
        batch_elapsed = time.time() - batch_start
        logger.info(
            f"{partition_prefix}Finished writing batch of {rowcount} CSV-formatted records"
            f"on partition {partition_idx} in {batch_elapsed:.3f}s"
        )
    except Exception as exc_batch:
        logger.error(f"{partition_prefix}ERROR writing batch/partition number {partition_idx}")
        logger.exception(exc_batch)
        raise exc_batch
    return [rowcount]


def copy_pandas_dfs_as_csv_to_pg(
    pandas_dfs: Iterable[pd.DataFrame],
    data_type_mapping: Dict[str, Dict[str, Union[StructField, Dtype]]],
    db_dsn: str,
    target_pg_table: str,
):
    """Process a partition of data records, converting them to in-memory CSV format and using SQL COPY to
    insert them into Postgres. Instantiate the psycopg2 DB connection only once per partition.
    """
    ensure_logging(logging_config_dict=LOGGING, formatter_class=AbbrevNamespaceUTCFormatter, logger_to_use=logger)
    psycopg2_dialect_conn_string = db_dsn.replace("postgres://", "postgresql+psycopg2://")
    sqlalchemy_engine = create_engine(psycopg2_dialect_conn_string, isolation_level="AUTOCOMMIT")
    sqlalchemy_connection = sqlalchemy_engine.connect()
    total_partitions = 0
    for partition_idx, pdf in enumerate(pandas_dfs):
        batch_start = time.time()
        partition_prefix = f"Partition#{partition_idx}: "
        logger.info(f"{partition_prefix}Starting write of a batch on partition {partition_idx}")
        try:
            with pd.option_context("display.max_rows", None):
                logger.info(f"{partition_prefix}Got Pandas DataFrame with {len(pdf)} rows and dtypes = {pdf.dtypes}")
                logger.info(f"{partition_prefix}Casting Pandas DataFrame with these types derived from source Spark DataFrame: {data_type_mapping}")
                # pdf = pdf.replace(np.nan, None)  # NaN values can't be type-cast
                pandas_types = {k: v["pandas_type"] for k, v in data_type_mapping.items()}
                pdf = pdf.astype(pandas_types)
                logger.info(f"{partition_prefix}Casted Pandas DataFrame now has these dtypes: {pdf.dtypes}")

            rowcount = insert_pandas_dataframe(df=pdf, table=target_pg_table, engine=sqlalchemy_connection, method="copy")
            # Yield new Pandas DF holding rowcount for this batch COPY
            yield pd.DataFrame(data={"rowcount": rowcount}, index=[0])
            batch_elapsed = time.time() - batch_start
            logger.info(
                f"{partition_prefix}Finished writing batch of {rowcount} CSV-formatted records"
                f"on partition {partition_idx} in {batch_elapsed:.3f}s"
            )
        except Exception as exc_batch:
            logger.error(f"{partition_prefix}ERROR writing batch/partition number {partition_idx}")
            logger.exception(exc_batch)
            raise exc_batch
        total_partitions += 1
    logger.info(f"Finished loading {total_partitions} batches of partitioned data")


def insert_pandas_dataframe(df: pd.DataFrame, table: str, engine: Engine, method: str = None):
    """Inserts a dataframe to the specified database table.

    Args:
        df (pd.DataFrame): data to insert
        table (str): name of table to insert to
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine that builds connections to the DB
        method: one of 'multi' or 'copy', if not None
            - 'multi': does a multi-value bulk insert (many value rows at once). It is efficient for analytics
                databases with few columns, and esp. if columnar storage, but not as efficient for
                row-oriented DBs, and slows considerably when many columns
            - 'copy': use database COPY command, and load from CSV in-memory string buffer
    """
    if method == "copy":
        method = _insert_pandas_dataframe_using_copy
    schema = None
    if "." in table:
        schema, table = tuple(table.split("."))
    rowcount = df.to_sql(name=table, con=engine, schema=schema, index=False, if_exists="append", method=method)
    if rowcount:  # returns may not be supported til Pandas 1.4.x+
        return rowcount
    return len(df.index)


def _insert_pandas_dataframe_using_copy(table: SQLTable, conn: Connection, fields: List[str], data: Iterable[Iterable]):
    """Callable concrete impl of the pandas.DataFrame.to_sql method parameter, which allows the given
    DataFrame's data to be buffered in-memory as a string in CSV format, and then loaded into the
    database via the given connection using COPY <table> (<cols>) FROM STDIN WITH CSV.

    Fastest way to get DataFrame data into a DB table.

    Args:
        table (pandas.io.sql.SQLTable): name of existing table to bulk insert into via COPY
        conn (sqlalchemy.engine.Connection): DB connection derived from the provided Engine, and handed to this
            function by the pandas.DataFrame.to_sql function
        fields (List[str]): column names
        data (Iterable[Iterable]): iterable data set, where each item is a collection of values for a data row
    """
    # Use DB API connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cursor:
        string_buffer = StringIO()
        writer = csv.writer(string_buffer)
        writer.writerows(data)
        string_buffer.seek(0)

        columns = ", ".join(f'"{f}"' for f in fields)
        if table.schema:
            table_name = f"{table.schema}.{table.name}"
        else:
            table_name = table.name

        sql = f"COPY {table_name} ({columns}) FROM STDIN WITH CSV"
        cursor.copy_expert(sql=sql, file=string_buffer)
        return cursor.rowcount
