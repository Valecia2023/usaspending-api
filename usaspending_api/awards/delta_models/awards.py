AWARDS_COLUMNS = {
    "awarding_agency_id": "INTEGER",
    "base_and_all_options_value": "NUMERIC(23,2)",
    "base_exercised_options_val": "NUMERIC(23,2)",
    "category": "STRING",
    "certified_date": "DATE",
    "create_date": "TIMESTAMP",
    "data_source": "STRING",
    "date_signed": "DATE",
    "description": "STRING",
    "earliest_transaction_id": "LONG",
    "fain": "STRING",
    "fiscal_year": "INTEGER",
    "fpds_agency_id": "STRING",
    "fpds_parent_agency_id": "STRING",
    "funding_agency_id": "INTEGER",
    "generated_unique_award_id": "STRING NOT NULL",
    "id": "LONG NOT NULL",
    "is_fpds": "BOOLEAN NOT NULL",
    "last_modified_date": "DATE",
    "latest_transaction_id": "LONG",
    "non_federal_funding_amount": "NUMERIC(23,2)",
    "officer_1_amount": "NUMERIC(23,2)",
    "officer_1_name": "STRING",
    "officer_2_amount": "NUMERIC(23,2)",
    "officer_2_name": "STRING",
    "officer_3_amount": "NUMERIC(23,2)",
    "officer_3_name": "STRING",
    "officer_4_amount": "NUMERIC(23,2)",
    "officer_4_name": "STRING",
    "officer_5_amount": "NUMERIC(23,2)",
    "officer_5_name": "STRING",
    "parent_award_piid": "STRING",
    "period_of_performance_current_end_date": "DATE",
    "period_of_performance_start_date": "DATE",
    "piid": "STRING",
    "subaward_count": "INTEGER NOT NULL",
    "total_funding_amount": "NUMERIC(23,2)",
    "total_indirect_federal_sharing": "NUMERIC(23,2)",
    "total_loan_value": "NUMERIC(23,2)",
    "total_obligation": "NUMERIC(23,2)",
    "total_subaward_amount": "NUMERIC(23,2)",
    "total_subsidy_cost": "NUMERIC(23,2)",
    "transaction_unique_id": "STRING NOT NULL",
    "type": "STRING",
    "type_description": "STRING",
    "update_date": "TIMESTAMP",
    "uri": "STRING",
}

awards_sql_string = rf"""
    CREATE OR REPLACE TABLE {{DESTINATION_TABLE}} (
        {", ".join([f'{key} {val}' for key, val in AWARDS_COLUMNS.items()])}
    )
    USING DELTA
    LOCATION 's3a://{{SPARK_S3_BUCKET}}/{{DELTA_LAKE_S3_PATH}}/{{DESTINATION_DATABASE}}/{{DESTINATION_TABLE}}'
    TBLPROPERTIES (delta.enableChangeDataFeed = {{CHANGE_DATA_FEED}})
"""
