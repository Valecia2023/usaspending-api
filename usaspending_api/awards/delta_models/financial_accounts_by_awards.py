FINANCIAL_ACCOUNTS_BY_AWARDS_COLUMNS = {
    "data_source": "STRING",
    "financial_accounts_by_awards_id": "INTEGER NOT NULL",
    "piid": "STRING",
    "parent_award_id": "STRING",
    "fain": "STRING",
    "uri": "STRING",
    "ussgl480100_undelivered_orders_obligations_unpaid_fyb": "NUMERIC(23,2)",
    "ussgl480100_undelivered_orders_obligations_unpaid_cpe": "NUMERIC(23,2)",
    "ussgl483100_undelivered_orders_oblig_transferred_unpaid_cpe": "NUMERIC(23,2)",
    "ussgl488100_upward_adjust_pri_undeliv_order_oblig_unpaid_cpe": "NUMERIC(23,2)",
    "ussgl490100_delivered_orders_obligations_unpaid_fyb": "NUMERIC(23,2)",
    "ussgl490100_delivered_orders_obligations_unpaid_cpe": "NUMERIC(23,2)",
    "ussgl493100_delivered_orders_oblig_transferred_unpaid_cpe": "NUMERIC(23,2)",
    "ussgl498100_upward_adjust_pri_deliv_orders_oblig_unpaid_cpe": "NUMERIC(23,2)",
    "ussgl480200_undelivered_orders_oblig_prepaid_advanced_fyb": "NUMERIC(23,2)",
    "ussgl480200_undelivered_orders_oblig_prepaid_advanced_cpe": "NUMERIC(23,2)",
    "ussgl483200_undeliv_orders_oblig_transferred_prepaid_adv_cpe": "NUMERIC(23,2)",
    "ussgl488200_up_adjust_pri_undeliv_order_oblig_ppaid_adv_cpe": "NUMERIC(23,2)",
    "ussgl490200_delivered_orders_obligations_paid_cpe": "NUMERIC(23,2)",
    "ussgl490800_authority_outlayed_not_yet_disbursed_fyb": "NUMERIC(23,2)",
    "ussgl490800_authority_outlayed_not_yet_disbursed_cpe": "NUMERIC(23,2)",
    "ussgl498200_upward_adjust_pri_deliv_orders_oblig_paid_cpe": "NUMERIC(23,2)",
    "obligations_undelivered_orders_unpaid_total_cpe": "NUMERIC(23,2)",
    "obligations_delivered_orders_unpaid_total_fyb": "NUMERIC(23,2)",
    "obligations_delivered_orders_unpaid_total_cpe": "NUMERIC(23,2)",
    "gross_outlays_undelivered_orders_prepaid_total_fyb": "NUMERIC(23,2)",
    "gross_outlays_undelivered_orders_prepaid_total_cpe": "NUMERIC(23,2)",
    "gross_outlays_delivered_orders_paid_total_fyb": "NUMERIC(23,2)",
    "gross_outlay_amount_by_award_fyb": "NUMERIC(23,2)",
    "gross_outlay_amount_by_award_cpe": "NUMERIC(23,2)",
    "obligations_incurred_total_by_award_cpe": "NUMERIC(23,2)",
    "ussgl487100_down_adj_pri_unpaid_undel_orders_oblig_recov_cpe": "NUMERIC(23,2)",
    "ussgl497100_down_adj_pri_unpaid_deliv_orders_oblig_recov_cpe": "NUMERIC(23,2)",
    "ussgl487200_down_adj_pri_ppaid_undel_orders_oblig_refund_cpe": "NUMERIC(23,2)",
    "ussgl497200_down_adj_pri_paid_deliv_orders_oblig_refund_cpe": "NUMERIC(23,2)",
    "deobligations_recoveries_refunds_of_prior_year_by_award_cpe": "NUMERIC(23,2)",
    "obligations_undelivered_orders_unpaid_total_fyb": "NUMERIC(23,2)",
    "gross_outlays_delivered_orders_paid_total_cpe": "NUMERIC(23,2)",
    "drv_award_id_field_type": "STRING",
    "drv_obligations_incurred_total_by_award": "NUMERIC(23,2)",
    "transaction_obligated_amount": "NUMERIC(23,2)",
    "reporting_period_start": "DATE",
    "reporting_period_end": "DATE",
    "last_modified_date": "DATE",
    "certified_date": "DATE",
    "create_date": "TIMESTAMP",
    "update_date": "TIMESTAMP",
    "award_id": "INTEGER",
    "object_class_id": "INTEGER",
    "program_activity_id": "INTEGER",
    "submission_id": "INTEGER NOT NULL",
    "treasury_account_id": "INTEGER",
    "distinct_award_key": "STRING NOT NULL",
    "disaster_emergency_fund_code": "STRING",
}

financial_accounts_by_awards_sql_string = rf"""
    CREATE OR REPLACE TABLE {{DESTINATION_TABLE}} (
        {", ".join([f'{key} {val}' for key, val in FINANCIAL_ACCOUNTS_BY_AWARDS_COLUMNS.items()])}
    )
    USING DELTA
    LOCATION 's3a://{{SPARK_S3_BUCKET}}/{{DELTA_LAKE_S3_PATH}}/{{DESTINATION_DATABASE}}/{{DESTINATION_TABLE}}'
"""
