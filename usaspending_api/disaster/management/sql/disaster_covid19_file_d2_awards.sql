SELECT
    award_search.generated_unique_award_id AS assistance_award_unique_key,
    award_search.fain AS award_id_fain,
    award_search.uri AS award_id_uri,
    latest_transaction.sai_number AS sai_number,
    0 AS disaster_emergency_fund_codes,
    (
       0
    ) AS "outlayed_amount_funded_by_COVID-19_supplementals",
    0 AS "obligated_amount_funded_by_COVID-19_supplementals",
    award_search.total_obligation AS total_obligated_amount,
    award_search.non_federal_funding_amount AS total_non_federal_funding_amount,
    award_search.total_funding_amount AS total_funding_amount,
    award_search.total_loan_value AS total_face_value_of_loan,
    award_search.total_subsidy_cost AS total_loan_subsidy_cost,
    award_search.date_signed AS award_base_action_date,
    EXTRACT (YEAR FROM (award_search.date_signed) + INTERVAL '3 months') AS award_base_action_date_fiscal_year,
    latest_transaction.action_date AS award_latest_action_date,
    EXTRACT (YEAR FROM (latest_transaction.action_date::DATE) + INTERVAL '3 months') AS award_latest_action_date_fiscal_year,
    award_search.period_of_performance_start_date AS period_of_performance_start_date,
    award_search.period_of_performance_current_end_date AS period_of_performance_current_end_date,
    latest_transaction.awarding_agency_code AS awarding_agency_code,
    latest_transaction.awarding_toptier_agency_name AS awarding_agency_name,
    latest_transaction.awarding_sub_tier_agency_c AS awarding_sub_agency_code,
    latest_transaction.awarding_subtier_agency_name AS awarding_sub_agency_name,
    latest_transaction.awarding_office_code AS awarding_office_code,
    latest_transaction.awarding_office_name AS awarding_office_name,
    latest_transaction.funding_agency_code AS funding_agency_code,
    latest_transaction.funding_toptier_agency_name AS funding_agency_name,
    latest_transaction.funding_sub_tier_agency_co AS funding_sub_agency_code,
    latest_transaction.funding_subtier_agency_name AS funding_sub_agency_name,
    latest_transaction.funding_office_code AS funding_office_code,
    latest_transaction.funding_office_name AS funding_office_name,
    array[]::text[] AS treasury_accounts_funding_this_award,
    array[]::text[] AS federal_accounts_funding_this_award,
    array[]::text[] AS object_classes_funding_this_award,
    array[]::text[] AS program_activities_funding_this_award,
    latest_transaction.recipient_unique_id AS recipient_duns,
    latest_transaction.recipient_uei AS recipient_uei,
    latest_transaction.recipient_name_raw AS recipient_name,
    latest_transaction.parent_recipient_unique_id AS recipient_parent_duns,
    latest_transaction.parent_uei AS recipient_parent_uei,
    latest_transaction.parent_recipient_name AS recipient_parent_name,
    latest_transaction.recipient_location_country_code AS recipient_country_code,
    latest_transaction.recipient_location_country_name AS recipient_country_name,
    latest_transaction.legal_entity_address_line1 AS recipient_address_line_1,
    latest_transaction.legal_entity_address_line2 AS recipient_address_line_2,
    latest_transaction.legal_entity_city_code AS recipient_city_code,
    latest_transaction.recipient_location_city_name AS recipient_city_name,
    latest_transaction.recipient_location_county_code AS recipient_county_code,
    latest_transaction.recipient_location_county_name AS recipient_county_name,
    latest_transaction.recipient_location_state_code AS recipient_state_code,
    latest_transaction.recipient_location_state_name AS recipient_state_name,
    latest_transaction.recipient_location_zip5 AS recipient_zip_code,
    latest_transaction.legal_entity_zip_last4 AS recipient_zip_last_4_code,
    latest_transaction.recipient_location_congressional_code AS recipient_congressional_district,
    latest_transaction.legal_entity_foreign_city AS recipient_foreign_city_name,
    latest_transaction.legal_entity_foreign_provi AS recipient_foreign_province_name,
    latest_transaction.legal_entity_foreign_posta AS recipient_foreign_postal_code,
    latest_transaction.place_of_performance_scope AS primary_place_of_performance_scope,
    latest_transaction.pop_country_code AS primary_place_of_performance_country_code,
    latest_transaction.pop_country_name AS primary_place_of_performance_country_name,
    latest_transaction.place_of_performance_code AS primary_place_of_performance_code,
    latest_transaction.pop_city_name AS primary_place_of_performance_city_name,
    latest_transaction.pop_county_code AS primary_place_of_performance_county_code,
    latest_transaction.pop_county_name AS primary_place_of_performance_county_name,
    latest_transaction.pop_state_name AS primary_place_of_performance_state_name,
    latest_transaction.place_of_performance_zip4a AS primary_place_of_performance_zip_4,
    latest_transaction.pop_congressional_code AS primary_place_of_performance_congressional_district,
    latest_transaction.place_of_performance_forei AS primary_place_of_performance_foreign_location,
    array[]::text[] AS cfda_numbers_and_titles,
    latest_transaction.type AS assistance_type_code,
    latest_transaction.type_description AS assistance_type_description,
    award_search.description AS prime_award_base_transaction_description,
    latest_transaction.business_funds_indicator AS business_funds_indicator_code,
    latest_transaction.business_funds_ind_desc AS business_funds_indicator_description,
    latest_transaction.business_types AS business_types_code,
    latest_transaction.business_types_desc AS business_types_description,
    latest_transaction.record_type AS record_type_code,
    latest_transaction.record_type_description AS record_type_description,
    award_search.officer_1_name AS highly_compensated_officer_1_name,
    award_search.officer_1_amount AS highly_compensated_officer_1_amount,
    award_search.officer_2_name AS highly_compensated_officer_2_name,
    award_search.officer_2_amount AS highly_compensated_officer_2_amount,
    award_search.officer_3_name AS highly_compensated_officer_3_name,
    award_search.officer_3_amount AS highly_compensated_officer_3_amount,
    award_search.officer_4_name AS highly_compensated_officer_4_name,
    award_search.officer_4_amount AS highly_compensated_officer_4_amount,
    award_search.officer_5_name AS highly_compensated_officer_5_name,
    award_search.officer_5_amount AS highly_compensated_officer_5_amount,
    CONCAT('https://www.usaspending.gov/award/', urlencode(award_search.generated_unique_award_id), '/') AS usaspending_permalink,
    latest_transaction.last_modified_date::TIMESTAMP WITH TIME ZONE AS last_modified_date
FROM rpt.award_search
INNER JOIN rpt.transaction_search AS latest_transaction
    ON (latest_transaction.is_fpds = FALSE AND award_search.latest_transaction_id = latest_transaction.transaction_id)
limit 21804792
