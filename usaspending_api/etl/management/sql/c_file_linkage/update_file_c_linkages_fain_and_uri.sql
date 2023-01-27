-- When both FAIN and URI are populated in File C, update File C assistance
-- records to link to a corresponding award if there is an single exact match
-- based on FAIN
WITH update_cte AS (
    UPDATE financial_accounts_by_awards AS faba
    SET
        award_id = (
            SELECT
                id
            FROM
                vw_awards AS aw
            WHERE
                UPPER(aw.fain) = UPPER(faba.fain)
        )
    WHERE
        faba.financial_accounts_by_awards_id = ANY(
            SELECT
                faba_sub.financial_accounts_by_awards_id
            FROM
                financial_accounts_by_awards AS faba_sub
            WHERE
                faba_sub.fain IS NOT NULL
                AND faba_sub.uri IS NOT NULL
                AND faba_sub.award_id IS NULL
                AND (
                    SELECT COUNT(*)
                    FROM vw_awards AS aw_sub
                    WHERE
                        UPPER(aw_sub.fain) = UPPER(faba_sub.fain)
                ) = 1
                {submission_id_clause}
        )
    RETURNING award_id
)
UPDATE
    {file_d_table} a
SET
    update_date = NOW()
FROM
    update_cte
WHERE
    a.award_id = update_cte.award_id
;

-- When both FAIN and URI are populated in File C, update File C assistance
-- records to link to a corresponding award if there is an single exact match
-- based on URI
WITH update_cte AS (
    UPDATE financial_accounts_by_awards AS faba
    SET
        award_id = (
            SELECT
                award_id
            FROM
                {file_d_table} AS aw
            WHERE
                UPPER(aw.uri) = UPPER(faba.uri)
        )
    WHERE
        faba.financial_accounts_by_awards_id = ANY(
            SELECT
                faba_sub.financial_accounts_by_awards_id
            FROM
                financial_accounts_by_awards AS faba_sub
            WHERE
                faba_sub.fain IS NOT NULL
                AND faba_sub.uri IS NOT NULL
                AND faba_sub.award_id IS NULL
                AND (
                    SELECT COUNT(*)
                    FROM {file_d_table} AS aw_sub
                    WHERE
                        UPPER(aw_sub.uri) = UPPER(faba_sub.uri)
                ) = 1
                {submission_id_clause}
            )
        RETURNING award_id
    )
UPDATE
    {file_d_table} a
SET
    update_date = NOW()
FROM
    update_cte
WHERE
    a.award_id = update_cte.award_id
;
