-- When PIID and Parent PIID are populated, update File C contract records to
--  link to a corresponding award if there is a single exact match based on PIID
--  and Parent PIID
WITH update_cte AS (
    UPDATE financial_accounts_by_awards AS faba
    SET
        award_id = (
            SELECT
                award_id
            FROM
                {file_d_table} AS aw
            WHERE
                UPPER(aw.piid) = UPPER(faba.piid)
                AND UPPER(aw.parent_award_piid) = UPPER(faba.parent_award_id)
        )
    WHERE
        faba.financial_accounts_by_awards_id = ANY(
            SELECT
                faba_sub.financial_accounts_by_awards_id
            FROM
                financial_accounts_by_awards AS faba_sub
            WHERE
                faba_sub.piid IS NOT NULL
                AND faba_sub.award_id IS NULL
                AND faba_sub.parent_award_id IS NOT NULL
                AND (
                    SELECT COUNT(*)
                    FROM {file_d_table} AS aw_sub
                    WHERE
                        UPPER(aw_sub.piid) = UPPER(faba_sub.piid)
                        AND UPPER(aw_sub.parent_award_piid) = UPPER(faba_sub.parent_award_id)
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

-- When PIID is populated and Parent PIID is NULL, update File C contract
--  records to link to a corresponding award if there is a single exact match
-- based on PIID
WITH update_cte AS (
    UPDATE financial_accounts_by_awards AS faba
    SET
        award_id = (
            SELECT
                award_id
            FROM
                {file_d_table} AS aw
            WHERE
                UPPER(aw.piid) = UPPER(faba.piid)
        )
    WHERE
        faba.financial_accounts_by_awards_id = ANY(
            SELECT
                faba_sub.financial_accounts_by_awards_id
            FROM
                financial_accounts_by_awards AS faba_sub
            WHERE
                faba_sub.piid IS NOT NULL
                AND faba_sub.award_id IS NULL
                AND faba_sub.parent_award_id IS NULL
                AND (
                    SELECT COUNT(*)
                    FROM {file_d_table} AS aw_sub
                    WHERE
                        UPPER(aw_sub.piid) = UPPER(faba_sub.piid)
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
