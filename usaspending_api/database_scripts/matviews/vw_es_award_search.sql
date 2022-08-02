DROP VIEW IF EXISTS vw_es_award_search CASCADE;

CREATE VIEW vw_es_award_search AS (
  SELECT * FROM award_search
  WHERE action_date >= '2007-10-01'
);

GRANT SELECT ON vw_es_award_search TO readonly;
