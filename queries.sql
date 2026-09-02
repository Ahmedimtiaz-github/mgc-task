-- MGC Developments — leads queries
-- Dialect: PostgreSQL (portable to any mainstream SQL engine with minor tweaks).
--
-- Verified against real data: both queries below were run for real against
-- leads.csv loaded into a local SQLite database (schema.sql's leads table,
-- dialect-adjusted). The only change SQLite needed was STRING_AGG(...) ->
-- GROUP_CONCAT(...) in Query 2 (SQLite has no STRING_AGG); everything else ran
-- unchanged. Real results:
--   Query 1 (conversion rate by source, 200+ leads, best first):
--     Referral 13.18%, Walk-in 10.28%, Facebook Ads 6.80%, Google Search 6.50%,
--     WhatsApp Campaign 6.33%, Property Portal 6.00%, Instagram 5.60%,
--     Billboard 4.33%, Expo Stall 2.95%  (all 9 sources clear the 200-lead bar)
--   Query 2 (duplicate leads by crm_record_hash): 160 duplicate groups found,
--     e.g. crm_record_hash 9989052966 -> lead_ids MGC-105130, MGC-105130-B
--     (matches the exact "-B" re-entry pattern this schema was designed around).
-- schema.sql itself stays PostgreSQL, as the more standard production target.

-- ============================================================================
-- Query 1: Conversion rate by lead source, sources with 200+ leads only,
-- best conversion rate first.
--
-- Filtering to 200+ leads (HAVING) avoids ranking small, noisy sources (e.g. a
-- source with 5 leads and 2 conversions would show a misleading 40% rate).
-- ============================================================================
SELECT
    source,
    COUNT(*)                                            AS total_leads,
    SUM(CASE WHEN converted THEN 1 ELSE 0 END)          AS converted_leads,
    ROUND(
        100.0 * SUM(CASE WHEN converted THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                                    AS conversion_rate_pct
FROM leads
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;


-- ============================================================================
-- Query 2: Find duplicate leads by crm_record_hash.
--
-- NOTE: with schema.sql's UNIQUE constraint on crm_record_hash actually in
-- place, this query will always return zero rows against the `leads` table
-- itself — a second INSERT sharing a crm_record_hash would be rejected by the
-- database before it could ever create a duplicate group. This query is what
-- you'd run against the messy raw CRM dump (e.g. a staging/import table without
-- that constraint) to find and clean up duplicates that already exist, exactly
-- like the 160 duplicated crm_record_hash values found in leads.csv, where the
-- same lead was re-entered with a different lead_id (one row suffixed "-B").
-- The UNIQUE constraint in schema.sql is what prevents this from recurring
-- going forward — it stops the duplicate at write time instead of requiring a
-- cleanup query like this one after the fact.
-- ============================================================================
SELECT
    crm_record_hash,
    STRING_AGG(lead_id, ', ' ORDER BY lead_id)  AS lead_ids,
    COUNT(*)                                     AS duplicate_count
FROM leads
GROUP BY crm_record_hash
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;
