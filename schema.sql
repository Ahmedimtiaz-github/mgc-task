-- MGC Developments — leads schema
-- Dialect: PostgreSQL (portable to any mainstream SQL engine with minor tweaks).
--
-- Design decision: ONE table. leads.csv is a single flat CRM export with no
-- repeating groups or child entities (a lead has exactly one source, one budget,
-- one set of call/response stats) — splitting it into multiple tables would just
-- add joins with no normalization benefit at this size (~9k rows).
--
-- Key decision: crm_record_hash, not lead_id, is the true identity of a lead.
-- Investigation of the raw CSV found 160 crm_record_hash values appearing twice,
-- each pair sharing every attribute but having two different lead_id values (one
-- suffixed "-B") — the same lead re-entered into the CRM by a second agent.
-- lead_id is therefore kept only as a CRM record reference / audit trail (which
-- agent/record created this row) and is intentionally NOT unique and NOT the
-- primary key. The UNIQUE constraint on crm_record_hash below is exactly what
-- would stop this duplication at write time: a second INSERT for the same lead
-- fails instead of silently creating a duplicate row.
--
-- The primary key is a surrogate id, so the audit-trail column (lead_id) and the
-- business dedup key (crm_record_hash) can both be plain columns with the
-- constraint that actually matters — UNIQUE — attached to the one that needs it.

CREATE TABLE leads (
    id                              BIGSERIAL PRIMARY KEY,
    lead_id                         VARCHAR(20) NOT NULL,                 -- CRM record reference, non-unique (dup rows share crm_record_hash, differ by a "-B" suffix here)
    crm_record_hash                 BIGINT NOT NULL UNIQUE,               -- true identity of the lead; UNIQUE prevents re-entry duplicates at write time
    created_at                      TIMESTAMP NOT NULL,
    source                          VARCHAR(50) NOT NULL,
    city                            VARCHAR(50),
    area                            VARCHAR(100),
    property_type                   VARCHAR(50) NOT NULL,
    budget_pkr_lac                  NUMERIC(10, 2),
    bedrooms                        SMALLINT,                             -- nullable: commercial shops/plots legitimately have no bedroom count
    first_response_minutes          NUMERIC(10, 2),
    calls_made                      INT NOT NULL DEFAULT 0,
    total_call_seconds              INT NOT NULL DEFAULT 0,
    whatsapp_replies                INT NOT NULL DEFAULT 0,
    site_visits                     INT NOT NULL DEFAULT 0,
    agent_experience_years          NUMERIC(4, 1),
    is_overseas                     BOOLEAN NOT NULL DEFAULT FALSE,
    referred_by_existing_client     BOOLEAN NOT NULL DEFAULT FALSE,
    has_financing_approved          BOOLEAN NOT NULL DEFAULT FALSE,
    token_amount_received_pkr       NUMERIC(14, 2) NOT NULL DEFAULT 0,
    converted                       BOOLEAN NOT NULL DEFAULT FALSE
);

-- Supporting indexes for the queries this schema is meant to answer.
CREATE INDEX idx_leads_source ON leads (source);
CREATE INDEX idx_leads_lead_id ON leads (lead_id);
