# backend/pipeline.py
import logging
import itertools

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

DEVICE = "cpu"
SIM_THRESHOLD = 0.6

# ─────────────────────────────────────────
# Shared model singleton (used by rag.py too)
# ─────────────────────────────────────────
_SENTENCE_MODEL = None

def get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        logger.info("Loading InLegalBERT embedding model...")
        _SENTENCE_MODEL = SentenceTransformer("law-ai/InLegalBERT", device=DEVICE)
        logger.info("InLegalBERT loaded.")
    return _SENTENCE_MODEL


# ─────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────
LABEL_MAP     = {"Neutral": 0, "Pro": 1, "Con": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Role bias multiplier: Employee perspective makes Con patterns heavier
ROLE_BIAS = {
    "Employee": 1.2,   # Con patterns weighted up — more protective
    "Employer": 0.8,   # Con patterns weighted down — employer perspective
}


# ─────────────────────────────────────────
# Weak labeler (direct classification — no training loop)
# ─────────────────────────────────────────
def weak_label_clause_v2(clause, role="Employee"):
    """
    Comprehensive weak labeler for ALL Indian legal contract types:
      - Employment agreements
      - Commercial / vendor / service agreements
      - Non-Disclosure Agreements (NDAs)
      - Lease / rental agreements
      - Loan / finance agreements
      - Partnership / LLP agreements
      - Franchise agreements
      - Software / SaaS / technology agreements
      - Construction / works contracts
      - Share purchase / M&A agreements
      - Consultancy agreements
      - Distribution / reseller agreements

    Scoring philosophy:
      Con   = one party holds unilateral power, no recourse for the other
      Pro   = rights explicitly granted, fair process, statutory protection,
              mutual obligations, cure periods, exit rights
      Neutral = definitional, procedural, boilerplate, mutual standard obligations

    Whitespace is normalised before matching so multiline text never
    breaks substring detection.
    """
    raw_text = clause["text"]
    text     = " ".join(raw_text.lower().split())
    section  = clause["section"].lower()

    score = 0.0
    bias  = ROLE_BIAS.get(role, 1.0)

    # ══════════════════════════════════════════════════════════════════
    # CON PATTERNS  — unfair, one-sided, rights-stripping language
    # Weight scale: 1.0 = mild signal, 2.0 = clear signal, 3.0 = critical
    # ══════════════════════════════════════════════════════════════════
    con_patterns = {

        # ── Universal one-sided power language ────────────────────────
        "sole discretion"                            : 2.5,
        "absolute discretion"                        : 2.5,
        "at its discretion"                          : 2.0,
        "at the discretion of"                       : 2.0,
        "at any time"                                : 1.5,
        "from time to time"                          : 1.0,   # unilateral change signal
        "reserves the right"                         : 2.0,
        "unilaterally"                               : 2.5,
        "without the consent"                        : 2.5,
        "without consent"                            : 2.0,
        "without prior approval"                     : 2.0,
        "without prior consent"                      : 2.0,

        # ── Without-notice family ──────────────────────────────────────
        "without notice"                             : 2.5,
        "without prior notice"                       : 2.5,
        "without prior warning"                      : 2.5,
        "without warning"                            : 2.0,
        "without any notice"                         : 2.5,
        "without further notice"                     : 2.0,

        # ── Without-compensation / payment family ─────────────────────
        "without compensation"                       : 2.5,
        "without any compensation"                   : 2.5,
        "without additional compensation"            : 2.0,
        "without any additional compensation"        : 2.5,
        "without remuneration"                       : 2.0,
        "without pay"                                : 2.5,
        "without any dues"                           : 2.5,
        "without payment"                            : 2.0,
        "no payment shall be due"                    : 2.5,
        "no claim shall lie"                         : 2.5,
        "no claim shall be entertained"              : 2.5,
        "no refund"                                  : 2.0,
        "non-refundable"                             : 2.0,

        # ── Without-process / inquiry family ──────────────────────────
        "without any inquiry"                        : 3.0,
        "without inquiry"                            : 2.5,
        "without any hearing"                        : 3.0,
        "without providing any opportunity"          : 3.0,
        "without issuing any charge sheet"           : 3.0,
        "without conducting any domestic inquiry"    : 3.0,
        "without show cause notice"                  : 2.5,
        "without any reason"                         : 2.5,
        "without reason"                             : 2.0,
        "without explanation"                        : 2.0,
        "for any reason or no reason"                : 3.0,
        "for any reason whatsoever"                  : 2.5,
        "no reason required"                         : 2.5,

        # ── Waiver of rights ──────────────────────────────────────────
        "waives all"                                 : 2.0,
        "waives any"                                 : 1.5,
        "waives all moral rights"                    : 2.5,
        "irrevocably waives"                         : 2.5,
        "hereby waives"                              : 2.0,
        "waive any right"                            : 2.0,
        "waiver of"                                  : 1.5,
        "no right of set-off"                        : 2.0,
        "no set-off"                                 : 2.0,

        # ── Liability exclusion / cap ──────────────────────────────────
        "in no event shall"                          : 2.0,
        "shall not be liable"                        : 1.5,
        "under no circumstances"                     : 2.0,
        "shall in no circumstances exceed"           : 2.0,
        "aggregate liability"                        : 1.5,
        "total liability shall not exceed"           : 2.0,
        "cap on liability"                           : 1.5,
        "without any liability"                      : 2.5,
        "no liability whatsoever"                    : 2.5,
        "excludes all liability"                     : 2.5,
        "consequential"                              : 1.5,   # consequential damages excluded
        "indirect damages"                           : 1.5,
        "special damages"                            : 1.5,
        "punitive damages"                           : 1.5,
        "loss of profits"                            : 1.5,   # excluded losses
        "loss of revenue"                            : 1.5,
        "loss of business"                           : 1.5,

        # ── Warranty disclaimer ────────────────────────────────────────
        "expressly disclaims"                        : 2.5,
        "disclaims all"                              : 2.5,
        "disclaims all implied warranties"           : 3.0,
        "as is"                                      : 1.5,   # "as-is" no warranty
        "as-is"                                      : 1.5,
        "without warranty"                           : 2.0,
        "no warranty"                                : 2.0,
        "makes no representation"                    : 1.5,
        "makes no warranty"                          : 2.0,
        "to the fullest extent permitted by law"     : 1.5,
        "to the maximum extent permitted"            : 1.5,

        # ── Forfeiture ────────────────────────────────────────────────
        "forfeit"                                    : 2.0,
        "forfeiture"                                 : 2.0,
        "forfeit any pending dues"                   : 3.0,
        "forfeit the deposit"                        : 2.5,
        "forfeit the security"                       : 2.5,
        "forfeit the earnest money"                  : 2.5,
        "shall be forfeited"                         : 2.5,
        "stands forfeited"                           : 2.5,

        # ── Unilateral modification ────────────────────────────────────
        "amend, modify, or vary"                     : 2.5,
        "amend or modify"                            : 2.0,
        "unilaterally amend"                         : 3.0,
        "without the employee's consent"             : 2.5,
        "without the other party's consent"          : 2.5,
        "take immediate effect"                      : 2.0,
        "constitute acceptance"                      : 2.5,
        "deemed acceptance"                          : 2.5,
        "deemed to have accepted"                    : 2.5,
        "effective date of any amendment"            : 2.0,
        "continued use shall constitute"             : 2.0,   # SaaS unilateral ToS change

        # ── IP overreach ──────────────────────────────────────────────
        "irrevocably assigns"                        : 2.5,
        "irrevocable assignment"                     : 2.5,
        "assigns all rights"                         : 2.0,
        "assigns all intellectual property"          : 2.5,
        "outside working hours"                      : 2.0,
        "using personal resources"                   : 1.5,
        "personal time"                              : 1.5,
        "personal equipment"                         : 1.5,
        "whether or not related to employment"       : 2.0,
        "worldwide, perpetual"                       : 2.0,
        "perpetual, irrevocable"                     : 2.0,
        "in perpetuity"                              : 2.0,

        # ── Non-compete / non-solicitation ────────────────────────────
        "shall not directly or indirectly engage"    : 2.5,
        "shall not engage in any business"           : 2.0,
        "shall not compete"                          : 2.0,
        "competes with the company"                  : 2.0,
        "competing business"                         : 1.5,
        "shall not solicit"                          : 2.0,
        "shall not approach"                         : 2.0,
        "entice, or hire"                            : 2.0,
        "poach"                                      : 2.0,
        "for competitive purposes"                   : 2.0,
        "territory of india"                         : 1.5,   # broad geographic scope
        "worldwide territory"                        : 2.0,

        # ── Employment-specific Con ────────────────────────────────────
        "as per management decision"                 : 2.0,
        "at the discretion of the management"        : 2.5,
        "management reserves"                        : 2.0,
        "management may"                             : 1.5,
        "without subsistence allowance"              : 2.5,
        "without any inquiry or hearing"             : 3.0,
        "no domestic inquiry"                        : 3.0,
        "suspension without pay"                     : 2.5,
        "termination at will"                        : 2.5,
        "at will"                                    : 2.0,

        # ── Lease / real estate Con ────────────────────────────────────
        "landlord may re-enter"                      : 2.5,
        "right of re-entry"                          : 2.5,
        "right to evict"                             : 2.5,
        "evict without notice"                       : 3.0,
        "forfeit the deposit"                        : 2.5,
        "no cooling-off"                             : 2.0,
        "tenant shall bear all"                      : 2.0,   # all repair costs on tenant
        "at landlord's discretion"                   : 2.5,
        "landlord may terminate"                     : 1.5,
        "increase rent at any time"                  : 2.5,
        "rent revision at landlord's discretion"     : 2.5,

        # ── Loan / finance Con ────────────────────────────────────────
        "acceleration clause"                        : 2.0,
        "entire outstanding amount shall become due" : 3.0,
        "cross-default"                              : 2.0,
        "cross default"                              : 2.0,
        "event of default"                           : 1.5,
        "lender may at its sole discretion"          : 2.5,
        "penalty interest"                           : 2.0,
        "penal interest"                             : 2.0,
        "prepayment penalty"                         : 2.0,
        "prepayment charge"                          : 1.5,
        "lender reserves the right"                  : 2.0,
        "unilateral recall"                          : 2.5,
        "demand loan"                                : 1.5,
        "on demand"                                  : 1.5,

        # ── Franchise / distribution Con ──────────────────────────────
        "franchisor may terminate"                   : 1.5,
        "franchisor reserves"                        : 2.0,
        "without any goodwill payment"               : 2.5,
        "no compensation on termination"             : 2.5,
        "change territory at any time"               : 2.5,
        "exclusive territory may be revised"         : 2.5,
        "minimum purchase obligation"                : 1.5,

        # ── SaaS / technology Con ─────────────────────────────────────
        "may modify the service"                     : 1.5,
        "may discontinue"                            : 2.0,
        "suspend access immediately"                 : 2.0,
        "revoke access"                              : 2.0,
        "without prior notice to the user"           : 2.5,
        "no uptime guarantee"                        : 2.0,
        "no sla"                                     : 1.5,
        "best efforts only"                          : 1.5,
        "data may be deleted"                        : 2.0,
        "data deletion without notice"               : 2.5,
        "user data may be shared"                    : 2.0,
        "sell user data"                             : 3.0,

        # ── Share purchase / M&A Con ──────────────────────────────────
        "drag-along"                                 : 2.0,
        "drag along"                                 : 2.0,
        "forced sale"                                : 2.5,
        "dilution without consent"                   : 2.5,
        "anti-dilution not applicable"               : 2.0,
        "board may issue shares"                     : 1.5,
        "without shareholder approval"               : 2.5,
        "promoter reserves the right"                : 2.0,

        # ── Punitive / penalty language ───────────────────────────────
        "liquidated damages"                         : 1.5,
        "penalty clause"                             : 2.0,
        "punitive"                                   : 1.5,
        "damages without proof"                      : 2.5,
        "irrespective of actual loss"                : 2.0,
        "without proof of actual damage"             : 2.0,

        # ── Broad indemnification against the weaker party ────────────
        "shall indemnify the company"                : 2.0,
        "indemnify the lessor"                       : 2.0,
        "indemnify the lender"                       : 2.0,
        "indemnify the franchisor"                   : 2.0,
        "without any limitation"                     : 2.5,
        "unlimited indemnity"                        : 2.5,
        "indemnify against all losses"               : 2.0,

        # ── Automatic renewal traps ────────────────────────────────────
        "automatically renew"                        : 1.5,
        "auto-renew"                                 : 1.5,
        "deemed renewed"                             : 2.0,
        "unless cancelled in writing"                : 1.5,
        "cancellation must be received"              : 1.5,
        "no cancellation after"                      : 2.0,
        "non-cancellable"                            : 2.0,

        # ── Dispute / governing law Con ───────────────────────────────
        "exclusive jurisdiction of courts chosen by" : 2.0,
        "arbitration costs borne by claimant"        : 2.0,
        "costs shall be borne by the employee"       : 2.5,
        "costs shall be borne by the tenant"         : 2.0,
        "costs shall be borne by the borrower"       : 2.0,
    }

    # ══════════════════════════════════════════════════════════════════
    # PRO PATTERNS  — rights granted, fair process, protective language
    # ══════════════════════════════════════════════════════════════════
    pro_patterns = {

        # ── Universal fairness signals ─────────────────────────────────
        "prior written notice"                       : 2.5,
        "written notice"                             : 1.5,
        "by either party"                            : 1.5,
        "either party may"                           : 1.5,
        "both parties"                               : 1.0,
        "mutual agreement"                           : 2.0,
        "mutually agreed"                            : 2.0,
        "by mutual consent"                          : 2.0,
        "good faith"                                 : 2.0,
        "in good faith"                              : 2.0,
        "best efforts"                               : 1.0,
        "reasonable efforts"                         : 1.5,
        "commercially reasonable"                    : 1.5,
        "not be unreasonably withheld"               : 2.0,
        "not unreasonably withheld"                  : 2.0,
        "not unreasonably delayed"                   : 1.5,
        "proportionate"                              : 1.5,

        # ── Rights explicitly granted ──────────────────────────────────
        "shall be entitled"                          : 1.5,
        "entitled to"                                : 1.5,
        "has the right to"                           : 1.5,
        "right to"                                   : 1.0,
        "rights of the"                              : 1.0,
        "shall have the right"                       : 1.5,
        "may exercise"                               : 1.0,
        "option to"                                  : 1.0,
        "shall be provided"                          : 1.5,

        # ── Notice periods ─────────────────────────────────────────────
        "thirty (30) days"                           : 2.0,
        "sixty (60) days"                            : 1.5,
        "ninety (90) days"                           : 1.5,
        "fifteen (15) days"                          : 1.5,
        "seven (7) days"                             : 1.0,
        "advance notice"                             : 1.5,
        "prior notice of"                            : 1.5,
        "notice period"                              : 1.5,

        # ── Cure / remedy periods ──────────────────────────────────────
        "cure period"                                : 2.5,
        "remedy such breach"                         : 2.0,
        "remedy the breach"                          : 2.0,
        "opportunity to cure"                        : 2.5,
        "right to cure"                              : 2.5,
        "cured within"                               : 2.0,
        "rectify the breach"                         : 2.0,
        "notice specifying the breach"               : 2.0,

        # ── Mutual amendment / consent requirement ─────────────────────
        "signed by both parties"                     : 2.5,
        "signed by authorised representatives"       : 2.0,
        "written consent of both"                    : 2.5,
        "consent of the other party"                 : 2.0,
        "prior written consent of the other"         : 2.0,
        "no amendment shall be valid unless"         : 2.5,
        "in writing and signed"                      : 2.0,
        "written agreement of both"                  : 2.5,

        # ── Fair payment / settlement ──────────────────────────────────
        "pro-rata"                                   : 2.0,
        "pro rata"                                   : 2.0,
        "full and final settlement"                  : 1.5,
        "full settlement"                            : 1.5,
        "salary in lieu"                             : 2.0,
        "notice pay"                                 : 2.0,
        "dues payable"                               : 1.5,
        "upon full payment"                          : 1.5,
        "within thirty days of"                      : 1.5,
        "within 30 days"                             : 1.5,
        "upon receipt of invoice"                    : 1.5,

        # ── Severability / preservation of rights ─────────────────────
        "severability"                               : 1.0,
        "remaining provisions shall continue"        : 1.0,
        "without prejudice to"                       : 1.5,
        "non-waiver"                                 : 1.0,
        "does not constitute a waiver"               : 1.5,

        # ── Dispute resolution fairness ────────────────────────────────
        "good faith negotiation"                     : 2.0,
        "amicable settlement"                        : 2.0,
        "mediation"                                  : 1.5,
        "arbitration and conciliation act"           : 2.0,
        "arbitration act, 1996"                      : 2.0,
        "sole arbitrator appointed by mutual"        : 2.5,
        "costs of arbitration shall be borne equally": 2.0,
        "equal costs"                                : 1.5,
        "each party shall bear its own costs"        : 1.5,

        # ── Force majeure protection ───────────────────────────────────
        "force majeure"                              : 1.5,
        "beyond the reasonable control"              : 1.5,
        "reasonable steps to mitigate"               : 2.0,
        "mitigation of loss"                         : 1.5,

        # ── Liability carve-outs (protective) ─────────────────────────
        "shall not apply in cases of fraud"          : 2.5,
        "does not apply to fraud"                    : 2.5,
        "wilful misconduct"                          : 2.0,
        "gross negligence"                           : 2.0,
        "fraud or wilful default"                    : 2.5,
        "death or personal injury"                   : 2.0,

        # ── Warranties (protecting the receiving party) ────────────────
        "warrants that"                              : 1.5,
        "represents and warrants"                    : 2.0,
        "warranty of"                                : 1.5,
        "fitness for purpose"                        : 1.5,
        "merchantability"                            : 1.0,
        "conform to specifications"                  : 1.5,
        "free from defects"                          : 1.5,
        "professional manner"                        : 1.5,
        "industry standards"                         : 1.5,
        "best practices"                             : 1.0,

        # ── IP fairness ────────────────────────────────────────────────
        "royalty-free"                               : 2.0,
        "non-exclusive"                              : 1.5,
        "licence to use"                             : 1.5,
        "vest in the client"                         : 2.5,
        "vest in and become"                         : 2.0,
        "exclusive property of the client"           : 2.5,
        "ownership shall transfer"                   : 2.0,
        "background ip"                              : 1.0,
        "retain ownership"                           : 1.5,
        "each party retains"                         : 1.5,

        # ── Indemnity in favour of the weaker party ────────────────────
        "service provider shall indemnify"           : 2.0,
        "indemnify and hold harmless the client"     : 2.0,
        "indemnify and hold harmless the employee"   : 2.5,
        "indemnify and hold harmless the tenant"     : 2.0,
        "third-party claims"                         : 1.5,
        "infringement of third-party"                : 1.5,

        # ── Confidentiality carve-outs (fair exceptions) ──────────────
        "publicly available"                         : 1.5,
        "publicly known"                             : 1.5,
        "independently developed"                    : 1.5,
        "rightfully received"                        : 1.5,
        "required by law to disclose"                : 1.5,
        "compelled by court order"                   : 1.5,
        "need-to-know basis"                         : 1.5,
        "need-to-know"                               : 1.5,

        # ── Employment — statutory Indian law references ───────────────
        "as per the industrial disputes act"         : 3.0,
        "industrial disputes act"                    : 2.5,
        "industrial disputes act, 1947"              : 3.0,
        "as per the payment of gratuity act"         : 3.0,
        "payment of gratuity act"                    : 2.5,
        "payment of gratuity act, 1972"              : 3.0,
        "provident fund"                             : 2.5,
        "employees' provident fund"                  : 3.0,
        "employees provident fund"                   : 3.0,
        "epf act"                                    : 2.5,
        "gratuity"                                   : 2.0,
        "esi"                                        : 2.0,
        "esic"                                       : 2.0,
        "employees' state insurance"                 : 2.5,
        "employees state insurance"                  : 2.5,
        "esic act"                                   : 2.5,
        "as per labour law"                          : 2.0,
        "applicable labour law"                      : 2.5,
        "retrenchment compensation"                  : 2.5,
        "fifteen (15) days wages"                    : 2.0,
        "bonus act"                                  : 2.0,
        "payment of bonus act"                       : 2.5,
        "minimum wages act"                          : 2.5,
        "factories act"                              : 2.0,
        "shops and establishments act"               : 2.0,
        "maternity benefit"                          : 2.5,
        "maternity benefit act"                      : 3.0,
        "maternity benefit act, 1961"                : 3.0,
        "maternity leave"                            : 2.5,
        "paternity leave"                            : 2.0,
        "paid leave"                                 : 2.0,
        "earned leave"                               : 2.0,
        "annual leave"                               : 1.5,
        "sick leave"                                 : 1.5,
        "casual leave"                               : 1.5,
        "medical insurance"                          : 1.5,
        "health insurance"                           : 1.5,
        "group insurance"                            : 1.5,
        "insurance coverage"                         : 1.5,
        "severance"                                  : 2.0,
        "severance pay"                              : 2.5,
        "fair domestic inquiry"                      : 3.0,
        "domestic inquiry"                           : 2.5,
        "written charge-sheet"                       : 2.5,
        "charge sheet"                               : 2.0,
        "charge-sheet and opportunity to respond"    : 3.0,
        "opportunity to respond"                     : 2.0,
        "opportunity to be heard"                    : 2.5,
        "principles of natural justice"              : 3.0,
        "right to raise a grievance"                 : 2.5,
        "grievance redressal"                        : 2.0,
        "grievance mechanism"                        : 2.0,
        "works committee"                            : 1.5,
        "labour court"                               : 2.0,

        # ── Lease / real estate Pro ────────────────────────────────────
        "quiet enjoyment"                            : 2.5,
        "peaceful possession"                        : 2.5,
        "right to peaceful enjoyment"                : 2.5,
        "landlord shall maintain"                    : 2.0,
        "structural repairs by landlord"             : 2.0,
        "security deposit refund"                    : 2.0,
        "refund of deposit"                          : 2.0,
        "deposit shall be refunded"                  : 2.5,
        "within 30 days of vacating"                 : 2.0,
        "rent control act"                           : 2.5,
        "transfer of property act"                   : 2.0,
        "notice before eviction"                     : 2.5,
        "tenant's right"                             : 2.0,
        "lessee shall have the right"                : 2.0,

        # ── Loan / finance Pro ────────────────────────────────────────
        "right of prepayment"                        : 2.0,
        "prepayment without penalty"                 : 2.5,
        "no prepayment penalty"                      : 2.5,
        "fixed interest rate"                        : 1.5,
        "interest shall not exceed"                  : 2.0,
        "interest at the rate of"                    : 1.0,
        "emi schedule"                               : 1.5,
        "moratorium period"                          : 1.5,
        "right to foreclose"                         : 1.0,
        "borrower may prepay"                        : 2.0,
        "rbi guidelines"                             : 2.5,
        "fair practices code"                        : 2.5,
        "banking regulation act"                     : 2.0,
        "consumer protection act"                    : 2.5,

        # ── Share purchase / M&A Pro ──────────────────────────────────
        "tag-along"                                  : 2.0,
        "tag along"                                  : 2.0,
        "right of first refusal"                     : 2.0,
        "rofr"                                       : 2.0,
        "anti-dilution"                              : 2.0,
        "anti dilution"                              : 2.0,
        "pre-emption right"                          : 2.0,
        "pre-emptive right"                          : 2.0,
        "drag-along rights of minority"              : 1.5,
        "minority protection"                        : 2.5,
        "shareholder approval"                       : 2.0,
        "board approval required"                    : 1.5,
        "supermajority"                              : 1.5,
        "affirmative vote"                           : 1.5,
        "companies act, 2013"                        : 2.0,
        "sebi regulations"                           : 2.0,

        # ── Franchise / distribution Pro ──────────────────────────────
        "exclusive territory"                        : 1.5,
        "protected territory"                        : 2.0,
        "goodwill payment on termination"            : 2.5,
        "right to renew"                             : 1.5,
        "renewal option"                             : 1.5,
        "right of first renewal"                     : 2.0,
        "franchisee shall be provided"               : 1.5,
        "training shall be provided"                 : 1.5,
        "support shall be provided"                  : 1.5,
        "reasonable notice before"                   : 2.0,

        # ── SaaS / technology Pro ─────────────────────────────────────
        "uptime guarantee"                           : 2.0,
        "service level agreement"                    : 2.0,
        "sla"                                        : 1.5,
        "data portability"                           : 2.0,
        "right to export data"                       : 2.0,
        "data shall not be shared"                   : 2.0,
        "gdpr"                                       : 1.5,
        "data protection"                            : 1.5,
        "privacy policy"                             : 1.0,
        "right to delete"                            : 2.0,
        "right to erasure"                           : 2.0,
        "it act, 2000"                               : 1.5,
        "information technology act"                 : 1.5,

        # ── Construction / works Pro ──────────────────────────────────
        "defect liability period"                    : 2.0,
        "retention money"                            : 1.5,
        "release of retention"                       : 2.0,
        "completion certificate"                     : 1.5,
        "contractor shall rectify"                   : 2.0,
        "snag list"                                  : 1.5,
        "arbitration under"                          : 1.5,
        "extension of time"                          : 1.5,
        "variation order"                            : 1.0,
        "approved variation"                         : 1.5,
        "engineer's decision"                        : 1.0,
    }

    # ══════════════════════════════════════════════════════════════════
    # HARD-NEUTRAL SECTION OVERRIDES
    # These section types are almost always definitional/procedural.
    # Override the score entirely — skip pattern matching.
    # ══════════════════════════════════════════════════════════════════
    neutral_sections = {
        # Universal
        "definitions", "interpretation", "governing law",
        "governing law and jurisdiction", "entire agreement",
        "commencement", "introduction", "preamble",
        "jurisdiction", "recitals", "background",
        "general provisions", "miscellaneous",
        "notices", "notice", "communication",
        "counterparts", "execution",
        "severability", "waiver",
        "assignment",   # assignment itself is usually neutral procedural
        "schedule", "annexure", "appendix",
        # Employment
        "place of work", "working hours",
        # Commercial
        "entire agreement", "further assurance",
        # Real estate
        "schedule of property", "description of premises",
        # Finance
        "representations", "conditions precedent",
    }
    if any(ns in section for ns in neutral_sections):
        return LABEL_MAP["Neutral"], 1.0

    # ══════════════════════════════════════════════════════════════════
    # HARD-CON SECTION OVERRIDES
    # Section names that are inherently one-sided by design.
    # ══════════════════════════════════════════════════════════════════
    con_sections = {
        "termination without notice",
        "termination without cause",
        "termination for convenience",   # often one-sided
        "dismissal",
        "suspension without pay",
        "forfeiture",
        "penalty",
        "liquidated damages",
    }
    if any(cs in section for cs in con_sections):
        return LABEL_MAP["Con"], 0.95

    # ══════════════════════════════════════════════════════════════════
    # SCORE COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    for pat, w in con_patterns.items():
        if pat in text:
            score -= w * bias

    for pat, w in pro_patterns.items():
        if pat in text:
            score += w

    # ── Section-specific score adjustments ────────────────────────────

    # Confidentiality: NDA language ("at any time", "without notice") fires
    # Con patterns but is normal in mutual NDAs. Push towards Neutral
    # unless genuinely one-sided indemnity/forfeiture language present.
    confidentiality_sections = {
        "confidentiality", "nda", "non-disclosure",
        "confidential information", "trade secrets",
    }
    if any(cs in section for cs in confidentiality_sections):
        score += 2.0

    # Warranty/representation sections: warranties favour the protected party.
    warranty_sections = {
        "warranties", "representations", "representations and warranties",
        "warranty", "indemnification", "indemnity",
    }
    if any(ws in section for ws in warranty_sections):
        score += 1.0   # slight Pro push — warranties are generally protective

    # Limitation of liability sections: almost always Con for weaker party.
    liability_sections = {
        "limitation of liability", "liability", "exclusion of liability",
        "cap on liability", "exclusions",
    }
    if any(ls in section for ls in liability_sections):
        score -= 1.5   # push towards Con

    # ── Final classification ───────────────────────────────────────────
    if score <= -1.5:
        return LABEL_MAP["Con"],     min(1.0, abs(score) / 6)
    elif score >= 1.5:
        return LABEL_MAP["Pro"],     min(1.0, abs(score) / 6)
    else:
        return LABEL_MAP["Neutral"], 0.3


# ─────────────────────────────────────────
# Graph builder — O(n log n) via FAISS top-k
# ─────────────────────────────────────────
def build_graph(clauses, embeddings):
    import faiss

    n = len(clauses)
    edge_src, edge_dst = [], []

    # Sequential edges
    for i in range(n - 1):
        edge_src.extend([i, i + 1])
        edge_dst.extend([i + 1, i])

    # Same-section edges
    for i, j in itertools.combinations(range(n), 2):
        if clauses[i]["section"] == clauses[j]["section"]:
            edge_src.extend([i, j])
            edge_dst.extend([j, i])

    # Semantic edges — FAISS top-5 instead of O(n²) cosine matrix
    emb_np = embeddings.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb_np)
    index = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)
    k = min(6, n)                          # top-5 neighbors + self
    scores, neighbors = index.search(emb_np, k)

    for i in range(n):
        for rank in range(1, k):           # skip rank 0 (self)
            j   = int(neighbors[i, rank])
            sim = float(scores[i, rank])
            if sim >= SIM_THRESHOLD and j != i:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return edge_index


# ─────────────────────────────────────────
# Amendment templates (India-aware)
# ─────────────────────────────────────────
def retrieve_best_template(clause_text):
    text = clause_text.lower()
    if any(k in text for k in ["terminate", "termination"]):
        return (
            "only after providing prior written notice of at least 30 days "
            "or notice pay in lieu thereof, as per the Industrial Disputes Act"
        )
    if any(k in text for k in ["dismiss", "misconduct"]):
        return (
            "only after a fair domestic inquiry with written charge-sheet, "
            "opportunity to respond, and as per applicable labour law"
        )
    if any(k in text for k in ["modify", "sole discretion"]):
        return (
            "only after prior written notice of 15 days and employee consultation, "
            "subject to applicable labour regulations"
        )
    if "suspend" in text:
        return (
            "only after written notice and opportunity for the employee to respond, "
            "with subsistence allowance as per applicable rules"
        )
    if any(k in text for k in ["forfeit", "forfeiture"]):
        return (
            "subject to applicable provisions of the Payment of Gratuity Act "
            "and relevant labour statutes"
        )
    return (
        "with prior written notice and fair review, "
        "in accordance with applicable Indian labour law"
    )


def build_amendment(original):
    template = retrieve_best_template(original)
    original = original.rstrip(".")
    return f"{original}, {template}."


# ─────────────────────────────────────────
# Main pipeline — NO training loop
# ─────────────────────────────────────────
def run_pipeline(clauses, role="Employee"):
    results = []
    if not clauses:
        return results

    logger.info(f"Running pipeline on {len(clauses)} clauses (role={role})")

    # ── Embed clauses ──────────────────────────────────────────────────
    model  = get_sentence_model()
    texts  = [c["text"] for c in clauses]

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).to(DEVICE)

    # ── Weak-label directly (replaces training loop) ───────────────────
    labels     = []
    confidences = []
    for c in clauses:
        lbl, conf = weak_label_clause_v2(c, role=role)
        labels.append(lbl)
        confidences.append(conf)

    # ── Build graph (kept for future use / GAT upgrade path) ──────────
    edge_index = build_graph(clauses, embeddings)
    logger.debug(f"Graph: {len(clauses)} nodes, {edge_index.shape[1]} edges")

    # ── Assemble results ───────────────────────────────────────────────
    for i, c in enumerate(clauses):
        pred       = labels[i]
        conf       = confidences[i]
        label_name = INV_LABEL_MAP[pred]

        results.append({
            "clause_id" : c["clause_id"],
            "section"   : c["section"],
            "label"     : label_name,
            "confidence": round(conf, 4),
            "original"  : c["text"],
            "amended"   : build_amendment(c["text"]) if label_name == "Con" else None,
        })

    con_count = sum(1 for r in results if r["label"] == "Con")
    logger.info(f"Pipeline complete — Con: {con_count}, Pro: {sum(1 for r in results if r['label']=='Pro')}, Neutral: {sum(1 for r in results if r['label']=='Neutral')}")
    return results