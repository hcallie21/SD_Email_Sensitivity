import yaml
from pathlib import Path
import re
from faker import Faker
import secrets
import random
import uuid
import string

fake = Faker()

ASSET_TYPES = ["Stock", "Share", "Bond", "Mutual Fund","Exchange-Traded Fund (ETF)",
    "Futures Contract", "Option", "Real Estate Property", "Intellectual Property", "Patent",
    "Trademark", "Copyright", "Certificate of Deposit","Index Fund","Pension Fund","Trust Fund"]
UNIVERSITY_NAMES = [
    "Cypress City University", "Northgate University", "Evergreen University",
    "Redwood State University", "Harborview University", "Riverbend University",
    "Summit Ridge University", "Stonebridge University", "Granite Peak University",
    "Blue Harbor University", "Highland University", "Beacon Hills University",
    "Prairie Plains University", "Pioneer Valley University", "Aurora University",
    "Metroline University", "Cedar Point University", "Silverline University",
    "Sunrise University", "OpenField University", "Catalyst University",
    "Brightwater University", "Lighthouse University", "Palisade University",
    "Heritage University", "Horizon University", "Everglade University",
    "Wilph Creek University", "Seaside University", "City University"
]
SCHOOL_NAMES = [
    "Cypress High School", "Northgate High School", "Evergreen High School",
    "Redwood High School", "Harborview High School", "Riverbend High School",
    "Summit Ridge High", "Stonebridge High", "Granite Peak High", "Blue Harbor High",
    "Highland High", "Beacon Hills High", "Prairie Plains High", "Pioneer Valley High",
    "Aurora High", "Metroline High", "Cedar Point High", "Silverline High",
    "Sunrise High", "OpenField High", "Catalyst High", "Brightwater High",
    "Lighthouse High", "Palisade High", "Heritage High", "Horizon High",
    "Wilph Creek High", "Seaside High", "Maple Grove High", "Hillcrest High"
]
BANK_NAMES = [
    "Number One Bank", "Totally Real Bank", "Pioneer Trust", "Brightwater Capital",
    "First Elm Bank", "Horizon National", "Cypress Credit Union", "Northgate Savings",
    "Blue Harbor Bank", "Evergreen Financial", "Summit Mutual", "Stonebridge Bank",
    "Redwood Trust", "Riverbend Bank", "Silverline Credit", "Prairie Bank & Trust",
    "Harborview Financial", "Lighthouse Bank", "Granite State Bank", "Highland Capital Bank",
    "Cedar Street Bank", "OpenField Credit Union", "Catalyst Bank", "Beacon Hill Bank",
    "Ironwood Financial", "Aurora Community Bank", "Palisade Trust", "Sunrise Bank",
    "Heritage Capital Bank", "Metroline Bank"
]
GOV_AGENCY_NAMES = [
    "National Health Authority", "Department of Public Works", "Office of Civic Services",
    "Bureau of Transportation", "Agency for Urban Planning", "Commission on Education",
    "Department of Revenue and Taxation", "Public Safety Commission", "Housing & Development Board",
    "Environmental Protection Office", "Maritime Security Agency", "Energy Regulation Commission",
    "Agency for Workforce Development", "Office of Small Business Support",
    "National Cybersecurity Directorate", "Food & Drug Oversight Bureau",
    "Public Utilities Commission", "Parks & Conservation Service",
    "Water Resources Authority", "Disaster Response Agency", "National Statistics Office",
    "Trade & Industry Council", "Consumer Protection Bureau", "Cultural Affairs Commission",
    "Office of Records & Archives", "Innovation & Technology Agency",
    "Transportation Safety Board", "Office of Elections & Ethics",
    "Border & Customs Service", "Rural Development Authority"
]
HOSPITAL_NAMES = [
    "Solace Medical Center", "Harborview General Hospital", "Evergreen Clinic",
    "Redwood Regional Hospital", "Cedar Grove Medical", "Blue Harbor Clinic",
    "Riverbend Community Hospital", "Summit Ridge Medical Center", "Stonebridge Clinic",
    "Granite Peak Hospital", "Highland Care Center", "Beacon Hills Medical",
    "Prairie Plains Hospital", "Pioneer Valley Health", "Aurora Medical Pavilion",
    "Metroline General", "Cedar Point Health", "Silverline Medical Center",
    "Sunrise Community Hospital", "OpenField Clinic", "Catalyst Health Center",
    "Brightwater Hospital", "Lighthouse Medical", "Palisade Regional Hospital",
    "Heritage Health", "Horizon Medical Center", "Wilph Creek Clinic",
    "Seaside Medical", "Maple Grove Hospital", "Hillcrest Clinic"
]
MEDICATIONS = [
    "Atorvastatin", "Metformin", "Lisinopril", "Levothyroxine", "Amlodipine",
    "Omeprazole", "Albuterol", "Sertraline"
]
PROCEDURES = [
    "appendectomy", "breast biopsy", "carotid endarterectomy", "cataract surgery", "cesarean section",
    "cholecystectomy", "coronary artery bypass", "debridement of wound", "dilation and curettage", "skin graft",
    "hemorrhoidectomy", "hysterectomy", "hysteroscopy", "inguinal hernia repair", "mastectomy", "prostatectomy",
    "tonsillectomy", "Cognitive Behavioral Therapy", "Dialectical Behavior Therapy",
    "Physical Therapy", "Occupational Therapy", "Speech Therapy"
]
SYMPTOMS = [
    "fever", "cough", "fatigue", "shortness of breath", "headache", "nausea", "dizziness"
]
RACES = [
    "Asian", "Black", "Hispanic or Latino", "White", "Native American",
    "Pacific Islander", "Middle Eastern or North African", "Multiracial", "Other"
]
RELIGIONS = [
    "Christianity", "Islam", "Hinduism", "Buddhism", "Judaism",
    "Sikhism", "Atheism", "Agnosticism", "Other"
]
FIELDS_OF_STUDY = [
    "Computer Science", "Mechanical Engineering", "Biology", "Economics", "Psychology",
    "Mathematics", "Chemistry", "Physics", "Political Science", "Sociology", "History",
    "Business Administration", "Nursing", "Education", "Environmental Science"
]
COURSES = [
    "Advanced Algorithms", "Thermodynamics II", "Organic Chemistry", "Microeconomics",
    "Developmental Psychology", "Linear Algebra", "Electromagnetism", "Comparative Politics",
    "Corporate Finance", "Medical Microbiology", "Instructional Design", "Data Visualization"
]
SUBJECTS = ["algebra","biology","chemistry","geometry","literature","world history","geography",
            "U.S. history","physics","pre-calculus","life science","health","language arts"]
DOCUMENT_TYPES = [
    "Invoice", "Receipt", "Memo", "Report", "Statement", "Policy", "Contract", "Certificate",
    "Application", "Permit", "Form", "Ticket"
]
CONTRACT_NAMES = [
    "Employment Agreement","Consulting Agreement",
    "Non-Disclosure Agreement (NDA)","Service Level Agreement (SLA)","Software Licensing Agreement", "Vendor Supply Contract",
    "Sales Agreement","Partnership Agreement","Lease Agreement","Loan Agreement","Purchase Order Contract",
    "Investment Agreement","Distribution Agreement","Maintenance Contract","Support Services Contract",
    "Research Collaboration Agreement","Confidentiality Agreement","Franchise Agreement","Merger Agreement",
    "Acquisition Contract","Joint Venture Agreement","Independent Contractor Agreement","Manufacturing Contract",
    "Outsourcing Agreement","Sponsorship Agreement","End-User License Agreement (EULA)", "Subscription Agreement",
    "Employment Termination Agreement","Renewal Contract","Settlement Agreement","Memorandum of Understanding (MOU)",
    "Service Renewal Contract", "Compliance Contract","Construction Contract",
    "Facility Management Agreement","Insurance Coverage Agreement", "Investment Subscription Contract",
    "Training Services Agreement", "Technology Transfer Agreement", "Consulting Retainer Agreement","Legal Representation Contract",
    "Data Processing Agreement (DPA)", "Security Services Contract"
]
LAW_FIRMS = [
    "Baxter & Cole LLP", "Hawthorne Legal Group", "Redwood & Finch", "Summit Law Partners",
    "Northgate & Fields", "Harbor & Stone Attorneys", "Evergreen Legal"
]
VENUES = [
    "Harborview Convention Center", "Summit Pavilion", "Granite Expo Hall",
    "Cedar Grove Center", "Blue Harbor Center", "Sunrise Ballroom"
]
SHOWS = [
        "Shadow Line","City of Echoes","The Last Harbor","Echo Valley","The Seventh Floor","Cold Meridian","Silent Verdict",
        "Hidden Agenda","Iron Bridge", "The Turning Point", "Kitchen Stories","Office Hours","The Neighbors Upstairs","Almost Famous Friends",
        "Weekend Plans","Room for Two","Startup Shenanigans","Late Checkout","Daily Grind","Three Floors Down","Chef’s Arena","The Big Renovation",
        "Ultimate Survival Challenge","Design Duel","Talent Quest Live","Next Big Brand","The Great Adventure Race",
        "Home Fix Nation","Planet Unseen","Voices of Tomorrow","Frontier Science","Behind the Data","True North",
        "Hidden Histories", "Cities in Motion","Deep Dive: The Ocean Frontier","Neon Horizon","Chronos Gate","The Outer Frame",
        "Wired Hearts","Parallel Skies", "Beyond Epsilon", "The Time Between Us"]

def rand_upper(n): return ''.join(random.choices(string.ascii_uppercase, k=n))
def rand_alphanum(n): return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))
def rand_hex(n): return ''.join(random.choices('0123456789abcdef', k=n))

def gen_short_id(prefix=None, digits=6):
    if prefix:
        return f"{prefix.upper()}-{random.randint(10**(digits-1), 10**digits - 1)}"
    return f"{random.randint(10**(digits-1), 10**digits - 1)}"
def get_amount(): return f'{round(random.uniform(1.0, 200000.0), 2)}'
def get_grade(): return f'{round(random.uniform(0.0, 100.0), 2)}'
def get_token(hexlen=32): return secrets.token_hex(hexlen//2) if hexlen % 2 == 0 else secrets.token_hex((hexlen+1)//2)
def get_uuid(): return str(uuid.uuid4())
def get_apikey():
    styles = [
        lambda: "AK-" + get_token(24),
        lambda: get_uuid().replace("-", ""),
        lambda: "".join(random.choices(string.ascii_letters + string.digits, k=32))
    ]
    return random.choice(styles)()
def get_application():
    nouns = [
        "Ocean", "Beehive", "Atlas", "Nimbus", "Vector", "Pilot", "Harbor",
        "Signal", "Zenith", "Vertex", "Meadow", "Cascade", "Nimbus", "Eclipse",
        "Frontier", "Beacon", "Tempo", "Pulse", "Voyager", "Halo", "Fusion"
    ]

    tech_prefixes = [
        "Meta", "Ultra", "Hyper", "Quantum", "Neuro", "Cyber", "Auto",
        "Prime", "Proto", "Info", "Data", "Aero", "Nano", "Omni", "Flex"
    ]

    suffixes = [
        "Soft", "Desk", "Hub", "Cloud", "Flow", "Works", "Suite", "System",
        "Logic", "Core", "Shift", "Link", "Engine", "Portal"
    ]

    syllables = [
        "vora", "luma", "tora", "nexa", "dara", "sura", "mira",
        "lena", "cora", "xira", "zuna", "vesta", "sera"
    ]

    pattern = random.choice(["noun", "compound", "prefixnoun", "nounsuffix", "syllable"])
    if pattern == "noun":return random.choice(nouns)
    if pattern == "compound":return random.choice(nouns) + random.choice(nouns)
    if pattern == "prefixnoun":return random.choice(tech_prefixes) + random.choice(nouns)
    if pattern == "nounsuffix":return random.choice(nouns) + random.choice(suffixes)
    if pattern == "syllable":return random.choice(nouns) + random.choice(syllables)

    return random.choice(nouns)  # fallback
def get_asset(): return str(random.choice(ASSET_TYPES)).lower()
def get_count(): return str(random.randint(1, 1000))
def get_date():
    random_date = fake.date_between(start_date='-100y', end_date='+100y')
    formats = [
        lambda d: d.strftime("%Y-%m-%d"),           # ISO
        lambda d: d.strftime("%b %d, %Y"),          # "Nov 10, 2025"
        lambda d: d.strftime("%B %d, %Y"),          # "November 10, 2025"
        lambda d: d.strftime("%m/%d/%Y"),           # US style
    ]
    return random.choice(formats)(random_date)
def get_judge(): return f'Judge {fake.name}'
def get_otp(): return str(random.randint(100000, 999999))
def get_pin(): return str(random.randint(1000, 9999))
def get_semester(): return random.choice(['summer','spring','fall','winter'])
def get_time(): return f'{random.randint(1,12)}:{random.randint(0,59)} {fake.am_pm()}'
def get_wallet_address(): return "0x" + secrets.token_hex(20)
def get_routing_number(): return str(random.randint(10**8, 10**9 - 1))
def get_school(): return random.choice(SCHOOL_NAMES)
def get_college(): return random.choice(UNIVERSITY_NAMES)
def get_bank(): return random.choice(BANK_NAMES)
def get_agency(): return random.choice(GOV_AGENCY_NAMES)
def get_provider():
    formats = [
        f'{random.choice(HOSPITAL_NAMES)}',
        f'Dr. {fake.last_name()}',
    ]
    return random.choice(formats)
def get_medication(): return random.choice(MEDICATIONS)
def get_treatment(): return random.choice(PROCEDURES)
def get_symptom(): return random.choice(SYMPTOMS)
def get_political_party(): return random.choice(["Democratic Party", "Republican Party", "Green Party", "Libertarian Party"])
def get_orientation(): return random.choice(["straight", "gay", "lesbian", "bisexual", "asexual", "pansexual", "queer"])
def get_race(): return random.choice(RACES)
def get_religion(): return random.choice(RELIGIONS)
def get_discipline(): return random.choice(FIELDS_OF_STUDY)
def get_course(): return random.choice(COURSES)
def get_subject(): return random.choice(SUBJECTS)
def get_document(): return random.choice(DOCUMENT_TYPES)
def get_contract(): return random.choice(CONTRACT_NAMES)
def get_lawfirm(): return random.choice(LAW_FIRMS)
def get_venue(): return random.choice(VENUES)
def get_department(): return random.choice(["Human Resources","Accounting","Finance","IT","Operations","Legal","Marketing","Sales","R&D","Customer Support"])
def get_court():
    formats = [
        f"{fake.city()} District Court",
        f"{fake.country()} Superior Court",
        f"{fake.state()} State Court",
    ]
    return random.choice(formats)
def get_platform(): return random.choice(["MySpace","LinkdIn","YouTube","Instagram","Facebook","Twitter","TikTok","Netflix","Hulu","Amazon Prime","Disney+"])
def get_password():
    formats = [
        f'{fake.password()}',
        f"{fake.word('noun')}{random.randint(0,999)}",
    ]
    return random.choice(formats)

def get_benefit_type(): return str(random.choice(["Unemployment", "Disability", "Housing Voucher", "Childcare Subsidy","Healthcare Allowance", "Retirement Pension", "Food Assistance"])).lower()
def get_broker(): return random.choice(["Stonebridge Capital", "Morrison Brokers", "Aurora Wealth","Northgate Securities", "Cedar Ridge Partners", "Halcyon Advisors", "Summit Brokerage Group"])
def get_case_number(): return f"{random.randint(19,25)}-{rand_upper(3)}-{gen_short_id(digits=6)}"
def get_citizenship_status(): return str(random.choice(["Citizen", "Lawful Permanent Resident", "Naturalized Citizen", "Temporary Resident", "Asylee", "Non-Resident Alien"])).lower()
def get_condition(): return str(random.choice(["Chronic back pain", "Type II diabetes", "Hypertension","Seasonal allergy", "Visual impairment", "Asthma"])).lower()
def get_crime(): return str(random.choice(["Fraud", "Burglary", "Theft", "Assault", "Vandalism", "Identity theft", "Money laundering", "Murder", "Arson"])).lower()
def get_event():return random.choice(["Annual Innovation Summit", "Community Health Fair", "Q3 Investor Briefing", "Spring Charity Gala","Project Kickoff Workshop", "Press Conference"])
def get_genre(): return random.choice(["Romance", "Comedy", "Drama", "Documentary", "Sci-fi", "Thriller", "True Crime"])
def get_investment_fund(): return random.choice(["Blue Harbor Growth Fund", "Atlas Venture Partners", "Summit Private Equity","Northfield Opportunity Fund", "Crescent Ridge Ventures", "Peregrine Capital"])
def get_organization():
    return random.choice([
        "Greenfield Research Institute", "Harbor Community Alliance",
        "Global Trade Network", "Axis Software Ltd.", "Frontier Logistics Inc.",
        "Metropolitan Arts Council"
    ])
def get_policy_document():
    return random.choice([
        "HR — Remote Work Policy (v2.1)", "Data Retention Policy",
        "Privacy & Data Use Guidelines", "Code of Conduct",
        "Travel Expense Policy (v1.0)", "Security Incident Response Plan"
    ])
def get_portfolio_id(): return f"PF-{rand_alphanum(random.randint(6,8))}"
def get_product():return f"{fake.word('adjective').capitalize()}{fake.word('noun').capitalize()}"
def get_product_type():
    return random.choice([
        "Consumer Electronics", "Home Appliance", "Software-as-a-Service",
        "Outdoor Gear", "Healthcare Device", "Personal Care", "Financial Software"
    ])
def get_project(): return random.choice(["Project Orion", "Platform Modernization", "Atlas Migration","Customer 360 Initiative", "Project Keystone", "NextGen Billing"])
def get_sentence():
    time = ["hours","days","months","years"]
    penal = ["of community service","in prison","without parole", "under house arrest"]
    return f"{random.randint(1,100)} {random.choice(time)} {random.choice(penal)}"
def get_server():
    env = random.choice(["prod", "dev", "stg", "qa"])
    role = random.choice(["web", "db", "app", "cache"])
    return f"{env}-{role}-{random.randint(1,99):02}"
def get_stock_symbol(): return rand_upper(random.randint(3,5))
def get_tax_form():return random.choice(["Form 1040", "Form W-2", "Form 1099-MISC", "Form 941", "Form 8862", "Schedule C"])
def get_thesis():
    return random.choice([
        "Machine Learning for Low-Resource Languages",
        "Urban Resilience and Climate Adaptation",
        "Optimizing Distributed Databases",
        "Narratives of Migration: A Qualitative Study",
        "Blockchain Applications in Supply Chains",
        "Comparative Study of Voting Systems"
    ])
def get_transaction_hash():return "0x" + rand_hex(16)
def get_visa_type():
    return random.choice([
        "H-1B (Specialty Occupation)", "F-1 (Student)", "B-2 (Visitor)",
        "L-1 (Intracompany Transfer)", "Permanent Resident (Green Card)", "K-1 (Fiancé(e))"
    ])
def get_show(): random.choice(SHOWS)

#============================
# Master Key
#============================
INDEX = {
    'access_code': lambda: gen_short_id(),
    "account": fake.user_name,
    "address": fake.street_address,
    'address2': fake.street_address,
    'amount':  get_amount,
    'api_key': get_apikey,
    'application': get_application,
    'asset_type': get_asset,
    'bank': get_bank,
    'bank_account': get_routing_number,
    'benefit_type': get_benefit_type,
    'brand': fake.company,
    'broker': get_broker,
    'card_number': fake.credit_card_number,
    'case_number': get_case_number,
    'citizenship_status': get_citizenship_status,
    'city': fake.city,
    'college': get_college,
    'college2': get_college,
    'company': fake.company,
    'company2': fake.company,
    'condition': get_condition,
    'contract': get_contract,
    'count': get_count,
    'country': fake.country,
    'course': get_course,
    'court': get_court,
    'crime': get_crime,
    'crypto_wallet': get_wallet_address,
    'crypto_wallet2': get_wallet_address,
    'currency': fake.currency_symbol,
    'cvv': lambda: gen_short_id(digits=3),
    'date': get_date,
    'date2': get_date,
    'department': get_department,
    'discipline': get_discipline,
    'document': get_document,
    'drivers_license': lambda: gen_short_id(rand_upper(1),8),
    'email': fake.email,
    'email2': fake.email,
    'employee_id': lambda: gen_short_id("EMPL",8),
    'ethnicity': get_race,
    'event': get_event,
    'genre': get_genre,
    'government_agency': get_agency,
    'grade': get_grade,
    'grade2': get_grade,
    'id_number': lambda: gen_short_id(digits=9),
     'investment_fund': get_investment_fund,
    'judge': get_judge,
    'legal_team': get_lawfirm,
    'login_link': fake.url,
    'medication': get_medication,
    'name': fake.name,
    'name2': fake.name,
    'organization': get_organization,
    'orientation': get_orientation,
    'orientation2': get_orientation,
    'otp': get_otp,
    'passport_number': fake.passport_number,
    'password': get_password,
    'phone': fake.phone_number,
    'phone2': fake.phone_number,
    'pin': get_pin,
    'platform': get_platform,
    'policy_document': get_policy_document,
    'political_party': get_political_party,
    'portfolio_id': get_portfolio_id,
    'portfolio_id2': get_portfolio_id,
    'position': fake.job,
    'product': get_product,
    'product_type': get_product_type,
    'program': get_application,
    'project': get_project,
    'provider': get_provider,
    'record_id': lambda: gen_short_id("REC",8),
    'religion': get_religion,
    'school': get_school,
    'semester': get_semester,
    'sentence': get_sentence,
    'server': get_server,
    'show': get_show,
    'stock_symbol': get_stock_symbol,
    'store': fake.company,
    'subject': get_subject,
    'symptom': get_symptom,
    'system': get_application,
    'tax_form': get_tax_form,
    'thesis': get_thesis,
    'time': get_time,
    'time2': get_time,
    'token': get_otp,
    'token_symbol': fake.cryptocurrency_code,
    'transaction_hash': get_transaction_hash,
    'treatment': get_treatment,
    'university': get_college,
    'venue': get_venue,
    'visa_type': get_visa_type,
    'website': fake.url,
}

#============================
# For Testing
#============================
def extract_placeholders_from_yaml_dir(template_dir="Templates"):
    all_placeholders = set()
    for path in Path(template_dir).rglob("*.yaml"):
        with open(path, "r") as f:
            try:
                data = yaml.safe_load(f)
                if not data:
                    continue
                for entry in data:
                    if isinstance(entry, dict):
                        text = entry.get("text", "")
                        all_placeholders.update(re.findall(r"\{([^}]+)}", text))
                        for span in entry.get("spans", []):
                            if isinstance(span, dict):
                                all_placeholders.update(re.findall(r"\{([^}]+)}", span.get("text", "")))
            except yaml.YAMLError:
                print(f"⚠️ Skipping invalid YAML file: {path}")
    return sorted(all_placeholders)

def build_placeholder_map(placeholders):
    """
    Accepts a list of placeholder names (strings) and returns a dict:
        { "placeholder": callable_that_returns_string }
    Each callable, when invoked, returns a single generated value (string).
    """
    mapping = {}
    for ph in placeholders:
        if ph in INDEX:
            gen = INDEX[ph]
        else:
            gen = "UNMAPPED"
        mapping[ph] = (lambda g=gen: (g() if callable(g) else g))
    return mapping
