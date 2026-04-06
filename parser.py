"""
cv_parser/parser.py
Pure parsing logic — no CLI, no FastAPI. Import and call parse_cv(text).
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def split_lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]

def first_match(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return clean(m.group(1)) if m and m.lastindex else (clean(m.group()) if m else None)


# ─────────────────────────────────────────────────────────────────────────────
# PDF text normalizer
# ─────────────────────────────────────────────────────────────────────────────

# Known section headings used to re-insert newlines before them
_SECTION_TITLES_RE = re.compile(
    r"(?<!\n)"  # not already preceded by newline
    r"(?<!Technical )"   # not a sub-label like "Technical Skills:"
    r"(?="
    r"(?:WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|EDUCATION|SKILLS?|PROJECTS?|"
    r"CERTIFICATIONS?|LANGUAGES?|VOLUNTEER(?:\s+ACTIVITIES)?|AWARDS?|SUMMARY|"
    r"PROFILE|OBJECTIVE|PUBLICATIONS?)"
    r"\b)",
    re.IGNORECASE,
)

# Bullet characters to split on
_BULLET_RE = re.compile(r"(?<!\n)(●|•|·|▸|▪|➤|➢|◆|▶|►)(?!\n)")


def normalize_pdf_text(text: str) -> str:
    """
    PDF extraction often collapses multiple visual lines into one long string,
    or splits single words onto separate lines.  This function reconstructs
    a clean, line-per-logical-unit layout.
    """
    lines = text.splitlines()
    # Pre-pass: split any line that contains multiple section headings inline
    # Only split on ALL-CAPS headings to avoid false positives like "Technical Skills:"
    _INLINE_SECTION_RE = re.compile(
        r"(?<!\A)\s{2,}(?="
        r"(?:WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|EDUCATION|SKILLS|PROJECTS|"
        r"CERTIFICATIONS|LANGUAGES|VOLUNTEER(?:\s+ACTIVITIES)?|AWARDS|SUMMARY|"
        r"PROFILE|OBJECTIVE|PUBLICATIONS)"
        r"(?:\s+[A-Z]|\s*$|\s*\n))",  # must be followed by another capital word or end
    )
    expanded: list[str] = []
    for raw in lines:
        parts = _INLINE_SECTION_RE.split(raw)
        expanded.extend(parts)
    lines = expanded

    merged: list[str] = []

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        # If line is a single short word/token that looks like a word fragment,
        # try to append it to the previous accumulated line
        if (
            merged
            and len(stripped.split()) <= 2
            and len(stripped) <= 20
            and not re.match(r"^(●|•|·|▸|▪|➤|➢|◆|▶|►)", stripped)
            and not re.match(
                r"^(WORK|EXPERIENCE|EDUCATION|SKILLS?|PROJECTS?|CERTIFICATIONS?|"
                r"LANGUAGES?|VOLUNTEER|AWARDS?|SUMMARY|PROFILE|OBJECTIVE)\b",
                stripped, re.I,
            )
        ):
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)

    # Re-join then surgically insert newlines before section headings and bullets
    combined = "\n".join(merged)

    # Fix split section headings like "WORK  \nEXPERIENCE  ..." → "WORK EXPERIENCE ..."
    combined = re.sub(
        r"\b(WORK|TECHNICAL|VOLUNTEER)\s*\n+\s*(EXPERIENCE|SKILLS?|ACTIVITIES?)\b",
        r"\1 \2",
        combined, flags=re.IGNORECASE,
    )

    # "...Technical \n\nSkills:" — orphaned sub-heading inside a job block.
    # Covers: "... Technical \n\nSkills:", "Technical \n\nSkills:", etc.
    combined = re.sub(
        r"Technical\s*\n[\s\n]*(Skills?\s*:)",
        r"Technical \1",   # join → "Technical Skills:"
        combined, flags=re.IGNORECASE,
    )

    combined = _SECTION_TITLES_RE.sub(r"\n\n", combined)
    combined = _BULLET_RE.sub(r"\n●", combined)
    # Collapse runs of blank lines
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Section splitting
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_HEADINGS = [
    r"work\s+experience", r"experience", r"employment(\s+history)?",
    r"professional\s+background", r"career\s+history",
    r"education(\s+&\s+training)?", r"academic(\s+background)?", r"qualifications",
    r"skills?(\s+&\s+technologies?)?", r"technical\s+skills?", r"core\s+competencies",
    r"tools?(\s+&\s+(technologies?|platforms?))?",
    r"certifications?(\s+&\s+licenses?)?", r"licenses?",
    r"languages?",
    r"projects?",
    r"awards?(\s+&\s+achievements?)?",
    r"summary|profile|objective|about(\s+me)?",
    r"publications?",
    r"volunteer(\s+activities?)?",
]

_HEADING_RE = re.compile(
    r"^(" + "|".join(_SECTION_HEADINGS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_sections(text: str) -> dict[str, str]:
    matches = list(_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        heading = clean(m.group()).lower().rstrip(":")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()
    return sections


def find_section(sections: dict[str, str], *keywords: str) -> Optional[str]:
    for key, val in sections.items():
        for kw in keywords:
            if kw in key:
                return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Personal info
# ─────────────────────────────────────────────────────────────────────────────

def parse_personal_info(text: str) -> dict:
    header = "\n".join(split_lines(text)[:30])

    email   = first_match(r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}", header)
    phone_m = re.search(
        r"(?:(?:\+|00)\d{1,3}[\s\-.]?)?"
        r"(?:\(?\d{1,4}\)?[\s\-.]?)"
        r"\d{3,5}[\s\-.]?\d{3,5}"
        r"(?:\s?(?:ext|x)[\s.]?\d{1,5})?",
        header,
    )
    phone = clean(phone_m.group()) if phone_m else None

    li_m     = re.search(r"linkedin\.com/in/([\w\-]+)", text, re.I)
    linkedin = f"linkedin.com/in/{li_m.group(1)}" if li_m else None

    gh_m   = re.search(r"github\.com/([\w\-]+)", text, re.I)
    github = f"github.com/{gh_m.group(1)}" if gh_m else None

    web_m   = re.search(r"https?://(?!(?:www\.)?(linkedin|github)\.com)[\w\-./~]+", text, re.I)
    website = web_m.group() if web_m else None

    location = {"city": None, "country": None}
    loc_m = re.search(
        r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*,\s*[A-Z][a-zA-Z]{1,20})\b", header
    )
    if loc_m:
        cand = loc_m.group().strip()
        bad  = {"linkedin", "github", "present", "current", "university", "institute", "college"}
        if not any(b in cand.lower() for b in bad) and len(cand) < 50:
            parts = [p.strip() for p in cand.split(",", 1)]
            location = {
                "city":    parts[0] if len(parts) > 0 else None,
                "country": parts[1] if len(parts) > 1 else None,
            }

    name = None
    _skip = re.compile(r"@|http|linkedin|github|phone|email|mobile|tel:|www\.", re.I)
    for line in split_lines(text)[:10]:
        words = line.split()
        # Skip lines with contact info markers or URLs
        if _skip.search(line):
            continue
        # Filter out words that look like phone numbers, emails, or separators
        name_words = [w for w in words if not re.search(r"\d{2,}|@|\|", w)]
        if 2 <= len(name_words) <= 5 and name_words == words[:len(name_words)] and name_words[0][0].isupper():
            # All kept words must start with a capital letter (proper name)
            if all(w[0].isupper() for w in name_words):
                name = clean(" ".join(name_words))
                break

    return {
        "full_name": name,
        "email":     email,
        "phone":     phone,
        "location":  location,
        "linkedin":  linkedin,
        "github":    github,
        "website":   website,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Work experience
# ─────────────────────────────────────────────────────────────────────────────

_DATE_PAT = (
    r"(?:"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|Q[1-4]\s*\d{4}"
    r"|\d{4}"
    r")"
)

PERIOD_RE = re.compile(
    rf"({_DATE_PAT})\s*[-–—to]+\s*({_DATE_PAT}|[Pp]resent|[Cc]urrent|[Nn]ow)",
    re.I,
)


def parse_work_experience(section_text: str) -> list[dict]:
    if not section_text:
        return []

    all_lines = split_lines(section_text)

    # Find every line that contains a date range — each anchors one job
    date_line_idxs = [i for i, line in enumerate(all_lines) if PERIOD_RE.search(line)]
    if not date_line_idxs:
        return []

    # Walk back from each date line to find where the job title starts
    job_start_idxs: list[int] = []
    for di in date_line_idxs:
        start = di
        for k in range(di - 1, max(-1, di - 10), -1):
            line = all_lines[k]
            if not line or re.match(r"^[•·▸▪\-\*➤➢●◆▶►]", line):
                break
            start = k
        if not job_start_idxs or job_start_idxs[-1] != start:
            job_start_idxs.append(start)

    jobs: list[dict] = []
    for ji, start in enumerate(job_start_idxs):
        end        = job_start_idxs[ji + 1] if ji + 1 < len(job_start_idxs) else len(all_lines)
        block      = all_lines[start:end]
        block_text = "\n".join(block)

        period_m   = PERIOD_RE.search(block_text)
        start_date = end_date = None
        current    = False
        if period_m:
            start_date = period_m.group(1)
            end_raw    = period_m.group(2)
            if re.match(r"present|current|now", end_raw, re.I):
                current = True
            else:
                end_date = end_raw

        title = company = None
        for line in block:
            if re.match(r"^[•·▸▪\-\*➤➢●◆▶►]", line):
                continue
            if PERIOD_RE.search(line):
                # Date range is on this line — company name precedes the date
                company_cand = PERIOD_RE.sub("", line).strip()
                company_cand = re.sub(r"\s*[—|]\s*$", "", company_cand).strip()
                if company_cand and not company:
                    company = clean(company_cand)
                continue
            # Skip lines that are clearly sentence continuations (wrapped text):
            # lowercase start, or very short fragments like "approach to problem-solving."
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].islower() or (len(stripped.split()) <= 4 and stripped[0].islower()):
                continue
            if title is None:
                title = clean(line)
            elif company is None:
                company = clean(re.sub(r"\s*[—|].*$", "", line))
                break

        if not title:
            continue

        loc_m = re.search(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?,\s*[A-Z]{2,3})\b", block_text)
        if loc_m:
            parts = [p.strip() for p in loc_m.group().split(",", 1)]
            location = {
                "city":    parts[0] if len(parts) > 0 else None,
                "country": parts[1] if len(parts) > 1 else None,
            }
        else:
            location = {"city": None, "country": None}

        highlights = [
            re.sub(r"^[•·▸▪\-\*➤➢●◆▶►]\s*", "", line).strip()
            for line in block
            if re.match(r"^[•·▸▪\-\*➤➢●◆▶►]", line)
        ]

        jobs.append({
            "company":    company,
            "title":      title,
            "location":   location,
            "start_date": start_date,
            "end_date":   end_date,
            "current":    current,
            "highlights": [h for h in highlights if h and len(h) > 10],
        })

    return [j for j in jobs if j["title"]]


# ─────────────────────────────────────────────────────────────────────────────
# Education
# ─────────────────────────────────────────────────────────────────────────────

_DEGREE_RE = re.compile(
    r"\b(Bachelor(?:'s)?|Master(?:'s)?|PhD|Ph\.D|Doctor(?:ate)?|"
    r"B\.?Sc?|M\.?Sc?|M\.?Eng|B\.?Eng|B\.?A|M\.?A|MBA|LLB|BEd|MEd|"
    r"Associate(?:'s)?|Diploma|Certificate|HND|BSN|MSN)\b",
    re.I,
)


def parse_education(section_text: str) -> list[dict]:
    if not section_text:
        return []

    # Split into blocks: prefer double-newline, fall back to date-range anchors
    raw_blocks = re.split(r"\n{2,}", section_text)
    if len(raw_blocks) <= 1:
        # PDF text often lacks blank lines — split at lines containing a date range
        raw_blocks = []
        current: list[str] = []
        for line in section_text.splitlines():
            if PERIOD_RE.search(line) and current:
                raw_blocks.append("\n".join(current) + "\n" + line)
                current = []
            else:
                current.append(line)
        if current:
            raw_blocks.append("\n".join(current))

    entries: list[dict] = []
    for block in raw_blocks:
        lines = split_lines(block)
        if not lines:
            continue

        degree_m = _DEGREE_RE.search(block)
        degree   = degree_m.group() if degree_m else None

        field = None
        if degree_m:
            after   = block[degree_m.end():].strip()
            field_m = re.match(r"(?:in|of)?\s*([A-Za-z\s&]+?)(?:\n|,|$)", after)
            if field_m:
                field = clean(field_m.group(1))

        # Prefer end year of a date range as graduation year
        period_m = PERIOD_RE.search(block)
        if period_m:
            end_raw = period_m.group(2)
            year_m2 = re.search(r"\b(20\d\d|19\d\d)\b", end_raw)
            graduation_year = year_m2.group() if year_m2 else None
        else:
            year_m = re.search(r"\b(20\d\d|19\d\d)\b", block)
            graduation_year = year_m.group() if year_m else None

        gpa_m = re.search(r"\bGPA[:\s]+([0-9.]+)", block, re.I)
        gpa   = gpa_m.group(1) if gpa_m else None

        institution = None
        for line in lines:
            # Strip date ranges from candidate institution lines
            stripped = PERIOD_RE.sub("", line).strip()
            if not stripped:
                continue
            if not _DEGREE_RE.match(stripped) and not re.match(r"^(20|19)\d\d", stripped):
                institution = clean(stripped)
                break

        if institution or degree:
            entries.append({
                "institution":     institution,
                "degree":          degree,
                "field":           field,
                "graduation_year": graduation_year,
                "gpa":             gpa,
            })

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Skills
# ─────────────────────────────────────────────────────────────────────────────

_TECH_KEYWORDS: set[str] = {
    "python","javascript","typescript","java","c++","c#","c","go","golang","rust",
    "ruby","php","swift","kotlin","scala","r","matlab","perl","bash","shell",
    "powershell","sql","nosql","html","css","sass","less","dart","flutter","lua",
    "haskell","elixir","clojure","groovy","vba","fortran","cobol","assembly",
    "react","angular","vue","svelte","next.js","nextjs","nuxt","gatsby","django",
    "flask","fastapi","spring","springboot","laravel","rails","express","node.js",
    "nodejs","tensorflow","pytorch","keras","scikit-learn","sklearn","pandas",
    "numpy","scipy","matplotlib","seaborn","d3.js","jquery","bootstrap","tailwind",
    "graphql","rest","soap","grpc","redux","mobx","rxjs",
    "postgresql","postgres","mysql","sqlite","mongodb","redis","elasticsearch",
    "cassandra","dynamodb","oracle","sql server","mssql","mariadb","neo4j",
    "influxdb","firebase","supabase",
    "aws","azure","gcp","google cloud","heroku","vercel","netlify","docker",
    "kubernetes","k8s","terraform","ansible","puppet","chef","jenkins","github actions",
    "gitlab ci","circleci","travis ci","prometheus","grafana","datadog","splunk",
    "nginx","apache","linux","ubuntu","debian","centos","macos","windows server",
    "git","github","gitlab","bitbucket","jira","confluence","notion","slack",
    "figma","sketch","postman","swagger","vs code","intellij","eclipse","xcode",
    "android studio","vim","emacs",
    # Data / BI tools
    "power bi","tableau","excel","hadoop","spark","apache spark","kafka","flink",
    "apache flink","hive","apache hive","snowflake","dbt","airbyte","fivetran",
    "apache nifi","nifi","dimensional modeling","data warehousing",
    "aws athena","athena","looker","qlik","metabase","superset",
}

_TOOL_KEYWORDS: set[str] = {
    "aws","azure","gcp","google cloud","docker","kubernetes","k8s","terraform",
    "ansible","jenkins","nginx","apache","linux","ubuntu","github actions","gitlab ci",
    "circleci","prometheus","grafana","datadog","splunk","git","github","gitlab",
    "bitbucket","jira","confluence","figma","sketch","postman","swagger",
    "intellij","vs code","android studio","xcode","firebase","heroku","vercel","netlify",
    # Data platform tools
    "power bi","tableau","hadoop","spark","apache spark","kafka","flink","apache flink",
    "hive","apache hive","snowflake","dbt","airbyte","fivetran","apache nifi","nifi",
    "aws athena","athena","looker","qlik","metabase","superset","excel",
}

_METHOD_KEYWORDS: set[str] = {
    "agile","scrum","kanban","waterfall","lean","devops","devsecops","tdd",
    "bdd","ddd","ci/cd","microservices","serverless","soa","oop","functional",
    "rest","restful","mvc","mvvm","solid","dry","pair programming","code review",
    "a/b testing","data driven","test driven",
}


def parse_skills(section_text: str, full_text: str) -> dict:
    source     = section_text if section_text else full_text
    raw_tokens = re.split(r"[,|•·;\n\t/●]+", source)
    tokens     = [clean(t).lower() for t in raw_tokens if 1 < len(t.strip()) < 40]

    technical: list[str] = []
    tools:     list[str] = []
    methods:   list[str] = []
    seen:      set[str]  = set()

    def _classify(tok: str) -> None:
        if tok in seen:
            return
        seen.add(tok)
        if tok in _METHOD_KEYWORDS:
            methods.append(tok.title())
        elif tok in _TECH_KEYWORDS:
            (tools if tok in _TOOL_KEYWORDS else technical).append(tok)

    for tok in tokens:
        _classify(tok)

    # Scan full text for ALL known keywords (handles multi-word like "Power BI")
    for kw in _TECH_KEYWORDS | _METHOD_KEYWORDS:
        if kw not in seen and re.search(r"\b" + re.escape(kw) + r"\b", full_text, re.I):
            _classify(kw)

    # De-duplicate: remove entries that are substrings of a longer entry already present
    def _dedup(lst: list[str]) -> list[str]:
        out = []
        s = sorted(set(lst), key=len, reverse=True)
        for item in s:
            # Keep item if no already-kept item fully contains it
            if not any(item != kept and item in kept for kept in out):
                out.append(item)
        return sorted(out)

    return {
        "technical":           _dedup(technical),
        "tools_and_platforms": _dedup(tools),
        "methodologies":       sorted(set(methods)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Certifications
# ─────────────────────────────────────────────────────────────────────────────

_CERT_ISSUERS: set[str] = {
    "aws","amazon","google","microsoft","azure","oracle","cisco","comptia",
    "pmi","isaca","isc2","isc²","red hat","salesforce","databricks","mongodb",
    "hashicorp","cncf","linux foundation","coursera","udemy","edx",
}


def parse_certifications(section_text: Optional[str]) -> list[dict]:
    if not section_text:
        return []

    certs: list[dict] = []
    for line in split_lines(section_text):
        if len(line) < 5:
            continue
        year_m = re.search(r"\b(20\d\d|19\d\d)\b", line)
        date   = year_m.group() if year_m else None
        name   = re.sub(r"\s*[\|\-–]\s*(?:20|19)\d\d.*$", "", line).strip()
        issuer = next((i.title() for i in _CERT_ISSUERS if i in name.lower()), None)
        certs.append({"name": clean(name), "issuer": issuer, "issued_date": date, "expiry_date": None})

    return certs


# ─────────────────────────────────────────────────────────────────────────────
# Languages
# ─────────────────────────────────────────────────────────────────────────────

_SPOKEN_LANGUAGES: set[str] = {
    "english","spanish","french","german","italian","portuguese","dutch","russian",
    "chinese","mandarin","cantonese","japanese","korean","arabic","hindi","bengali",
    "urdu","turkish","polish","swedish","norwegian","danish","finnish","czech",
    "slovak","romanian","hungarian","greek","hebrew","thai","vietnamese","indonesian",
    "malay","tagalog","swahili","persian","farsi","punjabi","gujarati","tamil",
    "telugu","marathi","kannada","ukrainian","catalan","serbian","croatian",
    "bulgarian","slovenian","latvian","lithuanian","estonian",
}

_PROFICIENCY_RE = re.compile(
    r"\b(native|fluent|proficient|advanced|intermediate|basic|elementary|"
    r"beginner|conversational|professional|mother\s+tongue|bilingual|"
    r"c[12]|b[12]|a[12])\b",
    re.I,
)


def parse_languages(section_text: str, full_text: str) -> list[dict]:
    result: list[dict] = []
    seen:   set[str]   = set()

    def _add(lang: str, prof: Optional[str] = None) -> None:
        key = lang.lower()
        if key in _SPOKEN_LANGUAGES and key not in seen:
            seen.add(key)
            result.append({"language": lang.title(), "proficiency": prof})

    for line in split_lines(section_text or ""):
        m = re.match(r"([A-Za-z\s]+?)(?:\s*[-–:]\s*|\s+)(.+)?$", line)
        if m and m.group(1).strip().lower() in _SPOKEN_LANGUAGES:
            prof_m = _PROFICIENCY_RE.search(line)
            _add(m.group(1).strip(), prof_m.group() if prof_m else None)

    for lang in _SPOKEN_LANGUAGES:
        if lang not in seen:
            m = re.search(r"\b" + re.escape(lang) + r"\b", full_text, re.I)
            if m:
                ctx    = full_text[max(0, m.start() - 30):m.end() + 40]
                prof_m = _PROFICIENCY_RE.search(ctx)
                _add(lang, prof_m.group() if prof_m else None)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_cv(text: str) -> dict:
    """Parse raw CV text and return structured dict."""
    text = normalize_pdf_text(text)
    sections = split_sections(text)

    work_sec  = find_section(sections, "experience", "employment", "career", "professional")
    edu_sec   = find_section(sections, "education", "academic", "qualification")
    skill_sec = find_section(sections, "skill", "competen", "technolog")
    cert_sec  = find_section(sections, "certif", "license")
    lang_sec  = find_section(sections, "language")

    return {
        "personal_info":   parse_personal_info(text),
        "work_experience": parse_work_experience(work_sec or ""),
        "education":       parse_education(edu_sec or ""),
        "skills":          parse_skills(skill_sec or "", text),
        "certifications":  parse_certifications(cert_sec),
        "languages":       parse_languages(lang_sec or "", text),
    }