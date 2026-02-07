"""Constants for the unified crawler."""

# Keywords that indicate capability/product pages
CAPABILITY_KEYWORDS = [
    "product", "platform", "solution", "feature", "capability",
    "module", "software", "technology", "integration", "api",
    "automation", "analytics", "dashboard", "workflow", "management",
    "service", "offering", "tool", "suite", "enterprise",
]

# Keywords for services/implementation pages
SERVICE_KEYWORDS = [
    "service", "implementation", "support", "migration", "training",
    "professional-services", "customer-success", "consulting", "onboarding",
    "deployment", "managed", "maintenance",
]

# Keywords that indicate customer proof
PROOF_SIGNALS = [
    "customer", "client", "case study", "case-study", "success story",
    "testimonial", "trusted by", "used by", "selected", "partnered",
    "logo", "enterprise", "fortune", "leading", "companies use",
]

# Hard exclude - always skip these
HARD_EXCLUDE = [
    "/login", "/signin", "/signup", "/register", "/auth",
    "/legal", "/privacy", "/terms", "/cookie", "/gdpr",
    "/careers", "/jobs", "/job-", "/apply",
    "/cart", "/checkout", "/account",
]

# Soft demote - penalize but don't exclude, can be promoted back with proof signals
SOFT_DEMOTE = [
    "/blog", "/news", "/press", "/event", "/webinar",
    "/podcast", "/newsletter", "/rss",
]

# Priority path patterns for discovery
PRIORITY_PATHS = [
    "/product", "/platform", "/solution", "/feature", "/capability",
    "/module", "/software", "/technology", "/integration", "/api",
    "/service", "/implementation", "/support", "/professional-services",
    "/customer", "/client", "/case-stud", "/success",
    "/about", "/company", "/who-we-are", "/team",
    "/pricing", "/plan", "/package",
    "/security", "/trust", "/compliance", "/soc", "/gdpr",
    "/doc", "/resource", "/whitepaper", "/datasheet", "/guide",
    "/industr", "/vertical", "/use-case", "/workflow",
    "/partner", "/ecosystem", "/marketplace",
]

# Hub pages that are good starting points for BFS
HUB_PATTERNS = [
    "/product", "/solution", "/platform", "/resource", "/customer",
    "/service", "/integration", "/partner",
]

# Coverage quotas: (min, max) pages per category
COVERAGE_QUOTAS = {
    "about": (1, 2),
    "product": (3, 6),
    "solutions": (3, 6),
    "customers": (2, 4),
    "pricing": (0, 1),
    "security": (0, 1),
    "services": (0, 2),
    "docs": (0, 3),
    "integrations": (1, 3),
    "other": (0, 2),
}

# Page type detection patterns
PAGE_TYPE_PATTERNS = {
    "product": ["/product", "/platform", "/module", "/software"],
    "solutions": ["/solution", "/use-case", "/workflow", "/industr", "/vertical"],
    "features": ["/feature", "/capability", "/function"],
    "integrations": ["/integration", "/partner", "/connect", "/api", "/ecosystem"],
    "customers": ["/customer", "/client", "/case-stud", "/success", "/testimonial"],
    "services": ["/service", "/implementation", "/support", "/training", "/professional"],
    "pricing": ["/pricing", "/plan", "/package", "/cost"],
    "security": ["/security", "/trust", "/compliance", "/soc", "/gdpr", "/hipaa"],
    "docs": ["/doc", "/resource", "/whitepaper", "/datasheet", "/guide", "/help"],
    "about": ["/about", "/company", "/who-we-are", "/team", "/leadership", "/history"],
}

# URL parameters to strip during normalization
STRIP_PARAMS = [
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "ref", "source", "mc_cid", "mc_eid",
]

# Common sitemap locations
SITEMAP_PATHS = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap/sitemap.xml",
    "/sitemap.xml.gz",
    "/sitemap_index.xml.gz",
    "/wp-sitemap.xml",
    "/page-sitemap.xml",
    "/post-sitemap.xml",
]

# Customer logo section indicators
LOGO_SECTION_INDICATORS = [
    "trusted by", "customers", "clients", "used by", "powering",
    "companies", "brands", "enterprises", "organizations", "partners",
    "logo", "testimonial", "success stor",
]

# Max limits
MAX_URLS_FROM_SITEMAP = 500
MAX_URLS_FROM_HOMEPAGE = 200
MAX_BFS_DEPTH = 2
MAX_PAGES_TO_PREVIEW = 100
MAX_PAGES_TO_CRAWL = 30
BATCH_SIZE = 5
REQUEST_DELAY = 0.5  # seconds between batches
