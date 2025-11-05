#!/usr/bin/env python3
"""
Two-step Pappers script:
- Step 1: use /v2/recherche-entreprise to fetch a list of companies (collect SIREN)
- Step 2: for each SIREN, call /v2/entreprise to retrieve detailed fields (site internet + linkedin, etc.)
- Output: CSV

Usage:
  PAPPERS_API_KEY=xxxxx python pappers_two_step.py --code-naf 58.29C --effectif-min 10 --effectif-max 49 --forme-juridique "SARL, société à responsabilité limitée,SA à conseil d'administration (s.a.i.),SAS, société par actions simplifiée"

Notes:
- This script is resilient to missing fields because Pappers sometimes renames/omits keys.
- You can add/adjust search filters in build_search_params().
"""
import os
import time
import csv
import argparse
import sys
from typing import Dict, Any, List, Optional
import requests

BASE_URL = "https://api.pappers.fr/v2"
SEARCH_ENDPOINT = f"{BASE_URL}/recherche-entreprise"
ENTREPRISE_ENDPOINT = f"{BASE_URL}/entreprise"

DEFAULT_TIMEOUT = 30
RETRY_STATUS = {429, 500, 502, 503, 504}

def get_api_key() -> str:
    key = os.getenv("PAPPERS_API_KEY")
    if key:
        return key
    # fallback: try to parse the key from the old script if it exists (best effort)
    fallback = None
    try:
        from pathlib import Path
        txt = Path("/mnt/data/test_auth.py").read_text(encoding="utf-8")
        import re
        m = re.search(r'API_KEY\s*=\s*"(.*?)"', txt)
        if m:
            fallback = m.group(1)
    except Exception:
        pass
    if fallback:
        return fallback
    print("Missing API key. Set PAPPERS_API_KEY.", file=sys.stderr)
    sys.exit(2)

def backoff_sleep(attempt: int) -> None:
    # capped exponential backoff, jitter
    delay = min(2 ** attempt, 30) + (0.05 * attempt)
    time.sleep(delay)

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def build_search_params(args) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "api_token": get_api_key(),
        "curseur": "*",
        "par_page": args.par_page,
        "entreprise_cessee": False if args.entreprise_cessee is None else args.entreprise_cessee
    }
    if args.code_naf:
        params["code_naf"] = args.code_naf
    if args.forme_juridique:
        params["forme_juridique"] = args.forme_juridique
    if args.effectif_min is not None:
        params["effectif_min"] = args.effectif_min
    if args.effectif_max is not None:
        params["effectif_max"] = args.effectif_max
    if args.q:
        params["q"] = args.q  # full-text query on denomination, dirigeants, etc.
    return params

def paged_search(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Iterate /recherche-entreprise using 'curseur' pagination."""
    out: List[Dict[str, Any]] = []
    curseur = params.get("curseur", "*")
    attempt = 0
    while True:
        params["curseur"] = curseur
        try:
            resp = requests.get(SEARCH_ENDPOINT, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as e:
            if attempt < 5:
                attempt += 1
                backoff_sleep(attempt)
                continue
            raise

        if resp.status_code in RETRY_STATUS and attempt < 5:
            attempt += 1
            backoff_sleep(attempt)
            continue
        resp.raise_for_status()
        attempt = 0  # reset on success

        data = resp.json()
        entreprises = data.get("entreprises") or data.get("resultats") or []
        if not isinstance(entreprises, list):
            entreprises = []

        out.extend(entreprises)

        next_curseur = data.get("curseur_suivant") or data.get("suivant")
        if not next_curseur:
            break
        curseur = next_curseur
    return out

def get_entreprise(siren: str, api_key: str) -> Optional[Dict[str, Any]]:
    params = {
        "api_token": api_key,
        "siren": siren,
        # "champs": ",".join([...])  # optionally restrict fields to speed up & reduce payload
    }
    attempt = 0
    while True:
        try:
            resp = requests.get(ENTREPRISE_ENDPOINT, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException:
            if attempt < 5:
                attempt += 1
                backoff_sleep(attempt)
                continue
            return None

        if resp.status_code in RETRY_STATUS and attempt < 5:
            attempt += 1
            backoff_sleep(attempt)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

def extract_sites_and_linkedin(detail: Dict[str, Any]) -> (List[str], Optional[str]):
    """
    Try various shapes Pappers might use.
    - 'site_internet' might be a string or list
    - 'sites_internet' may exist
    - 'reseaux_sociaux' may be a list of dicts with 'type'/'url'
    - sometimes there's a direct key like 'lien_linkedin'
    """
    sites: List[str] = []

    # common variations
    for key in ("sites_internet", "site_internet", "site_web"):
        v = detail.get(key)
        if isinstance(v, list):
            sites.extend([s for s in v if isinstance(s, str)])
        elif isinstance(v, str):
            sites.append(v)

    linkedin = None

    # direct key
    for key in ("lien_linkedin", "linkedin", "url_linkedin"):
        v = detail.get(key)
        if isinstance(v, str) and v:
            linkedin = v
            break

    # nested networks
    rs = detail.get("reseaux_sociaux") or detail.get("social") or detail.get("social_media")
    if isinstance(rs, list):
        for entry in rs:
            url = safe_get(entry, "url", default=None)
            typ = (safe_get(entry, "type", default="") or "").lower()
            if isinstance(url, str):
                sites.append(url) if "http" in url and "linkedin" not in url else None
                if "linkedin" in url.lower():
                    linkedin = linkedin or url
            if "linkedin" in typ and isinstance(url, str):
                linkedin = linkedin or url

    # dedupe, preserve order
    def dedupe(seq):
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out
    return dedupe(sites), linkedin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-naf", dest="code_naf", help="Filter by code NAF (e.g., 58.29C)")
    parser.add_argument("--forme-juridique", dest="forme_juridique", help="Comma-separated list per Pappers expectations")
    parser.add_argument("--effectif-min", dest="effectif_min", type=int)
    parser.add_argument("--effectif-max", dest="effectif_max", type=int)
    parser.add_argument("--q", help="Full-text query for the search endpoint")
    parser.add_argument("--par-page", dest="par_page", type=int, default=100)
    parser.add_argument("--entreprise-cessee", dest="entreprise_cessee", type=lambda x: x.lower()=="true")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between /entreprise calls")
    parser.add_argument("--out", default="entreprises_detail.csv", help="Output CSV filename")
    args = parser.parse_args()

    api_key = get_api_key()
    search_params = build_search_params(args)
    print("Searching /recherche-entreprise…", file=sys.stderr)
    companies = paged_search(search_params)
    print(f"Found {len(companies)} companies in search.", file=sys.stderr)

    # Prepare CSV
    fieldnames = [
        "siren",
        "nom_entreprise",
        "code_naf",
        "date_creation",
        "forme_juridique",
        "ville",
        "effectif",
        "annee_finances",
        "chiffre_affaires",
        "resultat",
        "sites_internet",
        "lien_linkedin",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, c in enumerate(companies, 1):
            siren = c.get("siren") or c.get("SIREN")
            if not siren:
                continue

            # Pull basic info from search hit (fast) in case entreprise call fails
            siege = c.get("siege", {}) if isinstance(c.get("siege"), dict) else {}
            row: Dict[str, Any] = {
                "siren": siren,
                "nom_entreprise": c.get("nom_entreprise") or c.get("denomination"),
                "code_naf": c.get("code_naf"),
                "date_creation": c.get("date_creation"),
                "forme_juridique": c.get("forme_juridique"),
                "ville": siege.get("ville") if isinstance(siege, dict) else None,
                "effectif": c.get("effectif"),
                "annee_finances": c.get("annee_finances"),
                "chiffre_affaires": c.get("chiffre_affaires"),
                "resultat": c.get("resultat"),
                "sites_internet": None,
                "lien_linkedin": None,
            }

            detail = get_entreprise(siren, api_key)
            if detail:
                sites, linkedin = extract_sites_and_linkedin(detail)
                # backfill missing values from detailed doc where possible
                row["nom_entreprise"] = row["nom_entreprise"] or detail.get("nom_entreprise") or detail.get("denomination")
                row["code_naf"] = row["code_naf"] or detail.get("code_naf")
                row["date_creation"] = row["date_creation"] or detail.get("date_creation")
                row["forme_juridique"] = row["forme_juridique"] or detail.get("forme_juridique")
                siege_d = detail.get("siege") if isinstance(detail.get("siege"), dict) else {}
                if siege_d and isinstance(siege_d, dict):
                    row["ville"] = row["ville"] or siege_d.get("ville")
                row["sites_internet"] = ", ".join(sites) if sites else ""
                row["lien_linkedin"] = linkedin or ""

            writer.writerow(row)
            if args.sleep:
                time.sleep(args.sleep)

            if i % 50 == 0:
                print(f"Processed {i}/{len(companies)}…", file=sys.stderr)

    print(f"Done. Wrote {args.out}.")

if __name__ == "__main__":
    main()
