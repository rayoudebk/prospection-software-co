import requests
import csv
import time

API_KEY = "0a9d1fd5a40a4053c6501ff07f9371b38cf5078dfb5c12f7"

# ÉTAPE 1: Recherche pour filtrer les entreprises
recherche_params = {
    "code_naf": "58.29C",
    "forme_juridique": (
        "SARL, société à responsabilité limitée,"
        "SA à conseil d'administration (s.a.i.),"
        "SAS, société par actions simplifiée"
    ),
    "effectif_min": 10,
    "effectif_max": 49,
    "entreprise_cessee": False,
    "curseur": "*",
    "par_curseur": 10,  # Passer à 10 entreprises pour le test
    "champs_supplementaires": "sites_internet,greffe,representants"
}

headers = {
    "api-key": API_KEY
}

all_companies = []

# ÉTAPE 1: Get list of companies with SIREN numbers using recherche endpoint
all_companies = []
target_companies = 1000
page = 1
par_curseur = 200

while len(all_companies) < target_companies:
    recherche_params = {
        "code_naf": "58.29C",
        "forme_juridique": (
            "SARL, société à responsabilité limitée,"
            "SA à conseil d'administration (s.a.i.),"
            "SAS, société par actions simplifiée"
        ),
        "effectif_min": 10,
        "effectif_max": 49,
        "entreprise_cessee": False,
        "curseur": "*",  # Toujours commencer par *
        "par_curseur": par_curseur,
        "page": page,  # Utiliser le numéro de page
        "champs_supplementaires": "sites_internet,greffe,representants"
    }
    
    try:
        response = requests.get("https://api.pappers.fr/v2/recherche", params=recherche_params, headers=headers, timeout=30)
        
        if response.status_code == 401:
            error_data = response.json()
            if "crédits" in error_data.get("message", ""):
                print("❌ ERROR: You don't have enough credits!")
                break
            else:
                print(f"❌ Authentication error: {error_data}")
                break
        else:
            response.raise_for_status()
            data = response.json()
            results = data.get("resultats", [])
            
            if not results:  # Plus d'entreprises disponibles
                print("✅ No more companies available")
                break
                
            print(f"✅ Retrieved {len(results)} companies from page {page} (Total: {len(all_companies) + len(results)})")
            
            # ÉTAPE 2: Get detailed info for each company using entreprise endpoint
            for i, company in enumerate(results):
                if len(all_companies) >= target_companies:
                    break
                    
                siren = company.get("siren")
                if siren:
                    print(f"🔍 Getting details for company {len(all_companies)+1}/{target_companies}: {company.get('nom_entreprise', 'Unknown')} (SIREN: {siren})")
                    
                    try:
                        # Call entreprise endpoint for detailed info WITH supplementary fields
                        entreprise_params = {
                            "siren": siren,
                            "champs_supplementaires": "sites_internet,finances"  # Supprimer lien_linkedin
                        }
                        
                        entreprise_response = requests.get(
                            "https://api.pappers.fr/v2/entreprise",
                            params=entreprise_params,
                            headers=headers,
                            timeout=30
                        )
                        
                        if entreprise_response.status_code == 200:
                            detailed_data = entreprise_response.json()
                            all_companies.append(detailed_data)
                            print(f"✅ Got details for {company.get('nom_entreprise', 'Unknown')}")
                            
                            # Supprimer tout le debug financier
                            # finances = detailed_data.get("finances", [])
                            # if finances:
                            #     print(f"🔍 Debug - Finances trouvées: {len(finances)} années")
                            #     for i, finance in enumerate(finances[:2]):
                            #         print(f"  Année {finance.get('annee', 'N/A')}: CA={finance.get('chiffre_affaires', 'N/A')}, Résultat={finance.get('resultat', 'N/A')}")
                            # else:
                            #     print("�� Debug - Aucune donnée financière trouvée")
                        else:
                            print(f"❌ Failed to get details for SIREN {siren}: {entreprise_response.status_code}")
                            # Fallback to basic data from recherche
                            all_companies.append(company)
                        
                        # Add a small delay to avoid rate limiting
                        time.sleep(0.1)
                        
                    except requests.exceptions.RequestException as e:
                        print(f"❌ Network error for SIREN {siren}: {e}")
                        # Fallback to basic data from recherche
                        all_companies.append(company)
                else:
                    print(f"❌ No SIREN found for company: {company.get('nom_entreprise', 'Unknown')}")
                    all_companies.append(company)
            
            # Passer à la page suivante
            page += 1
            
            # Arrêter si on a récupéré moins d'entreprises que par_curseur (dernière page)
            if len(results) < par_curseur:
                print("✅ Last page reached")
                break
                
        time.sleep(0.5)  # Pause entre les pages
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        break
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        break

print(f"📊 Total companies collected: {len(all_companies)}")

# Save results to CSV file
if all_companies:
    with open("entreprises_filtrees.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row - Supprimer lien_linkedin
        writer.writerow([
            "nom_entreprise", "domaine_activite", "date_creation", "forme_juridique",
            "ville_siege", "effectif", "annee_effectif",
            "capital", "sites_internet", 
            "chiffre_affaires", "resultat", "annee_finances"
        ])
        
        # Write company data
        for company in all_companies:
            siege = company.get("siege", {})
            
            # Handle sites_internet
            sites = company.get("sites_internet") or []
            if isinstance(sites, str):
                sites = [sites] if sites else []
            
            # Handle financial data from finances array
            finances = company.get("finances", [])
            chiffre_affaires = ""
            resultat = ""
            annee_finances = ""
            
            if finances and len(finances) > 0:
                # Prendre les données de la première année disponible
                first_finance = finances[0]
                chiffre_affaires = first_finance.get("chiffre_affaires", "")
                resultat = first_finance.get("resultat", "")
                annee_finances = first_finance.get("annee", "")
            
            writer.writerow([
                company.get("nom_entreprise"),
                company.get("domaine_activite"),
                company.get("date_creation"),
                company.get("forme_juridique"),
                siege.get("ville"),
                company.get("effectif"),
                company.get("annee_effectif"),
                company.get("capital"),
                ", ".join(sites) if sites else "",
                # Supprimer company.get("lien_linkedin", ""),
                chiffre_affaires,
                resultat,
                annee_finances
            ])   
    print(f"✅ CSV file 'entreprises_filtrees.csv' created with {len(all_companies)} companies!")
else:
    print("❌ No companies found to save.")