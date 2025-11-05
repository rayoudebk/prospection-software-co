import requests
import json

API_KEY = "0a9d1fd5a40a4053c6501ff07f9371b38cf5078dfb5c12f7"

# Utiliser le SIREN de GESTIMUM qu'on a déjà
siren = "853547644"

headers = {
    "api-key": API_KEY
}

print(f"🔍 Récupération des données complètes pour SIREN: {siren}")

try:
    # Appel à l'endpoint entreprise pour récupérer toutes les données
    response = requests.get(
        "https://api.pappers.fr/v2/entreprise",
        params={"siren": siren},
        headers=headers,
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Sauvegarder le JSON complet dans un fichier
        with open("entreprise_complete.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("✅ JSON complet sauvegardé dans 'entreprise_complete.json'")
        
        # Afficher la structure de haut niveau
        print("\n📋 Structure de haut niveau:")
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} avec {len(value)} éléments")
            else:
                print(f"  {key}: {value}")
        
        # Explorer les champs qui pourraient contenir sites internet et LinkedIn
        print("\n🔍 Exploration des champs potentiels pour sites internet:")
        
        # Vérifier etablissements
        if 'etablissements' in data and data['etablissements']:
            print("  📍 Etablissements:")
            for i, etab in enumerate(data['etablissements'][:2]):  # Limiter à 2 pour éviter trop de texte
                print(f"    Etablissement {i+1}:")
                for key, value in etab.items():
                    if 'site' in key.lower() or 'web' in key.lower() or 'url' in key.lower():
                        print(f"      {key}: {value}")
        
        # Vérifier representants
        if 'representants' in data and data['representants']:
            print("  👥 Représentants:")
            for i, rep in enumerate(data['representants'][:2]):  # Limiter à 2
                print(f"    Représentant {i+1}:")
                for key, value in rep.items():
                    if 'site' in key.lower() or 'web' in key.lower() or 'url' in key.lower() or 'linkedin' in key.lower():
                        print(f"      {key}: {value}")
        
        # Vérifier beneficiaires_effectifs
        if 'beneficiaires_effectifs' in data and data['beneficiaires_effectifs']:
            print("  🎯 Bénéficiaires effectifs:")
            for i, ben in enumerate(data['beneficiaires_effectifs'][:2]):  # Limiter à 2
                print(f"    Bénéficiaire {i+1}:")
                for key, value in ben.items():
                    if 'site' in key.lower() or 'web' in key.lower() or 'url' in key.lower() or 'linkedin' in key.lower():
                        print(f"      {key}: {value}")
        
        print(f"\n✅ Exploration terminée. Vérifiez le fichier 'entreprise_complete.json' pour voir toute la structure.")
        
    else:
        print(f"❌ Erreur: {response.status_code}")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"❌ Erreur réseau: {e}")
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")