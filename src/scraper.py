# src/scraper.py
import requests
from bs4 import BeautifulSoup
import os

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'DNT': '1',
}

def scrape_and_save(url, filename):
    try:
        print(f"Fetching {url}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find(id='main-content')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.body.get_text(separator='\n', strip=True)
        output_dir = 'data/raw'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully saved content to {filepath}\n")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return False

if __name__ == "__main__":
    pages_to_scrape = {
        'Flat_and_Grant_Eligibility.txt': 'https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility',
        'Couples_and_Families.txt': 'https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families',
        'Housing_Loan_from_HDB.txt': 'https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/housing-loan-options/housing-loan-from-hdb',
        'Application_for_HFE_Letter.txt': 'https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/application-for-an-hdb-flat-eligibility-hfe-letter',
        'Singles.txt': 'https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles',
        # --- YOUR NEW, MORE SPECIFIC EHG URL ---
        'EHG_Grant_Amounts.txt': 'https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/enhanced-cpf-housing-grant-families'
    }
    print("--- Starting Scraper ---")
    for filename, url in pages_to_scrape.items():
        scrape_and_save(url, filename)
    print("--- Scraper Finished ---")
