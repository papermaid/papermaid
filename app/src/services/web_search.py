import requests
from config import BING_SEARCH_URL, BING_SUBSCRIPTION_KEY

class BingSearchClient:
  def __init__(self):
      self.search_url = BING_SEARCH_URL
      self.headers = {"Ocp-Apim-Subscription-Key": BING_SUBSCRIPTION_KEY}

  def search(self, search_term, count=30, safe_search="Moderate"):
      params = {
          "q": search_term,
          "textDecorations": True,
          "textFormat": "HTML",
          "count": count,
          "setLang": "en-US",
          "safesearch": safe_search
      }

      try:
          response = requests.get(self.search_url, headers=self.headers, params=params)
          response.raise_for_status()
          return response.json()
      except requests.exceptions.HTTPError as e:
          print(f"HTTP error occurred: {e}")
          if response is not None:
              print(f"Response content: {response.text}")
          return None

  def format_search_results(self, response):
      if response is None:
          print("No results to display.")
          return

      print(f"Original Query: {response['queryContext']['originalQuery']}")
      print(f"Total Estimated Matches: {response['webPages']['totalEstimatedMatches']}")
      print(f"Web Search URL: {response['webPages']['webSearchUrl']}\n")
      
      print("Search Results:")
      for index, page in enumerate(response['webPages']['value'], start=1):
          print(f"{index}. {page['name']}")
          print(f"   URL: {page['url']}")
          print(f"   Snippet: {page['snippet']}")
          if 'datePublished' in page:
              print(f"   Date Published: {page['datePublished']}")
          if 'siteName' in page:
              print(f"   Site Name: {page['siteName']}")
          else:
              print("   Site Name: Not available")
          print()

def main():
  search_term = "What is RAG?"
  
  client = BingSearchClient()
  results = client.search(search_term)
  
  if results:
      client.format_search_results(results)
  else:
      print("No results returned from the search.")

if __name__ == "__main__":
  main()