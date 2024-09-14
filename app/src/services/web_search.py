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
            "safesearch": safe_search,
        }

        try:
            response = requests.get(
                self.search_url, headers=self.headers, params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            if response is not None:
                print(f"Response content: {response.text}")
            return None

    def format_url(self, response):
        if response is None:
            return "No url to display."

        urls = [page["url"] for page in response["webPages"]["value"]]
        return urls

    def format_snippet(self, response):
        if response is None:
            return "No results to display."

        snippets = [page["snippet"] for page in response["webPages"]["value"]]
        return snippets

    def format_search_results(self, response):
        if response is None:
            return "No results to display."

        formatted_results = []
        formatted_results.append(
            f"Original Query: {response['queryContext']['originalQuery']}"
        )
        formatted_results.append(
            f"Total Estimated Matches: {response['webPages']['totalEstimatedMatches']}"
        )
        formatted_results.append(
            f"Web Search URL: {response['webPages']['webSearchUrl']}\n"
        )

        formatted_results.append("Search Results:")
        for index, page in enumerate(response["webPages"]["value"], start=1):
            formatted_results.append(f"{index}. {page['name']}")
            formatted_results.append(f"   URL: {page['url']}")
            formatted_results.append(f"   Snippet: {page['snippet']}")
            if "datePublished" in page:
                formatted_results.append(f"   Date Published: {page['datePublished']}")
            if "siteName" in page:
                formatted_results.append(f"   Site Name: {page['siteName']}")
            else:
                formatted_results.append("   Site Name: Not available")
            formatted_results.append("")

        return "\n".join(formatted_results)


def main():
    search_term = "What is RAG?"  # query user

    client = BingSearchClient()
    results = client.search(search_term)

    if results:
        formatted_results = client.format_search_results(results)
        print(formatted_results)
        # Uncomment the following lines if you want to use other formatting methods
        # print(client.format_url(results))
        # print(client.format_snippet(results))
    else:
        print("No results returned from the search.")


if __name__ == "__main__":
    main()
