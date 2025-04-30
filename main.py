import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote
from datetime import datetime, timedelta

# ---------- CONFIG ----------
SUPPORTED_LANGS = ["en", "de", "fr", "es", "ar", "tr", "it", "ru", "pl"]
DE_ESTIMATE_FACTOR = 0.12
HEADERS = {"User-Agent": "WikipediaGapFinder/0.1 (your@email.com)"}


# ---------- FUNCTIONS ----------
def get_all_articles_recursive(category_name, lang="en", depth=2, limit=100):
    collected = []
    seen_cats = set()
    category_prefix = "Kategorie:" if lang == "de" else "Category:"
    API_URL = f"https://{lang}.wikipedia.org/w/api.php"

    def crawl(cat, level):
        if level > depth or len(collected) >= limit:
            return
        seen_cats.add(cat)

        PARAMS = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"{category_prefix}{cat}",
            "cmlimit": 50,
            "cmtype": "page|subcat"
        }
        R = requests.get(API_URL, params=PARAMS, headers=HEADERS)
        DATA = R.json()

        for item in DATA.get("query", {}).get("categorymembers", []):
            if item["title"].startswith(category_prefix):
                subcat = item["title"].replace(category_prefix, "")
                if subcat not in seen_cats:
                    crawl(subcat, level + 1)
            else:
                collected.append(item)
                if len(collected) >= limit:
                    break

    crawl(category_name, 0)
    return collected[:limit], len(collected)


def get_subcategories(category_name, lang="en"):
    category_prefix = "Kategorie:" if lang == "de" else "Category:"
    API_URL = f"https://{lang}.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"{category_prefix}{category_name}",
        "cmtype": "subcat",
        "cmlimit": 50,
        "format": "json"
    }
    R = requests.get(API_URL, params=PARAMS, headers=HEADERS)
    if R.status_code != 200:
        return []
    DATA = R.json()
    return [
        entry["title"].replace(category_prefix, "")
        for entry in DATA.get("query", {}).get("categorymembers", [])
    ]


def get_wikidata_id(article_title, lang="en"):
    URL = f"https://{lang}.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "pageprops",
        "titles": article_title,
        "format": "json"
    }
    R = requests.get(url=URL, params=PARAMS, headers=HEADERS)
    DATA = R.json()
    pages = DATA.get("query", {}).get("pages", {})
    for page_id in pages:
        if "pageprops" in pages[page_id] and "wikibase_item" in pages[page_id][
                "pageprops"]:
            return pages[page_id]["pageprops"]["wikibase_item"]
    return None


@st.cache_data(ttl=86400)
def get_languages(title, lang="en"):
    URL = f"https://{lang}.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "pageprops",
        "titles": title,
        "format": "json"
    }
    R = requests.get(url=URL, params=PARAMS, headers=HEADERS)
    DATA = R.json()
    pages = DATA.get("query", {}).get("pages", {})
    for page_id in pages:
        if "pageprops" in pages[page_id] and "wikibase_item" in pages[page_id][
                "pageprops"]:
            wikidata_id = pages[page_id]["pageprops"]["wikibase_item"]
            entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
            R = requests.get(entity_url, headers=HEADERS)
            if R.status_code != 200:
                return []
            entity_data = R.json()
            sitelinks = entity_data.get("entities",
                                        {}).get(wikidata_id,
                                                {}).get("sitelinks", {})
            langs = [
                k.replace("wiki", "") for k in sitelinks if k.endswith("wiki")
            ]
            langs_sorted = sorted(langs)
            return langs_sorted[:5] + (["..."]
                                       if len(langs_sorted) > 5 else [])
    return []


def has_german_link(wikidata_id):
    URL = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    R = requests.get(URL, headers=HEADERS)
    DATA = R.json()
    entity = DATA.get("entities", {}).get(wikidata_id, {})
    return "dewiki" in entity.get("sitelinks", {})


def normalize_title(title):
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "titles": title,
        "redirects": 1,
        "format": "json"
    }
    R = requests.get(url=URL, params=PARAMS, headers=HEADERS)
    DATA = R.json()
    pages = DATA["query"]["pages"]
    for page_id in pages:
        return pages[page_id]["title"].replace(" ", "_")
    return title.replace(" ", "_")


def get_pageviews(title, lang="en", days=30):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    encoded = quote(title, safe='')
    URL = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{encoded}/daily/{start_str}/{end_str}"
    R = requests.get(URL, headers=HEADERS)
    if R.status_code != 200:
        return 0
    DATA = R.json()
    return sum(item["views"] for item in DATA.get("items", []))


@st.cache_data(ttl=86400)
def article_exists_in_de(title, lang="en"):
    URL = f"https://{lang}.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "pageprops",
        "titles": title,
        "format": "json"
    }
    R = requests.get(URL, params=PARAMS, headers=HEADERS)
    DATA = R.json()
    pages = DATA.get("query", {}).get("pages", {})
    for page_id in pages:
        if "pageprops" in pages[page_id] and "wikibase_item" in pages[page_id][
                "pageprops"]:
            wikidata_id = pages[page_id]["pageprops"]["wikibase_item"]
            entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
            R = requests.get(entity_url, headers=HEADERS)
            if R.status_code != 200:
                return False
            entity_data = R.json()
            sitelinks = entity_data.get("entities",
                                        {}).get(wikidata_id,
                                                {}).get("sitelinks", {})
            return "dewiki" in sitelinks
    return False


def get_top_articles(lang="en", days=1, limit=100):
    titles = {}
    today = datetime.today()
    for delta in range(days):
        day = (today - timedelta(days=delta + 1)).strftime("%Y/%m/%d")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/{day}"
        R = requests.get(url, headers=HEADERS)
        if R.status_code != 200:
            continue
        items = R.json().get("items", [])[0].get("articles", [])
        for item in items:
            title = item["article"]
            views = item["views"]
            if ":" in title or title.lower() in ["hauptseite", "main_page"]:
                continue
            titles[title] = titles.get(title, 0) + views
    sorted_titles = sorted(titles.items(), key=lambda x: x[1], reverse=True)
    return sorted_titles[:limit]


@st.cache_data(ttl=86400)
def get_summary(title, lang="en"):
    encoded = quote(title.replace(" ", "_"))
    URL = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    R = requests.get(URL, headers=HEADERS)
    if R.status_code != 200:
        return ""
    DATA = R.json()
    return DATA.get("extract", "").split(".")[0] + "."


# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Wikipedia Gap Finder", layout="wide")
tab1, tab2, tab3 = st.tabs(
    ["🧩 Category Checker", "📉 Top 500 Missing in DE", "🔍 Most Searched (DE)"])

with tab1:
    st.header("🧩 Wikipedia Category Checker")
    category_input = st.text_input("Enter Wikipedia category name:",
                                   value="20th-century philosophers")
    lang_code = st.selectbox("Select Wikipedia language:",
                             options=SUPPORTED_LANGS,
                             index=0)
    limit = st.slider("Number of articles to check", 5, 500, 20)
    use_subcats = st.checkbox(
        "Include articles from subcategories (recursive)", value=True)

    if st.button("Check Category"):
        with st.spinner("Fetching and analyzing articles..."):
            if use_subcats:
                members, total_found = get_all_articles_recursive(
                    category_input, lang=lang_code, depth=2, limit=limit)
            else:
                members = get_category_members(category_input, limit)
                total_found = len(members)

            st.markdown(f"**{len(members)} articles found in category.**")

            subcats = get_subcategories(category_input, lang=lang_code)

            if len(members) < 5:
                st.warning(
                    f"⚠️ Only {len(members)} articles found. The category may be too narrow or abstract."
                )
                if subcats:
                    st.markdown("Try one of these related categories instead:")
                    for s in subcats:
                        st.markdown(f"- `{s}`")

            if total_found > 0:
                scan_pct = (len(members) / total_found) * 100
                st.markdown(
                    f"📊 Scanned {len(members)} of {total_found} total articles ({scan_pct:.3f}%)"
                )

            rows = []
            for member in members:
                title = member["title"]
                wikidata_id = get_wikidata_id(title)
                de_exists = has_german_link(
                    wikidata_id) if wikidata_id else False
                langs = get_languages(wikidata_id) if wikidata_id else []
                langs_str = ", ".join(langs)
                normalized_title = normalize_title(title)
                views_en = get_pageviews(normalized_title, lang=lang_code)
                est_views_de = int(views_en * DE_ESTIMATE_FACTOR)
                summary = get_summary(title)
                short_summary = summary[:180] + "..." if len(
                    summary) > 180 else summary
                wiki_url = f"https://{lang_code}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                link = f'<a href="{wiki_url}" target="_blank">{title}</a>'

                rows.append({
                    "Title": link,
                    "DE Exists": "✅" if de_exists else "❌",
                    "Languages": langs_str,
                    "Views (30d)": views_en,
                    "Estimated DE Views": est_views_de,
                    "Summary": short_summary
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df_display = df.sort_values(
                    by="Views (30d)", ascending=False).reset_index(drop=True)
                st.markdown(df_display.to_html(escape=False),
                            unsafe_allow_html=True)
                csv = df_display.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV",
                                   data=csv,
                                   file_name="wikipedia_gap_results.csv",
                                   mime="text/csv")
            else:
                st.warning("No articles found – nothing to display.")

            if subcats:
                st.markdown("---")
                st.subheader("🔎 Related Subcategories")
                st.markdown("Here are additional subcategories to explore:")
                for s in subcats:
                    st.markdown(f"- `{s}`")

with tab2:
    st.header("📉 Top 500 Missing in German Wikipedia")
    st.markdown(
        "This view shows the most viewed articles in another language Wikipedia that are missing in German Wikipedia."
    )

    selected_lang = st.selectbox("Select source language:",
                                 options=SUPPORTED_LANGS,
                                 index=0)

    if st.button(f"Load Top Missing ({selected_lang} → DE)"):
        with st.spinner(
                f"Fetching top viewed articles from {selected_lang}.wikipedia.org..."
        ):
            top_articles = get_top_articles(lang=selected_lang,
                                            days=30,
                                            limit=1000)
            missing_rows = []
            for title, views in top_articles:
                if not article_exists_in_de(title, lang=selected_lang):
                    langs = get_languages(title, lang=selected_lang)
                    langs_str = ", ".join(langs) if isinstance(langs,
                                                               list) else ""
                    est_views = int(views * DE_ESTIMATE_FACTOR)
                    summary = get_summary(title, lang=selected_lang)
                    wiki_url = f"https://{selected_lang}.wikipedia.org/wiki/{quote(title)}"
                    google_url = f"https://www.google.com/search?q=\"{quote(title)}\"+site:.de"
                    missing_rows.append({
                        "Title":
                        f'<a href="{wiki_url}" target="_blank">{title}</a>',
                        "Languages":
                        langs_str,
                        "Source Views":
                        views,
                        "Estimated DE Views":
                        est_views,
                        "Summary":
                        summary,
                        "Google DE 🔍":
                        f'<a href="{google_url}" target="_blank">Check</a>'
                    })

            if missing_rows:
                df = pd.DataFrame(missing_rows)
                st.markdown("### Top Missing Articles")
                st.markdown(df.to_html(escape=False, index=False),
                            unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV",
                                   data=csv,
                                   file_name="top500_missing_de.csv",
                                   mime="text/csv")
            else:
                st.info("All top articles exist in German Wikipedia.")

with tab3:
    st.header("🔍 Most Searched Terms in Wikipedia (Missing in DE)")
    lang_code = st.selectbox("Select Source Wikipedia language:",
                             options=SUPPORTED_LANGS,
                             index=0)
    period = st.selectbox("Select time period:",
                          ["Yesterday", "Past 30 Days (aggregated)"])
    limit = st.slider("Number of top articles to check", 10, 500, 100)

    if st.button("Find Missing Articles"):
        with st.spinner("Fetching and analyzing..."):
            days = 1 if period == "Yesterday" else 30
            top_articles = get_top_articles(lang=lang_code,
                                            days=days,
                                            limit=limit)
            all_rows = []

            for title, views in top_articles:
                exists = article_exists_in_de(title, lang=lang_code)
                link = f'<a href="https://{lang_code}.wikipedia.org/wiki/{quote(title)}" target="_blank">{title}</a>'
                summary = get_summary(title, lang=lang_code)
                est_views = int(views * DE_ESTIMATE_FACTOR)
                langs = get_languages(title, lang=lang_code)
                langs_str = ", ".join(langs)
                google_url = f"https://www.google.com/search?q=\"{quote(title)}\"+site:.de"
                google_link = f'<a href="{google_url}" target="_blank">Google DE 🔍</a>'

                all_rows.append({
                    "Title": link,
                    "Exists in DE": "✅" if exists else "❌",
                    "Languages": langs_str,
                    "Source Views": views,
                    "Estimated DE Views": est_views,
                    "Summary": summary,
                    "Google Check": google_link
                })

            if all_rows:
                df = pd.DataFrame(all_rows).sort_values(by="Source Views",
                                                        ascending=False)
                st.markdown("### 📊 Top Articles (with DE Status)")
                st.markdown(df.to_html(escape=False, index=False),
                            unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV",
                                   data=csv,
                                   file_name="top_articles_de_status.csv",
                                   mime="text/csv")
            else:
                st.info("No articles found.")
