import streamlit as st
import pandas as pd
import re
import requests
from urllib.parse import quote
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev

import logging
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)


st.set_page_config(page_title="Wikipedia Gap Finder", layout="wide")

# ---------- CONFIG ----------
SUPPORTED_LANGS = ["en", "de", "fr", "es", "ar", "tr", "it", "ru", "pl"]
DE_ESTIMATE_FACTOR = 0.12
HEADERS = {"User-Agent": "WikipediaGapFinder/0.1 (daniel.sigge@web.de)"}


# ---------- FUNCTIONS ----------

def process_articles_concurrent(titles, lang):
    def process_single(title):
        try:
            wikidata_id = get_wikidata_id(title, lang=lang)
            de_exists = has_german_link(wikidata_id) if wikidata_id else False
            langs = get_languages(wikidata_id) if wikidata_id else []
            langs_str = ", ".join(langs)

            normalized = normalize_title(title)
            views = get_pageviews(normalized, lang=lang)
            est_views = int(views * DE_ESTIMATE_FACTOR)
            avg, std_dev, cv, peak_ratio, virality = get_daily_views_stats(normalized, lang=lang)
            summary = get_summary(normalized, lang=lang)
            short_summary = summary[:180] + "..." if len(summary) > 180 else summary

            wiki_url = f"https://{lang}.wikipedia.org/wiki/{quote(title)}"

            return {
                    "Title": f'<a href="{wiki_url}" target="_blank">{title}</a>',
                    "Languages": langs_str,
                    "Views (30d)": views,
                    "Estimated DE Views": est_views,
                    "Exists in DE": "✅" if de_exists else "❌",
                    "CV": cv,
                    "Viralität": virality,
                    "Summary": short_summary
                }
        except Exception as e:
            return {
                "Title": title,
                "Languages": "❌",
                "Views (30d)": "❌",
                "Estimated DE Views": "❌",
                "Exists in DE": "❌",
                "Summary": f"Fehler: {e}"
            }

    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(process_single, titles))



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

    def get_category_members_paged(category_name, lang="en", max_articles=5000):
        category_prefix = "Kategorie:" if lang == "de" else "Category:"
        base_url = f"https://{lang}.wikipedia.org/w/api.php"
        collected = []
        cmcontinue = None

        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"{category_prefix}{category_name}",
                "cmtype": "page",
                "cmlimit": "500",
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            response = requests.get(base_url, params=params, headers=HEADERS)
            if response.status_code != 200:
                break

            data = response.json()
            members = data.get("query", {}).get("categorymembers", [])
            collected.extend(members)

            if "continue" in data and "cmcontinue" in data["continue"]:
                cmcontinue = data["continue"]["cmcontinue"]
            else:
                break

            if len(collected) >= max_articles:
                break

        return collected, len(collected)

def get_missing_titles_parallel(titles, lang):
    def check(title):
        return title if not article_exists_in_de(title, lang=lang) else None

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(check, titles))
    return [title for title in results if title]

def process_articles_with_progress(titles, lang):
    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(titles)

    for i, title in enumerate(titles):
        result = process_articles_concurrent([title], lang)
        if result:
            results.extend(result)
        progress.progress((i + 1) / total)
        status.text(f"{i+1}/{total} Artikel verarbeitet")

    return results


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


@st.cache_data(ttl=86400)
def get_sitelinks(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200 or not resp.text.strip():
            return {}
        data = resp.json()
        return data.get("entities", {}).get(qid, {}).get("sitelinks", {})
    except Exception as e:
        print(f"❌ Fehler beim Laden von {qid}: {e}")
        return {}


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


@st.cache_data(ttl=86400)
def get_all_frauenrot_lists():
    base_url = "https://de.wikipedia.org"
    master_url = f"{base_url}/wiki/Wikipedia:WikiProjekt_Frauen/Frauen_in_Rot/Listen"
    R = requests.get(master_url)
    if R.status_code != 200:
        return {}

    html = R.text
    matches = re.findall(
        r'href="/wiki/(Wikipedia:WikiProjekt_Frauen/Frauen_in_Rot/Fehlende_Artikel.*?)"',
        html)
    unique = sorted(set(matches))
    return {
        match.split("/")[-1].replace("_", " "): base_url + "/wiki/" + match
        for match in unique
    }


def extract_missing_names_from_list(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    red_links = soup.find_all("a", class_="new")
    names = [link.get("title") for link in red_links if link.get("title")]
    return list(set(names))


@st.cache_data(ttl=86400)
def extract_qids_from_list(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    qids = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 1:
            continue
        last_cell = cells[-1]
        text = last_cell.get_text(strip=True)
        if text.startswith("Q") and text[1:].isdigit():
            qids.append(text)

    return list(set(qids))

def process_articles(members_batch, lang_code):
    rows = []
    for member in members_batch:
        title = member["title"]

        # Ankerlinks (z. B. "Breakbeat#Progressive_breaks") überspringen
        if "#" in title:
            continue

        wikidata_id = get_wikidata_id(title, lang=lang_code)
        de_exists = has_german_link(wikidata_id) if wikidata_id else False
        langs = get_languages(wikidata_id) if wikidata_id else []
        langs_str = ", ".join(langs)

        normalized_title = normalize_title(title)
        views = get_pageviews(normalized_title, lang=lang_code)
        est_views = int(views * DE_ESTIMATE_FACTOR)

        summary = get_summary(normalized_title, lang=lang_code)
        short_summary = summary[:180] + "..." if len(summary) > 180 else summary

        wiki_url = f"https://{lang_code}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        link = f'<a href="{wiki_url}" target="_blank">{title}</a>'

        rows.append({
            "Title": link,
            "DE Exists": "✅" if de_exists else "❌",
            "Languages": langs_str,
            "Views (30d)": views,
            "Estimated DE Views": est_views,
            "Summary": short_summary
        })

    return rows

def process_articles(members, lang_code):
    rows = []
    for member in members:
        title = member["title"]
        wikidata_id = get_wikidata_id(title, lang=lang_code)
        de_exists = has_german_link(wikidata_id) if wikidata_id else False
        langs = get_languages(wikidata_id) if wikidata_id else []
        langs_str = ", ".join(langs)
        normalized_title = normalize_title(title)
        views_en = get_pageviews(normalized_title, lang=lang_code)
        est_views_de = int(views_en * DE_ESTIMATE_FACTOR)
        summary = get_summary(title, lang=lang_code)
        short_summary = summary[:180] + "..." if len(summary) > 180 else summary
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
    return rows


def get_daily_views_stats(title, lang="en", days=90):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    encoded = quote(title, safe='')
    URL = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{encoded}/daily/{start_str}/{end_str}"
    R = requests.get(URL, headers=HEADERS)
    if R.status_code != 200:
        return 0, 0, 0, 0, "❌ Fehler"

    items = R.json().get("items", [])
    daily_views = [i["views"] for i in items if "views" in i]
    if not daily_views or mean(daily_views) == 0:
        return 0, 0, 0, 0, "Keine Daten"

    avg = mean(daily_views)
    std_dev = stdev(daily_views)
    peak_ratio = max(daily_views) / avg
    cv = std_dev / avg

    if cv > 1.0:
        status = "🧨 Viral"
    elif cv < 0.3:
        status = "💎 Stable"
    else:
        status = "⚖️ Mixed"

    return round(avg), round(std_dev), round(cv, 2), round(peak_ratio, 2), status


# ---------- STREAMLIT APP ----------

st.title("Wikipedia Relevanz-Radar")

st.markdown("""
Der **Wikipedia Relevanz-Radar** ist ein Tool zur datenbasierten Erkennung von Artikeln, die in der deutschen Wikipedia fehlen, aber in anderen Sprachversionen sehr häufig aufgerufen werden.

Ziel ist es, Wikipedia-Autor:innen, Redaktionen und Interessierten zu helfen, **relevante Inhalte für die deutschsprachige Wikipedia zu identifizieren** – basierend auf tatsächlichen Nutzerinteressen.

---

""")

tab0, tab1, tab2, tab3, tab4, tab5= st.tabs([
    "🙇🏻‍♂️ Info & Anleitung", "🔎 1) Kategorie-Analyse", "🔎 2) Meistgelesen vs. DE (Schnell)",
    "🔎 3) Meistgelesen vs. DE (Gefiltert)", "🔎 4) Rotlink-Frauen-Projekt", "🔎 5) Eigene Artikelliste analysieren"
])

with tab0:

    st.markdown("### Wie funktioniert das Tool?")
    st.markdown("""
    Das Tool greift auf öffentlich zugängliche Wikipedia-Statistiken und Wikidata-Verknüpfungen zu, um Lücken zu erkennen. Es bietet drei Hauptfunktionen:

    **Viralität: CV (Coefficient of Variation) misst, wie stark die täglichen Seitenaufrufe eines Artikels im Verhältnis zum Durchschnitt schwanken. Daraus ergibt sich eine Einschätzung zur Viralität: stable (💎), mixed (⚖️) oder viral (🧨).**
    **Was sind "Estimated DE Views"? Mit einem Faktor 0.12 werden mögliche Aufrufe in der deutschsprachigen Wikipedia berechnet. Ein randomisiertes Sample aus englischen Wikipedia-Artikeln ergab, dass deutsche Artikel  
                            
    **1. Kategorie-Analyse:**  
    Gibt man eine Wikipedia-Kategorie ein (z. B. *20th-century philosophers*), zeigt das Tool an, welche Artikel in anderen Sprachen existieren, aber nicht in der deutschen Wikipedia – inkl. Seitenaufrufen, Kurzbeschreibung und Sprachen, in denen der Artikel vorhanden ist.

    **2. Meistgelesen vs. DE (Schnell):**  
    Zeigt eine täglich aktualisierte Liste der meistbesuchten Artikel auf z. B. Englisch oder Spanisch, die noch nicht in der deutschen Wikipedia existieren. So erkennt man besonders gefragte Themen.

    **3. Meistgelesen vs. DE (Gefiltert):**  
    Gleiche Abfrage wie in Tab 2, nur filtert das System hier in DE existierende Artikel schon heraus. Vorsicht, deutlich langsamer.

    **4. Rotlink-Frauen-Projekt:**
    Analysiert Listen des Frauen-in-Rot-Projekts und zeigt, in welchen Sprachen Artikel existieren, welche Version am längsten ist – und wie viele Aufrufe diese Version hatte.
         
    **5. Artikelliste analysieren** 
    Paste eine eigene kommagetrennte Artikelliste in das Feld und sieh, welche Artikel in der Deutschen Wikipedia fehlen.            
    """                       
                )

    st.markdown("### Hinweise")
    st.markdown("""
    - Alle Daten stammen aus offiziellen Wikimedia-APIs (Pageviews, Wikidata, Categories)  
    - Ergebnisse sind automatisch erzeugt und können redaktionelle Prüfung nicht ersetzen
    - Hast du Feedback? Ich bin auf Wikipedia unter *User:Fizzywater90* zu erreichen
    """)

with tab1:
    st.header("1) Kategorie-Analyse")
    category_input = st.text_input("Enter Wikipedia category name:", value="20th-century philosophers")
    lang_code = st.selectbox("Select Wikipedia language:", options=SUPPORTED_LANGS, index=0)
    use_subcats = st.checkbox("Include articles from subcategories (recursive)", value=True)

    # SessionState: Fortschritt zwischenspeichern
    if "category_results" not in st.session_state:
        st.session_state["category_results"] = []
        st.session_state["category_cursor"] = 0
        st.session_state["category_total"] = 0
        st.session_state["category_members"] = []

    if st.button("Kategorie analysieren & erste Artikel laden"):
        with st.spinner("Lade alle Mitglieder der Kategorie..."):
            if use_subcats:
                members, total_found = get_all_articles_recursive(
                    category_input, lang=lang_code, depth=2, limit=5000)
            else:
                members = get_category_members(category_input, limit=5000)
                total_found = len(members)

            # Nur Hauptartikel (ohne Anker/Abschnitte)
            filtered_members = [m for m in members if "#" not in m["title"]]

            st.session_state["category_members"] = filtered_members
            st.session_state["category_cursor"] = 0
            st.session_state["category_total"] = len(filtered_members)
            st.session_state["category_results"] = []

        if st.session_state["category_members"]:
            to_process = st.session_state["category_members"][:50]
            with st.spinner("Analysiere Artikel..."):
                titles = [a["title"] for a in to_process]
                rows = process_articles_concurrent(titles, lang_code)
                st.session_state["category_results"].extend(rows)
                st.session_state["category_results"].sort(key=lambda x: x["Views (30d)"], reverse=True)
                st.session_state["category_cursor"] += len(to_process)

    # Weitere Artikel laden
    if st.session_state["category_members"]:
        total = st.session_state["category_total"]
        current_cursor = st.session_state["category_cursor"]
        next_cursor = min(current_cursor + 50, total)
        to_process = st.session_state["category_members"][current_cursor:next_cursor]

        if to_process and st.button(f"Analysiere nächste {len(to_process)} Artikel"):
            with st.spinner("Analysiere weitere Artikel..."):
                titles = [a["title"] for a in to_process]
                rows = process_articles_concurrent(titles, lang_code)
                st.session_state["category_results"].extend(rows)
                st.session_state["category_results"].sort(key=lambda x: x["Views (30d)"], reverse=True)
                st.session_state["category_cursor"] += len(to_process)

        if st.session_state["category_results"]:
            df = pd.DataFrame(st.session_state["category_results"])
            st.markdown(f"**{st.session_state['category_cursor']} von {total} Artikeln analysiert.**")
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        if st.session_state["category_cursor"] < st.session_state["category_total"]:
            st.info("🔁 Es sind weitere Artikel verfügbar – klicke erneut auf „Analysiere nächste …“")
        else:
            st.success("✅ Alle Artikel in dieser Kategorie wurden analysiert.")

with tab2:
    st.header("🔍 2) Gefragte Artikel, die in DE fehlen (Schneller)")
    lang_code = st.selectbox("Select Source Wikipedia language:", options=SUPPORTED_LANGS, index=0)
    period = st.selectbox("Select time period:", ["Yesterday", "Past 30 Days (aggregated)"])
    limit = st.slider("Number of top articles to check", 10, 5000, 1000)

    if st.button("Find Missing Articles"):
        with st.spinner("Fetching and analyzing..."):
            days = 1 if period == "Yesterday" else 30
            top_articles = get_top_articles(lang=lang_code, days=days, limit=limit)
            titles = [title for title, _ in top_articles]
            results = process_articles_concurrent(titles, lang_code)

            if results:
                df = pd.DataFrame(results).sort_values(by="Views (30d)", ascending=False)
                st.markdown("### 📊 Top Articles (with DE Status)")
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv, file_name="top_articles_de_status.csv", mime="text/csv")
            else:
                st.info("Keine Artikel gefunden.")

with tab3:
    st.header("3) Gefragte Artikel, die in DE fehlen (Gefiltert, Langsamer)")
    st.markdown("Zeigt meistbesuchte Artikel in anderer Sprache, die in DE fehlen. Zeigt mit großer Wahrscheinlichkeit viele virale Artikel an. Kann gegebenenfalls Artikel enthalten, die es schon in DE gibt.")

    selected_lang = st.selectbox("Select source language:", options=SUPPORTED_LANGS, index=0)

    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Zeitraum (Tage)", [7, 14, 30, 90], index=2)
    with col2:
        limit = st.selectbox("Anzahl Artikel", [100, 250, 500, 1000], index=3)

    if st.button(f"Load Top Missing ({selected_lang} → DE)"):
        with st.spinner(f"Lade meistgelesene Artikel aus {selected_lang}.wikipedia.org..."):
            top_articles = get_top_articles(lang=selected_lang, days=days, limit=limit)
            titles = [title for title, _ in top_articles]

            st.info("🔍 Prüfe, welche Artikel auf Deutsch fehlen...")
            missing_titles = get_missing_titles_parallel(titles, selected_lang)

            if missing_titles:
                st.info("📊 Verarbeite fehlende Artikel...")
                results = process_articles_with_progress(missing_titles, selected_lang)

                df = pd.DataFrame(results)
                df = df.sort_values(by="Views (30d)", ascending=False)

                st.markdown("### Top Missing Articles")
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv, file_name="top_missing_articles.csv", mime="text/csv")
            else:
                st.success("🎉 Alle Top-Artikel existieren bereits auf Deutsch.")



from concurrent.futures import ThreadPoolExecutor, as_completed

with tab4:
    st.header("🔴 Rotfrauen-Projekt")
    st.markdown(
        "Wähle eine oder mehrere Listen aus dem Frauen-in-Rot-Projekt. Das Tool prüft, in welchen Sprachversionen Artikel existieren, "
        "welche Version am längsten ist – und wie viele Aufrufe diese Version hatte."
    )

    with st.spinner("Lade alle Listen des Frauen-in-Rot-Projekts..."):
        frauenrot_lists = get_all_frauenrot_lists()

    selected_lists = st.multiselect(
        "Wähle eine oder mehrere Frauenlisten aus:",
        options=list(frauenrot_lists.keys()))

    if selected_lists:
        if st.button("🔍 Relevanz analysieren"):
            with st.spinner("Analysiere Relevanz für ausgewählte Personen... (bitte Tab nicht wechseln)"):
                all_qids = set()
                for name in selected_lists:
                    url = frauenrot_lists[name]
                    try:
                        qids = extract_qids_from_list(url)
                        all_qids.update(qids)
                    except Exception as e:
                        print(f"Fehler beim Laden der Liste {name}: {e}")
                        continue

                st.markdown(f"**Gesamt: {len(all_qids)} Personen** werden analysiert.")

                def process_qid(qid, i, total):
                    try:
                        sitelinks = get_sitelinks(qid)
                        if not sitelinks:
                            return None

                        sizes = {}
                        for lang_key, link in sitelinks.items():
                            if not lang_key.endswith("wiki"):
                                continue
                            lang = lang_key.replace("wiki", "")
                            title = link["title"]
                            rev_url = f"https://{lang}.wikipedia.org/w/api.php"
                            rev_params = {
                                "action": "query",
                                "prop": "revisions",
                                "titles": title,
                                "rvprop": "size",
                                "format": "json"
                            }
                            try:
                                resp = requests.get(rev_url, params=rev_params, headers=HEADERS, timeout=5)
                                data = resp.json()
                                pages = data.get("query", {}).get("pages", {})
                                for page in pages.values():
                                    size = page.get("revisions", [{}])[0].get("size", 0)
                                    sizes[lang] = (size, title)
                            except Exception:
                                continue

                        if not sizes:
                            return None

                        max_lang, (max_bytes, max_title) = max(sizes.items(), key=lambda x: x[1][0])
                        views = get_pageviews(max_title, lang=max_lang)
                        summary = get_summary(max_title, lang=max_lang)
                        est_de = int(views * DE_ESTIMATE_FACTOR)
                        exists_de = article_exists_in_de(max_title)

                        wiki_url = f"https://{max_lang}.wikipedia.org/wiki/{quote(max_title)}"
                        google_url = f"https://www.google.com/search?q=\"{quote(max_title)}\"+site:.de"
                        langs_str = ", ".join(sorted([k.replace("wiki", "") for k in sitelinks.keys() if k.endswith("wiki")]))

                        return {
                            "Name": f'<a href="{wiki_url}" target="_blank">{max_title}</a>',
                            "Sprache (größte Version)": max_lang,
                            "Views (30d)": views,
                            "Estimated DE Views": est_de,
                            "DE Exists": "✅" if exists_de else "❌",
                            "Sprachen": langs_str,
                            "Summary": summary,
                            "Google": f'<a href="{google_url}" target="_blank">Suchen</a>'
                        }

                    except Exception as e:
                        print(f"❌ Fehler bei {qid}: {e}")
                        return None

                rows = []
                total = len(all_qids)
                progress = st.progress(0)

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(process_qid, qid, i, total): qid
                        for i, qid in enumerate(all_qids)
                    }
                    for i, future in enumerate(as_completed(futures)):
                        result = future.result()
                        if result:
                            rows.append(result)
                        progress.progress((i + 1) / total)

                if rows:
                    df = pd.DataFrame(rows)
                    st.markdown(f"""
                        <div style='height: 600px; overflow-y: auto'>
                            {df.to_html(escape=False, index=False)}
                        </div>
                    """, unsafe_allow_html=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ CSV herunterladen",
                        data=csv,
                        file_name="frauenbiografien_gapcheck.csv",
                        mime="text/csv")
                else:
                    st.info("Keine analysierbaren Artikel gefunden.") 
with tab5:
    st.header("🔎 Eigene Artikelliste analysieren")
    st.markdown("Gib eine Liste von Artikeln ein, getrennt durch Kommas. Das Tool prüft, ob es eine deutsche Version gibt, wie viele Views sie in der Quellsprache haben und schätzt das DE-Potenzial.")

    input_text = st.text_area("Artikel eingeben (kommagetrennt)", "Albert Einstein, Simone de Beauvoir, Yung Hurn")
    input_lang = st.selectbox("Sprache der Artikelliste", options=SUPPORTED_LANGS, index=0)

if st.button("🔍 Analysieren"):
    titles = [t.strip() for t in input_text.split(",") if t.strip()]
    if not titles:
        st.info("Bitte mindestens einen Artikelnamen eingeben.")
    else:
        with st.spinner("Analysiere Artikel..."):
            results = process_articles_concurrent(titles, lang=input_lang)

        if results:
            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values(by="Views (30d)", ascending=False)
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv, file_name="eigene_liste_check.csv", mime="text/csv")
            else:
                st.error("Keine Daten geladen – bitte überprüfe deine Artikelliste.")
        else:
            st.info("Keine gültigen Artikel erkannt.")

