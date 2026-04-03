import streamlit as st
import pandas as pd
import re
import requests
from urllib.parse import quote, quote_plus
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, stdev
import logging

logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

st.set_page_config(page_title="Wikipedia Relevanz-Radar", layout="wide")

# ---------- CONFIG ----------
SUPPORTED_LANGS = ["en", "de", "fr", "es", "ar", "tr", "it", "ru", "pl"]
DE_ESTIMATE_FACTOR = 0.12
HEADERS = {"User-Agent": "WikipediaGapFinder/0.3 (daniel.sigge@web.de)"}
REQUEST_TIMEOUT = 10


# ---------- HELPERS ----------
def safe_numeric(value, default=0):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if value in ("", "n/a", "N/A", "-", "❌", "None"):
            return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_get(url, params=None, timeout=REQUEST_TIMEOUT):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        return r
    except Exception:
        return None


def prepare_dataframe_for_sorting(df):
    if "Views (30d)" in df.columns:
        df["Views (30d)"] = pd.to_numeric(df["Views (30d)"], errors="coerce").fillna(0)
    if "Views (Yesterday)" in df.columns:
        df["Views (Yesterday)"] = pd.to_numeric(df["Views (Yesterday)"], errors="coerce").fillna(0)
    if "Estimated DE Views" in df.columns:
        df["Estimated DE Views"] = pd.to_numeric(df["Estimated DE Views"], errors="coerce").fillna(0)
    if "CV" in df.columns:
        df["CV"] = pd.to_numeric(df["CV"], errors="coerce")
    return df


def filter_missing_in_de(rows, only_missing=True):
    if not only_missing:
        return rows
    filtered = []
    for row in rows:
        exists_value = row.get("Exists in DE", row.get("DE Exists", "❌"))
        if exists_value == "❌":
            filtered.append(row)
    return filtered


def views_format(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


# ---------- WIKIPEDIA / WIKIDATA ----------
def get_wikidata_id(article_title, lang="en"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "titles": article_title,
        "redirects": 1,
        "format": "json"
    }
    r = safe_get(url, params=params)
    if not r:
        return None

    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "pageprops" in page and "wikibase_item" in page["pageprops"]:
            return page["pageprops"]["wikibase_item"]
    return None


@st.cache_data(ttl=86400)
def get_languages(wikidata_id):
    if not wikidata_id:
        return []

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    r = safe_get(url)
    if not r:
        return []

    data = r.json()
    sitelinks = data.get("entities", {}).get(wikidata_id, {}).get("sitelinks", {})
    langs = [k.replace("wiki", "") for k in sitelinks if k.endswith("wiki")]
    langs_sorted = sorted(langs)
    return langs_sorted[:8] + (["..."] if len(langs_sorted) > 8 else [])


def has_german_link(wikidata_id):
    if not wikidata_id:
        return False
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    r = safe_get(url)
    if not r:
        return False
    data = r.json()
    entity = data.get("entities", {}).get(wikidata_id, {})
    return "dewiki" in entity.get("sitelinks", {})


def exists_in_german_via_langlinks(title, source_lang="en"):
    url = f"https://{source_lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "langlinks",
        "lllang": "de",
        "redirects": 1,
        "format": "json"
    }
    r = safe_get(url, params=params)
    if not r:
        return False

    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if page.get("langlinks"):
            return True
    return False


@st.cache_data(ttl=86400)
def article_exists_in_de(title, lang="en"):
    wikidata_id = get_wikidata_id(title, lang=lang)
    via_wikidata = has_german_link(wikidata_id) if wikidata_id else False
    via_langlinks = exists_in_german_via_langlinks(title, source_lang=lang)
    return via_wikidata or via_langlinks


def normalize_title(title, lang="en"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "redirects": 1,
        "format": "json"
    }
    r = safe_get(url, params=params)
    if not r:
        return title.replace(" ", "_")

    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("title", title).replace(" ", "_")
    return title.replace(" ", "_")


def get_pageviews(title, lang="en", days=30):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    encoded = quote(title, safe="")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{encoded}/daily/{start_str}/{end_str}"
    r = safe_get(url)
    if not r:
        return 0

    data = r.json()
    return sum(item.get("views", 0) for item in data.get("items", []))


def get_yesterday_views(title, lang="en"):
    end_date = datetime.today() - timedelta(days=1)
    day_str = end_date.strftime("%Y%m%d")
    encoded = quote(title, safe="")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{encoded}/daily/{day_str}/{day_str}"
    r = safe_get(url)
    if not r:
        return 0
    data = r.json()
    items = data.get("items", [])
    if not items:
        return 0
    return items[0].get("views", 0)


@st.cache_data(ttl=86400)
def get_summary(title, lang="en"):
    encoded = quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    r = safe_get(url)
    if not r:
        return ""

    data = r.json()
    extract = data.get("extract", "")
    return extract[:300] if extract else ""


def get_daily_views_stats(title, lang="en", days=90):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    encoded = quote(title, safe="")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{encoded}/daily/{start_str}/{end_str}"
    r = safe_get(url)
    if not r:
        return 0, 0, 0, 0, "❌ Fehler"

    items = r.json().get("items", [])
    daily_views = [i["views"] for i in items if "views" in i]

    if not daily_views or mean(daily_views) == 0:
        return 0, 0, 0, 0, "Keine Daten"

    avg = mean(daily_views)
    std_dev = stdev(daily_views) if len(daily_views) > 1 else 0
    peak_ratio = max(daily_views) / avg if avg else 0
    cv = std_dev / avg if avg else 0

    if cv > 1.0:
        status = "🧨 Viral"
    elif cv < 0.3:
        status = "💎 Stable"
    else:
        status = "⚖️ Mixed"

    return round(avg), round(std_dev), round(cv, 2), round(peak_ratio, 2), status


# ---------- CATEGORY / TOP ----------
def get_category_members(category_name, lang="en", limit=5000):
    category_prefix = "Kategorie:" if lang == "de" else "Category:"
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
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

        r = safe_get(api_url, params=params)
        if not r:
            break

        data = r.json()
        collected.extend(data.get("query", {}).get("categorymembers", []))

        if "continue" in data and "cmcontinue" in data["continue"]:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break

        if len(collected) >= limit:
            break

    return collected[:limit]


def get_all_articles_recursive(category_name, lang="en", depth=2, limit=5000):
    collected = []
    seen_cats = set()
    category_prefix = "Kategorie:" if lang == "de" else "Category:"
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

    def crawl(cat, level):
        if level > depth or len(collected) >= limit:
            return
        seen_cats.add(cat)

        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"{category_prefix}{cat}",
            "cmlimit": 50,
            "cmtype": "page|subcat"
        }

        r = safe_get(api_url, params=params)
        if not r:
            return

        data = r.json()
        for item in data.get("query", {}).get("categorymembers", []):
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


def get_top_articles(lang="en", days=1, limit=100):
    titles = {}
    today = datetime.today()

    for delta in range(days):
        day = (today - timedelta(days=delta + 1)).strftime("%Y/%m/%d")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/{day}"
        r = safe_get(url)
        if not r:
            continue

        items = r.json().get("items", [])
        if not items:
            continue

        for item in items[0].get("articles", []):
            title = item.get("article")
            views = item.get("views", 0)
            if not title:
                continue
            if ":" in title or title.lower() in ["hauptseite", "main_page"]:
                continue
            titles[title] = titles.get(title, 0) + views

    return sorted(titles.items(), key=lambda x: x[1], reverse=True)[:limit]


def get_missing_titles_parallel(titles, lang):
    def check(title):
        return title if not article_exists_in_de(title, lang=lang) else None

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(check, titles))
    return [title for title in results if title]


# ---------- FRAUEN IN ROT ----------
@st.cache_data(ttl=86400)
def get_all_frauenrot_lists():
    base_url = "https://de.wikipedia.org"
    master_url = f"{base_url}/wiki/Wikipedia:WikiProjekt_Frauen/Frauen_in_Rot/Listen"
    r = safe_get(master_url)
    if not r:
        return {}

    html = r.text
    matches = re.findall(
        r'href="/wiki/(Wikipedia:WikiProjekt_Frauen/Frauen_in_Rot/Fehlende_Artikel.*?)"',
        html
    )
    unique = sorted(set(matches))
    return {
        match.split("/")[-1].replace("_", " "): base_url + "/wiki/" + match
        for match in unique
    }


@st.cache_data(ttl=86400)
def extract_qids_from_list(url):
    r = safe_get(url)
    if not r:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    qids = []

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        text = cells[-1].get_text(strip=True)
        if text.startswith("Q") and text[1:].isdigit():
            qids.append(text)

    return list(set(qids))


@st.cache_data(ttl=86400)
def get_sitelinks(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    r = safe_get(url)
    if not r:
        return {}
    data = r.json()
    return data.get("entities", {}).get(qid, {}).get("sitelinks", {})


# ---------- CORE PROCESSING ----------
def process_articles_concurrent(titles, lang):
    def process_single(title):
        try:
            wikidata_id = get_wikidata_id(title, lang=lang)
            de_exists_wikidata = has_german_link(wikidata_id) if wikidata_id else False
            de_exists_langlinks = exists_in_german_via_langlinks(title, source_lang=lang)
            de_exists = de_exists_wikidata or de_exists_langlinks

            langs = get_languages(wikidata_id) if wikidata_id else []
            langs_str = ", ".join(langs)

            normalized = normalize_title(title, lang=lang)
            views = get_pageviews(normalized, lang=lang)
            est_views = int(views * DE_ESTIMATE_FACTOR)

            avg, std_dev, cv, peak_ratio, virality = get_daily_views_stats(normalized, lang=lang)
            summary = get_summary(normalized, lang=lang)
            short_summary = summary[:180] + "..." if len(summary) > 180 else summary

            wiki_url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

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
                "Languages": "",
                "Views (30d)": 0,
                "Estimated DE Views": 0,
                "Exists in DE": "❌",
                "CV": None,
                "Viralität": "Fehler",
                "Summary": f"Fehler: {e}"
            }

    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(process_single, titles))


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


@st.cache_data(ttl=21600)
def get_top_viral_articles(lang="en", limit=10, source_pool=100):
    top_articles = get_top_articles(lang=lang, days=1, limit=source_pool)
    rows = []

    for title, _ in top_articles:
        normalized = normalize_title(title, lang=lang)
        yesterday_views = get_yesterday_views(normalized, lang=lang)
        avg, std_dev, cv, peak_ratio, virality = get_daily_views_stats(normalized, lang=lang)

        wiki_url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

        rows.append({
            "Title": f'<a href="{wiki_url}" target="_blank">{title}</a>',
            "Views (Yesterday)": yesterday_views,
            "CV": cv,
            "Viralität": virality
        })

    df = pd.DataFrame(rows)
    df = prepare_dataframe_for_sorting(df)
    df = df.sort_values(by=["CV", "Views (Yesterday)"], ascending=[False, False]).head(limit)
    return df


# ---------- UI ----------
st.title("Wikipedia Relevanz-Radar")

tab_start, tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Start",
    "🙇🏻‍♂️ Info",
    "1) Kategorie-Analyse",
    "2) Meistgelesen vs. DE (Schnell)",
    "3) Meistgelesen vs. DE (Gefiltert)",
    "4) Rotlink-Frauen-Projekt",
    "5) Eigene Artikelliste"
])

with tab_start:
    st.header("Aktuelle virale Top-Artikel von gestern")
    st.markdown("Automatische Übersicht der aktuell auffälligen Artikel in **DE** und **EN**, sortiert nach Viralitäts-Score (CV).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("DE")
        with st.spinner("Lade DE-Trends..."):
            df_de = get_top_viral_articles(lang="de", limit=10, source_pool=100)
        if not df_de.empty:
            display_de = df_de.copy()
            display_de["Views (Yesterday)"] = display_de["Views (Yesterday)"].apply(views_format)
            st.markdown(display_de.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("Keine Daten verfügbar.")

    with col2:
        st.subheader("EN")
        with st.spinner("Lade EN-Trends..."):
            df_en = get_top_viral_articles(lang="en", limit=10, source_pool=100)
        if not df_en.empty:
            display_en = df_en.copy()
            display_en["Views (Yesterday)"] = display_en["Views (Yesterday)"].apply(views_format)
            st.markdown(display_en.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("Keine Daten verfügbar.")

with tab0:
    st.markdown("""
Der **Wikipedia Relevanz-Radar** hilft dabei, relevante Artikel zu finden, die in anderen Sprachversionen stark gelesen werden.

Neu:
- pro Tab kannst du jetzt **selbst umschalten**, ob Artikel mit vorhandener DE-Version angezeigt werden sollen
- zusätzlicher Start-Tab mit **Top-Viral-Artikeln von gestern** in DE und EN
""")

with tab1:
    st.header("1) Kategorie-Analyse")
    category_input = st.text_input("Wikipedia-Kategorie", value="20th-century philosophers")
    lang_code = st.selectbox("Quellsprache", options=SUPPORTED_LANGS, index=0, key="tab1_lang")
    use_subcats = st.checkbox("Unterkategorien einbeziehen", value=True)
    only_missing_tab1 = st.checkbox("Artikel mit vorhandener DE-Version ausblenden", value=True, key="tab1_only_missing")

    if "category_results" not in st.session_state:
        st.session_state["category_results"] = []
        st.session_state["category_cursor"] = 0
        st.session_state["category_total"] = 0
        st.session_state["category_members"] = []

    if st.button("Kategorie analysieren & erste Artikel laden"):
        with st.spinner("Lade Kategorie..."):
            if use_subcats:
                members, _ = get_all_articles_recursive(category_input, lang=lang_code, depth=2, limit=5000)
            else:
                members = get_category_members(category_input, lang=lang_code, limit=5000)

            members = [m for m in members if "#" not in m["title"]]
            st.session_state["category_members"] = members
            st.session_state["category_cursor"] = 0
            st.session_state["category_total"] = len(members)
            st.session_state["category_results"] = []

        if st.session_state["category_members"]:
            to_process = st.session_state["category_members"][:50]
            with st.spinner("Analysiere Artikel..."):
                titles = [a["title"] for a in to_process]
                rows = process_articles_concurrent(titles, lang_code)
                st.session_state["category_results"].extend(rows)
                st.session_state["category_results"].sort(
                    key=lambda x: safe_numeric(x.get("Views (30d)", 0)),
                    reverse=True
                )
                st.session_state["category_cursor"] += len(to_process)

    if st.session_state["category_members"]:
        total = st.session_state["category_total"]
        current_cursor = st.session_state["category_cursor"]
        next_cursor = min(current_cursor + 50, total)
        to_process = st.session_state["category_members"][current_cursor:next_cursor]

        if to_process and st.button(f"Nächste {len(to_process)} Artikel analysieren"):
            with st.spinner("Analysiere weitere Artikel..."):
                titles = [a["title"] for a in to_process]
                rows = process_articles_concurrent(titles, lang_code)
                st.session_state["category_results"].extend(rows)
                st.session_state["category_results"].sort(
                    key=lambda x: safe_numeric(x.get("Views (30d)", 0)),
                    reverse=True
                )
                st.session_state["category_cursor"] += len(to_process)

        display_rows = filter_missing_in_de(st.session_state["category_results"], only_missing_tab1)

        if display_rows:
            df = pd.DataFrame(display_rows)
            df = prepare_dataframe_for_sorting(df)
            df = df.sort_values(by="Views (30d)", ascending=False)
            st.markdown(f"**{st.session_state['category_cursor']} von {total} Artikeln analysiert.**")
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

with tab2:
    st.header("2) Meistgelesen vs. DE (Schnell)")
    lang_code = st.selectbox("Quellsprache", options=SUPPORTED_LANGS, index=0, key="tab2_lang")
    period = st.selectbox("Zeitraum", ["Yesterday", "Past 30 Days (aggregated)"])
    limit = st.slider("Anzahl Top-Artikel", 10, 5000, 1000)
    only_missing_tab2 = st.checkbox("Artikel mit vorhandener DE-Version ausblenden", value=True, key="tab2_only_missing")

    if st.button("Artikel laden", key="tab2_button"):
        with st.spinner("Analysiere..."):
            days = 1 if period == "Yesterday" else 30
            top_articles = get_top_articles(lang=lang_code, days=days, limit=limit)
            titles = [title for title, _ in top_articles]
            results = process_articles_concurrent(titles, lang_code)
            results = filter_missing_in_de(results, only_missing_tab2)

            if results:
                df = pd.DataFrame(results)
                df = prepare_dataframe_for_sorting(df)
                df = df.sort_values(by="Views (30d)", ascending=False)
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("CSV herunterladen", data=csv, file_name="top_articles.csv", mime="text/csv")
            else:
                st.info("Keine passenden Artikel gefunden.")

with tab3:
    st.header("3) Meistgelesen vs. DE (Gefiltert)")
    selected_lang = st.selectbox("Quellsprache", options=SUPPORTED_LANGS, index=0, key="tab3_lang")
    only_missing_tab3 = st.checkbox("Artikel mit vorhandener DE-Version ausblenden", value=True, key="tab3_only_missing")

    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Zeitraum (Tage)", [7, 14, 30, 90], index=2)
    with col2:
        limit = st.selectbox("Anzahl Artikel", [100, 250, 500, 1000], index=3)

    if st.button(f"Top Missing laden ({selected_lang} → DE)", key="tab3_button"):
        with st.spinner(f"Lade Top-Artikel aus {selected_lang}.wikipedia.org..."):
            top_articles = get_top_articles(lang=selected_lang, days=days, limit=limit)
            titles = [title for title, _ in top_articles]

            if only_missing_tab3:
                titles = get_missing_titles_parallel(titles, selected_lang)

            if titles:
                results = process_articles_with_progress(titles, selected_lang)
                results = filter_missing_in_de(results, only_missing_tab3)

                if results:
                    df = pd.DataFrame(results)
                    df = prepare_dataframe_for_sorting(df)
                    df = df.sort_values(by="Views (30d)", ascending=False)
                    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("CSV herunterladen", data=csv, file_name="top_missing_articles.csv", mime="text/csv")
                else:
                    st.info("Keine passenden Artikel gefunden.")
            else:
                st.success("Keine passenden Artikel gefunden.")

with tab4:
    st.header("4) Rotlink-Frauen-Projekt")
    only_missing_tab4 = st.checkbox("Artikel mit vorhandener DE-Version ausblenden", value=True, key="tab4_only_missing")

    with st.spinner("Lade Listen..."):
        frauenrot_lists = get_all_frauenrot_lists()

    selected_lists = st.multiselect(
        "Frauen-in-Rot-Listen auswählen",
        options=list(frauenrot_lists.keys()),
        key="tab4_multiselect"
    )

    if selected_lists and st.button("Relevanz analysieren", key="tab4_button"):
        with st.spinner("Analysiere Listen..."):
            all_qids = set()
            for name in selected_lists:
                url = frauenrot_lists[name]
                qids = extract_qids_from_list(url)
                all_qids.update(qids)

            rows = []
            failed_qids = []
            total = len(all_qids)
            progress = st.progress(0)
            status_text = st.empty()

            def process_qid_robust(qid):
                try:
                    sitelinks = get_sitelinks(qid)
                    if not sitelinks:
                        raise ValueError("Keine Sprachversionen")

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
                            "redirects": 1,
                            "format": "json"
                        }
                        resp = safe_get(rev_url, params=rev_params, timeout=5)
                        if not resp:
                            continue
                        data = resp.json()
                        pages = data.get("query", {}).get("pages", {})
                        for page in pages.values():
                            size = page.get("revisions", [{}])[0].get("size", 0)
                            sizes[lang] = (size, title)

                    if not sizes:
                        raise ValueError("Keine Artikelgröße")

                    max_lang, (max_bytes, max_title) = max(sizes.items(), key=lambda x: x[1][0])

                    views = get_pageviews(max_title, lang=max_lang)
                    summary = get_summary(max_title, lang=max_lang)
                    est_de = int(views * DE_ESTIMATE_FACTOR)
                    exists_de = article_exists_in_de(max_title, lang=max_lang)

                    wiki_url = f"https://{max_lang}.wikipedia.org/wiki/{quote(max_title.replace(' ', '_'))}"
                    query = f'"{max_title}" site:.de'
                    google_url = f"https://www.google.com/search?q={quote_plus(query)}"

                    langs_str = ", ".join(sorted(
                        [k.replace("wiki", "") for k in sitelinks.keys() if k.endswith("wiki")]
                    ))

                    return {
                        "Name": f'<a href="{wiki_url}" target="_blank">{max_title}</a>',
                        "Sprache (größte Version)": max_lang,
                        "Views (30d)": views,
                        "Estimated DE Views": est_de,
                        "Exists in DE": "✅" if exists_de else "❌",
                        "Sprachen": langs_str,
                        "Summary": summary,
                        "Google": f'<a href="{google_url}" target="_blank">Suchen</a>'
                    }

                except Exception:
                    failed_qids.append(qid)
                    return None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(process_qid_robust, qid): qid for qid in all_qids}
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result:
                        rows.append(result)
                    progress.progress((i + 1) / total if total else 1)
                    status_text.text(f"{i+1}/{total} verarbeitet...")

            rows = filter_missing_in_de(rows, only_missing_tab4)

            if rows:
                df = pd.DataFrame(rows)
                df = prepare_dataframe_for_sorting(df)
                df = df.sort_values(by="Views (30d)", ascending=False)
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("CSV herunterladen", data=csv, file_name="frauenrot_gapcheck.csv", mime="text/csv")
            else:
                st.info("Keine passenden Artikel gefunden.")

with tab5:
    st.header("5) Eigene Artikelliste")
    only_missing_tab5 = st.checkbox("Artikel mit vorhandener DE-Version ausblenden", value=True, key="tab5_only_missing")
    input_text = st.text_area("Artikelliste (kommagetrennt)", "Albert Einstein, Simone de Beauvoir, Yung Hurn")
    input_lang = st.selectbox("Quellsprache", options=SUPPORTED_LANGS, index=0, key="tab5_lang")

    if st.button("Analysieren", key="tab5_button"):
        titles = [t.strip() for t in input_text.split(",") if t.strip()]
        if not titles:
            st.info("Bitte mindestens einen Artikelnamen eingeben.")
        else:
            with st.spinner("Analysiere Artikel..."):
                results = process_articles_concurrent(titles, input_lang)
                results = filter_missing_in_de(results, only_missing_tab5)

            if results:
                df = pd.DataFrame(results)
                df = prepare_dataframe_for_sorting(df)
                df = df.sort_values(by="Views (30d)", ascending=False)
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("CSV herunterladen", data=csv, file_name="eigene_liste_check.csv", mime="text/csv")
            else:
                st.info("Keine passenden Artikel gefunden.")
