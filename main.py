import streamlit as st
import pandas as pd
import re
import requests
import time
from urllib.parse import quote, quote_plus
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, stdev

st.set_page_config(page_title="Wikipedia Relevanz-Radar", layout="wide")

# ---------- CONFIG ----------
SUPPORTED_LANGS = ["en", "de", "fr", "es", "ar", "tr", "it", "ru", "pl"]
DE_ESTIMATE_FACTOR = 0.12
HEADERS = {"User-Agent": "WikipediaGapFinder/0.6 (daniel.sigge@web.de)"}
REQUEST_TIMEOUT = 10

MW_TITLES_PER_REQUEST = 50
WD_IDS_PER_REQUEST = 50


# ---------- HELPERS ----------
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def safe_numeric(value, default=0):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if value in ("", "n/a", "N/A", "-", "❌", "❓", "None"):
            return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def views_format(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


def safe_get(url, params=None, timeout=REQUEST_TIMEOUT, retries=3, pause=0.6):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(pause * (attempt + 1))
                continue
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
            else:
                return None
    return None


def prepare_dataframe_for_sorting(df):
    for col in ["Views (30d)", "Views (Yesterday)", "Estimated DE Views", "CV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def reorder_columns(df, context="default"):
    preferred_orders = {
        "default": [
            "Title",
            "CV",
            "Viralität",
            "Views (30d)",
            "Views (Yesterday)",
            "Estimated DE Views",
            "Exists in DE",
            "German Title",
            "Summary",
        ],
        "frauen": [
            "Name",
            "CV",
            "Viralität",
            "Views (30d)",
            "Estimated DE Views",
            "Exists in DE",
            "German Title",
            "Sprache (größte Version)",
            "Summary",
            "Google",
        ],
        "start": [
            "Title",
            "CV",
            "Viralität",
            "Views (Yesterday)",
        ],
    }

    order = preferred_orders.get(context, preferred_orders["default"])
    existing = [col for col in order if col in df.columns]
    remaining = [col for col in df.columns if col not in existing]
    return df[existing + remaining]


def filter_missing_in_de(rows, only_missing=True):
    if not only_missing:
        return rows
    return [row for row in rows if row.get("Exists in DE") == "❌"]


# ---------- BASIC WIKIPEDIA ----------
def normalize_title_fallback(title: str) -> str:
    return title.replace(" ", "_")


@st.cache_data(ttl=86400)
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
        return normalize_title_fallback(title)

    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("title", title).replace(" ", "_")
    return normalize_title_fallback(title)


@st.cache_data(ttl=86400)
def get_pageviews(title, lang="en", days=30):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    encoded = quote(title, safe="")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/all-agents/{encoded}/daily/{start_str}/{end_str}"
    )
    r = safe_get(url)
    if not r:
        return None

    data = r.json()
    return sum(item.get("views", 0) for item in data.get("items", []))


@st.cache_data(ttl=86400)
def get_summary(title, lang="en"):
    encoded = quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    r = safe_get(url)
    if not r:
        return ""
    data = r.json()
    return data.get("extract", "")


@st.cache_data(ttl=86400)
def get_daily_views_stats(title, lang="en", days=30):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    encoded = quote(title, safe="")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/all-agents/{encoded}/daily/{start_str}/{end_str}"
    )
    r = safe_get(url)
    if not r:
        return None, None, None, None, "Fehler"

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


# ---------- BATCH METADATA ----------
def get_wikidata_sitelinks_batch(qids):
    result = {}
    for chunk in chunks(list(set([q for q in qids if q])), WD_IDS_PER_REQUEST):
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "sitelinks",
            "format": "json"
        }
        r = safe_get(url, params=params)
        if not r:
            continue
        data = r.json()
        entities = data.get("entities", {})
        for qid, entity in entities.items():
            result[qid] = entity.get("sitelinks", {})
    return result


def get_batch_article_info(titles, lang="en"):
    info = {
        t: {
            "normalized_title": normalize_title_fallback(t),
            "qid": None,
            "de_title": None,
            "lookup_failed": False,
        }
        for t in titles
    }

    for chunk_titles in chunks(titles, MW_TITLES_PER_REQUEST):
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": "|".join(chunk_titles),
            "redirects": 1,
            "prop": "pageprops|langlinks",
            "lllang": "de",
            "format": "json"
        }

        r = safe_get(url, params=params)
        if not r:
            for t in chunk_titles:
                info[t]["lookup_failed"] = True
            continue

        data = r.json()
        query = data.get("query", {})

        alias_map = {}
        for entry in query.get("normalized", []):
            alias_map[entry["from"]] = entry["to"]
        for entry in query.get("redirects", []):
            alias_map[entry["from"]] = entry["to"]

        pages = query.get("pages", {})
        page_by_title = {}
        for page in pages.values():
            page_title = page.get("title")
            if page_title:
                page_by_title[page_title] = page

        for original in chunk_titles:
            resolved = alias_map.get(original, original)
            page = (
                page_by_title.get(resolved)
                or page_by_title.get(resolved.replace("_", " "))
                or page_by_title.get(resolved.replace(" ", "_"))
            )

            if not page:
                continue

            normalized_title = page.get("title", original).replace(" ", "_")
            qid = page.get("pageprops", {}).get("wikibase_item")

            de_title = None
            langlinks = page.get("langlinks", [])
            if langlinks:
                de_title = langlinks[0].get("*")

            info[original]["normalized_title"] = normalized_title
            info[original]["qid"] = qid
            info[original]["de_title"] = de_title

    qids = [v["qid"] for v in info.values() if v["qid"]]
    sitelinks_map = get_wikidata_sitelinks_batch(qids)

    for original, meta in info.items():
        qid = meta["qid"]
        if not qid:
            continue

        sitelinks = sitelinks_map.get(qid, {})
        if not meta["de_title"] and "dewiki" in sitelinks:
            meta["de_title"] = sitelinks["dewiki"].get("title")

    return info


@st.cache_data(ttl=86400)
def article_exists_in_de(title, lang="en"):
    info = get_batch_article_info([title], lang=lang)
    row = info.get(title, {})
    return row.get("de_title") is not None


# ---------- TOP ARTICLES ----------
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

    sorted_titles = sorted(titles.items(), key=lambda x: x[1], reverse=True)
    return sorted_titles[:limit]


# ---------- CATEGORY ----------
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
    sitelinks = get_wikidata_sitelinks_batch([qid])
    return sitelinks.get(qid, {})


# ---------- CORE PROCESSING ----------
def process_articles_batch(
    titles,
    lang,
    known_views=None,
    view_column="Views (30d)",
    include_summary=True,
    include_stats=True,
    max_workers=4,
):
    batch_info = get_batch_article_info(titles, lang=lang)

    def enrich(original_title):
        meta = batch_info.get(original_title, {})
        normalized_title = meta.get("normalized_title", normalize_title_fallback(original_title))
        de_title = meta.get("de_title")
        lookup_failed = meta.get("lookup_failed", False)

        if de_title:
            exists_in_de = "✅"
        elif lookup_failed:
            exists_in_de = "❓"
        else:
            exists_in_de = "❌"

        wiki_url = f"https://{lang}.wikipedia.org/wiki/{quote(normalized_title)}"

        row = {
            "Title": f'<a href="{wiki_url}" target="_blank">{normalized_title.replace("_", " ")}</a>',
            "CV": None,
            "Viralität": "",
            "German Title": de_title if de_title else "",
            "Exists in DE": exists_in_de,
        }

        if known_views is not None:
            views = known_views.get(original_title)
        else:
            views = get_pageviews(normalized_title, lang=lang, days=30)

        row[view_column] = views
        row["Estimated DE Views"] = int(views * DE_ESTIMATE_FACTOR) if isinstance(views, (int, float)) else None

        if include_summary:
            summary = get_summary(normalized_title, lang=lang) or ""
            row["Summary"] = summary[:180] + "..." if len(summary) > 180 else summary

        if include_stats:
            avg, std_dev, cv, peak_ratio, virality = get_daily_views_stats(normalized_title, lang=lang, days=30)
            row["CV"] = cv
            row["Viralität"] = virality

        return row

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(enrich, titles))

    return rows


@st.cache_data(ttl=21600)
def get_top_viral_articles(lang="en", limit=10, source_pool=20):
    top_articles = get_top_articles(lang=lang, days=1, limit=source_pool)
    titles = [title for title, _ in top_articles]
    known_views = {title: views for title, views in top_articles}

    def enrich(title):
        normalized = normalize_title(title, lang=lang)
        _, _, cv, _, virality = get_daily_views_stats(normalized, lang=lang, days=30)
        wiki_url = f"https://{lang}.wikipedia.org/wiki/{quote(normalized)}"
        return {
            "Title": f'<a href="{wiki_url}" target="_blank">{normalized.replace("_", " ")}</a>',
            "Views (Yesterday)": known_views.get(title),
            "CV": cv,
            "Viralität": virality,
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(enrich, titles))

    df = pd.DataFrame(rows)
    df = prepare_dataframe_for_sorting(df)
    df = reorder_columns(df, context="start")
    df = df.sort_values(by=["CV", "Views (Yesterday)"], ascending=[False, False])
    return df.head(limit)


# ---------- UI ----------
st.title("Wikipedia Relevanz-Radar")

tab_start, tab_info, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Start",
    "🕵🏻 Info",
    "1) Kategorie-Analyse",
    "2) Meistgelesen vs. DE (Schnell)",
    "3) Meistgelesen vs. DE (Gefiltert)",
    "4) Rotlink-Frauen-Projekt",
    "5) Eigene Artikelliste"
])

with tab_start:
    st.header("Aktuelle virale Top-Artikel von gestern")
    st.markdown("DE und EN, sortiert nach CV. Lädt nur auf Knopfdruck, damit die App schnell bleibt.")

    if st.button("Top virale Artikel laden", key="load_start_tab"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("DE")
            with st.spinner("Lade DE-Trends..."):
                df_de = get_top_viral_articles(lang="de", limit=10, source_pool=20)
            if not df_de.empty:
                display_de = df_de.copy()
                display_de["Views (Yesterday)"] = display_de["Views (Yesterday)"].apply(views_format)
                st.markdown(display_de.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("Keine Daten verfügbar.")

        with col2:
            st.subheader("EN")
            with st.spinner("Lade EN-Trends..."):
                df_en = get_top_viral_articles(lang="en", limit=10, source_pool=20)
            if not df_en.empty:
                display_en = df_en.copy()
                display_en["Views (Yesterday)"] = display_en["Views (Yesterday)"].apply(views_format)
                st.markdown(display_en.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("Keine Daten verfügbar.")
    else:
        st.info("Klicke auf „Top virale Artikel laden“.")

with tab_info:
    st.markdown("""
Dieses Tool sucht nach relevanten Artikeln in anderen Sprachversionen und prüft, ob sie in der deutschen Wikipedia existieren.

Bedeutung:
- **✅** deutscher Artikel gefunden
- **❌** kein deutscher Artikel gefunden
- **❓** technische Unsicherheit bei der Prüfung

Wenn der Filter aktiv ist, werden nur bestätigte **❌** angezeigt.
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

    if st.button("Kategorie analysieren & erste Artikel laden", key="tab1_first_load"):
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
                rows = process_articles_batch(
                    titles,
                    lang=lang_code,
                    known_views=None,
                    view_column="Views (30d)",
                    include_summary=True,
                    include_stats=True,
                    max_workers=4,
                )
                st.session_state["category_results"].extend(rows)
                st.session_state["category_results"].sort(
                    key=lambda x: safe_numeric(x.get("Views (30d)"), 0),
                    reverse=True
                )
                st.session_state["category_cursor"] += len(to_process)

    if st.session_state["category_members"]:
        total = st.session_state["category_total"]
        current_cursor = st.session_state["category_cursor"]
        next_cursor = min(current_cursor + 50, total)
        to_process = st.session_state["category_members"][current_cursor:next_cursor]

        if to_process and st.button(f"Nächste {len(to_process)} Artikel analysieren", key="tab1_next_load"):
            with st.spinner("Analysiere weitere Artikel..."):
                titles = [a["title"] for a in to_process]
                rows = process_articles_batch(
                    titles,
                    lang=lang_code,
                    known_views=None,
                    view_column="Views (30d)",
                    include_summary=True,
                    include_stats=True,
                    max_workers=4,
                )
                st.session_state["category_results"].extend(rows)
                st.session_state["category_results"].sort(
                    key=lambda x: safe_numeric(x.get("Views (30d)"), 0),
                    reverse=True
                )
                st.session_state["category_cursor"] += len(to_process)

        display_rows = filter_missing_in_de(st.session_state["category_results"], only_missing_tab1)

        if display_rows:
            df = pd.DataFrame(display_rows)
            df = prepare_dataframe_for_sorting(df)
            df = reorder_columns(df, context="default")
            df = df.sort_values(by="Views (30d)", ascending=False, na_position="last")
            st.markdown(f"**{st.session_state['category_cursor']} von {total} Artikeln analysiert.**")
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("Keine passenden Artikel gefunden.")

with tab2:
    st.header("2) Meistgelesen vs. DE (Schnell)")
    lang_code = st.selectbox("Quellsprache", options=SUPPORTED_LANGS, index=0, key="tab2_lang")
    period = st.selectbox("Zeitraum", ["Yesterday", "Past 30 Days (aggregated)"])
    limit = st.slider("Anzahl Top-Artikel", 10, 5000, 300)
    only_missing_tab2 = st.checkbox("Artikel mit vorhandener DE-Version ausblenden", value=True, key="tab2_only_missing")

    if st.button("Artikel laden", key="tab2_button"):
        with st.spinner("Analysiere..."):
            days = 1 if period == "Yesterday" else 30
            top_articles = get_top_articles(lang=lang_code, days=days, limit=limit)
            titles = [title for title, _ in top_articles]
            known_views = {title: views for title, views in top_articles}
            view_col = "Views (Yesterday)" if days == 1 else "Views (30d)"

            results = process_articles_batch(
                titles,
                lang=lang_code,
                known_views=known_views,
                view_column=view_col,
                include_summary=False,
                include_stats=True,
                max_workers=4,
            )

            results = filter_missing_in_de(results, only_missing_tab2)

            if results:
                df = pd.DataFrame(results)
                df = prepare_dataframe_for_sorting(df)
                df = reorder_columns(df, context="default")
                df = df.sort_values(by=view_col, ascending=False, na_position="last")
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
        limit = st.selectbox("Anzahl Artikel", [100, 250, 500, 1000], index=1)

    if st.button(f"Top Missing laden ({selected_lang} → DE)", key="tab3_button"):
        with st.spinner(f"Lade Top-Artikel aus {selected_lang}.wikipedia.org..."):
            top_articles = get_top_articles(lang=selected_lang, days=days, limit=limit)
            titles = [title for title, _ in top_articles]
            known_views = {title: views for title, views in top_articles}

            if only_missing_tab3:
                batch_info = get_batch_article_info(titles, lang=selected_lang)
                titles = [
                    t for t in titles
                    if batch_info.get(t, {}).get("de_title") is None
                    and not batch_info.get(t, {}).get("lookup_failed", False)
                ]
                known_views = {t: known_views[t] for t in titles if t in known_views}

            if titles:
                results = process_articles_batch(
                    titles,
                    lang=selected_lang,
                    known_views=known_views,
                    view_column="Views (30d)",
                    include_summary=True,
                    include_stats=True,
                    max_workers=4,
                )
                results = filter_missing_in_de(results, only_missing_tab3)

                if results:
                    df = pd.DataFrame(results)
                    df = prepare_dataframe_for_sorting(df)
                    df = reorder_columns(df, context="default")
                    df = df.sort_values(by="Views (30d)", ascending=False, na_position="last")
                    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("CSV herunterladen", data=csv, file_name="top_missing_articles.csv", mime="text/csv")
                else:
                    st.info("Keine passenden Artikel gefunden.")
            else:
                st.info("Keine passenden Artikel gefunden.")

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

                    max_lang, (_, max_title) = max(sizes.items(), key=lambda x: x[1][0])

                    title_for_api = max_title.replace(" ", "_")
                    views = get_pageviews(title_for_api, lang=max_lang, days=30)
                    summary = get_summary(title_for_api, lang=max_lang)
                    est_de = int(views * DE_ESTIMATE_FACTOR) if isinstance(views, (int, float)) else None
                    _, _, cv, _, virality = get_daily_views_stats(title_for_api, lang=max_lang, days=30)

                    single_info = get_batch_article_info([max_title], lang=max_lang).get(max_title, {})
                    de_title = single_info.get("de_title")
                    if de_title:
                        exists_de = "✅"
                    elif single_info.get("lookup_failed", False):
                        exists_de = "❓"
                    else:
                        exists_de = "❌"

                    wiki_url = f"https://{max_lang}.wikipedia.org/wiki/{quote(title_for_api)}"
                    query = f'"{max_title}" site:.de'
                    google_url = f"https://www.google.com/search?q={quote_plus(query)}"

                    return {
                        "Name": f'<a href="{wiki_url}" target="_blank">{max_title}</a>',
                        "CV": cv,
                        "Viralität": virality,
                        "German Title": de_title if de_title else "",
                        "Sprache (größte Version)": max_lang,
                        "Views (30d)": views,
                        "Estimated DE Views": est_de,
                        "Exists in DE": exists_de,
                        "Summary": summary[:180] + "..." if len(summary) > 180 else summary,
                        "Google": f'<a href="{google_url}" target="_blank">Suchen</a>'
                    }

                except Exception:
                    return {
                        "Name": qid,
                        "CV": None,
                        "Viralität": "Fehler",
                        "German Title": "",
                        "Sprache (größte Version)": "",
                        "Views (30d)": None,
                        "Estimated DE Views": None,
                        "Exists in DE": "❓",
                        "Summary": "Fehler",
                        "Google": ""
                    }

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_qid_robust, qid): qid for qid in all_qids}
                for i, future in enumerate(as_completed(futures)):
                    rows.append(future.result())
                    progress.progress((i + 1) / total if total else 1)
                    status_text.text(f"{i+1}/{total} verarbeitet...")

            rows = filter_missing_in_de(rows, only_missing_tab4)

            if rows:
                df = pd.DataFrame(rows)
                df = prepare_dataframe_for_sorting(df)
                df = reorder_columns(df, context="frauen")
                df = df.sort_values(by="Views (30d)", ascending=False, na_position="last")
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
                results = process_articles_batch(
                    titles,
                    lang=input_lang,
                    known_views=None,
                    view_column="Views (30d)",
                    include_summary=True,
                    include_stats=True,
                    max_workers=4,
                )
                results = filter_missing_in_de(results, only_missing_tab5)

            if results:
                df = pd.DataFrame(results)
                df = prepare_dataframe_for_sorting(df)
                df = reorder_columns(df, context="default")
                df = df.sort_values(by="Views (30d)", ascending=False, na_position="last")
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("CSV herunterladen", data=csv, file_name="eigene_liste_check.csv", mime="text/csv")
            else:
                st.info("Keine passenden Artikel gefunden.")
