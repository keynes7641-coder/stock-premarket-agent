import argparse
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

import exchange_calendars as xcals
import firebase_admin
import pytz
import requests
import yfinance as yf
from firebase_admin import credentials, firestore
from google import genai
from google.cloud.firestore_v1.base_query import FieldFilter
from google.genai import types

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

FLASH_MODEL = os.environ.get("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
PRO_MODEL = os.environ.get("GEMINI_PRO_MODEL", FLASH_MODEL)
FALLBACK_MODELS = list(
    dict.fromkeys(
        [
            FLASH_MODEL,
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]
    )
)
_grounding_enabled = True
_exhausted_models: set[str] = set()


def init_services() -> tuple[Any, genai.Client]:
    """初始化 Firebase Firestore 與 Gemini API 客戶端。"""
    sa_raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
    if not sa_raw:
        raise ValueError("缺少環境變數 FIREBASE_SERVICE_ACCOUNT。")

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise ValueError("缺少環境變數 GEMINI_API_KEY。")

    if not firebase_admin._apps:
        if sa_raw.startswith("{"):
            cred_dict = json.loads(sa_raw)
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate(sa_raw)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    client = genai.Client(
        api_key=gemini_key,
        http_options=types.HttpOptions(timeout=45000),
    )
    return db, client


def is_market_open(market: str) -> bool:
    """精準判斷台股 (XTAI) 或美股 (XNYS) 今日是否開市。"""
    if market == "TW":
        tz = pytz.timezone("Asia/Taipei")
        cal = xcals.get_calendar("XTAI")
    else:
        tz = pytz.timezone("America/New_York")
        cal = xcals.get_calendar("XNYS")

    today_str = datetime.now(tz).strftime("%Y-%m-%d")
    is_open = cal.is_session(today_str)
    print(f"[{market}] 當地日期 {today_str} 是否為交易日: {is_open}")
    return is_open


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    retries: int = 2,
) -> Any:
    """針對暫時性 503/500 尖峰錯誤進行短暫退避重試。"""
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            err_msg = str(exc)
            is_transient = "503" in err_msg or "UNAVAILABLE" in err_msg or "500" in err_msg
            if is_transient and attempt < retries - 1:
                wait_sec = 2
                print(f"[提示] {model} 暫時繁忙 (503/500)，{wait_sec} 秒後重試 ({attempt + 1}/{retries})...")
                time.sleep(wait_sec)
                continue
            raise


def get_macro_summary(client: genai.Client, market: str) -> str:
    """抓取大盤、費半、台積電ADR、VIX 與美債殖利率走勢，並生成總經風向摘要。"""
    indices = {
        "^GSPC": "標普500",
        "^SOX": "費城半導體",
        "TSM": "台積電ADR",
        "^VIX": "VIX恐慌指數",
        "^TNX": "美國10年期公債殖利率",
    }
    if market == "TW":
        indices["^TWII"] = "台灣加權指數"

    macro_lines = []
    for symbol, name in indices.items():
        try:
            df = yf.Ticker(symbol).history(period="5d")
            if len(df) >= 2:
                last = df.iloc[-1]["Close"]
                prev = df.iloc[-2]["Close"]
                chg = ((last - prev) / prev) * 100
                unit = "%" if symbol == "^TNX" else ""
                macro_lines.append(f"- {name} ({symbol}): {last:.2f}{unit} ({chg:+.2f}%)")
        except Exception as exc:
            print(f"[警告] 無法抓取指數 {symbol}: {exc}")
            continue

    raw_macro = "\n".join(macro_lines) if macro_lines else "- 暫無最新隔夜指數資料"
    prompt = f"""
    你是買方總經策略師。請根據以下隔夜市場數據，為今日【{market}】開盤提供 3 句話以內的「盤前總經風向與風險提示」：
    {raw_macro}

    要求：點出半導體/科技股氣氛、美債殖利率或 VIX 是否帶來估值壓力，以及今日操作該偏積極或保守控管倉位。
    """
    summary_text = "總經 AI 摘要生成暫時異常，請參考上方指數數據。"
    for model_name in FALLBACK_MODELS:
        try:
            res = _generate_with_retry(
                client,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (res.text or "").strip()
            if text:
                print(f"[成功] 總經風向摘要已由 {model_name} 生成")
                summary_text = text
                break
        except Exception as exc:
            print(f"[警告] 總經風向摘要使用 {model_name} 失敗: {exc}")

    return f"{raw_macro}\n\n**💡 總經風向判讀**：\n{summary_text}"


def get_price_summary(yf_symbol: str) -> str:
    """抓取標的近期收盤價、20MA 與成交量資訊。"""
    try:
        df = yf.Ticker(yf_symbol).history(period="1mo")
        if len(df) < 2:
            return "無法取得近期股價"
        latest, prev = df.iloc[-1], df.iloc[-2]
        chg_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
        ma20 = df["Close"].tail(20).mean()
        ma20_status = "站上月線" if latest["Close"] > ma20 else "跌破月線"
        return (
            f"最新收盤: {latest['Close']:.2f} ({chg_pct:+.2f}%) | "
            f"20MA: {ma20:.2f} ({ma20_status}) | 量: {int(latest['Volume']):,}"
        )
    except Exception as exc:
        return f"股價抓取異常: {exc}"


def get_tw_finmind_data(ticker: str) -> str:
    """透過 FinMind API 抓取台股近期三大法人買賣超與最新月營收。"""
    url = "https://api.finmindtrade.com/api/v4/data"
    parts = []

    # 1. 三大法人買賣超
    inst_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        res = requests.get(
            url,
            params={
                "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
                "data_id": ticker,
                "start_date": inst_start,
            },
            timeout=10,
        ).json()
        if res.get("data"):
            latest_date = res["data"][-1]["date"]
            day_records = [r for r in res["data"] if r["date"] == latest_date]
            details = [
                f"{r['name']}: {(r['buy'] - r['sell']) // 1000:+d}張"
                for r in day_records
            ]
            parts.append(f"[{latest_date} 法人] " + ", ".join(details))
    except Exception:
        pass

    # 2. 最新月營收
    rev_start = (datetime.now() - timedelta(days=75)).strftime("%Y-%m-%d")
    try:
        res = requests.get(
            url,
            params={
                "dataset": "TaiwanStockMonthRevenue",
                "data_id": ticker,
                "start_date": rev_start,
            },
            timeout=10,
        ).json()
        if res.get("data"):
            latest = res["data"][-1]
            rev_year = latest.get("revenue_year")
            rev_month = latest.get("revenue_month")
            revenue = latest.get("revenue")
            if revenue is not None:
                rev_billion = revenue / 1e8
                parts.append(f"[{rev_year}/{rev_month:02d} 營收] {rev_billion:.2f}億元")
    except Exception:
        pass

    return " | ".join(parts) if parts else "無近期法人與月營收資料"


def get_us_finnhub_news(ticker: str) -> str:
    """透過 Finnhub API 抓取美股過去 48 小時公司新聞標題。"""
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return "依賴 Google Search 檢索"
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={yesterday}&to={today}&token={api_key}"
    )
    try:
        articles = requests.get(url, timeout=10).json()
        if isinstance(articles, list) and articles:
            return "\n".join(
                [f"- {a.get('headline', '')} ({a.get('source', 'N/A')})" for a in articles[:5]]
            )
    except Exception:
        pass
    return "近兩日無 API 新聞記錄"


def get_google_news_rss(stock_name: str, ticker: str, market: str, max_items: int = 5) -> str:
    """透過免費 Google News RSS 抓取近兩日相關新聞標題（不消耗 Gemini Grounding 配額）。"""
    if market == "TW":
        q = f"{stock_name} {ticker} when:2d"
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    else:
        q = f"{ticker} stock OR {stock_name} when:2d"
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            return "暫無 RSS 新聞"
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")[:max_items]
        lines = []
        for item in items:
            title = (item.findtext("title") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if title:
                lines.append(f"- {title}" + (f" [{pub_date}]" if pub_date else ""))
        return "\n".join(lines) if lines else "近兩日無重大相關新聞"
    except Exception:
        return "暫無 RSS 新聞"


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    """容錯解析 Gemini 回傳的 JSON 字串，去除多餘的 Markdown 標記。"""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data

    raise ValueError(f"無法解析回傳內容為 JSON 物件: {cleaned[:200]}")


def analyze_stock(
    client: genai.Client,
    market: str,
    stock: dict[str, Any],
    macro_context: str,
) -> dict[str, Any]:
    """分析單一持股：優先嘗試 Search Grounding，若免費配額受限則自動以 RSS/籌碼數據降級完成分析。"""
    price_info = get_price_summary(stock["yf_symbol"])
    extra_data = (
        get_tw_finmind_data(stock["ticker"])
        if market == "TW"
        else get_us_finnhub_news(stock["ticker"])
    )
    rss_news = get_google_news_rss(stock["name"], stock["ticker"], market)

    today_str = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d")
    prompt = f"""
    今天是 {today_str}。你是買方證券分析師，請綜合以下過去 24~48 小時新聞、籌碼、技術面與大盤總經環境，為「{stock['name']} ({stock['ticker']})」提供盤前分析：

    - 大盤總經背景：{macro_context}
    - 標的：{stock['name']} ({stock['ticker']}) | 持有均價：{stock['avg_cost']}
    - 關注主軸：{stock.get('focus', '基本面與重大訂單動態')}
    - 價量技術面：{price_info}
    - 籌碼/API新聞摘要：{extra_data}
    - 近期即時新聞標題 (Google News RSS)：
    {rss_news}

    請嚴格輸出 JSON 格式，包含以下欄位（不要加 Markdown 標記）：
    {{
      "sentiment": "偏多 / 中性 / 偏空 三選一",
      "price_status": "最新價、損益狀態與均線/法人精簡一句話",
      "key_events": "過去24H關鍵事件摘要（若無實質影響股價的新聞或財報，直接填「無重大新事件」）",
      "action_advice": "結合持有均價、大盤氣氛與技術面的具體操作建議（限 80 字內）"
    }}
    """

    global _grounding_enabled

    # 嘗試順序：若尚有 Grounding 配額先嘗試 Grounding，否則依序走已驗證可用的 Gemini 3.x Flash 備援鏈
    attempts: list[tuple[str, bool]] = []
    if _grounding_enabled:
        attempts.append((PRO_MODEL, True))
    for m in FALLBACK_MODELS:
        if m not in _exhausted_models:
            attempts.append((m, False))

    last_err: Exception | None = None
    for model_name, use_search in attempts:
        mode_label = "Grounding" if use_search else "RSS/純數據模式"
        try:
            config_kwargs: dict[str, Any] = {
                "temperature": 0.2,
                "response_mime_type": "application/json",
            }
            if use_search:
                config_kwargs["tools"] = [{"google_search": {}}]

            try:
                response = _generate_with_retry(
                    client,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except Exception as first_err:
                err_msg = str(first_err)
                if any(
                    k in err_msg
                    for k in ("404", "429", "503", "504", "NOT_FOUND", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")
                ):
                    raise
                if use_search:
                    print(f"[提示] {stock['ticker']} ({model_name}) 切換純文字 JSON 提示重試")
                    response = _generate_with_retry(
                        client,
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            tools=[{"google_search": {}}],
                            temperature=0.2,
                        ),
                    )
                else:
                    raise

            parsed = _parse_json_response(response.text or "")
            print(f"[成功] {stock['name']} ({stock['ticker']}) 使用 {model_name} ({mode_label}) 分析完成")
            return {
                "ticker": stock["ticker"],
                "name": stock["name"],
                "avg_cost": stock["avg_cost"],
                "sentiment": str(parsed.get("sentiment") or "中性").strip(),
                "price_status": str(parsed.get("price_status") or price_info).strip(),
                "key_events": str(parsed.get("key_events") or "無重大新事件").strip(),
                "action_advice": str(
                    parsed.get("action_advice") or "暫無具體建議，請留意盤中量價變化。"
                ).strip(),
            }
        except Exception as exc:
            last_err = exc
            err_msg = str(exc)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                if use_search:
                    _grounding_enabled = False
                    print("[提示] 本次執行偵測到 Grounding 配額不足，後續標的改直接採用 RSS/純數據模式")
                else:
                    _exhausted_models.add(model_name)
                    print(f"[提示] 模型 {model_name} 配額暫滿，本次後續標的自動略過")
            print(f"[警告] {stock['name']} ({stock['ticker']}) 使用 {model_name} ({mode_label}) 失敗: {exc}")

    print(f"[錯誤] 分析 {stock['name']} ({stock['ticker']}) 所有模式皆失敗: {last_err}")
    try:
        avail = [
            m.name
            for m in client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or [])
        ]
        print(f"[診斷] 目前 API Key 可用模型清單: {', '.join(avail[:20])}")
    except Exception:
        pass
    return {
        "ticker": stock["ticker"],
        "name": stock["name"],
        "avg_cost": stock["avg_cost"],
        "sentiment": "中性",
        "price_status": price_info,
        "key_events": "AI 分析超時或解析異常，請稍後重試或手動檢視新聞。",
        "action_advice": "本檔標的本次分析失敗，建議開盤先觀望並留意關鍵支撐壓力位。",
    }


def send_telegram_notification(
    report_id: str,
    market: str,
    analyzed_stocks: list[dict[str, Any]],
) -> None:
    """選用功能：若有設定 Telegram 環境變數，推播精簡盤前摘要與儀表板連結。"""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return

    market_label = "台股" if market == "TW" else "美股"
    dashboard_url = os.environ.get("DASHBOARD_URL", "").strip()

    lines = [f"📊 <b>{html.escape(report_id)} {market_label}盤前情報已更新</b>", ""]
    for stock in analyzed_stocks:
        sentiment = str(stock.get("sentiment", "中性"))
        emoji = "🔴" if "多" in sentiment else ("🟢" if "空" in sentiment else "⚪")
        name = html.escape(str(stock.get("name", "")))
        ticker = html.escape(str(stock.get("ticker", "")))
        advice = html.escape(str(stock.get("action_advice", "")))
        lines.append(
            f"{emoji} <b>{name} ({ticker})</b> [{html.escape(sentiment)}]\n"
            f"   建議：{advice}"
        )

    if dashboard_url:
        safe_url = html.escape(dashboard_url, quote=True)
        lines.extend(["", f'👉 <a href="{safe_url}">開啟完整盤前儀表板</a>'])

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "\n".join(lines),
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.ok:
            print("Telegram 盤前摘要推播成功！")
        else:
            print(f"[警告] Telegram 推播失敗 ({resp.status_code}): {resp.text}")
    except Exception as exc:
        print(f"[警告] Telegram 推播發生例外: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="台美股 AI 盤前／盤後情報與重大事件 Agent")
    parser.add_argument(
        "--market",
        choices=["TW", "US", "ALL"],
        required=True,
        help="目標市場 (TW、US 或 ALL，ALL 僅用於 events 模式)",
    )
    parser.add_argument(
        "--mode",
        choices=["premarket", "afterhours", "events"],
        default="premarket",
        help="執行模式：premarket (盤前)、afterhours (盤後)、events (重大事件行事曆)",
    )
    parser.add_argument("--force", action="store_true", help="忽略休市日或快取強制執行")
    args = parser.parse_args()

    if args.mode in ("premarket", "afterhours"):
        if args.market == "ALL":
            raise ValueError("premarket 與 afterhours 模式請明確指定 --market TW 或 US。")
        if not args.force and not is_market_open(args.market):
            print(f"今日 {args.market} 休市，跳過執行。")
            return

    db, client = init_services()

    if args.market == "ALL":
        docs = db.collection("portfolio").stream()
    else:
        docs = (
            db.collection("portfolio")
            .where(filter=FieldFilter("market", "==", args.market))
            .stream()
        )
    portfolio = [doc.to_dict() for doc in docs]
    if not portfolio:
        print(f"Firestore 中無 {args.market} 持股資料，結束執行。")
        return

    if args.mode == "afterhours":
        from afterhours_events import run_afterhours_analysis

        run_afterhours_analysis(
            db=db,
            client=client,
            market=args.market,
            portfolio=portfolio,
            fallback_models=FALLBACK_MODELS,
            exhausted_models=_exhausted_models,
            generate_with_retry=_generate_with_retry,
            parse_json_response=_parse_json_response,
            get_us_finnhub_news=get_us_finnhub_news,
        )
        return

    if args.mode == "events":
        from afterhours_events import run_events_sync

        run_events_sync(
            db=db,
            client=client,
            portfolio=portfolio,
            force_refresh=args.force,
            fallback_models=FALLBACK_MODELS,
            exhausted_models=_exhausted_models,
            generate_with_retry=_generate_with_retry,
            parse_json_response=_parse_json_response,
        )
        return

    print("正在生成大盤總經摘要...")
    macro_summary = get_macro_summary(client, args.market)

    analyzed_stocks = []
    for idx, stock in enumerate(portfolio):
        print(f"正在分析 {stock['name']} ({stock['ticker']})...")
        res = analyze_stock(client, args.market, stock, macro_summary)
        analyzed_stocks.append(res)
        if idx < len(portfolio) - 1:
            time.sleep(2)

    today_str = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d")
    report_id = f"{today_str}_{args.market}"
    db.collection("reports").document(report_id).set(
        {
            "date": today_str,
            "market": args.market,
            "macro_summary": macro_summary,
            "stocks": analyzed_stocks,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
    )
    print(f"報告已成功寫入 Firestore ({report_id})！")

    send_telegram_notification(report_id, args.market, analyzed_stocks)


if __name__ == "__main__":
    main()
