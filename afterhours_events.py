import json
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote

import pytz
import requests
import yfinance as yf
from firebase_admin import firestore
from google import genai
from google.genai import types

TW_TZ = pytz.timezone("Asia/Taipei")


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _roc_to_iso(roc_str: str) -> str | None:
    """將民國日期 (例如 1151208 或 115/12/08) 轉為 YYYY-MM-DD。"""
    digits = re.sub(r"\D", "", str(roc_str or ""))
    if len(digits) == 7:
        y = int(digits[:3]) + 1911
        m = int(digits[3:5])
        d = int(digits[5:7])
        try:
            return date(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            return None
    if len(digits) == 8:
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _fetch_google_news_rss(query: str, market: str, max_items: int = 8) -> list[dict[str, str]]:
    """抓取 Google News RSS，回傳標題與發布時間清單。"""
    if market == "TW":
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    else:
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            return []
        root = ET.fromstring(resp.text)
        out: list[dict[str, str]] = []
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            link = (item.findtext("link") or "").strip()
            if title:
                out.append({"title": title, "pubDate": pub_date, "link": link})
        return out
    except Exception:
        return []


def _call_gemini_json(
    client: genai.Client,
    prompt: str,
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
    parse_json_response: Any,
) -> dict[str, Any]:
    """使用備援模型鏈呼叫 Gemini 並回傳解析後的 JSON 物件。"""
    last_err: Exception | None = None
    for model_name in fallback_models:
        if model_name in exhausted_models:
            continue
        try:
            res = generate_with_retry(
                client,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            return parse_json_response(res.text or "")
        except Exception as exc:
            last_err = exc
            err_msg = str(exc)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                exhausted_models.add(model_name)
            print(f"[警告] Gemini JSON 呼叫 ({model_name}) 失敗: {exc}")
    raise RuntimeError(f"所有模型呼叫失敗: {last_err}")


def _call_gemini_text(
    client: genai.Client,
    prompt: str,
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
) -> str:
    """使用備援模型鏈呼叫 Gemini 並回傳純文字。"""
    for model_name in fallback_models:
        if model_name in exhausted_models:
            continue
        try:
            res = generate_with_retry(
                client,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (res.text or "").strip()
            if text:
                return text
        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                exhausted_models.add(model_name)
            print(f"[警告] Gemini 文字呼叫 ({model_name}) 失敗: {exc}")
    return ""


# =====================================================================
# 模組一：盤後收盤分析 (After-Hours Closing Analysis)
# =====================================================================


def _classify_candle(o: float, h: float, l: float, c: float, prev_c: float) -> str:
    """判讀當日日 K 棒型態。"""
    total_range = max(h - l, 1e-6)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    chg_pct = ((c - prev_c) / prev_c * 100) if prev_c > 0 else 0.0

    if body / total_range < 0.12:
        if lower > upper * 2:
            return "長下影十字線 (低檔承接)"
        if upper > lower * 2:
            return "長上影十字線 (高檔遇壓)"
        return "十字變盤線"
    if c >= o:
        if chg_pct >= 3.0 and body / total_range >= 0.6:
            return "強勢長紅K"
        if upper / total_range >= 0.45:
            return "留長上影紅K (沖高遇壓)"
        if lower / total_range >= 0.45:
            return "留長下影紅K (開低走高)"
        return "實體紅K"
    else:
        if chg_pct <= -3.0 and body / total_range >= 0.6:
            return "弱勢長黑K"
        if lower / total_range >= 0.45:
            return "留長下影黑K (下檔有撐)"
        if upper / total_range >= 0.45:
            return "留長上影黑K (反彈遇壓)"
        return "實體黑K"


def _calc_rsi14(closes: list[float]) -> float | None:
    if len(closes) <= 14:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, 15):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / 14.0
    avg_loss = losses / 14.0
    for i in range(15, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * 13 + (diff if diff > 0 else 0.0)) / 14.0
        avg_loss = (avg_loss * 13 + (-diff if diff < 0 else 0.0)) / 14.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def get_closing_technical_snapshot(yf_symbol: str) -> dict[str, Any]:
    """抓取收盤完整日 K 資訊、K 棒型態、均線、RSI、量比與支撐壓力。"""
    try:
        df = yf.Ticker(yf_symbol).history(period="4mo")
        if len(df) < 2:
            return {"summary": "無法取得收盤數據", "candle_pattern": "無資料"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        o = float(latest["Open"])
        h = float(latest["High"])
        l = float(latest["Low"])
        c = float(latest["Close"])
        prev_c = float(prev["Close"])
        vol = int(latest["Volume"])
        chg_pct = ((c - prev_c) / prev_c) * 100 if prev_c > 0 else 0.0
        amp_pct = ((h - l) / prev_c) * 100 if prev_c > 0 else 0.0

        closes = [float(x) for x in df["Close"].tolist()]
        ma5 = sum(closes[-5:]) / min(len(closes), 5)
        ma20 = sum(closes[-20:]) / min(len(closes), 20)
        ma60 = sum(closes[-60:]) / min(len(closes), 60)
        rsi14 = _calc_rsi14(closes)

        prev5_vols = [float(x) for x in df["Volume"].iloc[-6:-1].tolist() if x > 0]
        avg5_vol = sum(prev5_vols) / len(prev5_vols) if prev5_vols else 0.0
        vol_ratio = (vol / avg5_vol) if avg5_vol > 0 else 1.0

        high20 = float(df["High"].tail(20).max())
        low20 = float(df["Low"].tail(20).min())
        candle = _classify_candle(o, h, l, c, prev_c)
        ma20_status = "站上月線" if c >= ma20 else "跌破月線"

        vol_desc = f"量: {vol:,} (5日量比 {vol_ratio:.2f}x)"
        rsi_str = f"{rsi14:.1f}" if rsi14 is not None else "N/A"
        summary = (
            f"開:{o:.2f} 高:{h:.2f} 低:{l:.2f} 收:{c:.2f} ({chg_pct:+.2f}%, 振幅{amp_pct:.2f}%) | "
            f"型態:{candle} | {vol_desc} | "
            f"MA5:{ma5:.2f} / MA20:{ma20:.2f}({ma20_status}) / MA60:{ma60:.2f} | "
            f"RSI14:{rsi_str} | 20日區間:{low20:.2f}~{high20:.2f}"
        )
        return {
            "close_price": round(c, 2),
            "change_pct": round(chg_pct, 2),
            "candle_pattern": candle,
            "volume_info": vol_desc,
            "support_default": f"{min(ma20, low20):.2f} ~ {max(ma20, low20) if c > ma20 else low20:.2f}",
            "resistance_default": f"{high20:.2f}",
            "summary": summary,
        }
    except Exception as exc:
        return {
            "close_price": None,
            "change_pct": None,
            "candle_pattern": "抓取異常",
            "volume_info": "無量能資料",
            "support_default": "--",
            "resistance_default": "--",
            "summary": f"收盤資料抓取異常: {exc}",
        }


def get_tw_afterhours_chips(ticker: str) -> str:
    """抓取台股收盤後三大法人買賣超與融資融券變化。"""
    url = "https://api.finmindtrade.com/api/v4/data"
    start_date = (datetime.now(TW_TZ) - timedelta(days=7)).strftime("%Y-%m-%d")
    parts: list[str] = []

    try:
        res = requests.get(
            url,
            params={
                "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
                "data_id": ticker,
                "start_date": start_date,
            },
            timeout=10,
        ).json()
        if res.get("data"):
            latest_date = res["data"][-1]["date"]
            day_recs = [r for r in res["data"] if r["date"] == latest_date]
            foreign = sum((r["buy"] - r["sell"]) for r in day_recs if r["name"] in ("Foreign_Investor", "Foreign_Dealer_Self")) // 1000
            trust = sum((r["buy"] - r["sell"]) for r in day_recs if r["name"] == "Investment_Trust") // 1000
            dealer = sum((r["buy"] - r["sell"]) for r in day_recs if r["name"] in ("Dealer_self", "Dealer_Hedging")) // 1000
            total = foreign + trust + dealer
            parts.append(
                f"[{latest_date} 三大法人] 外資:{foreign:+d}張, 投信:{trust:+d}張, 自營:{dealer:+d}張, 合計:{total:+d}張"
            )
    except Exception:
        pass

    try:
        res = requests.get(
            url,
            params={
                "dataset": "TaiwanStockMarginPurchaseShortSale",
                "data_id": ticker,
                "start_date": start_date,
            },
            timeout=10,
        ).json()
        if res.get("data"):
            m = res["data"][-1]
            m_bal = int(m.get("MarginPurchaseTodayBalance", 0))
            m_diff = m_bal - int(m.get("MarginPurchaseYesterdayBalance", 0))
            s_bal = int(m.get("ShortSaleTodayBalance", 0))
            s_diff = s_bal - int(m.get("ShortSaleYesterdayBalance", 0))
            ratio = (s_bal / m_bal * 100) if m_bal > 0 else 0.0
            parts.append(
                f"[{m['date']} 資券] 融資:{m_bal:,}({m_diff:+d}), 融券:{s_bal:,}({s_diff:+d}), 券資比:{ratio:.2f}%"
            )
    except Exception:
        pass

    return " | ".join(parts) if parts else "無最新法人與融資融券資料"


def get_closing_market_summary(
    client: genai.Client,
    market: str,
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
) -> str:
    """抓取收盤大盤指數並生成盤後大盤總結。"""
    if market == "TW":
        indices = {
            "^TWII": "台灣加權指數",
            "^GSPC": "標普500 (前夜)",
            "^SOX": "費城半導體 (前夜)",
            "TSM": "台積電ADR (前夜)",
        }
    else:
        indices = {
            "^GSPC": "標普500",
            "^IXIC": "那斯達克",
            "^SOX": "費城半導體",
            "TSM": "台積電ADR",
            "^VIX": "VIX恐慌指數",
            "^TNX": "美國10年期公債殖利率",
        }

    lines: list[str] = []
    for sym, name in indices.items():
        try:
            df = yf.Ticker(sym).history(period="5d")
            if len(df) >= 2:
                last = float(df.iloc[-1]["Close"])
                prev = float(df.iloc[-2]["Close"])
                chg = ((last - prev) / prev) * 100
                unit = "%" if sym == "^TNX" else ""
                lines.append(f"- {name} ({sym}): {last:.2f}{unit} ({chg:+.2f}%)")
        except Exception:
            continue

    raw_text = "\n".join(lines) if lines else "- 暫無最新收盤指數數據"
    prompt = f"""
    你是買方台美股盤後策略師。請根據以下【{market}】最新收盤指數表現，提供 3~4 句話的「當日盤後大盤總結與下個交易日風險提示」：
    {raw_text}

    要求：點出今日收盤強弱格局、資金流向（權值／半導體／科技股）、以及下個交易日應留意的關鍵風向。
    """
    ai_text = _call_gemini_text(client, prompt, fallback_models, exhausted_models, generate_with_retry)
    if not ai_text:
        ai_text = "盤後 AI 總結暫時無法生成，請參考上方收盤指數數據。"
    return f"{raw_text}\n\n**🌆 盤後大盤總結**：\n{ai_text}"


def run_afterhours_analysis(
    db: Any,
    client: genai.Client,
    market: str,
    portfolio: list[dict[str, Any]],
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
    parse_json_response: Any,
    get_us_finnhub_news: Any,
) -> str:
    """執行盤後分析並寫入 Firestore afterhours_reports/{YYYY-MM-DD}_{market}。"""
    print(f"正在生成 {market} 盤後大盤收盤總結...")
    market_summary = get_closing_market_summary(
        client, market, fallback_models, exhausted_models, generate_with_retry
    )

    today_str = datetime.now(TW_TZ).strftime("%Y-%m-%d")
    analyzed: list[dict[str, Any]] = []

    for idx, stock in enumerate(portfolio):
        print(f"正在進行盤後個股分析：{stock['name']} ({stock['ticker']})...")
        snap = get_closing_technical_snapshot(stock["yf_symbol"])
        chips_or_news = (
            get_tw_afterhours_chips(stock["ticker"])
            if market == "TW"
            else get_us_finnhub_news(stock["ticker"])
        )
        rss_query = (
            f"{stock['name']} {stock['ticker']} when:1d"
            if market == "TW"
            else f"{stock['ticker']} stock when:1d"
        )
        rss_items = _fetch_google_news_rss(rss_query, market, max_items=5)
        rss_text = "\n".join([f"- {it['title']}" for it in rss_items]) if rss_items else "當日無重大即時新聞"

        prompt = f"""
        今天是 {today_str}（收盤後）。你是買方證券分析師，請根據今日完整收盤價量、K 棒型態、均線/RSI、收盤後籌碼與新聞，為「{stock['name']} ({stock['ticker']})」撰寫盤後覆盤與明日操作建議：

        - 大盤盤後背景：{market_summary}
        - 標的：{stock['name']} ({stock['ticker']}) | 持有均價：{stock['avg_cost']}
        - 關注主軸：{stock.get('focus', '基本面與重大訂單動態')}
        - 當日收盤價量與技術結構：{snap['summary']}
        - 盤後籌碼／新聞：{chips_or_news}
        - 當日即時新聞標題：
        {rss_text}

        請嚴格輸出 JSON 格式（不要加 Markdown 標記）：
        {{
          "sentiment": "偏多 / 中性 / 偏空 三選一",
          "close_summary": "收盤價、損益狀態、K棒型態與量價位階一句話總結",
          "intraday_review": "今日盤中走勢強弱與法人／融資籌碼結構解讀（80字內）",
          "key_events": "今日盤中或盤後重要消息摘要（無則填「無重大新事件」）",
          "next_day_strategy": "結合持有均價、今日收盤結構與籌碼的下個交易日具體操作建議（100字內）",
          "support_level": "短線參考支撐價位（例：980 元或 20MA）",
          "resistance_level": "短線參考壓力價位（例：1050 元或前高）"
        }}
        """

        try:
            parsed = _call_gemini_json(
                client, prompt, fallback_models, exhausted_models, generate_with_retry, parse_json_response
            )
            analyzed.append(
                {
                    "ticker": stock["ticker"],
                    "name": stock["name"],
                    "avg_cost": stock["avg_cost"],
                    "close_price": snap.get("close_price"),
                    "change_pct": snap.get("change_pct"),
                    "candle_pattern": snap.get("candle_pattern", ""),
                    "volume_info": snap.get("volume_info", ""),
                    "sentiment": str(parsed.get("sentiment") or "中性").strip(),
                    "close_summary": str(parsed.get("close_summary") or snap["summary"]).strip(),
                    "intraday_review": str(parsed.get("intraday_review") or chips_or_news).strip(),
                    "key_events": str(parsed.get("key_events") or "無重大新事件").strip(),
                    "next_day_strategy": str(
                        parsed.get("next_day_strategy") or "暫無具體策略，留意開盤量價變化。"
                    ).strip(),
                    "support_level": str(parsed.get("support_level") or snap.get("support_default", "--")).strip(),
                    "resistance_level": str(parsed.get("resistance_level") or snap.get("resistance_default", "--")).strip(),
                }
            )
            print(f"[成功] {stock['name']} ({stock['ticker']}) 盤後分析完成")
        except Exception as exc:
            print(f"[錯誤] {stock['name']} ({stock['ticker']}) 盤後分析失敗: {exc}")
            analyzed.append(
                {
                    "ticker": stock["ticker"],
                    "name": stock["name"],
                    "avg_cost": stock["avg_cost"],
                    "close_price": snap.get("close_price"),
                    "change_pct": snap.get("change_pct"),
                    "candle_pattern": snap.get("candle_pattern", ""),
                    "volume_info": snap.get("volume_info", ""),
                    "sentiment": "中性",
                    "close_summary": snap["summary"],
                    "intraday_review": chips_or_news,
                    "key_events": "AI 盤後分析暫時異常，請參考上方收盤價量與籌碼數據。",
                    "next_day_strategy": "本次分析失敗，下個交易日建議先觀察月線與近期高低點支撐壓力。",
                    "support_level": snap.get("support_default", "--"),
                    "resistance_level": snap.get("resistance_default", "--"),
                }
            )

        if idx < len(portfolio) - 1:
            time.sleep(2)

    report_id = f"{today_str}_{market}"
    db.collection("afterhours_reports").document(report_id).set(
        {
            "date": today_str,
            "market": market,
            "market_summary": market_summary,
            "stocks": analyzed,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
    )
    print(f"盤後報告已成功寫入 Firestore (afterhours_reports/{report_id})！")
    return report_id


# =====================================================================
# 模組二：重大事件行事曆 + 財報／法說會深度分析 (Events Calendar)
# =====================================================================


def _fetch_twse_upcoming_exdiv() -> dict[str, list[dict[str, Any]]]:
    """抓取證交所預告除權除息清單 (TWT48U_ALL)。"""
    by_code: dict[str, list[dict[str, Any]]] = {}
    try:
        res = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL", timeout=12).json()
        if isinstance(res, list):
            for row in res:
                code = str(row.get("Code", "")).strip().upper()
                iso_date = _roc_to_iso(row.get("Date", ""))
                if code and iso_date:
                    by_code.setdefault(code, []).append(
                        {
                            "ex_date": iso_date,
                            "type_label": str(row.get("Exdividend", "息")).strip(),
                            "cash_dividend": _safe_float(row.get("CashDividend")),
                            "stock_dividend": _safe_float(row.get("StockDividendRatio")),
                        }
                    )
    except Exception as exc:
        print(f"[警告] 無法抓取 TWSE 預告除權息表: {exc}")
    return by_code


def _fetch_tw_quarterly_financials(ticker: str) -> list[dict[str, Any]]:
    """透過 FinMind 抓取台股近期最多 4 季完整損益表摘要（供財報事件瀏覽與 AI 分析）。"""
    start_date = (datetime.now(TW_TZ) - timedelta(days=560)).strftime("%Y-%m-%d")
    try:
        res = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={
                "dataset": "TaiwanStockFinancialStatements",
                "data_id": ticker,
                "start_date": start_date,
            },
            timeout=12,
        ).json()
        rows = res.get("data") or []
        if not rows:
            return []

        by_date: dict[str, dict[str, float]] = {}
        for r in rows:
            d = str(r.get("date", ""))
            t = str(r.get("type", ""))
            v = _safe_float(r.get("value"))
            if d and t and v is not None:
                by_date.setdefault(d, {})[t] = v

        sorted_dates = sorted(by_date.keys())
        quarters: list[dict[str, Any]] = []
        for idx, d in enumerate(sorted_dates):
            m = by_date[d]
            rev = m.get("Revenue")
            gp = m.get("GrossProfit")
            op = m.get("OperatingIncome")
            ni = m.get("IncomeAfterTaxes") or m.get("IncomeFromContinuingOperations")
            eps = m.get("EPS")
            if rev is None and eps is None:
                continue

            gm = (gp / rev * 100) if (gp is not None and rev and rev > 0) else None
            om = (op / rev * 100) if (op is not None and rev and rev > 0) else None
            nm = (ni / rev * 100) if (ni is not None and rev and rev > 0) else None

            # 計算季增 QoQ 與年增 YoY
            prev_q_rev = by_date[sorted_dates[idx - 1]].get("Revenue") if idx >= 1 else None
            prev_y_date = f"{int(d[:4]) - 1}{d[4:]}"
            prev_y_rev = by_date.get(prev_y_date, {}).get("Revenue")
            rev_qoq = ((rev - prev_q_rev) / prev_q_rev * 100) if (rev and prev_q_rev and prev_q_rev > 0) else None
            rev_yoy = ((rev - prev_y_rev) / prev_y_rev * 100) if (rev and prev_y_rev and prev_y_rev > 0) else None

            # 轉為季別標籤 (如 2026 Q3)
            month_num = int(d[5:7])
            q_num = (month_num - 1) // 3 + 1
            quarters.append(
                {
                    "period_end": d,
                    "quarter": f"{d[:4]} Q{q_num}",
                    "currency": "TWD",
                    "revenue": round(rev / 1e8, 2) if rev is not None else None,  # 億元
                    "gross_profit": round(gp / 1e8, 2) if gp is not None else None,
                    "operating_income": round(op / 1e8, 2) if op is not None else None,
                    "net_income": round(ni / 1e8, 2) if ni is not None else None,
                    "eps": round(eps, 2) if eps is not None else None,
                    "gross_margin": round(gm, 2) if gm is not None else None,
                    "operating_margin": round(om, 2) if om is not None else None,
                    "net_margin": round(nm, 2) if nm is not None else None,
                    "rev_qoq": round(rev_qoq, 2) if rev_qoq is not None else None,
                    "rev_yoy": round(rev_yoy, 2) if rev_yoy is not None else None,
                }
            )
        return quarters[-4:]
    except Exception as exc:
        print(f"[警告] 抓取 {ticker} FinMind 財報失敗: {exc}")
        return []


def _fetch_us_quarterly_financials(yf_symbol: str) -> list[dict[str, Any]]:
    """透過 yfinance 抓取美股近期最多 4 季損益表摘要（單位：百萬美元）。"""
    try:
        df = yf.Ticker(yf_symbol).quarterly_income_stmt
        if df is None or df.empty:
            return []

        cols = sorted(list(df.columns))
        raw_by_col: list[tuple[Any, dict[str, float | None]]] = []
        for col in cols:
            def _pick(keys: list[str]) -> float | None:
                for k in keys:
                    if k in df.index:
                        val = _safe_float(df.loc[k, col])
                        if val is not None:
                            return val
                return None

            raw_by_col.append(
                (
                    col,
                    {
                        "rev": _pick(["Total Revenue", "Operating Revenue"]),
                        "gp": _pick(["Gross Profit"]),
                        "op": _pick(["Operating Income", "EBIT"]),
                        "ni": _pick(["Net Income", "Net Income Common Stockholders"]),
                        "eps": _pick(["Diluted EPS", "Basic EPS"]),
                    },
                )
            )

        quarters: list[dict[str, Any]] = []
        for idx, (col, m) in enumerate(raw_by_col):
            d_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
            rev, gp, op, ni, eps = m["rev"], m["gp"], m["op"], m["ni"], m["eps"]
            if rev is None and eps is None:
                continue
            gm = (gp / rev * 100) if (gp is not None and rev and rev > 0) else None
            om = (op / rev * 100) if (op is not None and rev and rev > 0) else None
            nm = (ni / rev * 100) if (ni is not None and rev and rev > 0) else None

            prev_q_rev = raw_by_col[idx - 1][1]["rev"] if idx >= 1 else None
            rev_qoq = ((rev - prev_q_rev) / prev_q_rev * 100) if (rev and prev_q_rev and prev_q_rev > 0) else None

            month_num = int(d_str[5:7])
            q_num = (month_num - 1) // 3 + 1
            quarters.append(
                {
                    "period_end": d_str,
                    "quarter": f"{d_str[:4]} Q{q_num}",
                    "currency": "USD",
                    "revenue": round(rev / 1e6, 2) if rev is not None else None,  # 百萬美元
                    "gross_profit": round(gp / 1e6, 2) if gp is not None else None,
                    "operating_income": round(op / 1e6, 2) if op is not None else None,
                    "net_income": round(ni / 1e6, 2) if ni is not None else None,
                    "eps": round(eps, 2) if eps is not None else None,
                    "gross_margin": round(gm, 2) if gm is not None else None,
                    "operating_margin": round(om, 2) if om is not None else None,
                    "net_margin": round(nm, 2) if nm is not None else None,
                    "rev_qoq": round(rev_qoq, 2) if rev_qoq is not None else None,
                    "rev_yoy": None,
                }
            )
        return quarters[-4:]
    except Exception as exc:
        print(f"[警告] 抓取 {yf_symbol} 美股財報失敗: {exc}")
        return []


def _collect_structured_events_for_stock(
    stock: dict[str, Any],
    start_date: date,
    end_date: date,
    twse_exdiv_map: dict[str, list[dict[str, Any]]],
    financials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """從 FinMind、TWSE OpenAPI 與 yfinance 蒐集過去 90 天與未來 90 天的財報與股息股利事件。"""
    market = stock["market"]
    ticker = stock["ticker"]
    name = stock["name"]
    yf_symbol = stock["yf_symbol"]
    events: list[dict[str, Any]] = []

    def _in_window(d_str: str) -> bool:
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
            return start_date <= d <= end_date
        except Exception:
            return False

    # 1. 台股除權息 (FinMind + TWSE 預告表)
    if market == "TW":
        try:
            div_res = requests.get(
                "https://api.finmindtrade.com/api/v4/data",
                params={
                    "dataset": "TaiwanStockDividend",
                    "data_id": ticker,
                    "start_date": (start_date - timedelta(days=90)).strftime("%Y-%m-%d"),
                },
                timeout=10,
            ).json()
            for r in div_res.get("data") or []:
                ex_date = (r.get("CashExDividendTradingDate") or r.get("StockExDividendTradingDate") or "").strip()
                pay_date = (r.get("CashDividendPaymentDate") or "").strip()
                cash_div = (_safe_float(r.get("CashEarningsDistribution")) or 0.0) + (
                    _safe_float(r.get("CashStatutorySurplus")) or 0.0
                )
                stock_div = (_safe_float(r.get("StockEarningsDistribution")) or 0.0) + (
                    _safe_float(r.get("StockStatutorySurplus")) or 0.0
                )
                year_label = str(r.get("year") or "").strip()
                if ex_date and _in_window(ex_date):
                    parts = []
                    if cash_div > 0:
                        parts.append(f"現金股利 {cash_div:.2f} 元")
                    if stock_div > 0:
                        parts.append(f"股票股利 {stock_div:.2f} 元")
                    div_desc = "、".join(parts) if parts else "除權息"
                    events.append(
                        {
                            "market": market,
                            "ticker": ticker,
                            "name": name,
                            "event_type": "dividend",
                            "event_date": ex_date,
                            "title": f"{year_label} 除權息 ({div_desc})".strip(),
                            "summary": f"除權息交易日：{ex_date}" + (f"｜發放日：{pay_date}" if pay_date else ""),
                            "details": {
                                "year": year_label,
                                "cash_dividend": round(cash_div, 4),
                                "stock_dividend": round(stock_div, 4),
                                "ex_date": ex_date,
                                "payment_date": pay_date,
                            },
                        }
                    )
        except Exception as exc:
            print(f"[警告] FinMind 股利查詢 {ticker} 失敗: {exc}")

        for item in twse_exdiv_map.get(ticker, []):
            ex_date = item["ex_date"]
            if _in_window(ex_date) and not any(
                e["event_type"] == "dividend" and e["event_date"] == ex_date for e in events
            ):
                cash_d = item.get("cash_dividend") or 0.0
                stock_d = item.get("stock_dividend") or 0.0
                events.append(
                    {
                        "market": market,
                        "ticker": ticker,
                        "name": name,
                        "event_type": "dividend",
                        "event_date": ex_date,
                        "title": f"證交所預告除{item.get('type_label', '息')}" + (f" ({cash_d:.2f} 元)" if cash_d > 0 else ""),
                        "summary": f"預告除權息日期：{ex_date}",
                        "details": {
                            "cash_dividend": cash_d,
                            "stock_dividend": stock_d,
                            "ex_date": ex_date,
                            "payment_date": "",
                        },
                    }
                )

    # 2. yfinance 財報日 (Earnings Dates / Calendar) 與美股股息
    try:
        tk = yf.Ticker(yf_symbol)
        # 2a. earnings_dates (涵蓋過去與未來財報發表日與 EPS 預期/實際)
        try:
            ed_df = tk.earnings_dates
            if ed_df is not None and not ed_df.empty:
                for ts, row in ed_df.iterrows():
                    d_str = ts.strftime("%Y-%m-%d")
                    if not _in_window(d_str):
                        continue
                    eps_est = _safe_float(row.get("EPS Estimate"))
                    eps_act = _safe_float(row.get("Reported EPS"))
                    surp = _safe_float(row.get("Surprise(%)"))
                    summary_parts = []
                    if eps_act is not None:
                        summary_parts.append(f"實際 EPS: {eps_act:.2f}")
                    if eps_est is not None:
                        summary_parts.append(f"預估 EPS: {eps_est:.2f}")
                    if surp is not None:
                        summary_parts.append(f"驚喜度: {surp:+.2f}%")
                    events.append(
                        {
                            "market": market,
                            "ticker": ticker,
                            "name": name,
                            "event_type": "earnings",
                            "event_date": d_str,
                            "title": f"{name} 季度財報發表",
                            "summary": "｜".join(summary_parts) if summary_parts else "季度財務報告公告日",
                            "details": {
                                "eps_estimate": eps_est,
                                "eps_actual": eps_act,
                                "surprise_pct": surp,
                                "financials": financials,
                            },
                        }
                    )
        except Exception:
            pass

        # 2b. calendar (未來財報日與除息日)
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed_list = cal.get("Earnings Date") or []
                if not isinstance(ed_list, list):
                    ed_list = [ed_list]
                for ed in ed_list:
                    d_str = ed.strftime("%Y-%m-%d") if hasattr(ed, "strftime") else str(ed)[:10]
                    if _in_window(d_str) and not any(
                        e["event_type"] == "earnings" and abs((datetime.strptime(e["event_date"], "%Y-%m-%d").date() - datetime.strptime(d_str, "%Y-%m-%d").date()).days) <= 7
                        for e in events
                    ):
                        events.append(
                            {
                                "market": market,
                                "ticker": ticker,
                                "name": name,
                                "event_type": "earnings",
                                "event_date": d_str,
                                "title": f"{name} 預計發表季度財報",
                                "summary": "預估財報公告日程（Yahoo Finance）",
                                "details": {"financials": financials},
                            }
                        )

                ex_div = cal.get("Ex-Dividend Date")
                if ex_div:
                    d_str = ex_div.strftime("%Y-%m-%d") if hasattr(ex_div, "strftime") else str(ex_div)[:10]
                    if _in_window(d_str) and not any(
                        e["event_type"] == "dividend" and e["event_date"] == d_str for e in events
                    ):
                        pay_d = cal.get("Dividend Date")
                        pay_str = pay_d.strftime("%Y-%m-%d") if hasattr(pay_d, "strftime") else ""
                        events.append(
                            {
                                "market": market,
                                "ticker": ticker,
                                "name": name,
                                "event_type": "dividend",
                                "event_date": d_str,
                                "title": f"{name} 除息交易日",
                                "summary": f"除息日：{d_str}" + (f"｜發放日：{pay_str}" if pay_str else ""),
                                "details": {"ex_date": d_str, "payment_date": pay_str},
                            }
                        )
        except Exception:
            pass

        # 2c. 美股歷史配息
        if market == "US":
            try:
                divs = tk.dividends
                if divs is not None and len(divs) > 0:
                    for ts, amt in divs.items():
                        d_str = ts.strftime("%Y-%m-%d")
                        if _in_window(d_str) and not any(
                            e["event_type"] == "dividend" and e["event_date"] == d_str for e in events
                        ):
                            val = _safe_float(amt) or 0.0
                            events.append(
                                {
                                    "market": market,
                                    "ticker": ticker,
                                    "name": name,
                                    "event_type": "dividend",
                                    "event_date": d_str,
                                    "title": f"{name} 除息 USD {val:.3f}",
                                    "summary": f"美股除息交易日：{d_str}｜配息 USD {val:.3f}",
                                    "details": {"cash_dividend": round(val, 4), "ex_date": d_str},
                                }
                            )
            except Exception:
                pass
    except Exception as exc:
        print(f"[警告] yfinance 事件查詢 {yf_symbol} 失敗: {exc}")

    # 3. 若台股近三個月有公告最新季報（例如 Q3 結束於 09-30，通常 10~11 月公告），但 yfinance 無紀錄，自動補入最新財報事件
    if market == "TW" and financials and not any(e["event_type"] == "earnings" for e in events):
        latest_q = financials[-1]
        p_end = datetime.strptime(latest_q["period_end"], "%Y-%m-%d").date()
        est_pub = min(datetime.now(TW_TZ).date(), p_end + timedelta(days=40))
        if start_date <= est_pub <= end_date:
            events.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "name": name,
                    "event_type": "earnings",
                    "event_date": est_pub.strftime("%Y-%m-%d"),
                    "title": f"{name} {latest_q['quarter']} 財報公告",
                    "summary": f"營收 {latest_q.get('revenue', '--')} 億｜毛利率 {latest_q.get('gross_margin', '--')}%｜EPS {latest_q.get('eps', '--')} 元",
                    "details": {
                        "period": latest_q["quarter"],
                        "eps_actual": latest_q.get("eps"),
                        "financials": financials,
                    },
                }
            )

    return events


def _discover_conference_and_news_events(
    client: genai.Client,
    stock: dict[str, Any],
    start_date: date,
    end_date: date,
    existing_events: list[dict[str, Any]],
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
    parse_json_response: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """從 Google News RSS 萃取法說會及公告中的財報／股息重大日程（嚴格限制於新聞證據，杜絕臆測）。"""
    market = stock["market"]
    ticker = stock["ticker"]
    name = stock["name"]

    if market == "TW":
        conf_rss = _fetch_google_news_rss(f"{name} {ticker} 法說會 when:90d", "TW", max_items=10)
        extra_rss = _fetch_google_news_rss(f"{name} {ticker} (財報 OR 除息 OR 股利) when:60d", "TW", max_items=6)
    else:
        conf_rss = _fetch_google_news_rss(f"{ticker} (earnings call OR investor day OR conference) when:90d", "US", max_items=10)
        extra_rss = _fetch_google_news_rss(f"{ticker} (earnings OR dividend) when:60d", "US", max_items=6)

    all_news = conf_rss + [n for n in extra_rss if n["title"] not in {c["title"] for c in conf_rss}]
    if not all_news:
        return [], []

    news_text = "\n".join([f"- {n['title']} [{n['pubDate']}]" for n in all_news[:14]])
    existing_brief = "\n".join(
        [f"- {e['event_date']} {e['event_type']}: {e['title']}" for e in existing_events]
    ) or "無"

    today_str = datetime.now(TW_TZ).strftime("%Y-%m-%d")
    prompt = f"""
    今天是 {today_str}。請從以下關於「{name} ({ticker})」的新聞標題與發布日期，萃取在 {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} 區間內，明確提到的【法說會 (conference)】、以及既有清單尚未收錄的【財報發表 (earnings)】或【配息除權息 (dividend)】事件。

    既有已收錄事件（請勿重複輸出同類型且日期相差 5 天內的事件）：
    {existing_brief}

    新聞標題與發布日：
    {news_text}

    嚴格規範：
    1. 僅輸出新聞中有明確證據的事件；若新聞僅提到法說會已舉行但無精確日期，可使用該法說會新聞發布日作為 event_date。
    2. 絕對不可編造新聞沒提到的未來日期；若無新增事件請回傳空陣列。
    3. 最多回傳 4 筆最重要事件。

    嚴格輸出 JSON 格式：
    {{
      "events": [
        {{
          "event_type": "conference | earnings | dividend",
          "event_date": "YYYY-MM-DD",
          "title": "事件名稱（25字內，例如：2026 Q3 法人說明會）",
          "summary": "一句話重點摘要"
        }}
      ]
    }}
    """
    try:
        parsed = _call_gemini_json(
            client, prompt, fallback_models, exhausted_models, generate_with_retry, parse_json_response
        )
        extracted: list[dict[str, Any]] = []
        for item in parsed.get("events") or []:
            etype = str(item.get("event_type", "")).strip().lower()
            edate = str(item.get("event_date", "")).strip()
            title = str(item.get("title", "")).strip()
            if etype not in ("conference", "earnings", "dividend") or not re.match(r"^\d{4}-\d{2}-\d{2}$", edate) or not title:
                continue
            d_obj = datetime.strptime(edate, "%Y-%m-%d").date()
            if not (start_date <= d_obj <= end_date):
                continue
            # 去重：同類型且日期在 7 天內視為同一事件
            if any(
                e["event_type"] == etype
                and abs((datetime.strptime(e["event_date"], "%Y-%m-%d").date() - d_obj).days) <= 7
                for e in existing_events + extracted
            ):
                continue
            extracted.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "name": name,
                    "event_type": etype,
                    "event_date": edate,
                    "title": title,
                    "summary": str(item.get("summary") or "").strip(),
                    "details": {},
                }
            )
        return extracted, all_news
    except Exception as exc:
        print(f"[警告] {ticker} 新聞事件萃取失敗: {exc}")
        return [], all_news


def _generate_event_ai_analysis(
    client: genai.Client,
    stock: dict[str, Any],
    event: dict[str, Any],
    financials: list[dict[str, Any]],
    related_news: list[dict[str, str]],
    today_date: date,
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
    parse_json_response: Any,
) -> dict[str, Any]:
    """針對已發生或即將發生的重大事件生成結構化 AI 深度分析（財報／法說會／股息）。"""
    ev_date = datetime.strptime(event["event_date"], "%Y-%m-%d").date()
    is_past = ev_date <= today_date
    etype = event["event_type"]
    name = stock["name"]
    ticker = stock["ticker"]
    focus = stock.get("focus", "基本面與成長動能")

    fin_text = json.dumps(financials[-4:], ensure_ascii=False) if financials else "無財報數字"
    news_text = "\n".join([f"- {n['title']} [{n['pubDate']}]" for n in related_news[:10]]) if related_news else "無額外新聞"

    if not is_past:
        days_left = (ev_date - today_date).days
        prompt = f"""
        你是買方研究員。標的「{name} ({ticker})」（持有均價 {stock.get('avg_cost')}、關注主軸：{focus}）將於 {event['event_date']}（還有 {days_left} 天）迎來【{event['title']}】（類型：{etype}）。
        近期財報數據：{fin_text}
        近期新聞：{news_text}

        請輸出 JSON（不加 Markdown）：
        {{
          "verdict": "事前觀察",
          "preview_focus": "事件前最關鍵的 3 個觀察指標與市場預期（120字內）",
          "extra_insights": "事件前後的持股風險控管或布局提醒（80字內）"
        }}
        """
    elif etype == "earnings":
        prompt = f"""
        你是買方資深財報分析師。請針對「{name} ({ticker})」於 {event['event_date']} 發布的財報【{event['title']}】進行深度財報分析：
        - 標的關注主軸：{focus} | 持有均價：{stock.get('avg_cost')}
        - 事件摘要：{event.get('summary', '')}
        - 近期各季損益表（台股單位億元、美股單位百萬美元）：{fin_text}
        - 相關新聞：{news_text}

        請嚴格輸出 JSON（不加 Markdown）：
        {{
          "verdict": "優於預期 / 符合預期 / 低於預期 / 中性觀察 四選一",
          "highlights": "財報核心亮點：點出最新季營收、毛利率/營益率/淨利率(三率)與 EPS 的關鍵變化（140字內）",
          "outlook": "獲利趨勢與基本面強弱評析：結合季增/年增動能評估公司成長性（120字內）",
          "extra_insights": "其他重要補充與投資建議：隱憂、估值對照與持股操作建議（120字內）"
        }}
        """
    elif etype == "conference":
        prompt = f"""
        你是買方產業研究員。請針對「{name} ({ticker})」於 {event['event_date']} 舉辦的法說會【{event['title']}】，整理深度法說會重點分析：
        - 標的關注主軸：{focus} | 持有均價：{stock.get('avg_cost')}
        - 最新季財報背景：{fin_text}
        - 法說會相關新聞報導：
        {news_text}

        請嚴格輸出 JSON（不加 Markdown）：
        {{
          "verdict": "展望偏多 / 中性穩健 / 展望保守 三選一",
          "highlights": "法說會重點事項：管理層對上季表現、下季財測指引、毛利率與訂單能見度的關鍵重點（160字內）",
          "outlook": "公司未來營運重點：中長期成長動能、新產品/技術進度、產能擴充或資本支出規劃（160字內）",
          "extra_insights": "其他重要補充事項：法人提問焦點、匯率/地緣/競爭風險、以及給持股者的具體操作啟示（150字內）"
        }}
        """
    else:
        prompt = f"""
        你是買方存股與事件分析師。請針對「{name} ({ticker})」於 {event['event_date']} 的股息股利事件【{event['title']}】（{event.get('summary', '')}，持有均價 {stock.get('avg_cost')}）提供精簡評估。
        近期財報背景：{fin_text}

        請嚴格輸出 JSON（不加 Markdown）：
        {{
          "verdict": "殖利率具吸引力 / 穩健配息 / 偏成長少配息 三選一",
          "highlights": "本次配息水準與現金/股票股利結構重點（100字內）",
          "outlook": "填息動能與基本面獲利支撐度評估（100字內）",
          "extra_insights": "參與除權息與持有成本攤提的實用提醒（90字內）"
        }}
        """

    try:
        return _call_gemini_json(
            client, prompt, fallback_models, exhausted_models, generate_with_retry, parse_json_response
        )
    except Exception as exc:
        print(f"[警告] 事件 AI 分析 ({ticker} {etype} {event['event_date']}) 失敗: {exc}")
        return {
            "verdict": "待補充",
            "highlights": event.get("summary") or "AI 分析暫時異常，請點擊下方財報數據或稍後重試。",
            "outlook": "",
            "extra_insights": "",
        }


def run_events_sync(
    db: Any,
    client: genai.Client,
    portfolio: list[dict[str, Any]],
    force_refresh: bool,
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
    parse_json_response: Any,
) -> int:
    """同步所有持股「過去 90 天 ～ 未來 90 天」重大事件，並對已發生財報／法說會產出深度分析。"""
    today_date = datetime.now(TW_TZ).date()
    start_date = today_date - timedelta(days=90)
    end_date = today_date + timedelta(days=90)

    has_tw = any(s.get("market") == "TW" for s in portfolio)
    twse_exdiv_map = _fetch_twse_upcoming_exdiv() if has_tw else {}

    # 讀取既有 stock_events 以避免重複呼叫 Gemini 分析歷史事件
    existing_docs: dict[str, dict[str, Any]] = {}
    for doc in db.collection("stock_events").stream():
        existing_docs[doc.id] = doc.to_dict()

    saved_count = 0

    # 先同步美國聯準會 FOMC 利率決策會議（過去 90 天 ～ 未來 90 天）
    saved_count += _sync_fomc_events(
        db=db,
        client=client,
        portfolio=portfolio,
        today_date=today_date,
        start_date=start_date,
        end_date=end_date,
        existing_docs=existing_docs,
        force_refresh=force_refresh,
        fallback_models=fallback_models,
        exhausted_models=exhausted_models,
        generate_with_retry=generate_with_retry,
        parse_json_response=parse_json_response,
    )

    for stock in portfolio:
        market = stock["market"]
        ticker = stock["ticker"]
        name = stock["name"]
        print(f"正在掃描重大事件行事曆：{name} ({market}:{ticker})...")

        financials = (
            _fetch_tw_quarterly_financials(ticker)
            if market == "TW"
            else _fetch_us_quarterly_financials(stock["yf_symbol"])
        )

        events = _collect_structured_events_for_stock(
            stock, start_date, end_date, twse_exdiv_map, financials
        )
        news_events, related_news = _discover_conference_and_news_events(
            client,
            stock,
            start_date,
            end_date,
            events,
            fallback_models,
            exhausted_models,
            generate_with_retry,
            parse_json_response,
        )
        events.extend(news_events)

        for ev in events:
            doc_id = f"{market}_{ticker}_{ev['event_type']}_{ev['event_date']}"
            ev_date = datetime.strptime(ev["event_date"], "%Y-%m-%d").date()
            status = "past" if ev_date <= today_date else "upcoming"

            if ev["event_type"] == "earnings" and financials:
                ev.setdefault("details", {})["financials"] = financials

            prev_doc = existing_docs.get(doc_id)
            prev_ai = (prev_doc or {}).get("ai_analysis") or {}
            has_valid_prev_ai = bool(
                prev_ai.get("highlights") or prev_ai.get("preview_focus")
            ) and prev_ai.get("verdict") != "待補充"

            # 若先前為 upcoming 但今天已變成 past，必須重新執行事件後深度分析！
            became_past = (prev_doc or {}).get("status") == "upcoming" and status == "past"

            if has_valid_prev_ai and not became_past and not force_refresh:
                ai_analysis = prev_ai
                print(f"  [快取] {doc_id} 沿用既有 AI 分析")
            else:
                print(f"  [分析] 生成 {doc_id} ({status}) AI 深度解析...")
                ai_analysis = _generate_event_ai_analysis(
                    client,
                    stock,
                    ev,
                    financials,
                    related_news,
                    today_date,
                    fallback_models,
                    exhausted_models,
                    generate_with_retry,
                    parse_json_response,
                )
                time.sleep(1.5)

            db.collection("stock_events").document(doc_id).set(
                {
                    "market": market,
                    "ticker": ticker,
                    "name": name,
                    "event_type": ev["event_type"],
                    "event_date": ev["event_date"],
                    "status": status,
                    "title": ev["title"],
                    "summary": ev.get("summary", ""),
                    "details": ev.get("details", {}),
                    "ai_analysis": ai_analysis,
                    "source": "auto",
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            saved_count += 1

    print(f"重大事件行事曆同步完成，共更新 {saved_count} 筆事件！")
    return saved_count


# =====================================================================
# 模組三：美國聯準會 FOMC 會議行事曆與總經／產業深度解析
# =====================================================================

# 官方 Federal Reserve 2026–2027 FOMC 會議排程
# 格式：(開始日, 決議公布日(美東), 是否含 SEP 利率點陣圖, 台灣公布時間說明)
FOMC_OFFICIAL_SCHEDULE: list[tuple[str, str, bool, str]] = [
    ("2026-01-27", "2026-01-28", False, "台灣時間 2026-01-29 03:00"),
    ("2026-03-17", "2026-03-18", True, "台灣時間 2026-03-19 02:00"),
    ("2026-04-28", "2026-04-29", False, "台灣時間 2026-04-30 02:00"),
    ("2026-06-16", "2026-06-17", True, "台灣時間 2026-06-18 02:00"),
    ("2026-07-28", "2026-07-29", False, "台灣時間 2026-07-30 02:00"),
    ("2026-09-15", "2026-09-16", True, "台灣時間 2026-09-17 02:00"),
    ("2026-10-27", "2026-10-28", False, "台灣時間 2026-10-29 02:00"),
    ("2026-12-08", "2026-12-09", True, "台灣時間 2026-12-10 03:00"),
    ("2027-01-26", "2027-01-27", False, "台灣時間 2027-01-28 03:00"),
    ("2027-03-16", "2027-03-17", True, "台灣時間 2027-03-18 02:00"),
    ("2027-04-27", "2027-04-28", False, "台灣時間 2027-04-29 02:00"),
    ("2027-06-08", "2027-06-09", True, "台灣時間 2027-06-10 02:00"),
    ("2027-07-27", "2027-07-28", False, "台灣時間 2027-07-29 02:00"),
    ("2027-09-14", "2027-09-15", True, "台灣時間 2027-09-16 02:00"),
    ("2027-11-16", "2027-11-17", False, "台灣時間 2027-11-18 03:00"),
]


def _get_macro_rates_snapshot() -> str:
    """抓取當前美債殖利率、美元指數、費半與標普做為 FOMC 分析背景數據。"""
    symbols = {
        "^TNX": "美國10年期公債殖利率",
        "^IRX": "美國13週國庫券利率",
        "DX-Y.NYB": "美元指數 DXY",
        "^GSPC": "標普500",
        "^SOX": "費城半導體",
        "^VIX": "VIX恐慌指數",
    }
    lines: list[str] = []
    for sym, label in symbols.items():
        try:
            df = yf.Ticker(sym).history(period="5d")
            if len(df) >= 2:
                last = float(df.iloc[-1]["Close"])
                prev = float(df.iloc[-2]["Close"])
                chg = ((last - prev) / prev) * 100
                unit = "%" if sym in ("^TNX", "^IRX") else ""
                lines.append(f"- {label}: {last:.2f}{unit} ({chg:+.2f}%)")
        except Exception:
            continue
    return "\n".join(lines) if lines else "暫無最新利率與總經指數快照"


def _call_fomc_gemini_json(
    client: genai.Client,
    prompt: str,
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
    parse_json_response: Any,
) -> dict[str, Any]:
    """優先嘗試開啟 Google Search Grounding 取得最新聯準會官方利率決議，失敗則自動降級純 RSS 模式。"""
    for use_search in (True, False):
        for model_name in fallback_models:
            if not use_search and model_name in exhausted_models:
                continue
            try:
                kwargs: dict[str, Any] = {
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                }
                if use_search:
                    kwargs["tools"] = [{"google_search": {}}]
                try:
                    res = generate_with_retry(
                        client,
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(**kwargs),
                    )
                except Exception as first_err:
                    err_msg = str(first_err)
                    if any(k in err_msg for k in ("404", "429", "503", "504", "NOT_FOUND", "RESOURCE_EXHAUSTED")):
                        raise
                    if use_search:
                        res = generate_with_retry(
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
                return parse_json_response(res.text or "")
            except Exception as exc:
                err_msg = str(exc)
                if not use_search and ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg):
                    exhausted_models.add(model_name)
                print(f"[警告] FOMC 分析 ({model_name}, grounding={use_search}) 失敗: {exc}")
    raise RuntimeError("FOMC 分析所有模型皆失敗")


def _sync_fomc_events(
    db: Any,
    client: genai.Client,
    portfolio: list[dict[str, Any]],
    today_date: date,
    start_date: date,
    end_date: date,
    existing_docs: dict[str, dict[str, Any]],
    force_refresh: bool,
    fallback_models: list[str],
    exhausted_models: set[str],
    generate_with_retry: Any,
    parse_json_response: Any,
) -> int:
    """同步過去 90 天 ～ 未來 90 天內的美國聯準會 FOMC 會議並產出總經與產業影響分析。"""
    macro_snapshot = _get_macro_rates_snapshot()
    holdings_brief = "、".join(
        [f"{s.get('name', '')}({s.get('ticker', '')})" for s in portfolio[:15]]
    ) or "台美科技與權值股"

    count = 0
    for m_start, m_decision, has_sep, tw_time in FOMC_OFFICIAL_SCHEDULE:
        d_obj = datetime.strptime(m_decision, "%Y-%m-%d").date()
        if not (start_date <= d_obj <= end_date):
            continue

        doc_id = f"MACRO_FOMC_fomc_{m_decision}"
        status = "past" if d_obj <= today_date else "upcoming"
        sep_tag = "（含 SEP 利率點陣圖）" if has_sep else ""
        month_label = f"{d_obj.year}年{d_obj.month}月"
        title = f"{month_label} 美國聯準會 FOMC 利率決策會議{sep_tag}"

        prev_doc = existing_docs.get(doc_id)
        prev_ai = (prev_doc or {}).get("ai_analysis") or {}
        prev_details = (prev_doc or {}).get("details") or {}
        has_valid_prev = bool(
            (prev_ai.get("highlights") or prev_ai.get("preview_focus"))
            and prev_ai.get("macro_impact")
            and prev_ai.get("sector_impact")
            and prev_ai.get("verdict") != "待補充"
        )
        became_past = (prev_doc or {}).get("status") == "upcoming" and status == "past"

        if has_valid_prev and not became_past and not force_refresh:
            details = prev_details
            ai_analysis = prev_ai
            summary = (prev_doc or {}).get("summary") or f"會議日期：{m_start} ~ {m_decision}｜{tw_time}"
            print(f"  [快取] {doc_id} 沿用既有 FOMC 分析")
        else:
            print(f"  [分析] 生成 FOMC 會議 {doc_id} ({status}) 總經與產業深度解析...")
            en_month = d_obj.strftime("%B %Y")
            rss_en = _fetch_google_news_rss(f"Federal Reserve FOMC interest rate decision {en_month}", "US", max_items=8)
            rss_zh = _fetch_google_news_rss(f"聯準會 FOMC 利率決議 {month_label}", "TW", max_items=6)
            rss_lines = "\n".join(
                [f"- {n['title']} [{n['pubDate']}]" for n in (rss_en + rss_zh)[:12]]
            ) or "無近期 RSS 新聞"

            if status == "past":
                prompt = f"""
                今天是 {today_date.strftime('%Y-%m-%d')}。你是資深總體經濟與全球跨產業策略師。
                請針對已結束的【{title}】（美東會議：{m_start} ~ {m_decision}，{tw_time}）檢索並彙整實際利率決議結果，深入分析其對「總體經濟」與「主要產業面」的總體影響：

                - 目前市場利率與指數背景：
                {macro_snapshot}
                - 使用者目前持股清單：{holdings_brief}
                - 相關新聞參考：
                {rss_lines}

                請嚴格輸出 JSON 格式（不要加 Markdown 標記）：
                {{
                  "rate_action": "實際利率調整動作（例如：降息 1 碼 (-25bps) / 維持利率不變 / 升息 1 碼）",
                  "target_rate_range": "決議後的聯邦基金目標利率區間（例如：3.75% - 4.00%）",
                  "policy_stance": "鴿派利多 / 中性偏鴿 / 中性穩健 / 中性偏鷹 / 鷹派承壓 五選一",
                  "summary": "一句話精簡總結本次會議結果（45字內）",
                  "highlights": "會議決議與利率調整重點：說明利率調整幅度、投票意向、會後聲明措辭變化、鮑爾/主席記者會基調，若含 SEP 則說明最新利率點陣圖中位數與通膨/GDP預測變化（180字內）",
                  "macro_impact": "對總體經濟之總體影響：解析對美國通膨、就業、經濟軟著陸機率、10年期美債殖利率、美元指數、以及全球與台美股資金流動性／本益比估值的影響（180字內）",
                  "sector_impact": "對主要產業面總體影響：分點說明(1)半導體與AI科技成長股、(2)金融與銀行業、(3)高股息/公用事業/債券防禦族群、(4)中小型成長股與航太國防等利率敏感產業的受惠或承壓方向（220字內）",
                  "extra_insights": "結合使用者目前持股（{holdings_brief}）的具體資產配置、倉位與風險操作啟示（140字內）"
                }}
                """
            else:
                days_left = (d_obj - today_date).days
                prompt = f"""
                今天是 {today_date.strftime('%Y-%m-%d')}。你是資深總體經濟與全球跨產業策略師。
                美國聯準會將於 {m_start} ~ {m_decision}（{tw_time}，距離今日還有 {days_left} 天）召開【{title}】。
                請根據目前利率環境與市場最新定價，提供會前總經與產業面情境分析：

                - 目前市場利率與指數背景：
                {macro_snapshot}
                - 使用者目前持股清單：{holdings_brief}
                - 相關新聞參考：
                {rss_lines}

                請嚴格輸出 JSON 格式（不要加 Markdown 標記）：
                {{
                  "rate_action": "市場目前主流預期（例如：市場預期降息 1 碼 / 市場預期按兵不動）",
                  "target_rate_range": "目前基準利率區間（例如：3.75% - 4.00%）",
                  "policy_stance": "事前觀察",
                  "summary": "一句話會前焦點總結（45字內）",
                  "preview_focus": "會前市場預期與 3 大關鍵觀察指標：利率期貨定價機率、最新 CPI/PCE 通膨與非農就業數據指引、以及本次聲明／點陣圖最關鍵的變數（180字內）",
                  "macro_impact": "不同利率情境對總經之可能影響：對照『符合預期』vs『意外偏鷹／偏鴿』時，美債殖利率、美元與全球股市估值的可能反應（180字內）",
                  "sector_impact": "各大產業會前影響評估：說明半導體/AI科技股、金融股、中小型成長股與防禦型資產在不同利率結果下的強弱輪動方向（200字內）",
                  "extra_insights": "針對使用者目前持股（{holdings_brief}）的會前倉位控管與避險提醒（140字內）"
                }}
                """

            try:
                parsed = _call_fomc_gemini_json(
                    client, prompt, fallback_models, exhausted_models, generate_with_retry, parse_json_response
                )
                details = {
                    "meeting_range": f"{m_start} ~ {m_decision}",
                    "tw_release_time": tw_time,
                    "has_sep": has_sep,
                    "rate_action": str(parsed.get("rate_action") or "依聯準會公告").strip(),
                    "target_rate_range": str(parsed.get("target_rate_range") or "--").strip(),
                }
                summary = str(parsed.get("summary") or f"會議日期：{m_start} ~ {m_decision}｜{tw_time}").strip()
                ai_analysis = {
                    "verdict": str(parsed.get("policy_stance") or ("事前觀察" if status == "upcoming" else "中性穩健")).strip(),
                    "highlights": str(parsed.get("highlights") or "").strip(),
                    "preview_focus": str(parsed.get("preview_focus") or "").strip(),
                    "macro_impact": str(parsed.get("macro_impact") or "").strip(),
                    "sector_impact": str(parsed.get("sector_impact") or "").strip(),
                    "extra_insights": str(parsed.get("extra_insights") or "").strip(),
                }
            except Exception as exc:
                print(f"[警告] FOMC 會議 {doc_id} 分析失敗: {exc}")
                details = {
                    "meeting_range": f"{m_start} ~ {m_decision}",
                    "tw_release_time": tw_time,
                    "has_sep": has_sep,
                    "rate_action": "待補充",
                    "target_rate_range": "--",
                }
                summary = f"會議日期：{m_start} ~ {m_decision}｜{tw_time}"
                ai_analysis = {
                    "verdict": "待補充",
                    "highlights": "AI 分析暫時異常，請稍後重試。",
                    "macro_impact": "",
                    "sector_impact": "",
                    "extra_insights": "",
                }
            time.sleep(1.5)

        db.collection("stock_events").document(doc_id).set(
            {
                "market": "MACRO",
                "ticker": "FOMC",
                "name": "美國聯準會 (Fed)",
                "event_type": "fomc",
                "event_date": m_decision,
                "status": status,
                "title": title,
                "summary": summary,
                "details": details,
                "ai_analysis": ai_analysis,
                "source": "auto",
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        count += 1

    print(f"美國聯準會 FOMC 會議同步完成，共處理 {count} 場會議！")
    return count

