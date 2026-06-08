"""
generate_report.py
──────────────────────────────────────────────────────────────────
fatigue.db 에서 세션 데이터를 읽어 HTML 리포트를 생성하고 브라우저로 엽니다.
외부 라이브러리 없이 동작합니다 (순수 HTML/CSS/SVG).

사용법:
  python generate_report.py            # 가장 최근 세션
  python generate_report.py --all      # 전체 세션 통합
  python generate_report.py --list     # 세션 목록만 출력
"""

import sqlite3
import os
import sys
import webbrowser
import argparse
import math
from datetime import datetime

_BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_BASE, "fatigue.db")
OUT_PATH = os.path.join(_BASE, "report.html")


# ──────────────────────────────────────────────────────────────────
#  DB 쿼리
# ──────────────────────────────────────────────────────────────────
def fetch_sessions(conn):
    return conn.execute(
        "SELECT session_id, start_time, end_time, total_blinks, avg_fatigue, max_fatigue "
        "FROM sessions ORDER BY start_time DESC"
    ).fetchall()

def fetch_blinks(conn, sids):
    ph = ",".join("?"*len(sids))
    return conn.execute(
        f"SELECT timestamp, completeness, fatigue_index, ear "
        f"FROM blink_events WHERE session_id IN ({ph}) ORDER BY timestamp", sids
    ).fetchall()


# ──────────────────────────────────────────────────────────────────
#  SVG 헬퍼
# ──────────────────────────────────────────────────────────────────
def svg_line_chart(data, labels, w=800, h=160, color="#f87171", ylabel="%", ymax=100):
    """단순 SVG 라인 차트"""
    if not data:
        return f'<svg width="{w}" height="{h}"><text x="{w//2}" y="{h//2}" fill="#64748b" text-anchor="middle" font-size="13">데이터 없음</text></svg>'

    pad_l, pad_r, pad_t, pad_b = 42, 16, 12, 28
    gw = w - pad_l - pad_r
    gh = h - pad_t - pad_b
    ymin = 0
    scale_y = gh / max(ymax - ymin, 1)
    scale_x = gw / max(len(data) - 1, 1)

    def px(i): return pad_l + i * scale_x
    def py(v): return pad_t + gh - (v - ymin) * scale_y

    # 격자
    lines = []
    for gv in range(0, int(ymax) + 1, max(1, int(ymax) // 5)):
        gy = py(gv)
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+gw}" y2="{gy:.1f}" stroke="#2a2d3a" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l-4}" y="{gy+4:.1f}" fill="#475569" font-size="9" text-anchor="end">{gv}{ylabel}</text>')

    # 채우기
    pts = [(px(i), py(v)) for i, v in enumerate(data)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    fill_pts = f"{pad_l:.1f},{pad_t+gh:.1f} " + poly + f" {px(len(data)-1):.1f},{pad_t+gh:.1f}"
    fill_col = color.replace("#", "")
    lines.append(f'<polygon points="{fill_pts}" fill="{color}" fill-opacity="0.12"/>')

    # 라인
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    lines.append(f'<path d="{path}" stroke="{color}" stroke-width="2" fill="none" stroke-linejoin="round"/>')

    # X축 레이블 (최대 8개)
    step = max(1, len(labels) // 8)
    for i in range(0, len(labels), step):
        lbl = labels[i] if i < len(labels) else ""
        lines.append(f'<text x="{px(i):.1f}" y="{pad_t+gh+16}" fill="#475569" font-size="9" text-anchor="middle">{lbl}</text>')

    # 축
    lines.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+gh}" stroke="#334155" stroke-width="1"/>')
    lines.append(f'<line x1="{pad_l}" y1="{pad_t+gh}" x2="{pad_l+gw}" y2="{pad_t+gh}" stroke="#334155" stroke-width="1"/>')

    return f'<svg width="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{"".join(lines)}</svg>'


def svg_bar_chart(values, labels, w=500, h=160, normal_range=(15, 20)):
    if not values:
        return f'<svg width="{w}" height="{h}"><text x="{w//2}" y="{h//2}" fill="#64748b" text-anchor="middle" font-size="13">데이터 없음</text></svg>'

    pad_l, pad_r, pad_t, pad_b = 30, 10, 12, 28
    gw = w - pad_l - pad_r
    gh = h - pad_t - pad_b
    n = len(values)
    bar_w = max(4, gw / n - 4)
    ymax = max(max(values) * 1.2, normal_range[1] * 1.5, 10)

    lines = []
    # 격자
    for gv in range(0, int(ymax) + 1, 10):
        gy = pad_t + gh - (gv / ymax) * gh
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+gw}" y2="{gy:.1f}" stroke="#2a2d3a" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l-4}" y="{gy+4:.1f}" fill="#475569" font-size="9" text-anchor="end">{gv}</text>')

    # 정상 범위 표시
    y_hi = pad_t + gh - (normal_range[1] / ymax) * gh
    y_lo = pad_t + gh - (normal_range[0] / ymax) * gh
    lines.append(f'<rect x="{pad_l}" y="{y_hi:.1f}" width="{gw}" height="{y_lo-y_hi:.1f}" fill="#4ade80" fill-opacity="0.07"/>')
    lines.append(f'<text x="{pad_l+gw-2}" y="{y_hi-2:.1f}" fill="#4ade80" font-size="8" text-anchor="end">정상 구간</text>')

    for i, v in enumerate(values):
        x = pad_l + i * (gw / n) + (gw / n - bar_w) / 2
        bh = (v / ymax) * gh
        by = pad_t + gh - bh
        col = "#4ade80" if normal_range[0] <= v <= normal_range[1] else ("#f87171" if v < 10 else "#fb923c")
        lines.append(f'<rect x="{x:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{col}" rx="3"/>')
        lbl = labels[i] if i < len(labels) else ""
        lines.append(f'<text x="{x+bar_w/2:.1f}" y="{pad_t+gh+16}" fill="#475569" font-size="9" text-anchor="middle">{lbl}분</text>')
        if bh > 14:
            lines.append(f'<text x="{x+bar_w/2:.1f}" y="{by-3:.1f}" fill="{col}" font-size="8" text-anchor="middle">{v}</text>')

    lines.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+gh}" stroke="#334155" stroke-width="1"/>')
    lines.append(f'<line x1="{pad_l}" y1="{pad_t+gh}" x2="{pad_l+gw}" y2="{pad_t+gh}" stroke="#334155" stroke-width="1"/>')

    return f'<svg width="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{"".join(lines)}</svg>'


def svg_donut(values, labels, colors, w=260, h=200):
    total = sum(values)
    if total == 0:
        return "<p style='color:#64748b'>데이터 없음</p>"

    cx, cy, r, ri = 90, 100, 75, 40
    start = -math.pi / 2
    paths = []
    for v, lbl, col in zip(values, labels, colors):
        angle = 2 * math.pi * v / total
        end = start + angle
        lx1 = cx + r * math.cos(start)
        ly1 = cy + r * math.sin(start)
        lx2 = cx + r * math.cos(end)
        ly2 = cy + r * math.sin(end)
        sx1 = cx + ri * math.cos(end)
        sy1 = cy + ri * math.sin(end)
        sx2 = cx + ri * math.cos(start)
        sy2 = cy + ri * math.sin(start)
        large = 1 if angle > math.pi else 0
        pct = round(v / total * 100)
        paths.append(
            f'<path d="M {lx1:.1f} {ly1:.1f} A {r} {r} 0 {large} 1 {lx2:.1f} {ly2:.1f} '
            f'L {sx1:.1f} {sy1:.1f} A {ri} {ri} 0 {large} 0 {sx2:.1f} {sy2:.1f} Z" '
            f'fill="{col}" stroke="#1a1d27" stroke-width="2">'
            f'<title>{lbl}: {v}회 ({pct}%)</title></path>'
        )
        start = end

    # 범례
    legend = []
    for i, (lbl, col, v) in enumerate(zip(labels, colors, values)):
        pct = round(v / total * 100)
        ly = 30 + i * 28
        legend.append(f'<rect x="180" y="{ly}" width="12" height="12" fill="{col}" rx="2"/>')
        legend.append(f'<text x="196" y="{ly+10}" fill="#cbd5e1" font-size="11">{lbl}</text>')
        legend.append(f'<text x="196" y="{ly+22}" fill="#64748b" font-size="10">{v}회 ({pct}%)</text>')

    return (f'<svg width="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(paths) + "".join(legend) + "</svg>")


# ──────────────────────────────────────────────────────────────────
#  HTML 생성
# ──────────────────────────────────────────────────────────────────
def build_html(sessions, blinks, session_summaries):
    # 데이터 집계
    bpm_buckets = {}
    fi_buckets  = {}
    comp = inc = micro = 0
    ear_pts = []

    for b in blinks:
        ts, completeness, fi, ear = b
        m = int(ts / 60)
        bpm_buckets[m] = bpm_buckets.get(m, 0) + 1
        key = round(ts / 30) * 0.5
        fi_buckets.setdefault(key, []).append(fi)
        ear_pts.append((ts, ear))
        if completeness > 0.9:   comp += 1
        elif completeness > 0.5: inc += 1
        else:                    micro += 1

    bpm_labels = sorted(bpm_buckets.keys())
    bpm_values = [bpm_buckets[m] for m in bpm_labels]

    fi_keys   = sorted(fi_buckets.keys())
    fi_values = [round(sum(fi_buckets[k]) / len(fi_buckets[k]) * 100, 1) for k in fi_keys]
    fi_labels = [f"{k:.1f}분" for k in fi_keys]

    # EAR (최대 200 샘플)
    step = max(1, len(ear_pts) // 200)
    ear_sample = ear_pts[::step]
    ear_values = [round(e[1], 3) for e in ear_sample]
    ear_labels = [f"{int(e[0])}s" for e in ear_sample]
    ear_ymax   = round(max(ear_values) * 1.3, 2) if ear_values else 0.5

    # 카드 데이터 (첫 번째 세션)
    s = session_summaries[0]
    fi_avg = s["avg_fatigue"]
    bpm_val = s["bpm"]
    fi_color  = "#4ade80" if fi_avg < 30 else "#fb923c" if fi_avg < 60 else "#f87171"
    bpm_color = "#4ade80" if 15 <= bpm_val <= 20 else "#fb923c"

    # 인사이트
    insights = []
    if bpm_val < 10:
        insights.append(("⚠️", f"분당 깜빡임 <b>{bpm_val}회</b>로 매우 낮습니다. 정상(15~20회)보다 부족해 안구건조 위험이 높습니다."))
    elif 15 <= bpm_val <= 20:
        insights.append(("✅", f"분당 깜빡임 <b>{bpm_val}회</b>로 정상 범위(15~20회) 안에 있습니다."))
    else:
        insights.append(("🟡", f"분당 깜빡임 <b>{bpm_val}회</b>입니다. 정상 범위(15~20회)를 살짝 벗어났습니다."))

    total_b = comp + inc + micro or 1
    inc_pct = round(inc / total_b * 100)
    if inc_pct > 50:
        insights.append(("🔴", f"깜빡임의 <b>{inc_pct}%가 불완전</b>합니다. 눈꺼풀이 완전히 닫히지 않아 눈물막 보충이 부족했습니다."))
    elif inc_pct > 30:
        insights.append(("🟡", f"깜빡임의 <b>{inc_pct}%가 불완전</b>합니다. 집중 작업 중 깜빡임 품질이 저하됩니다."))

    micro_pct = round(micro / total_b * 100)
    if micro_pct > 15:
        insights.append(("🔴", f"마이크로 깜빡임이 <b>{micro_pct}%</b>입니다. 극도의 집중 또는 피로 상태의 패턴입니다."))

    if fi_avg > 60:
        insights.append(("⚠️", f"평균 피로도 <b>{fi_avg}%</b> — 고피로 구간. 20-20-20 휴식을 권장합니다."))
    elif fi_avg > 30:
        insights.append(("🟡", f"평균 피로도 <b>{fi_avg}%</b> — 주의 구간. 작업 강도를 조절하세요."))

    insight_html = "\n".join(
        f'<div class="insight"><span class="iicon">{icon}</span><span>{text}</span></div>'
        for icon, text in insights
    )

    # 세션 테이블
    table_rows = ""
    for sv in session_summaries:
        fi_v = sv["avg_fatigue"]
        pclass = "normal" if fi_v < 30 else "caution" if fi_v < 60 else "danger"
        table_rows += (
            f'<tr><td>{sv["datetime"]}</td>'
            f'<td class="sid">{sv["id"]}</td>'
            f'<td>{sv["duration_min"]}분</td>'
            f'<td>{sv["total_blinks"]}회</td>'
            f'<td>{sv["bpm"]}</td>'
            f'<td>{sv["avg_fatigue"]}%</td>'
            f'<td>{sv["max_fatigue"]}%</td>'
            f'<td><span class="pill {pclass}">{"NORMAL" if fi_v<30 else "CAUTION" if fi_v<60 else "DANGER"}</span></td></tr>\n'
        )

    # SVG 차트
    svg_fatigue = svg_line_chart(fi_values, fi_labels, color="#f87171", ymax=100)
    svg_bpm     = svg_bar_chart(bpm_values, bpm_labels, normal_range=(15, 20))
    svg_donut_c = svg_donut(
        [comp, inc, micro],
        ["완전", "불완전", "마이크로"],
        ["#4ade80", "#fb923c", "#f87171"]
    )
    svg_ear = svg_line_chart(
        ear_values, ear_labels, color="#60a5fa",
        ylabel="", ymax=max(0.5, round(max(ear_values)*1.2, 1) if ear_values else 0.5)
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>눈 건강 리포트 — DeFB</title>
<style>
  :root {{
    --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3a;
    --text:#e2e8f0; --muted:#64748b;
    --green:#4ade80; --orange:#fb923c; --red:#f87171; --blue:#60a5fa;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:32px 24px;max-width:1080px;margin:0 auto}}
  h1{{font-size:1.75rem;font-weight:700;margin-bottom:6px}}
  h1 span{{color:var(--green)}}
  .sub{{color:var(--muted);font-size:.86rem;margin-bottom:28px}}

  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:24px}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 16px}}
  .card .label{{font-size:.72rem;color:var(--muted);margin-bottom:8px}}
  .card .value{{font-size:2rem;font-weight:700;line-height:1}}
  .card .unit{{font-size:.73rem;color:var(--muted);margin-top:5px}}

  .section{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:18px}}
  .section h3{{font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:16px}}

  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}

  .insight{{display:flex;gap:10px;align-items:flex-start;margin-bottom:10px;font-size:.88rem;line-height:1.55}}
  .iicon{{font-size:1.05rem;flex-shrink:0;margin-top:1px}}

  table{{width:100%;border-collapse:collapse;font-size:.83rem}}
  th,td{{padding:9px 12px;text-align:left}}
  th{{color:var(--muted);font-weight:500;border-bottom:1px solid var(--border)}}
  tr:not(:last-child) td{{border-bottom:1px solid var(--border)}}
  .sid{{color:var(--muted);font-size:.76rem}}
  .pill{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:600}}
  .pill.normal{{background:rgba(74,222,128,.15);color:var(--green)}}
  .pill.caution{{background:rgba(251,146,60,.15);color:var(--orange)}}
  .pill.danger{{background:rgba(248,113,113,.15);color:var(--red)}}

  @media(max-width:640px){{.two-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<h1>👁 눈 건강 <span>리포트</span></h1>
<p class="sub">지능형 눈 깜빡임 감지 솔루션 · DeFB &nbsp;·&nbsp; {s["datetime"]}</p>

<!-- 요약 카드 -->
<div class="cards">
  <div class="card">
    <div class="label">총 깜빡임 횟수</div>
    <div class="value" style="color:{bpm_color}">{s["total_blinks"]}</div>
    <div class="unit">회</div>
  </div>
  <div class="card">
    <div class="label">분당 깜빡임 (BPM)</div>
    <div class="value" style="color:{bpm_color}">{s["bpm"]}</div>
    <div class="unit">정상 15 ~ 20 회/분</div>
  </div>
  <div class="card">
    <div class="label">세션 시간</div>
    <div class="value" style="color:#a78bfa">{s["duration_min"]}</div>
    <div class="unit">분</div>
  </div>
  <div class="card">
    <div class="label">평균 피로도</div>
    <div class="value" style="color:{fi_color}">{fi_avg}%</div>
    <div class="unit">최대 {s["max_fatigue"]}%</div>
  </div>
</div>

<!-- 피로도 타임라인 -->
<div class="section">
  <h3>⚡ 피로도 타임라인</h3>
  {svg_fatigue}
</div>

<!-- 2열 -->
<div class="two-col">
  <div class="section">
    <h3>👁 분당 깜빡임 (BPM)</h3>
    {svg_bpm}
  </div>
  <div class="section">
    <h3>🔄 깜빡임 완전성</h3>
    {svg_donut_c}
  </div>
</div>

<!-- EAR 타임라인 -->
<div class="section">
  <h3>📈 EAR (눈 열림 비율) 타임라인</h3>
  {svg_ear}
</div>

<!-- 인사이트 -->
<div class="section">
  <h3>💡 분석 인사이트</h3>
  {insight_html}
</div>

<!-- 세션 테이블 -->
<div class="section">
  <h3>📋 세션 기록</h3>
  <table>
    <thead>
      <tr><th>시각</th><th>세션 ID</th><th>시간</th><th>깜빡임</th><th>BPM</th><th>평균 피로도</th><th>최대 피로도</th><th>상태</th></tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

</body>
</html>"""
    return html


# ──────────────────────────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--db",   default=DB_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[오류] DB 없음: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    sessions = fetch_sessions(conn)

    if not sessions:
        print("[오류] 저장된 세션이 없습니다.")
        conn.close(); sys.exit(1)

    if args.list:
        print(f"\n{'#':>3}  {'세션 ID':10}  {'시각':20}  {'시간':>6}  {'깜빡임':>6}  {'피로도':>8}")
        print("-" * 65)
        for i, s in enumerate(sessions):
            sid, st, et, tb, af, mf = s
            dur = (et - st) / 60 if et else 0
            dt  = datetime.fromtimestamp(st).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i+1:>3}  {sid:10}  {dt:20}  {dur:>5.1f}분  {tb or 0:>5}회  {(af or 0)*100:>7.1f}%")
        conn.close(); return

    target = sessions if args.all else [sessions[0]]
    sids   = [s[0] for s in target]
    blinks = fetch_blinks(conn, sids)
    conn.close()

    # 세션 요약 목록
    summaries = []
    for s in sessions:  # 테이블용 전체 세션
        sid, st, et, tb, af, mf = s
        dur = (et - st) / 60 if et else 0
        bpm = round(tb / dur, 1) if dur > 0 else 0
        summaries.append({
            "id": sid,
            "datetime": datetime.fromtimestamp(st).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": round(dur, 1),
            "total_blinks": tb or 0,
            "bpm": bpm,
            "avg_fatigue": round((af or 0) * 100, 1),
            "max_fatigue": round((mf or 0) * 100, 1),
        })

    mode = "전체 통합" if args.all else f"최근 세션 ({sessions[0][0]})"
    print(f"[INFO] {mode}  깜빡임 {len(blinks)}건")

    html = build_html(sessions, blinks, summaries)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[완료] {OUT_PATH}")
    webbrowser.open(f"file:///{OUT_PATH.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
