"""
generate_report.py
──────────────────────────────────────────────────────────────────
fatigue.db 에서 세션 데이터를 읽어 HTML 리포트를 생성하고 브라우저로 엽니다.

사용법:
  python generate_report.py            # 가장 최근 세션 자동 선택
  python generate_report.py --all      # 전체 세션 통합 리포트
  python generate_report.py --list     # 세션 목록만 출력
"""

import sqlite3
import json
import os
import sys
import webbrowser
import argparse
from datetime import datetime

_BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_BASE, "fatigue.db")
OUT_PATH = os.path.join(_BASE, "report.html")


# ──────────────────────────────────────────────────────────────────
#  DB 쿼리
# ──────────────────────────────────────────────────────────────────
def fetch_sessions(conn):
    rows = conn.execute(
        """SELECT session_id, start_time, end_time, total_blinks, avg_fatigue, max_fatigue
           FROM sessions ORDER BY start_time DESC"""
    ).fetchall()
    return rows


def fetch_blink_events(conn, session_ids):
    ph = ",".join("?" * len(session_ids))
    return conn.execute(
        f"""SELECT session_id, timestamp, ear, stage, completeness, fatigue_index
            FROM blink_events WHERE session_id IN ({ph}) ORDER BY timestamp""",
        session_ids,
    ).fetchall()


def fetch_snapshots(conn, session_ids):
    ph = ",".join("?" * len(session_ids))
    return conn.execute(
        f"""SELECT session_id, time_sec, fatigue_index, perclos, inc_ratio, yawn_rpm, head_droop
            FROM fatigue_snapshots WHERE session_id IN ({ph}) ORDER BY time_sec""",
        session_ids,
    ).fetchall()


# ──────────────────────────────────────────────────────────────────
#  데이터 가공
# ──────────────────────────────────────────────────────────────────
def process(sessions, blinks, snaps):
    """세션·이벤트 데이터를 차트용 JSON 형태로 가공"""

    # 세션 요약
    session_summaries = []
    for s in sessions:
        sid, st, et, tb, af, mf = s
        dur = (et - st) / 60 if et else 0
        dt  = datetime.fromtimestamp(st).strftime("%Y-%m-%d %H:%M:%S")
        bpm = tb / dur if dur > 0 else 0
        session_summaries.append({
            "id": sid,
            "datetime": dt,
            "duration_min": round(dur, 1),
            "total_blinks": tb or 0,
            "bpm": round(bpm, 1),
            "avg_fatigue": round((af or 0) * 100, 1),
            "max_fatigue": round((mf or 0) * 100, 1),
        })

    # 전체 통계 (첫 번째 세션 기준 또는 합산)
    total_blinks = sum(s["total_blinks"] for s in session_summaries)
    total_dur    = sum(s["duration_min"] for s in session_summaries)

    # 깜빡임 단계 분포
    stage_counts = {"Normal": 0, "Onset": 0, "Valley": 0, "Offset": 0}
    for b in blinks:
        stage = b[3]
        if stage in stage_counts:
            stage_counts[stage] += 1

    # 완전성 분포
    comp_counts = {"Complete": 0, "Incomplete": 0, "Micro": 0}
    for b in blinks:
        c = b[4]
        if c > 0.9:   comp_counts["Complete"] += 1
        elif c > 0.5: comp_counts["Incomplete"] += 1
        else:         comp_counts["Micro"] += 1

    # 피로도 타임라인 (스냅샷 기준)
    snap_labels = [round(r[1] / 60, 2) for r in snaps]  # 분 단위
    snap_fatigue = [round(r[2] * 100, 1) for r in snaps]
    snap_perclos = [round(r[3] * 100, 1) for r in snaps]
    snap_inc     = [round(r[4] * 100, 1) for r in snaps]

    # 깜빡임 이벤트 타임라인 (1분 단위 BPM 집계)
    bpm_buckets = {}
    for b in blinks:
        minute = int(b[1] / 60)
        bpm_buckets[minute] = bpm_buckets.get(minute, 0) + 1
    bpm_labels  = sorted(bpm_buckets.keys())
    bpm_values  = [bpm_buckets[m] for m in bpm_labels]

    # EAR 타임라인 (이벤트 샘플링 — 최대 300개)
    ear_events = blinks if len(blinks) <= 300 else blinks[::max(1, len(blinks)//300)]
    ear_labels = [round(b[1], 1) for b in ear_events]
    ear_values = [round(b[2], 4) for b in ear_events]

    return {
        "sessions": session_summaries,
        "total_blinks": total_blinks,
        "total_dur": round(total_dur, 1),
        "stage_counts": stage_counts,
        "comp_counts": comp_counts,
        "snap_labels": snap_labels,
        "snap_fatigue": snap_fatigue,
        "snap_perclos": snap_perclos,
        "snap_inc": snap_inc,
        "bpm_labels": bpm_labels,
        "bpm_values": bpm_values,
        "ear_labels": ear_labels,
        "ear_values": ear_values,
    }


# ──────────────────────────────────────────────────────────────────
#  HTML 생성
# ──────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>눈 건강 리포트</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #4ade80;
    --warn: #fb923c;
    --danger: #f87171;
    --blue: #60a5fa;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 32px 24px;
    max-width: 1100px;
    margin: 0 auto;
  }}

  /* 헤더 */
  .header {{ margin-bottom: 36px; }}
  .header h1 {{ font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }}
  .header h1 span {{ color: var(--accent); }}
  .header p {{ color: var(--muted); margin-top: 6px; font-size: 0.88rem; }}

  /* 세션 탭 */
  .session-tabs {{
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 28px;
  }}
  .session-tab {{
    padding: 6px 14px; border-radius: 20px; font-size: 0.80rem;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--muted); cursor: pointer; transition: all .15s;
  }}
  .session-tab.active {{
    border-color: var(--accent); color: var(--accent); background: rgba(74,222,128,.08);
  }}

  /* 요약 카드 */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px; margin-bottom: 28px;
  }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 18px;
  }}
  .card .label {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 6px; }}
  .card .value {{ font-size: 1.7rem; font-weight: 700; line-height: 1; }}
  .card .unit  {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; }}
  .card.good   {{ border-color: rgba(74,222,128,.3); }}
  .card.warn   {{ border-color: rgba(251,146,60,.3); }}
  .card.danger {{ border-color: rgba(248,113,113,.3); }}

  /* 차트 그리드 */
  .charts {{ display: grid; gap: 20px; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .chart-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 22px;
  }}
  .chart-box h3 {{
    font-size: 0.85rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 18px;
  }}
  .chart-box canvas {{ max-height: 220px; }}
  .chart-box.wide {{ grid-column: 1 / -1; }}
  .chart-box.wide canvas {{ max-height: 180px; }}

  /* 세션 테이블 */
  .table-wrap {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 22px; margin-top: 20px; overflow-x: auto;
  }}
  .table-wrap h3 {{
    font-size: 0.85rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 16px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th, td {{ padding: 10px 12px; text-align: left; }}
  th {{ color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); }}
  tr:not(:last-child) td {{ border-bottom: 1px solid var(--border); }}
  .pill {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
  }}
  .pill.normal  {{ background: rgba(74,222,128,.15); color: var(--accent); }}
  .pill.caution {{ background: rgba(251,146,60,.15); color: var(--warn); }}
  .pill.danger  {{ background: rgba(248,113,113,.15); color: var(--danger); }}

  @media (max-width: 640px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>👁 눈 건강 <span>리포트</span></h1>
  <p>지능형 눈 깜빡임 감지 솔루션 · DeFB</p>
</div>

<!-- 세션 탭 -->
<div class="session-tabs" id="tabs"></div>

<!-- 요약 카드 -->
<div class="cards" id="cards"></div>

<!-- 차트 -->
<div class="charts">
  <div class="chart-box wide">
    <h3>⚡ 피로도 타임라인</h3>
    <canvas id="chartFatigue"></canvas>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h3>👁 분당 깜빡임 (BPM)</h3>
      <canvas id="chartBPM"></canvas>
    </div>
    <div class="chart-box">
      <h3>🔄 깜빡임 완전성</h3>
      <canvas id="chartComp"></canvas>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h3>📊 단계 분포</h3>
      <canvas id="chartStage"></canvas>
    </div>
    <div class="chart-box">
      <h3>📈 EAR 타임라인</h3>
      <canvas id="chartEAR"></canvas>
    </div>
  </div>
</div>

<!-- 세션 목록 테이블 -->
<div class="table-wrap">
  <h3>📋 세션 기록</h3>
  <table id="sessionTable">
    <thead>
      <tr>
        <th>시각</th><th>세션 ID</th><th>시간</th>
        <th>깜빡임</th><th>BPM</th><th>평균 피로도</th><th>최대 피로도</th><th>상태</th>
      </tr>
    </thead>
    <tbody id="sessionTbody"></tbody>
  </table>
</div>

<script>
const DATA = __DATA_JSON__;

let charts = {{}};
let currentIdx = 0;

function fatiguePill(val) {{
  if (val < 30) return `<span class="pill normal">NORMAL</span>`;
  if (val < 60) return `<span class="pill caution">CAUTION</span>`;
  return `<span class="pill danger">DANGER</span>`;
}}

function renderTabs() {{
  const el = document.getElementById('tabs');
  DATA.sessions.forEach((s, i) => {{
    const btn = document.createElement('button');
    btn.className = 'session-tab' + (i === 0 ? ' active' : '');
    btn.textContent = s.datetime + ' (' + s.duration_min + '분)';
    btn.onclick = () => selectSession(i);
    el.appendChild(btn);
  }});
}}

function selectSession(idx) {{
  currentIdx = idx;
  document.querySelectorAll('.session-tab').forEach((b, i) =>
    b.classList.toggle('active', i === idx));
  renderCards(idx);
  renderCharts(idx);
}}

function renderCards(idx) {{
  const s = DATA.sessions[idx];
  const fi = s.avg_fatigue;
  const fiClass = fi < 30 ? 'good' : fi < 60 ? 'warn' : 'danger';

  document.getElementById('cards').innerHTML = `
    <div class="card">
      <div class="label">총 깜빡임 횟수</div>
      <div class="value" style="color:#60a5fa">${{s.total_blinks}}</div>
      <div class="unit">회</div>
    </div>
    <div class="card">
      <div class="label">분당 깜빡임 (BPM)</div>
      <div class="value" style="color:#4ade80">${{s.bpm}}</div>
      <div class="unit">회/분 (정상 15~20)</div>
    </div>
    <div class="card">
      <div class="label">세션 시간</div>
      <div class="value" style="color:#a78bfa">${{s.duration_min}}</div>
      <div class="unit">분</div>
    </div>
    <div class="card ${{fiClass}}">
      <div class="label">평균 피로도</div>
      <div class="value" style="color:${{fi<30?'#4ade80':fi<60?'#fb923c':'#f87171'}}">${{fi}}%</div>
      <div class="unit">최대 ${{s.max_fatigue}}%</div>
    </div>
  `;
}}

function renderCharts(idx) {{
  // 차트 destroy
  Object.values(charts).forEach(c => c && c.destroy());
  charts = {{}};

  const d = DATA;

  // 공통 옵션
  const gridColor = 'rgba(255,255,255,0.06)';
  const textColor = '#94a3b8';
  const baseFont  = {{ family: 'Segoe UI, system-ui, sans-serif', size: 11 }};

  // 1. 피로도 타임라인
  if (d.snap_labels.length) {{
    charts.fatigue = new Chart(document.getElementById('chartFatigue'), {{
      type: 'line',
      data: {{
        labels: d.snap_labels.map(v => v.toFixed(1) + '분'),
        datasets: [
          {{
            label: '피로도 (%)',
            data: d.snap_fatigue,
            borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,.10)',
            fill: true, tension: 0.4, pointRadius: 3,
          }},
          {{
            label: 'PERCLOS (%)',
            data: d.snap_perclos,
            borderColor: '#60a5fa', backgroundColor: 'transparent',
            tension: 0.4, pointRadius: 2,
          }},
          {{
            label: '불완전 깜빡임 비율 (%)',
            data: d.snap_inc,
            borderColor: '#fb923c', backgroundColor: 'transparent',
            borderDash: [4, 4], tension: 0.4, pointRadius: 2,
          }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: true,
        plugins: {{ legend: {{ labels: {{ color: textColor, font: baseFont }} }} }},
        scales: {{
          x: {{ ticks: {{ color: textColor, font: baseFont }}, grid: {{ color: gridColor }} }},
          y: {{
            min: 0, max: 100,
            ticks: {{ color: textColor, font: baseFont, callback: v => v + '%' }},
            grid: {{ color: gridColor }},
          }},
        }},
      }},
    }});
  }}

  // 2. BPM (분당 깜빡임)
  if (d.bpm_labels.length) {{
    charts.bpm = new Chart(document.getElementById('chartBPM'), {{
      type: 'bar',
      data: {{
        labels: d.bpm_labels.map(v => v + '분'),
        datasets: [{{
          label: '깜빡임 수',
          data: d.bpm_values,
          backgroundColor: d.bpm_values.map(v =>
            v >= 15 && v <= 20 ? 'rgba(74,222,128,.7)' :
            v < 10              ? 'rgba(248,113,113,.7)' : 'rgba(251,146,60,.7)'
          ),
          borderRadius: 4,
        }}],
      }},
      options: {{
        responsive: true, maintainAspectRatio: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              afterLabel: ctx => {{
                const v = ctx.parsed.y;
                if (v >= 15 && v <= 20) return '✅ 정상';
                if (v < 10)             return '⚠️ 부족';
                return '🟡 주의';
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ ticks: {{ color: textColor, font: baseFont }}, grid: {{ color: gridColor }} }},
          y: {{
            beginAtZero: true,
            ticks: {{ color: textColor, font: baseFont }},
            grid: {{ color: gridColor }},
          }},
        }},
      }},
    }});
  }}

  // 3. 완전성 도넛
  const cc = d.comp_counts;
  charts.comp = new Chart(document.getElementById('chartComp'), {{
    type: 'doughnut',
    data: {{
      labels: ['완전 깜빡임', '불완전', '마이크로'],
      datasets: [{{
        data: [cc.Complete, cc.Incomplete, cc.Micro],
        backgroundColor: ['rgba(74,222,128,.8)', 'rgba(251,146,60,.8)', 'rgba(248,113,113,.8)'],
        borderColor: '#1a1d27', borderWidth: 3,
      }}],
    }},
    options: {{
      responsive: true, maintainAspectRatio: true,
      plugins: {{
        legend: {{ position: 'right', labels: {{ color: textColor, font: baseFont, padding: 14 }} }},
      }},
    }},
  }});

  // 4. 단계 분포 도넛
  const sc = d.stage_counts;
  charts.stage = new Chart(document.getElementById('chartStage'), {{
    type: 'doughnut',
    data: {{
      labels: ['Normal', 'Onset', 'Valley', 'Offset'],
      datasets: [{{
        data: [sc.Normal, sc.Onset, sc.Valley, sc.Offset],
        backgroundColor: [
          'rgba(74,222,128,.8)', 'rgba(96,165,250,.8)',
          'rgba(167,139,250,.8)', 'rgba(251,146,60,.8)'
        ],
        borderColor: '#1a1d27', borderWidth: 3,
      }}],
    }},
    options: {{
      responsive: true, maintainAspectRatio: true,
      plugins: {{
        legend: {{ position: 'right', labels: {{ color: textColor, font: baseFont, padding: 14 }} }},
      }},
    }},
  }});

  // 5. EAR 타임라인
  if (d.ear_labels.length) {{
    charts.ear = new Chart(document.getElementById('chartEAR'), {{
      type: 'line',
      data: {{
        labels: d.ear_labels.map(v => v.toFixed(0) + 's'),
        datasets: [{{
          label: 'EAR',
          data: d.ear_values,
          borderColor: 'rgba(96,165,250,.8)',
          backgroundColor: 'rgba(96,165,250,.05)',
          fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
        }}],
      }},
      options: {{
        responsive: true, maintainAspectRatio: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: textColor, font: baseFont, maxTicksLimit: 8 }}, grid: {{ color: gridColor }} }},
          y: {{ ticks: {{ color: textColor, font: baseFont }}, grid: {{ color: gridColor }} }},
        }},
      }},
    }});
  }}
}}

function renderTable() {{
  const tbody = document.getElementById('sessionTbody');
  DATA.sessions.forEach(s => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{s.datetime}}</td>
      <td style="color:#64748b;font-size:.78rem">${{s.id}}</td>
      <td>${{s.duration_min}}분</td>
      <td>${{s.total_blinks}}회</td>
      <td>${{s.bpm}}</td>
      <td>${{s.avg_fatigue}}%</td>
      <td>${{s.max_fatigue}}%</td>
      <td>${{fatiguePill(s.avg_fatigue)}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

// 초기화
renderTabs();
renderCards(0);
renderCharts(0);
renderTable();
</script>
</body>
</html>
"""


def generate_html(data: dict) -> str:
    return HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))


# ──────────────────────────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",  action="store_true", help="전체 세션 통합")
    parser.add_argument("--list", action="store_true", help="세션 목록만 출력")
    parser.add_argument("--db",   default=DB_PATH,     help="DB 경로")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[오류] DB 파일을 찾을 수 없습니다: {args.db}")
        print("  test_6.py 를 먼저 실행해서 데이터를 생성해주세요.")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    sessions = fetch_sessions(conn)

    if not sessions:
        print("[오류] 저장된 세션이 없습니다. test_6.py 를 먼저 실행해주세요.")
        conn.close()
        sys.exit(1)

    # --list
    if args.list:
        print(f"\n{'#':>3}  {'세션 ID':10}  {'시각':20}  {'시간':>6}  {'깜빡임':>6}  {'피로도 avg':>10}")
        print("-" * 70)
        for i, s in enumerate(sessions):
            sid, st, et, tb, af, mf = s
            dur = (et - st) / 60 if et else 0
            dt  = datetime.fromtimestamp(st).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i+1:>3}  {sid:10}  {dt:20}  {dur:>5.1f}분  {tb or 0:>6}회  {(af or 0)*100:>9.1f}%")
        conn.close()
        return

    # 세션 선택
    if args.all:
        target_sessions = sessions
        print(f"[INFO] 전체 {len(sessions)}개 세션 통합 리포트 생성")
    else:
        target_sessions = [sessions[0]]  # 최신 세션
        sid = sessions[0][0]
        dt  = datetime.fromtimestamp(sessions[0][1]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] 최근 세션 리포트 생성: {sid} ({dt})")

    session_ids = [s[0] for s in target_sessions]
    blinks = fetch_blink_events(conn, session_ids)
    snaps  = fetch_snapshots(conn, session_ids)
    conn.close()

    print(f"  깜빡임 이벤트: {len(blinks)}건  스냅샷: {len(snaps)}건")

    # 가공 & HTML 생성
    data = process(target_sessions, blinks, snaps)
    html = generate_html(data)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[완료] 리포트 저장: {OUT_PATH}")
    webbrowser.open(f"file:///{OUT_PATH.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
