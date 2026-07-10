# -*- coding: utf-8 -*-
"""每晚汇总:触发次数、时间分布、各素材止叫成功率。用法: python summarize.py [日期 YYYY-MM-DD]"""
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime

LOG = "logs/triggers.csv"
day = sys.argv[1] if len(sys.argv) > 1 else None

rows = []
with open(LOG, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if day is None or r["timestamp"].startswith(day):
            rows.append(r)

if not rows:
    sys.exit("没有匹配的记录。")

print(f"触发总次数: {len(rows)}")

hours = Counter(datetime.fromisoformat(r["timestamp"]).hour for r in rows)
print("\n时间分布(小时 -> 次数):")
for h in sorted(hours):
    print(f"  {h:02d}:00  {'#' * hours[h]} {hours[h]}")

stats = defaultdict(lambda: [0, 0])  # material -> [effective, total_judged]
for r in rows:
    if r["effective"] in ("yes", "no"):
        stats[r["material"]][1] += 1
        if r["effective"] == "yes":
            stats[r["material"]][0] += 1

print("\n各素材止叫成功率:")
for m, (ok, tot) in stats.items():
    print(f"  {m}: {ok}/{tot} = {ok / tot:.0%}" if tot else f"  {m}: 无判定数据")

classes = Counter(r["class"] for r in rows)
print("\n触发类别分布:", dict(classes))
if hours and max(hours, key=hours.get) in (3, 4, 5):
    print("\n提示: 触发集中在凌晨 3-5 点,大概率是讨食——考虑定时喂食器。")
