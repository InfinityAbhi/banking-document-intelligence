"""Generate reproducible, synthetic loan reports and six deliberately seeded issues."""
import csv
import hashlib
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
FONT_CANDIDATES = [
    Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
]
FONT = next(p for p in FONT_CANDIDATES if p.exists())
body = ImageFont.truetype(str(FONT), 28)
heading = ImageFont.truetype(str(FONT), 36)
docs, records, questions = [], [], []
for account in range(1, 13):
    for quarter in (1, 2):
        aid, period = f"ACCT-{account:03d}", f"2026-Q{quarter}"
        doc_id = f"loan-{account:03d}-q{quarter}"
        principal = f"{250000 + account * 13750 - quarter * 2500:.2f}"
        minimum = f"{1.10 + (account % 4) * .05:.2f}"
        due = "2026-04-30" if quarter == 1 else "2026-07-31"
        lines = [
            "SYNTHETIC PORTFOLIO DATA", "Quarterly loan review report", "",
            f"Account: {aid}", f"Period: {period}", "Currency: USD",
            f"Outstanding principal: {principal}", f"Minimum coverage ratio: {minimum}",
            f"Next reporting due: {due}", "", "Review notes",
        ]
        narrative = (
            f"This simulated account holds a term facility secured by equipment. "
            f"The minimum coverage ratio is {minimum}. The next reporting due date is {due}. "
            "The analyst should reconcile the principal in this report with the structured "
            "portfolio record before preparing a review packet. Any mismatch requires manual "
            "investigation; the application must not approve a loan or infer creditworthiness. "
            "Store the source document version, reporting period, and extracted evidence for "
            "later review. Figures are fictional and generated for software testing only. "
            "No customers, institutions, or transactions in this report represent real entities."
        )
        lines.extend(textwrap.wrap(narrative, 82))
        im = Image.new('RGB', (1600, 1150), 'white')
        draw = ImageDraw.Draw(im)
        y = 60
        for idx, line in enumerate(lines):
            draw.text((70, y), line, fill=(20, 30, 40), font=heading if idx == 1 else body)
            y += 44
        path = DATA / f"{doc_id}.png"
        im.save(path)
        fields = dict(account_id=aid, period=period, currency="USD", principal=principal)
        docs.append(dict(id=doc_id, title=f"Loan review {aid} {period}", file=path.name,
                         fields=fields, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        records.append(dict(doc_id=doc_id, **fields))
        questions.extend([
            dict(question=f"What is the outstanding principal for {aid} in {period}?",
                 doc_id=doc_id, expected_text=principal),
            dict(question=f"What is the minimum coverage ratio for {aid} in {period}?",
                 doc_id=doc_id, expected_text=minimum),
        ])
# One schema-valid row intentionally disagrees with the corresponding document.
records[0]["principal"] = f"{float(records[0]['principal']) + 100:.2f}"
records.extend([
    dict(records[1]),
    {**records[2], "principal": "-1.00"},
    {**records[3], "currency": "ZZZ"},
    {**records[4], "doc_id": "missing-document"},
    {**records[5], "account_id": ""},
])
with (DATA / 'portfolio.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
(DATA / 'manifest.json').write_text(json.dumps({
    'synthetic': True, 'font': FONT.name, 'documents': docs, 'questions': questions,
    'expected_issues': {'invalid_record': 5, 'document_mismatch': 1},
    'expected_review_doc': docs[0]['id'],
    'benchmark_limit': 'Clean rendered scans, repeated template, explicit account/period filters. Not a general OCR or RAG benchmark.'
}, indent=2))
print(json.dumps({'documents':len(docs), 'rows':len(records), 'queries':len(questions)}))
