"""Exercise the actual OCR/API pipeline and save reproducible benchmark evidence."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from bankdocs.api import create_app, DEMO_KEY

parser=argparse.ArgumentParser()
parser.add_argument('--db', default=str(ROOT / '.runtime/evaluation.sqlite'))
args=parser.parse_args()
db=Path(args.db)
if db.exists():
    raise SystemExit('Evaluation requires a fresh database. Choose a new --db path.')
manifest=json.loads((ROOT/'data/manifest.json').read_text())
client=TestClient(create_app(db_path=db, api_key=DEMO_KEY, adapters=(None,None,None)))
headers={'X-API-Key':DEMO_KEY}
matches=0; field_total=0; upload_ms=[]; details=[]
for doc in manifest['documents']:
    source=ROOT/'data'/doc['file']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==doc['sha256']
    start=time.perf_counter()
    response=client.post('/api/documents', headers=headers,
        data={'doc_id':doc['id'],'title':doc['title']},
        files={'file':(source.name,source.read_bytes(),'image/png')})
    upload_ms.append((time.perf_counter()-start)*1000)
    assert response.status_code==200, response.text
    result=response.json()
    comparison={k:result['fields'].get(k)==v for k,v in doc['fields'].items()}
    matches+=sum(comparison.values()); field_total+=len(comparison)
    details.append({'doc_id':doc['id'], 'fields':comparison,'source_sha256':doc['sha256']})
with (ROOT/'data/portfolio.csv').open() as f:
    rows=list(csv.DictReader(f))
response=client.post('/api/records', headers=headers,json={'rows':rows})
assert response.status_code==200,response.text
records=response.json()
issues=client.get('/api/issues',headers=headers).json()
observed=dict(Counter(i['code'] for i in issues))
doc_hits=0; value_hits=0; retrieval_ms=[]; query_results=[]
for q in manifest['questions']:
    start=time.perf_counter()
    response=client.post('/api/search',headers=headers,json={'question':q['question']})
    retrieval_ms.append((time.perf_counter()-start)*1000)
    assert response.status_code==200,response.text
    hits=response.json()
    correct=bool(hits and hits[0]['doc_id']==q['doc_id'])
    found=any(q['expected_text'] in h['text'] for h in hits if h['doc_id']==q['doc_id'])
    doc_hits+=correct;value_hits+=found
    query_results.append({**q,'correct_document_at_1':correct,'answer_value_present_at_3':found})
review=client.post('/api/ask',headers=headers,json={'question':manifest['questions'][0]['question']}).json()
unknown=client.post('/api/ask',headers=headers,json={'question':'What is the principal for ACCT-999 in 2026-Q2?'}).json()
quotes=client.post('/api/ask',headers=headers,json={'question':manifest['questions'][2]['question']}).json()
results={
    'generated_at_utc':datetime.now(timezone.utc).isoformat(),
    'environment':{'python':platform.python_version(),'platform':platform.platform(),
                   'tesseract':subprocess.run(['tesseract','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]},
    'dataset_sha256':hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest(),
    'scope':{'synthetic_documents':len(details),'structured_rows':len(rows),'queries':len(query_results)},
    'ocr':{'matched_fields':matches,'total_fields':field_total,'exact_match_rate':matches/field_total,
           'median_ingestion_ms':round(statistics.median(upload_ms),2),'documents':details},
    'validation':{'summary':records,'observed_issues':observed,
                  'all_seeded_issues_detected':observed==manifest['expected_issues']},
    'retrieval':{'correct_document_at_1':doc_hits,'answer_value_present_at_3':value_hits,
                 'total_queries':len(query_results),'median_ms':round(statistics.median(retrieval_ms),2),
                 'method':'TF-IDF with explicit account/period metadata filters','results':query_results},
    'controls':{'mismatch_blocks_answer':review['status']=='needs_review',
                'unknown_account_returns_no_evidence':unknown['status']=='no_evidence',
                'valid_answer_includes_source_hash_and_page':bool(quotes['citations'] and quotes['citations'][0]['source_sha'] and quotes['citations'][0]['page']==1)},
    'database_stats':client.get('/api/stats',headers=headers).json(),
    'cloud_execution':{'textract':False,'bedrock':False,'snowflake_cortex':False,'ecs_deployed':False},
    'limitations':[manifest['benchmark_limit'],'Local latency is not production throughput or a cloud SLA.',
                   'No LLM factuality or live cloud accuracy measured. TF-IDF is lexical retrieval, not semantic embeddings.']
}
output=ROOT/'evaluation/results.json';output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(results,indent=2))
assert matches==field_total, 'OCR did not match every fixture field; inspect results'
assert observed==manifest['expected_issues'], 'Seeded issue detection mismatch'
assert all(results['controls'].values()), 'Control failure'
print(json.dumps({k:v for k,v in results.items() if k in ['scope','validation','controls','database_stats']},indent=2))
print(f'OCR {matches}/{field_total}; retrieval documents {doc_hits}/48; answer values {value_hits}/48')
