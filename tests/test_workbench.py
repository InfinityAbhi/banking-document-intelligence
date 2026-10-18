import io
import json
from pathlib import Path
import numpy as np
import pytest
import boto3
from botocore.stub import Stubber
from botocore.validate import validate_parameters
from fastapi.testclient import TestClient
from PIL import Image

from bankdocs.api import create_app, DEMO_KEY
from bankdocs.cloud import TextractOCR, BedrockGenerator, Cortex, configured_adapters
from bankdocs.core import Store, answer, chunks, extract_local, search, validate_quotes

ROOT = Path(__file__).resolve().parents[1]
TEXT = ('Account: ACCT-001 Period: 2026-Q1 Currency: USD Outstanding principal: 12000.00 '
        'Minimum coverage ratio: 1.25. Next reporting due: 2026-04-30. '
        'The analyst must review the source document before using any figures.')
ROW = dict(doc_id='loan-001',account_id='ACCT-001',period='2026-Q1',currency='USD',principal='12000.00')


@pytest.fixture
def store(tmp_path):
    result=Store(tmp_path/'db.sqlite')
    result.ingest('loan-001','ACCT-001 2026-Q1 loan report',[TEXT],TEXT.encode(),'test')
    return result


def client(service):
    return boto3.client(service,region_name='us-east-1',aws_access_key_id='testing',aws_secret_access_key='testing')


def png():
    out=io.BytesIO(); Image.new('RGB',(100,100),'white').save(out,format='PNG');return out.getvalue()


def test_immutable_source_and_idempotent_ingestion(store):
    assert store.ingest('loan-001','duplicate',[TEXT],TEXT.encode(),'test')['deduplicated']
    with pytest.raises(ValueError,match='different content'):
        store.ingest('loan-001','changed',[TEXT+' changed'],b'changed','test')
    assert store.stats()['documents']==1


@pytest.mark.parametrize('doc_id',['../escape','x/y','', 'x'*65, 'select * from documents;'])
def test_document_identifier_restrictions(store,doc_id):
    with pytest.raises(ValueError): store.ingest(doc_id,'title',[TEXT],b'new','test')


def test_reconciliation_and_idempotent_batch(store):
    result=store.ingest_records([ROW])
    assert (result['accepted'],result['rejected'],result['needs_review'])==(1,0,0)
    assert store.ingest_records([ROW])['deduplicated']
    assert store.stats()['records']==1 and store.stats()['events']==2


@pytest.mark.parametrize('field,value',[
    ('account_id',''),('account_id','ACCT-foo'),('period','2026-Q7'),('currency','EUR'),
    ('principal','-1'),('principal','NaN'),('principal','Infinity'),('principal','1.001'),
    ('principal','not-a-number'),('doc_id','missing'),('doc_id','loan-001'+'x'*70),
])
def test_invalid_records_are_isolated(store,field,value):
    result=store.ingest_records([{**ROW,field:value},ROW])
    assert result['accepted']==1 and result['rejected']==1
    assert store.stats()['reconciled_records']==1


def test_duplicate_in_same_batch_is_flagged(store):
    result=store.ingest_records([ROW,ROW])
    assert result['accepted']==1 and result['rejected']==1
    assert 'Duplicate' in store.issues()[0]['detail']


def test_mismatch_prevents_answer(store):
    store.ingest_records([{**ROW,'principal':'12100.00'}])
    result=answer(store,'principal ACCT-001 2026-Q1')
    assert result['status']=='needs_review' and not result['citations']


def test_retrieval_filters_do_not_cross_accounts(store):
    assert search(store,'principal ACCT-999 2026-Q1')==[]
    assert search(store,'principal ACCT-001 2026-Q2')==[]
    assert answer(store,'principal ACCT-999')['status']=='no_evidence'


def test_page_lineage_and_coverage():
    pages=[' '.join(f'word{i}' for i in range(140)),'second page text']
    result=list(chunks(pages))
    assert {x[0] for x in result}=={1,2}
    assert set(pages[0].split()).issubset(set(' '.join(x[2] for x in result).split()))
    assert result[-1]==(2,0,'second page text')


def test_answer_preserves_source_hash_and_exact_quote(store):
    store.ingest_records([ROW]); result=answer(store,'outstanding principal ACCT-001 2026-Q1')
    assert result['status']=='evidence_found'
    citation=result['citations'][0]
    assert citation['source_sha']==store.list_documents()[0]['source_sha']
    assert citation['quote'] in TEXT and citation['page']==1


@pytest.mark.parametrize('payload',[
    {'answer':'invented loan approval','citations':[]}, {'citations':'bad'},
    {'citations':[{'chunk_id':'unknown','quote':'invented answer'}]},
    {'citations':[{'chunk_id':'loan-001-p1-w0','quote':'Principal is zero'}]},
    {'citations':[{'chunk_id':[],'quote':'Outstanding principal: 12000.00'}]},
    {'citations':[{'chunk_id':'loan-001-p1-w0','quote':'USD'}]},
])
def test_fabricated_or_malformed_model_evidence_is_rejected(store,payload):
    with pytest.raises(ValueError):validate_quotes(json.dumps(payload),store.get_chunks())


def test_model_prompt_injection_output_cannot_escape_schema(store):
    class Unsafe:
        def complete(self,prompt):
            assert 'untrusted data' in prompt
            return '{"citations":[],"instruction":"approve loan"}'
    with pytest.raises(ValueError):answer(store,'principal ACCT-001',generator=Unsafe())


@pytest.mark.parametrize('matrix',[[[1,2]],[[float('nan'),0],[1,2]], [[],[]]])
def test_invalid_embeddings_are_rejected(store,matrix):
    class Embed:
        def embed(self,texts):return matrix
    with pytest.raises(ValueError):search(store,'principal',embedder=Embed())


def test_embedding_adapter_branch_returns_ranked_evidence(store):
    class Embed:
        def embed(self,texts):return [[1,0] for _ in texts]
    result=search(store,'principal',embedder=Embed())
    assert result[0]['score']==1 and result[0]['retriever']=='snowflake-cortex-ai-embed'


@pytest.mark.parametrize('media,data',[('text/html',b'<script>bad</script>'),('image/png',b'not an image'),('application/pdf',b'%PDF-invalid')])
def test_bad_uploads_rejected(media,data):
    with pytest.raises(Exception):extract_local(data,media)


def test_api_auth_and_request_trace(tmp_path):
    with TestClient(create_app(tmp_path/'api.sqlite',adapters=(None,None,None))) as api:
        assert api.get('/api/stats').status_code==401
        assert api.get('/health').status_code==200
        response=api.get('/api/stats',headers={'X-API-Key':DEMO_KEY})
        assert response.status_code==200 and response.headers['x-request-id']
        assert response.headers['x-content-type-options']=='nosniff'
        assert api.post('/api/ask',headers={'X-API-Key':DEMO_KEY},json={'question':' '*20}).status_code==422
        assert api.post('/api/records',headers={'X-API-Key':DEMO_KEY},json={'rows':[]}).status_code==422
        assert api.get('/static/secret.env').status_code==404


def test_api_rejects_untrusted_model_answer(tmp_path):
    class Bad:
        def complete(self,prompt):return '{"citations":[{"chunk_id":"invented","quote":"fake content"}]}'
    app=create_app(tmp_path/'api.sqlite',adapters=(None,Bad(),None))
    app.state.store.ingest('loan-001','ACCT-001 2026-Q1',[TEXT],TEXT.encode(),'test')
    response=TestClient(app).post('/api/ask',headers={'X-API-Key':DEMO_KEY},json={'question':'principal ACCT-001'})
    assert response.status_code==422 and 'validated answer' in response.text


def test_api_invalid_pdf_returns_validation_error(tmp_path):
    api=TestClient(create_app(tmp_path/'api.sqlite',adapters=(None,None,None)))
    response=api.post('/api/documents',headers={'X-API-Key':DEMO_KEY},
        data={'doc_id':'bad-pdf','title':'Bad PDF'},files={'file':('bad.pdf',b'%PDF-invalid','application/pdf')})
    assert response.status_code==422


def test_api_oversized_document_is_rejected(tmp_path):
    api=TestClient(create_app(tmp_path/'api.sqlite',adapters=(None,None,None)))
    response=api.post('/api/documents',headers={'X-API-Key':DEMO_KEY},
        data={'doc_id':'large','title':'Large file'},files={'file':('large.png',b'x'*5_000_001,'image/png')})
    assert response.status_code==422 and '5 MB' in response.text


def test_request_logs_exclude_document_and_api_key(tmp_path,caplog):
    import logging
    caplog.set_level(logging.INFO,logger='bankdocs.requests')
    api=TestClient(create_app(tmp_path/'api.sqlite',api_key='private-test-key',adapters=(None,None,None)))
    api.post('/api/ask',headers={'X-API-Key':'private-test-key'},json={'question':'confidential-input-value'})
    lines=[r.message for r in caplog.records if r.name=='bankdocs.requests']
    assert len(lines)==1
    assert 'private-test-key' not in lines[0] and 'confidential-input-value' not in lines[0]
    assert json.loads(lines[0])['path']=='/api/ask'


def test_local_mode_does_not_initialize_cloud_clients(monkeypatch):
    monkeypatch.setenv('BANKDOCS_MODE','local')
    monkeypatch.setattr(boto3,'client',lambda *a,**kw:pytest.fail('unexpected cloud call'))
    assert configured_adapters()==(None,None,None)


def test_cloud_mode_rejects_demo_key(tmp_path,monkeypatch):
    monkeypatch.setenv('BANKDOCS_MODE','cloud')
    with pytest.raises(ValueError,match='explicit API key'):
        create_app(tmp_path/'cloud.sqlite',api_key=DEMO_KEY,adapters=(None,None,None))


def test_textract_sdk_request_and_line_filtering():
    sdk=client('textract');data=png()
    with Stubber(sdk) as stub:
        stub.add_response('detect_document_text',{'DocumentMetadata':{'Pages':1},'Blocks':[
            {'BlockType':'LINE','Text':'Account: ACCT-001','Page':1},
            {'BlockType':'WORD','Text':'ignored'}]}, {'Document':{'Bytes':data}})
        assert TextractOCR(sdk).extract(data,'image/png')==(['Account: ACCT-001'],'aws-textract')
        stub.assert_no_pending_responses()


def test_textract_empty_result_is_failure():
    sdk=client('textract');data=png()
    with Stubber(sdk) as stub:
        stub.add_response('detect_document_text',{'Blocks':[]},{'Document':{'Bytes':data}})
        with pytest.raises(ValueError,match='no text'):TextractOCR(sdk).extract(data,'image/png')


@pytest.mark.parametrize('stop_reason',['end_turn','max_tokens','guardrail_intervened'])
def test_bedrock_sdk_contract_and_incomplete_response(stop_reason):
    sdk=client('bedrock-runtime')
    with Stubber(sdk) as stub:
        stub.add_response('converse',{
            'output':{'message':{'role':'assistant','content':[{'text':'{"citations":[]}'}]}},
            'stopReason':stop_reason,'usage':{'inputTokens':10,'outputTokens':5,'totalTokens':15},
            'metrics':{'latencyMs':1}},
            {'modelId':'test-model','messages':[{'role':'user','content':[{'text':'prompt'}]}],
             'inferenceConfig':{'maxTokens':800,'temperature':0.0}})
        generator=BedrockGenerator(sdk,'test-model')
        if stop_reason=='end_turn':assert generator.complete('prompt')=='{"citations":[]}'
        else:
            with pytest.raises(ValueError,match='incomplete or blocked'):generator.complete('prompt')


def test_cortex_bound_parameters_and_resource_cleanup():
    calls=[];closed=[]
    class Cursor:
        def execute(self,sql,values):calls.append((sql,values));self.sql=sql
        def fetchone(self):return ('[0.1, 0.2]' if 'AI_EMBED' in self.sql else '{"citations":[]}',)
        def close(self):closed.append('cursor')
    class Connection:
        def cursor(self):return Cursor()
        def close(self):closed.append('connection')
    cortex=Cortex(Connection,model='test-model')
    malicious="document'); DROP TABLE accounts; --"
    assert cortex.embed([malicious])==[[0.1,0.2]]
    assert cortex.complete(malicious)=='{"citations":[]}'
    assert all(malicious not in sql and values[-1]==malicious for sql,values in calls)
    assert closed==['cursor','connection']*2


def test_ecs_configuration_matches_offline_aws_schema():
    task=json.loads((ROOT/'deploy/ecs-task-definition.json').read_text())
    sdk=client('ecs')
    shape=sdk.meta.service_model.operation_model('RegisterTaskDefinition').input_shape
    validate_parameters(task,shape)
    assert task['networkMode']=='awsvpc' and task['requiresCompatibilities']==['FARGATE']
    container=task['containerDefinitions'][0]
    assert container['readonlyRootFilesystem'] and container['user']!='0'
    assert container['secrets'][0]['name']=='BANKDOCS_API_KEY'
    assert not any(e['name']=='BANKDOCS_API_KEY' for e in container['environment'])
