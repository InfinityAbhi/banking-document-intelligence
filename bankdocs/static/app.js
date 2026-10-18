const byId = (id) => document.getElementById(id);
async function request(path, options = {}) {
  const response = await fetch(path, {...options, headers: {'X-API-Key': byId('key').value, ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function element(tag, text, className) {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function refresh() {
  try {
    const [stats, issues, health] = await Promise.all([request('/api/stats'), request('/api/issues'), request('/health')]);
    byId('mode').textContent = `${health.ocr} / ${health.retrieval} / ${health.generation}`;
    byId('stats').replaceChildren();
    for (const [key, label] of [['documents','Documents'],['chunks','Indexed chunks'],['reconciled_records','Reconciled records'],['issues','Exceptions']]) {
      const box = element('div', '', 'stat'); box.append(element('b', String(stats[key])),element('span',label)); byId('stats').append(box);
    }
    byId('issues').replaceChildren();
    for (const issue of issues) {
      const row = document.createElement('tr');
      for (const value of [issue.row_num,issue.doc_id,issue.code,issue.detail]) row.append(element('td',String(value)));
      byId('issues').append(row);
    }
  } catch (error) { byId('mode').textContent = error.message; }
}
byId('refresh').addEventListener('click', refresh);
byId('query-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = event.target.querySelector('button'); button.disabled = true;
  byId('answer').replaceChildren(element('p','Retrieving...'));
  try {
    const data = await request('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:byId('question').value})});
    byId('answer').replaceChildren(element('p',data.status.replaceAll('_',' '),data.status==='needs_review'?'warning':''));
    if (data.reason) byId('answer').append(element('p',data.reason));
    for (const citation of data.citations) {
      byId('answer').append(element('blockquote',citation.quote),element('p',`${citation.doc_id} / page ${citation.page} / source SHA-256 ${citation.source_sha}`,'source'));
    }
    if(data.limitation) byId('answer').append(element('p',data.limitation,'source'));
  } catch(error) { byId('answer').replaceChildren(element('p',error.message,'warning')); }
  finally {button.disabled=false;}
});
byId('upload-form').addEventListener('submit',async(event)=>{
  event.preventDefault(); const form=new FormData();form.append('doc_id',byId('doc-id').value);form.append('title',byId('doc-title').value);form.append('file',byId('doc-file').files[0]);
  try{byId('upload-result').textContent=JSON.stringify(await request('/api/documents',{method:'POST',body:form}),null,2);await refresh();}
  catch(error){byId('upload-result').textContent=error.message;}
});
refresh();
