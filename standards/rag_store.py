from __future__ import annotations
import json,pickle,re
from pathlib import Path
from typing import Any
import numpy as np
from models.engineering_contract import Criterion, CriterionType
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'standards'; INDEX=DATA/'index'

def _norm(s:str)->str:return re.sub(r'[^a-z0-9_]+',' ',str(s).lower()).strip()

def _criterion_type(name:str,parameter:str,topic:str='')->CriterionType:
    n=f'{name} {parameter} {topic}'.lower()
    if 'hazen' in n or 'hazen_williams_c' in n or parameter.lower()=='hazen_williams_c': return CriterionType.MATERIAL
    if 'velocity' in n: return CriterionType.VELOCITY
    if 'pressure' in n or 'prv' in n: return CriterionType.PRESSURE
    if 'friction' in n or 'head loss' in n or 'headloss' in n or 'darcy' in n: return CriterionType.HEAD_LOSS
    if 'demand' in n or 'flow' in n: return CriterionType.DEMAND
    if 'hot' in n or 'recirculation' in n or 'temperature' in n: return CriterionType.HOT_WATER
    if 'npsh' in n:return CriterionType.NPSH
    if 'method' in n or 'hydraulic' in n:return CriterionType.HYDRAULIC_METHOD
    if 'pump' in n:return CriterionType.PUMP
    if 'valve' in n or 'cutoff' in n:return CriterionType.CONTROL
    return CriterionType.OTHER

def _load_json(path:Path,default): return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

class RAGStore:
    def __init__(self,*,top_k:int=8):
        self.top_k=top_k; self.criteria=self._load_criteria(); self.chunks=self._load_chunks(); self.sources={x['source_id']:x for x in _load_json(DATA/'sources'/'source_registry.json',[])}; self.vectorizer=None; self.matrix=None; self.metadata=[]; self._load_or_build_index()
    def _load_criteria(self):
        rows=_load_json(DATA/'criteria'/'criteria_registry_v2.json',[]); out=[]
        for r in rows:
            try:
                val=r.get('value'); rng=val if isinstance(val,list) and len(val)==2 and all(isinstance(x,(int,float)) for x in val) else None
                out.append(Criterion(criterion_id=str(r['criterion_id']),criterion_type=_criterion_type(str(r.get('name','')),str(r.get('parameter',r.get('name',''))),str(r.get('topic',''))),parameter=str(r.get('parameter') or r.get('name') or r['criterion_id']),value=val if isinstance(val,(int,float)) else None,min_value=rng[0] if rng else r.get('min_value'),max_value=rng[1] if rng else r.get('max_value'),unit=r.get('unit'),application=r.get('application'),service=r.get('service'),jurisdiction=r.get('jurisdiction') or r.get('source_location'),applicability=str(r.get('applicability') or r.get('scope') or r.get('applicability_context') or 'source-defined'),method=r.get('method'),source_reference_id=str(r.get('source_id') or 'UNSPECIFIED'),source_status=str(r.get('source_status') or r.get('status') or 'UNVERIFIED'),edition_year=str(r.get('edition_year') or '') or None,section=r.get('section') or r.get('reference'),page=r.get('page'),notes=str(r.get('notes') or ''),system=r.get('system'),subsystem=r.get('subsystem'),flow_type=r.get('flow_type'),topic=r.get('topic'),source_location=r.get('source_location'),applicability_context=r.get('applicability_context'),retrieval_tags=r.get('retrieval_tags',[])))
            except Exception: continue
        return out
    def _load_chunks(self):
        p=DATA/'corpus'/'chunks_v2.jsonl'; return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []
    def _load_or_build_index(self):
        pkl=INDEX/'tfidf_vectorizer.pkl'; npz=INDEX/'tfidf_matrix.npz'; meta=INDEX/'metadata.jsonl'
        if pkl.exists() and npz.exists() and meta.exists():
            from scipy.sparse import load_npz
            self.vectorizer=pickle.loads(pkl.read_bytes()); self.matrix=load_npz(npz).tocsr(); self.metadata=[json.loads(x) for x in meta.read_text(encoding='utf-8').splitlines() if x.strip()]; return
        self._build_index()
    def _build_index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import save_npz
        records=[]
        for c in self.chunks:
            txt=' '.join(str(c.get(k) or '') for k in ['heading','text','topic','system','subsystem','flow_type','application','source_location','applicability_context'])+' '+' '.join(c.get('retrieval_tags',[])); records.append({'record_id':c['chunk_id'],'record_type':'chunk','text':txt,'metadata':c})
        for c in self.criteria:
            txt=' '.join(str(c.model_dump(mode='json').get(k) or '') for k in ['criterion_id','parameter','application','jurisdiction','applicability','notes','system','subsystem','flow_type','topic','source_location'])+' '+' '.join(c.retrieval_tags); records.append({'record_id':c.criterion_id,'record_type':'criterion','text':txt,'metadata':c.model_dump(mode='json')})
        self.metadata=records; self.vectorizer=TfidfVectorizer(ngram_range=(1,2),lowercase=True,norm='l2'); self.matrix=self.vectorizer.fit_transform([r['text'] for r in records]).tocsr(); INDEX.mkdir(parents=True,exist_ok=True); (INDEX/'tfidf_vectorizer.pkl').write_bytes(pickle.dumps(self.vectorizer)); save_npz(INDEX/'tfidf_matrix.npz',self.matrix); (INDEX/'metadata.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in records)+'\n',encoding='utf-8')
    def _topic_candidates(self,q):
        q=_norm(q); mapping={'PIPE_VELOCITY':['velocity','velocities','speed'],'PIPE_FRICTION':['hazen','friction','head loss','headloss','darcy'],'PIPE_SIZING':['pipe size','diameter','sizing','size selection'],'PRESSURE':['pressure','residual','terminal','prv'],'DEMAND_FLOW':['flow','demand','peak','transfer time'],'STORAGE':['tank','oht','ugt','storage','reservoir'],'PUMP_HEAD':['pump head','tdh','total dynamic head'],'PUMP_PROTECTION':['booster','cutoff','relief'],'SOURCE_CAPACITY':['well yield','yield','source'],'VALVES_AIR_MANAGEMENT':['air valve','check valve','non return'],'HOT_WATER_RECIRCULATION':['hot water','recirculation','return temperature'],'NPSH':['npsh','cavitation'],'PIPE_MATERIAL':['material','steel','hdpe','pvc','ppr','copper'],'WATER_QUALITY_PROTECTION':['backflow','potable','cross connection'],'GOVERNING_CODE':['code','ahj','ipc','bcp']}; return {t for t,terms in mapping.items() if any(x in q for x in terms)}
    def _application_context(self,application,service,query,material):
        a=(application or '').upper(); q=(query or '').lower(); topics=self._topic_candidates(q)
        if a=='TRANSFER': topics|={'PIPE_VELOCITY','PIPE_FRICTION','PIPE_SIZING','PUMP_HEAD','STORAGE','DEMAND_FLOW'}
        elif a=='BOOSTER': topics|={'PIPE_VELOCITY','PIPE_FRICTION','PIPE_SIZING','PRESSURE','PUMP_HEAD'}
        elif a=='SUBMERSIBLE': topics|={'PIPE_VELOCITY','PIPE_FRICTION','PIPE_SIZING','PUMP_HEAD','SOURCE_CAPACITY','NPSH'}
        elif a=='HOT_WATER_RECIRCULATION': topics|={'HOT_WATER_RECIRCULATION','PIPE_FRICTION','PIPE_SIZING','PUMP_HEAD'}
        subsystem=None
        if a=='TRANSFER': subsystem='RISING_MAIN'
        elif a=='BOOSTER': subsystem='BOOSTER_SYSTEM'
        elif a=='HOT_WATER_RECIRCULATION': subsystem='HOT_WATER_LOOP'
        return topics,subsystem
    def _search(self,query,top_k):
        q=self.vectorizer.transform([query]); scores=(self.matrix@q.T).toarray().ravel(); order=np.argsort(-scores); return [dict(self.metadata[i],score=float(scores[i])) for i in order[:top_k] if scores[i]>0]
    def retrieve(self,query,*,application=None,service=None,jurisdiction=None,system='WATER_SUPPLY',flow_type='PRESSURIZED',topic=None,subsystem=None,material=None,top_k=None):
        app_topics,app_subsystem=self._application_context(application,service,query,material); topics={topic} if topic else (self._topic_candidates(query) | app_topics); subsystem=subsystem or app_subsystem
        # Search once per requested engineering topic, then merge. This prevents
        # a generic storage/keyword match from crowding out critical velocity or
        # friction criteria.
        queries=[query+' '+str(material or '')]
        queries += [f"{query} {t} {material or ''}" for t in sorted(topics)]
        pool={}
        for qq in queries:
            for r in self._search(qq,max(12,(top_k or self.top_k)*2)):
                rid=(r.get('record_type'),r.get('record_id'))
                if rid not in pool or r['score']>pool[rid]['score']:
                    pool[rid]=r
        # Add metadata-matching criteria deterministically so critical topics cannot be lost to TF-IDF wording.
        for c in self.criteria:
            md=c.model_dump(mode='json')
            if md.get('system') and md.get('system')!=system: continue
            if flow_type and md.get('flow_type') and md.get('flow_type') not in {flow_type,'PRESSURIZED_FORCE_MAIN'}: continue
            if topics and md.get('topic') not in topics: continue
            score=0.02
            tags=' '.join(str(x) for x in c.retrieval_tags).lower()
            name=f"{c.parameter} {c.applicability} {c.application or ''}".lower()
            if material:
                mt=material.lower(); aliases=[]
                if 'hdpe' in mt or 'pe100' in mt or re.search(r'\bpe\b',mt): aliases=['hdpe','pe / polyethylene','plastic','plastic pipe']
                elif 'pvc' in mt: aliases=['pvc','plastic','plastic pipe']
                elif 'ppr' in mt or 'pp-r' in mt: aliases=['ppr','plastic','plastic pipe']
                elif 'steel' in mt or 'ms' in mt or 'carbon' in mt: aliases=['steel','di/ms']
                elif 'copper' in mt: aliases=['copper','copper tubing']
                if any(a in tags or a in name for a in aliases): score += 0.50
            if subsystem and c.subsystem==subsystem: score += 0.25
            rid=('criterion',c.criterion_id)
            if rid not in pool or score>pool[rid].get('score',0): pool[rid]={'record_id':c.criterion_id,'record_type':'criterion','text':'','metadata':md,'score':score}
        candidates=sorted(pool.values(),key=lambda x:-x['score'])
        selected=[]
        for r in candidates:
            m=r.get('metadata',{}); ms=m.get('system')
            if ms and ms!=system: continue
            mf=m.get('flow_type')
            if flow_type and mf and mf not in {flow_type,'PRESSURIZED_FORCE_MAIN'}: continue
            mt=m.get('topic')
            if topics and mt not in topics: continue
            if subsystem and m.get('subsystem') and m.get('subsystem')!=subsystem and m.get('subsystem') not in {'DISTRIBUTION_MAIN',None}: continue
            bonus=0.0
            tags=' '.join(str(x) for x in m.get('retrieval_tags',[])).lower(); name=str(m.get('name','')).lower()
            if material and any(tok in (tags+' '+name) for tok in _norm(material).split() if len(tok)>2): bonus+=0.35
            if application and application.lower() in str(m.get('application','')).lower(): bonus+=0.10
            if jurisdiction and jurisdiction.lower() in str(m.get('source_location','')).lower(): bonus+=0.05
            r['score']+=bonus; selected.append(r)
        selected=sorted(selected,key=lambda x:-x['score'])[:(top_k or self.top_k)]
        crit=[]; refs={}
        for r in selected:
            if r['record_type']=='criterion':
                c=next((x for x in self.criteria if x.criterion_id==r['record_id']),None)
                if c: crit.append(c.model_dump(mode='json')); refs[c.source_reference_id]=self.sources.get(c.source_reference_id)
        return {'criteria':crit,'chunks':[r.get('metadata',{})|{'score':r.get('score')} for r in selected if r['record_type']=='chunk'],'references':[x for x in refs.values() if x]}
    def retrieve_criteria(self,*,application,service=None,jurisdiction=None,query='engineering criteria',topic=None,subsystem=None,material=None):
        return [Criterion.model_validate(x) for x in self.retrieve(f'{application} {service or ""} {query} {material or ""}',application=application,service=service,jurisdiction=jurisdiction,topic=topic,subsystem=subsystem,material=material)['criteria']]
    def retriever(self,**kwargs): return self.retrieve_criteria(**kwargs)
