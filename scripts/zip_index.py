"""Lee solo el directorio central del zip remoto (HTTP Range) y guarda el indice."""
import io, sys, zipfile, collections, csv, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
api=KaggleApi(); api.authenticate()
with api.build_kaggle_client() as k:
    r=ApiDownloadDataFilesRequest(); r.competition_name='rsna-knee-abnormality-detection'
    resp=k.competitions.competition_api_client.download_data_files(r); url=resp.url; resp.close()

class HttpFile(io.RawIOBase):
    def __init__(s,url):
        s.url=url; s.pos=0; s.size=int(requests.head(url).headers['Content-Length']); s.cache={}
        s.fetched=0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s,o,w=0):
        s.pos={0:o,1:s.pos+o,2:s.size+o}[w]; return s.pos
    def read(s,n=-1):
        if n<0: n=s.size-s.pos
        end=min(s.size,s.pos+n)-1
        if end<s.pos: return b''
        d=requests.get(s.url,headers={'Range':f'bytes={s.pos}-{end}'}).content
        s.fetched+=len(d); s.pos+=len(d); return d
    def readinto(s,b):
        d=s.read(len(b)); b[:len(d)]=d; return len(d)

f=HttpFile(url)
z=zipfile.ZipFile(io.BufferedReader(f,buffer_size=64<<20))
infos=z.infolist()
print('entries',len(infos),'fetched MB',f.fetched/1e6)
top=collections.defaultdict(lambda:[0,0,0])
with open(sys.argv[1],'w',newline='') as out:
    w=csv.writer(out); w.writerow(['name','offset','csize','usize','method'])
    for i in infos:
        t=i.filename.split('/')[0]; top[t][0]+=1; top[t][1]+=i.compress_size; top[t][2]+=i.file_size
        w.writerow([i.filename,i.header_offset,i.compress_size,i.file_size,i.compress_type])
for t,(n,c,u) in top.items(): print(t,n,'files comp GB',round(c/1e9,2),'uncomp GB',round(u/1e9,2))
