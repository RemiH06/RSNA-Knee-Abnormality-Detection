import io, sys, zlib, struct, collections, requests, pandas as pd, pydicom
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
api=KaggleApi(); api.authenticate()
with api.build_kaggle_client() as k:
    r=ApiDownloadDataFilesRequest(); r.competition_name='rsna-knee-abnormality-detection'
    resp=k.competitions.competition_api_client.download_data_files(r); url=resp.url; resp.close()
z=pd.read_csv(sys.argv[1]); z=z[z.name.str.endswith('.dcm')].sample(40,random_state=0)
c=collections.Counter()
for _,m in z.iterrows():
    h=requests.get(url,headers={'Range':f'bytes={m.offset}-{m.offset+29}'}).content
    n,e=struct.unpack('<HH',h[26:30]); start=m.offset+30+n+e
    raw=requests.get(url,headers={'Range':f'bytes={start}-{start+m.csize-1}'}).content
    d=pydicom.dcmread(io.BytesIO(zlib.decompress(raw,-15)))
    ok='ok'
    try: d.pixel_array
    except Exception as ex: ok=type(ex).__name__
    c[(d.file_meta.TransferSyntaxUID.name[:28],d.Rows,d.Columns,ok)]+=1
print(collections.Counter((a,o) for (a,_,_,o) in c.elements())); import statistics; print(sorted(r for (_,r,_,_) in c.elements()))
