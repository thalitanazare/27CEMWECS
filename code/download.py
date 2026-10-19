"""Download only the public SWELL records and optional OCEAN model used here."""
import argparse
import hashlib
import json
import subprocess
from common import ROOT


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',choices=['L0','L1','L2','L3','L4','L5'],action='append')
    p.add_argument('--ocean',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    records=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    records=[r for r in records if (r['layout'] in (args.layout or ['L0','L1','L2','L3','L4','L5']))
             or (args.ocean and r['layout']=='reference')]
    for r in records:
        target=ROOT/'data'/('reference' if r['layout']=='reference' else 'raw/'+r['layout'])/r['filename']
        if not target.exists() or sha256(target)!=r['sha256']:
            if args.verify_only:raise RuntimeError('Missing or corrupt data: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True)
            part=target.with_suffix(target.suffix+'.part')
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3',
                            '--max-time','300',r['url'],'-o',str(part)],check=True)
            if sha256(part)!=r['sha256']:raise RuntimeError('Checksum mismatch: '+str(part))
            part.replace(target)
        print('Verified',r['layout'],r['filename'],flush=True)

if __name__=='__main__':main()