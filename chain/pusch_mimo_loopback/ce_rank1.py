#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np

CASE="case_0_m64_k1_r96"
WEIGHTS=("factor_b_re.bin","factor_b_im.bin","cube_time_fused_d0_re.bin",
         "cube_time_fused_d0_im.bin","cube_time_fused_d1_re.bin",
         "cube_time_fused_d1_im.bin","ce_pack_gather_index.bin","weight_model.bin")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(root:Path)->None:
    gold=root/"golden"/CASE; manifest={"schema_version":1,"operator":"channel_est_lmmse_mimo","profile":{"nr":64,"layers":1,"rank":96,"observation":"COMB2_798","variant":"cube_time_fused"},"weights":{}}
    for name in WEIGHTS:
        p=gold/name
        if not p.is_file() or p.stat().st_size==0 or not any(p.read_bytes()):
            raise RuntimeError(f"missing/empty/all-zero CE weight: {p}")
        manifest["weights"][name]={"bytes":p.stat().st_size,"sha256":sha(p)}
    (root/"weight_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
def link(root:Path,ls:Path,slot:int)->None:
    gold=root/"golden"/CASE
    for src_name,dst_name in ((f"slot{slot:02d}_h_re.bin","h_ls_re.bin"),(f"slot{slot:02d}_h_im.bin","h_ls_im.bin"),("pilot_count.bin","pilot_count.bin"),("pilot_sc.bin","pilot_sc.bin")):
        source=ls/src_name; target=gold/dst_name
        if target.name.startswith("pilot_") and target.read_bytes()!=source.read_bytes():
            raise RuntimeError(f"CE weight observation model disagrees with DMRS-LS: {target.name}")
        shutil.copyfile(source,target)
    for name,count in (("h_ls_re.bin",64*1*2*832),("h_ls_im.bin",64*1*2*832)):
        a=np.fromfile(gold/name,np.float16)
        if a.size!=count or not np.any(a):raise RuntimeError(f"invalid actual CE input {name}")
def collect(root:Path,out:Path,slot:int)->None:
    src=root/"ascend_output"/CASE
    out.mkdir(parents=True,exist_ok=True)
    for plane in ("re","im"):
        p=src/f"h_cube_time_fused_{plane}.bin"
        a=np.fromfile(p,np.float16)
        if a.size!=64*16*14*1664:raise RuntimeError("CE output shape mismatch")
        if not np.any(a.reshape(64,16,14,1664)[:,0,:,:1596]):raise RuntimeError("CE active layer all-zero")
        if np.any(a.reshape(64,16,14,1664)[:,1:]):raise RuntimeError("CE inactive detector layers nonzero")
        shutil.copyfile(p,out/f"slot{slot:02d}_h_{plane}.bin")
def main():
 p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","link","collect"));p.add_argument("--root",type=Path,required=True);p.add_argument("--ls",type=Path);p.add_argument("--out",type=Path);p.add_argument("--slot",type=int,default=0);a=p.parse_args()
 if a.action=="prepare":prepare(a.root)
 elif a.action=="link":link(a.root,a.ls,a.slot)
 else:collect(a.root,a.out,a.slot)
if __name__=="__main__":main()
