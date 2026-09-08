#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 info=np.fromfile(r/'artifacts/tx_bits.bin',np.int8)
 if info.size!=143*8448 or not np.any(info):raise RuntimeError('tx_bits missing/wrong/all-zero')
 matched=np.fromfile(r/'decoder/output_matched/decoded_bits.bin',np.int8)
 wrong=np.fromfile(r/'decoder/output_wrong/decoded_bits.bin',np.int8)
 if matched.size!=info.size or wrong.size!=info.size:raise RuntimeError('decoder output shape mismatch')
 me=int(np.count_nonzero(matched!=info));we=int(np.count_nonzero(wrong!=info));mb=me/info.size;wb=we/info.size
 result={'schema_version':1,'status':'PASS' if me==0 and .4<=wb<=.6 else 'HARD_FAIL','rank':1,'slots':23,
  'matched_cell_id':321,'wrong_cell_id':322,'tx_bits_nonzero':int(np.count_nonzero(info)),'tx_bits_sha256':sha(r/'artifacts/tx_bits.bin'),
  'matched_info_errors':me,'matched_info_ber':mb,'wrong_cell_info_errors':we,'wrong_cell_info_ber':wb,
  'ldpc_input_stride':26112,'device_stages':['descramble_mimo','rate_dematch_mimo','ldpc_decode'],
  'complete_coded_e2e':me==0 and .4<=wb<=.6}
 out=r/'artifacts/rank1_triangle_result.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
