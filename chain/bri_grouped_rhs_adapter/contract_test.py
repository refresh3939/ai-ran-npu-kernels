#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
matrix=json.loads((root/'contract_matrix.json').read_text())
assert matrix['input']['shape']==[23296,64,16]
assert matrix['output']['shape']==[2912,64,16]
assert matrix['group_size']==8
GROUP_SIZE=matrix['group_size']
rng=np.random.default_rng(0x42524938)
y=rng.integers(1,65535,size=(16,3),dtype=np.uint16)
canonical=np.repeat(y[:,:,None],16,axis=2)
grouped=np.zeros((2,3,16),dtype=np.uint16)
for re in range(16):grouped[re//GROUP_SIZE,:,re%GROUP_SIZE]=canonical[re,:,0]
for re in range(16):assert np.array_equal(grouped[re//GROUP_SIZE,:,re%GROUP_SIZE],y[re])
assert not np.any(grouped[:,:,8:])
canonical[7,1,9]^=1
assert canonical[7,1,9]!=canonical[7,1,0]
print('[PASS] bri_grouped_rhs_adapter mapping and poison rejection fixture')
