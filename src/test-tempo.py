
from base_external_packages import *
from funct_data import load_dict
from const_project import DIRS_DATA_TOPO

new_all = r'C:\dev\phd\ModelHealer\data\ss-134-7\DesignsNew.pickle'
designs_all = load_dict(new_all)
filtered_ids = load_dict(DIRS_DATA_TOPO + "/filtered_id.pickle")

kk = 0
filtered_designs_all = dict()

for i,t in enumerate(designs_all):

    failure = t.failures
    one_design_failure = dict()
    for k,v in failure.items():
        if k in filtered_ids:
            one_design_failure.update({k:v})
        else:
            continue
    filtered_designs_all.update({i:one_design_failure})


for ii,d in enumerate(filtered_designs_all.keys()):
    res = True
    fail_count = 0
    for k,v in filtered_designs_all[d].items():
        if v :
            res = False
            fail_count+=1
    
    if fail_count == 0:
        print (ii, str(fail_count))
print('t')
    
