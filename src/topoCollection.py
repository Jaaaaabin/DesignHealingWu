#
# topoCollection.py
#

# import modules
from const_project import DIRS_DATA_TOPO, NAME_INSTANCE_COLLECTION, NAME_TOPO_INSTANCES, NAME_TOPO_PARAMETER
from funct_topo import *

def topoCollect():

    
    for topo_inst in NAME_TOPO_INSTANCES:

        # set JSON files
        js_file_instance =  DIRS_DATA_TOPO + NAME_INSTANCE_COLLECTION + topo_inst + '.json'

        # load the dict from Revit
        with open(js_file_instance, encoding="utf-8-sig") as json_file: # encoding = 'utf-8-sig' for special characters.
            revit_instances = json.load(json_file)
        
        # create class objects for each dict.
        cls_objs_instances = convert_revitdict_to_clsobjs(revit_instances, class_name=topo_inst)

        if topo_inst == 'parameter':

            # instances: global parameter
            df_instances = build_instance_df(cls_objs_instances, topo_inst, final_index_name='id')
            df_instances.to_csv(DIRS_DATA_TOPO+'\df_'+ topo_inst +'.csv')
            
            with open(DIRS_DATA_TOPO + NAME_TOPO_PARAMETER + 'host.txt', 'w') as f:
                df_as_string = df_instances['name'].to_string(header=False, index=False)
                df_as_string = df_as_string.replace(" ", "")
                f.write(df_as_string)
            with open(DIRS_DATA_TOPO + NAME_TOPO_PARAMETER + 'objects.txt', 'w') as f:
                df_as_string = df_instances['elementifcguid'].to_string(header=False, index=False)
                f.write(df_as_string)
            
        elif topo_inst =='separationline':

            # instances: room separation line
            df_instances = build_instance_df(cls_objs_instances, topo_inst, final_index_name='id')
            df_instances.index = df_instances.index.map(str) if df_instances.index.is_integer() else df_instances.index
            df_instances.to_csv(DIRS_DATA_TOPO+'\df_'+ topo_inst +'.csv')
            
        else:
            # instances: non-parameter building elements/spaces.
            # df_instances = build_instance_df(cls_objs_instances, topo_inst, final_index_name='ifcguid')
            df_instances = build_instance_df(cls_objs_instances, topo_inst, final_index_name='id')
            df_instances.to_csv(DIRS_DATA_TOPO+'\df_'+ topo_inst +'.csv', encoding = 'utf-8-sig') # encoding = 'utf-8-sig' for special characters.