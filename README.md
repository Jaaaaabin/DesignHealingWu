# ModelHealer Project

## Introduction

Introduction for ModelHealer Project.

## Author

**By** Jiabin Wu at the [Chair of Computational Modeling and Simulation](https://www.cee.ed.tum.de/cms/home/), [Technical University of Munich](https://www.tum.de/)

## Project Structure

ModelHealer\\
|   .gitignore\
|   LICENSE\
|   main.py\
|   README.md\
|   tree.txt\
|
+---archive\
|   +---archivedyns\
|   |   |   00_ini_checking.dyn\
|   |   |   10_autoParametrizing.dyn\
|   |   |   11_delParameters.dyn\
|   |   |   21_extractTopologyObjectsParameters.dyn\
|   |   |   22_extractObjectsAndSpaces.dyn\
|   |   |   23_extractTopologyObjects.dyn\
|   |   |   24_extractTopologyObjectSpace.dyn\
|   |   |\
|   |   +---save_old\
|   |   |       calculateLongestExitDistance.dyn\
|   |   |       healing_jn_checking.dyn\
|   |   |\
|   |   \---save_old_healing\
|   |       |   healing_jn_adjusting.dyn\
|   |       |   healing_jn_checking.dyn\
|   |       |   healing_jn_modeling.dyn\
|   |       |\
|   |       \---rule_examples\
|   |           |   1 Press Play for More Information about Dynamo Player.dyn\
|   |           |   Calculate Longest Exit Distance.dyn\
|   |           |   Calculate Room Occupancy Load.dyn\
|   |           |   Calculate Total Length of Selected Lines.dyn\
|   |           |   Delete Mark Values from MEP Content.dyn\
|   |           |\
|   |           \---save\
|   |                   1 Press Play for More Information about Dynamo Player.dyn\
|   |                   Add Levels Above Selected Level (Imperial).dyn\
|   |                   Add Levels Above Selected Level (Metric).dyn\
|   |                   Calculate Longest Exit Distance.dyn\
|   |                   Calculate Room Occupancy Load.dyn\
|   |                   Calculate Total Length of Selected Lines.dyn\
|   |                   Delete Mark Values from MEP Content.dyn\
|   |                   Samples.zip\
|   |                   Select All Not Keynoted in the Active View.dyn\
|   |                   Select All Taggable Elements Not Tagged in the Active View.dyn\
|   |                   Update Sheet Names to Upper Case.dyn\
|   |\
|   \---archivenodes\
|           autoCheck.dyf\
|           autoParametrize.dyf\
|           extract-OB-SP.dyf\
|           extract-Topology-OB-OB.dyf\
|           extract-Topology-OB-PA.dyf\
|           extract-Topology-OB-SP.dyf\
|           varyGPs.dyf\
|
+---data\
|   |   control.rvt\
|   |   ini_gps_varytest.csv\
|   |\
|   +---data_0_ini\
|   +---data_1_param\
|   |   |   Execute_autoParametrize.dyn\
|   |   |   Execute_delParameters.dyn\
|   |   |\
|   |   \---res\
|   |           ini-parametrized-testE1.rvt\
|   |           ini-parametrized.rvt\
|   |\
|   +---data_2_topo\
|   |   |   31_collectData.py\
|   |   |   32_createNetworkx.py\
|   |   |   33_collectParameters.py\
|   |   |   34_duplicateModel.py\
|   |   |   Execute_extractTopology.dyn\
|   |   |   Execute_visualizeSelection.dyn\
|   |   |   summary_of_GPs.xlsx\
|   |   |\
|   |   \---res\
|   |       |   collected_GPs.json\
|   |       |   collected_Instances_Door.json\
|   |       |   collected_Instances_Slab.json\
|   |       |   collected_Instances_Space.json\
|   |       |   collected_Instances_Wall.json\
|   |       |   collected_Instances_Window.json\
|   |       |   collected_topology_GP_host.txt\
|   |       |   collected_topology_GP_objects.txt\
|   |       |   collected_topology_space_doors.txt\
|   |       |   collected_topology_space_host.txt\
|   |       |   collected_topology_space_seplines.txt\
|   |       |   collected_topology_space_walls.txt\
|   |       |   collected_topology_wall_host.txt\
|   |       |   collected_topology_wall_inserts.txt\
|   |       |   collected_topology_wall_slabs.txt\
|   |       |   collected_topology_wall_walls.txt\
|   |       |   df_AssociatedGPs.csv\
|   |       |   df_Doors.csv\
|   |       |   df_Parameters.csv\
|   |       |   df_Slabs.csv\
|   |       |   df_Spaces.csv\
|   |       |   df_Walls.csv\
|   |       |   df_Windows.csv\
|   |       |   ini_gps.csv\
|   |       |\
|   |       +---excludeLinksViaExternalWalls_k1\
|   |       |       df_AssociatedGPs.csv\
|   |       |       df_FailureNeighbors.csv\
|   |       |       df_InitialFailures.csv\
|   |       |\
|   |       +---excludeLinksViaExternalWalls_k2\
|   |       +---includeLinksViaExternalWalls_k1\
|   |       \---includeLinksViaExternalWalls_k2\
|   \---data_3_sa\
|       |   Execute_autoCheck.dyn\
|       |   Execute_varyGPs.dyn\
|       |   testnewchecking.py\
|       |\
|       +---dups\
|       |       1.rvt\
|       |       10.rvt\
|       |       11.rvt\
|       |       12.rvt\
|       |       13.rvt\
|       |       14.rvt\
|       |       15.rvt\
|       |       2.rvt\
|       |       3.rvt\
|       |       4.rvt\
|       |       5.rvt\
|       |       6.rvt\
|       |       7.rvt\
|       |       8.rvt\
|       |       9.rvt\
|       |\
|       +---res\
|       |       1.h5\
|       |       10.h5\
|       |       11.h5\
|       |       12.h5\
|       |       13.h5\
|       |       14.h5\
|       |       15.h5\
|       |       2.h5\
|       |       3.h5\
|       |       4.h5\
|       |       5.h5\
|       |       6.h5\
|       |       7.h5\
|       |       8.h5\
|       |       9.h5\
|       |\
|       \---vary\
|               1.rvt\
|               10.rvt\
|               11.rvt\
|               12.rvt\
|               13.rvt\
|               14.rvt\
|               15.rvt\
|               2.rvt\
|               3.rvt\
|               4.rvt\
|               5.rvt\
|               6.rvt\
|               7.rvt\
|               8.rvt\
|               9.rvt\
|
+---ini\
|   |   00_ini_checking.dyn\
|   |\
|   +---model\
|   |       ini.rvt\
|   |\
|   +---presteps\
|   |       00_BLD225071_Automating_Occupancy_Handout.pdf\
|   |       01_setSpaceFunctions.dyn\
|   |       02_copyOccupantLoadFactors.dyn\
|   |       setSpaceFunctions.xlsx\
|   |\
|   +---res\
|   |       0.h5\
|   |\
|   \---rules\
|           1010_1_1.pdf\
|           1017.pdf\
|           1020.pdf\
|           1207_3.pdf\
|           706_4.pdf\
|
+---src\
|       base_classes.py\
|       base_external_packages.py\
|       base_functions.py\
|       const_analysis.py\
|       const_ibcrule.py\
|       const_project.py\
|       funct_pca.py\
|       funct_plot.py\
|       funct_region.py\
|       funct_sensi.py\
|       main.py\
|       __init__.py\
|
\---tests
## Packages