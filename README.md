# ModelHealer Project

$ tree
.
├── archive
│   ├── archivedyns
│   └── archivenodes
│       ├── autoCheck.dyf
│       ├── autoParametrize.dyf
│       ├── extract-OB-SP.dyf
│       ├── extract-Topology-OB-OB.dyf
│       ├── extract-Topology-OB-PA.dyf
│       ├── extract-Topology-OB-SP.dyf
│       └── varyGPs.dyf
├── dir2
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir3
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir4
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir5
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir6
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir7
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── file_in_root.ext
└── README.md

% ├── archive
% │   │ 
% │   ├── <onestorey>                                 # all data for the one storey model
% │   │    ├── log                                    # log files for dynamo scripts
% │   │    ├── res                                    # temporary results
% │   │    ├── planfloor                              # planfloor
% │   │    ├── plansite                               # plansite
% │   │    ├── 0_basemodel.rvt                        # zero model without modeling
% │   │    ├── 0_variationData.txt                    # variation information in each iteration
% │   │    ├── save.rvt                               # backup
% │   │    ├── initial_design.rvt                     # initial design in .rvt (parametrized)
% │   │    ├── 1_MODEL.dyn                            # .dyn that creates the initial design
% │   │    ├── 1_CHECK.dyn                            # .dyn that execute code compliance checking
% │   │    ├── 3_ADJUST.dyn                           # .dyn that adjust the model for healing
% │   │
% │ 
% ├── src
% │   │ 
% │   ├── <godyn>/<onestorey>                         #........ codes for Dynamo scripts
% │   │ 		├── 01_getFileDirectory.py           
% │   │ 		├── 02_setGlobalParameters.py
% │   │ 		├── 03_constraints2Storey.py
% │   │ 		├── 04_constraints2Zones.py
% │   │ 		├── 05_constraints2Spaces.py
% │   │ 		├── 06_integrateConstraints.py
% │   │ 		├── 07_createReferences.py
% │   │ 		├── 08_createGeometry.py
% │   │ 		├── 09_exitReferences.py
% │   │ 		├── 10_createOpenings.py
% │   │ 		├── .....py
% │   │ 		├── 21_freshParameters.py
% │   │ 		├── 31_savectInput.py
% │   │ 		├── 32_checkCompliance.py
% │   │ 		├── 33_saveCheckingResults.py
% │   │ 		├── 34_analyzeModel.py
% │   │
% │   ├── <gorvt>/<HealingRVT>
% │   │    ├── AutoDynaRunAdjustPlugin/bin/Debug/AutoDynaRunAdjustPlugin.dll      # Revit add-in for auto adjusting.
% │   │    ├── AutoDynaRunCheckPlugin/bin/Debug/AutoDynaRunCheckPlugin.dll        # Revit add-in for auto checking.
% │   │    ├── StartProcess/bin/Debug/StartProcess.exe                            # external executer for Revit launching and killing.
% │   │    ├── ...
% │   │ 
% │   │ 
% │   ├── <healing>/*
% │   │    ├── __init__.py
% │   │    ├── __main__.py
% │   │    ├── base_classes.py                # define basic classes
% │   │    ├── base_constants.py              # define basic constant variables
% │   │    ├── base_functions.py              # define basic functions
% │   │    ├── external_packages.py           # import external packages
% │   │    ├── plot_functions.py              # define rendering functions
% │   │    ├── sa_functions.py                # define sensitivity analysis functions
% │   │    ├── yah_1.py                       # build project and complete the first sampling.
% │   │    ├── yah_2.py                       # analyze the first sampling results and complete the SA.
% │   │    ├── yah_3.py                       # analyze the first sampling results and complete the PCA.
% │   │
% │
% ├── tests
% │   ├── ...
% │
% ├── .gitignore
% ├── main.py
% ├── transfer.json
% └── README.md