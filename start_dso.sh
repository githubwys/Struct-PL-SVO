#!/bin/bash
cd bin
./run_pipeline -trajout=tra.txt -mapout=map.txt -files=../config/dataset_path_dso.txt -calib_path=../config -calib=/dataset_params_dso.yaml
