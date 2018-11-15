## Celldom CLI

Overview TBD

#### Troubleshooting

###### Frozen Processor

Every once in a while, processing seems to freeze with cpu usage at 100%.  Using 
```faulthandler``` to register a tracedump (issued via ```kill -SIGUSER1 PID```) shows a trace like:

    Thread 0x00007f106d7ff700 (most recent call first):
      File "/usr/lib/python3.5/threading.py", line 297 in wait
      File "/usr/lib/python3.5/threading.py", line 549 in wait
      File "/usr/local/lib/python3.5/dist-packages/tqdm/_monitor.py", line 66 in run
      File "/usr/lib/python3.5/threading.py", line 914 in _bootstrap_inner
      File "/usr/lib/python3.5/threading.py", line 882 in _bootstrap
    
    Current thread 0x00007f127ceb8700 (most recent call first):
      File "/usr/local/lib/python3.5/dist-packages/numpy/linalg/linalg.py", line 528 in inv
      File "/usr/local/lib/python3.5/dist-packages/skimage/transform/_geometric.py", line 688 in estimate
      File "/usr/local/lib/python3.5/dist-packages/skimage/transform/_warps.py", line 165 in resize
      File "/usr/local/lib/python3.5/dist-packages/mrcnn/utils.py", line 579 in unmold_mask
      File "/usr/local/lib/python3.5/dist-packages/mrcnn/model.py", line 2482 in unmold_detections
      File "/usr/local/lib/python3.5/dist-packages/mrcnn/model.py", line 2538 in detect
      File "/lab/repos/celldom/python/source/celldom/extract/cell_extraction.py", line 29 in extract
      File "/lab/repos/celldom/python/source/celldom/extract/apartment_extraction.py", line 199 in extract
      File "/lab/repos/celldom/python/source/celldom/core/cytometry.py", line 348 in analyze
      File "/lab/repos/celldom/python/source/celldom/execute/processing.py", line 38 in run_cytometer
      File "/usr/local/bin/celldom", line 132 in run_processor
      File "/usr/local/lib/python3.5/dist-packages/fire/core.py", line 542 in _CallCallable
      File "/usr/local/lib/python3.5/dist-packages/fire/core.py", line 366 in _Fire
      File "/usr/local/lib/python3.5/dist-packages/fire/core.py", line 127 in Fire
      File "/usr/local/bin/celldom", line 238 in <module>
      
 This may be related to a variety of issues where BLAS/LAPACK libraries are known to do this with inf/nan 
 matrix values ([Numpy issue 7461](https://github.com/numpy/numpy/issues/7461)).  Or it may not -- I need
 to gather more of these traces during freezes to see if the problem is really in linalg all the time.      