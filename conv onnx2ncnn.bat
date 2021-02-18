python -m onnxsim --skip-fuse-bn ufast_lane_det.onnx ufast_lane_det-sim.onnx

D:\GitHub\ncnn\build\tools\onnx\onnx2ncnn.exe  ufast_lane_det-sim.onnx ufast_lane_det-sim.param ufast_lane_det-sim.bin
rem D:\GitHub\ncnn\build\tools\onnx\onnx2ncnn.exe  ufast_lane_det.onnx ufast_lane_det.param ufast_lane_det.bin

D:\GitHub\ncnn\build\tools\ncnnoptimize.exe ufast_lane_det-sim.param ufast_lane_det-sim.bin ufast_lane_det-sim-opt.param ufast_lane_det-sim-opt.bin 1

pause