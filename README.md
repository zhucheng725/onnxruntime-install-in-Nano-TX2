### FOR JETPACK 4.4 TENSORRT7 

# Install onnxruntime step by step(if your network is ok):

```
#https://github.com/Microsoft/onnxruntime/blob/master/BUILD.md
1.
git clone --single-branch --recursive --branch v1.1.2 https://github.com/Microsoft/onnxruntime

2.
export CUDACXX="/usr/local/cuda/bin/nvcc"

3.Modify tools/ci_build/build.py
- "-Donnxruntime_DEV_MODE=" + ("OFF" if args.android else "ON"),
+ "-Donnxruntime_DEV_MODE=" + ("OFF" if args.android else "OFF"),

4.Modify cmake/CMakeLists.txt
-  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
+  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_53,code=sm_53") # Jetson TX1/Nano 
+  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_62,code=sm_62") # Jetson TX2

5.
./build.sh --config Release --update --build --build_wheel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu
```

# Install onnxruntime step by step(if your network is bad):

```
1. download from website: https://github.com/microsoft/onnxruntime/tree/Jetson-arm64-CI-tag-v1.2.0
#Jetpack 4.4 can be used in v1.2.0

2.download from website:

1.DNNLibrary : https://github.com/JDAI-CV/DNNLibrary/tree/e17f11e966b2cce7d747799b76bb9843813d4b01
2.FeaturizersLibrary : https://github.com/microsoft/FeaturizersLibrary/tree/afebe4c9d49ed74918bf8879aaa7aa93ccc7e47c
3.cub : https://github.com/NVlabs/cub/tree/c3cceac115c072fb63df1836ff46d8c60d9eb304
4.gemmlowp : https://github.com/google/gemmlowp/tree/42c53187a56c12dc5518cc25e778e5e3e7dbaf72
5.date : https://github.com/HowardHinnant/date/tree/e7e1482087f58913b80a20b04d5c58d9d6d90155
6.eigen : https://gitlab.com/libeigen/eigen.git
7.googletest : https://github.com/google/googletest/tree/703bd9caab50b139428cea1aaff9974ebee5742e
8.json : https://github.com/nlohmann/json/tree/d98bf0278d6f59a58271425963a8422ff48fe249
9.mimalloc : https://github.com/microsoft/mimalloc/tree/2d54553b7a78c7c35620b827e7e5ab2228ecb495
10.nsync : https://github.com/google/nsync/tree/436617053d0f39a1019a371c3a9aa599b3cb2cea
11.onnx : https://github.com/onnx/onnx/tree/1facb4c1bb9cc2107d4dbaf9fd647fefdbbeb0ab
12.onnx-tensorrt : https://github.com/stevenlix/onnx-tensorrt/tree/5a7cba1a768c3bb01cbf323e3acdeb8e29e3beca
13.protobuf : https://github.com/protocolbuffers/protobuf/tree/498de9f761bef56a032815ee44b6e6dbe0892cc4
14.re2 : https://github.com/google/re2/tree/30cad267151fa8f1b17da8c1ef0571da6da9a8f1
15.tvm : https://github.com/microsoft/onnxruntime-tvm/tree/c6e3efcdb09aeda961a6badf76093ceac69db64d
16.wil : https://github.com/microsoft/wil/tree/e8c599bca6c56c44b6730ad93f6abbc9ecd60fc1
```
and replace these files in 
```
/home/nvidia/onnxruntime-Jetson-arm64-CI-tag-v1.2.0/cmake/external/
```


