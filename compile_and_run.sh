g++ -x c++  -E -dM -< /dev/null | grep __cplusplus
cd ./build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cd ..
./build/SubgraphMatching -d test/D_0 -q test/Q_1