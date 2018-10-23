EIGEN=/usr/include/eigen3

cd ./binaries
for file in ../src/*cpp; do
	echo $EIGEN
	echo $file
	g++ -pg -std=c++11 -fPIC -c -I$EIGEN -I../include -I~/pf/include -O3 $file;
done
ar crv libssme.a *.o
cd ..

