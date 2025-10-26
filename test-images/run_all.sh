

mkdir -p output/NPRportrait1
cd NPRportrait1
for img in *
do
    echo "Processing image: $img";
    python ../../main.py $img --output "../output/NPRportrait1/$img" --params ../params.json --orientation-vector structural
done
cd ..

mkdir -p output/NPRportrait2
cd NPRportrait2
for img in *
do
    echo "Processing image: $img";
    python ../../main.py $img --output "../output/NPRportrait2/$img" --params ../params.json --orientation-vector structural
done
cd ..


mkdir -p output/NPRportrait3
cd NPRportrait3
for img in *
do
    echo "Processing image: $img";
    python ../../main.py $img --output "../output/NPRportrait3/$img" --params ../params.json --orientation-vector structural
done
cd ..

mkdir -p output/benchmark_unstylized
cd benchmark_unstylized
for img in *
do
    echo "Processing image: $img";
    python ../../main.py $img --output "../output/benchmark_unstylized/$img" --params ../params.json --orientation-vector gradient
done
cd ..