
mkdir -p output/NPRportrait1
cd NPRportrait1
foreach ($img in Get-ChildItem *)
{
    echo "Processing image: $img";
    python ../../main.py $img "../output/NPRportrait1/$($img.BaseName).png" --params ..\params.json --orientation-vector structural
}
cd ..

mkdir -p output/NPRportrait2
cd NPRportrait2
foreach ($img in Get-ChildItem *)
{
    echo "Processing image: $img";
    python ../../main.py $img "../output/NPRportrait2/$($img.BaseName).png" --params ..\params.json --orientation-vector structural
}
cd ..


mkdir -p output/NPRportrait3
cd NPRportrait3
foreach ($img in Get-ChildItem *)
{
    echo "Processing image: $img";
    python ../../main.py $img "../output/NPRportrait3/$($img.BaseName).png" --params ..\params.json --orientation-vector structural
}
cd ..

mkdir -p output/benchmark_unstylized
cd benchmark_unstylized
foreach ($img in Get-ChildItem *)
{
    echo "Processing image: $img";
    python ../../main.py $img "../output/benchmark_unstylized/$($img.BaseName).png" --params ..\params.json --orientation-vector structural
}
cd ..