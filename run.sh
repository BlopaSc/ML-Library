#! binbash

echo Running Pablo Sauma-Chacon Homework

echo Ensemble Learning exercises

cd EnsambleLearning

{
python Main.py
} || {
python3 Main.py
}

echo Linear Regression LMS exercise

cd ../LinearRegression

{
python Main.py
} || {
python3 Main.py
}