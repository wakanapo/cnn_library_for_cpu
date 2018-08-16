#!/bin/sh
python src/python/make_first_generation.py 3 50 p3g50 random
./bin/ga p3g50 --max_generation=100
python src/python/make_first_generation.py 4 50 p4g50 random
./bin/ga p4g50 --max_generation=100
python src/python/make_first_generation.py 6 50 p6g50 random
./bin/ga p6g50 --max_generation=100
python src/python/make_first_generation.py 7 50 p7g50 random
./bin/ga p7g50 --max_generation=100
python src/python/make_first_generation.py 8 50 p8g50 random
./bin/ga p8g50 --max_generation=100
python src/python/make_first_generation.py 15 50 p15g50 random
./bin/ga p15g50 --max_generation=100
python src/python/make_first_generation.py 16 50 p16g50 random
./bin/ga p16g50 --max_generation=100
