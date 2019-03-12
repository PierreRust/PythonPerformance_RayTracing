
init:
	pip install -r requirements.txt

test:
	pytest tests

clean:
	rm -f ./*.c
	rm -f ./*.out
	rm -f ./*.svg
	rm -f ./*.prof
	rm -f ./*.png
	rm -f ./*.so

check:
	mypy raytracer