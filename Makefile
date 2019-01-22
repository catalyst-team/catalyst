.PHONY: check-style

check-style:
	flake8 catalyst/ setup.py --count --ignore=E126,E226,E704,E731,W503,W504 --max-complexity=16 --show-source --statistics
	flake8 catalyst/ setup.py --count --exit-zero --max-complexity=10 --statistics
