
echo "Uses coverage.py (https://coverage.readthedocs.io/) to run tests and measure coverage"
echo "**************"
echo "Running tests:"
coverage run unit_tests.py
echo "**************"
echo "Report on test coverage:"
coverage report -m
