PYTHON3 := $(shell which python3 2>/dev/null)
COVERAGE3 := $(shell which coverage3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=hafnian --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest hafnian

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install Hafnian"
	@echo "  libperm            to compile the Fortran permanent library"
	@echo "  wheel              to build the Hafnian wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  coverage           to generate a coverage report"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install Hafnian you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

.PHONY : clean
clean:
	make -C src clean
	rm -rf hafnian/__pycache__
	rm -rf hafnian/tests/__pycache__
	rm -rf dist
	rm -rf build

doc:
	make -C docs html

.PHONY : clean-docs
clean-docs:
	make -C docs clean

test:
	$(PYTHON) $(TESTRUNNER)

coverage:
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)

libperm:
	make libperm -C src
