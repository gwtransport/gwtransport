import pytest
import coverage

def test_coverage():
    cov = coverage.Coverage(source=["src"])
    cov.start()
    pytest.main(["--cov=src", "--cov-report=xml", "--cov-report=html"])
    cov.stop()
    cov.save()
    cov.html_report(directory="htmlcov")
    cov.xml_report(outfile="coverage.xml")
