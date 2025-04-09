import coverage
import os
import subprocess

def run_example_scripts():
    example_dir = "examples"
    example_files = [f for f in os.listdir(example_dir) if f.endswith(".py")]

    for example_file in example_files:
        example_path = os.path.join(example_dir, example_file)
        subprocess.run(["python", example_path], check=True)

if __name__ == "__main__":
    cov = coverage.Coverage(source=["src"])
    cov.start()
    run_example_scripts()
    cov.stop()
    cov.save()
    cov.html_report(directory="htmlcov")
    cov.xml_report(outfile="coverage.xml")
