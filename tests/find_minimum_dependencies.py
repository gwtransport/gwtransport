#!/usr/bin/env python3
"""
Find minimum compatible versions for package dependencies.

This script systematically tests different versions of each dependency to find
the lowest compatible version that allows the package to install and pass tests.

Process
-------
1. Reads dependencies from pyproject.toml
2. For each dependency, creates temporary environments with different versions
3. Tests installation and runs the test suite
4. Uses cascading major→minor→patch search to efficiently find the minimum
5. Records results in a JSON file

Search Strategy
---------------
Cascading Major→Minor→Patch approach:

* **Phase 1**: Test the latest version to ensure it works
* **Phase 2**: Walk down major versions (newest to oldest) until one fails
* **Phase 3**: Walk down minor versions within the last working major until one fails
* **Phase 4**: Walk down patch versions within the last working minor until one fails

The search stops immediately at each level when encountering a failure, as older
versions are assumed to also fail. This approach is efficient and minimizes tests.

Requirements
------------
This script requires additional dependencies beyond the main package:

* requests: For fetching package versions from PyPI
* packaging: For version comparison

Install with::

    pip install requests packaging

Command-Line Usage
------------------
.. code-block:: bash

    python tests/find_minimum_dependencies.py [OPTIONS]

Options
-------
--dependency PACKAGE
    Test only this specific dependency

--python-version VERSION
    Python version to use (e.g., 3.11, 3.12). Default: 3.11

--include-optional GROUP
    Include optional dependency groups along with main dependencies.
    Can be specified multiple times.
    Example: ``--include-optional test --include-optional docs``

--only-optional GROUP
    Test ONLY optional dependency groups (excludes main dependencies).
    Cannot be used with ``--include-optional``.
    Example: ``--only-optional test``

--parallel N
    Number of parallel workers for testing multiple packages. Default: 1.
    Each package is tested in isolation using temporary virtual environments,
    so parallel processing is safe and won't cause conflicts.

--output FILE
    Output JSON file. Default: min_dependencies.json

--test-command CMD
    Custom test command. Default: ``pytest tests/src -v -x``
    The ``-x`` flag makes pytest stop on first failure for faster testing.

--verbose, -v
    Enable verbose logging (DEBUG level)

Examples
--------
Test all main dependencies (6 packages)::

    python tests/find_minimum_dependencies.py

Test main + test dependencies (14 packages) in parallel::

    python tests/find_minimum_dependencies.py --include-optional test --parallel 14

Test ONLY test dependencies (8 packages), excluding main::

    python tests/find_minimum_dependencies.py --only-optional test --parallel 8

Test ONLY docs dependencies (7 packages)::

    python tests/find_minimum_dependencies.py --only-optional docs --parallel 7

Test specific dependency from optional group::

    python tests/find_minimum_dependencies.py --only-optional test --dependency pytest

Test with Python 3.12::

    python tests/find_minimum_dependencies.py --python-version 3.12 --only-optional test --parallel 8

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import argparse
import json
import logging
import multiprocessing
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import requests
from packaging.version import parse as parse_version

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VersionTestResult:
    """Result of testing a specific version."""

    package: str
    version: str
    install_success: bool
    test_success: bool
    error_message: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()


@dataclass
class MinimumVersionResult:
    """Final result for a package's minimum version."""

    package: str
    minimum_version: Optional[str]
    tested_versions: list[str]
    install_failures: list[str]
    test_failures: list[str]
    total_tests: int
    success: bool


class PyPIVersionFinder:
    """Fetch available versions from PyPI."""

    @staticmethod
    def get_versions(package: str) -> list[str]:
        """Get all available versions for a package from PyPI.

        Parameters
        ----------
        package : str
            Package name

        Returns
        -------
        list[str]
            List of version strings, sorted from oldest to newest
        """
        try:
            response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=10)
            response.raise_for_status()
            data = response.json()
            versions = list(data["releases"].keys())

            # Filter out pre-releases and development versions
            versions = [
                v
                for v in versions
                if not any(pre in v.lower() for pre in ["a", "b", "rc", "dev", "pre", "alpha", "beta"])
            ]

            # Sort by version
            versions.sort(key=parse_version)
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning("Could not fetch versions for %s: %s", package, e)
            return []
        else:
            return versions


class VersionStepper:
    """Smart version stepping logic for cascading major→minor→patch search."""

    @staticmethod
    def parse_version(version: str) -> tuple[int, int, int]:
        """Parse version string into (major, minor, patch).

        Parameters
        ----------
        version : str
            Version string like "1.2.3"

        Returns
        -------
        tuple[int, int, int]
            Parsed version as (major, minor, patch)
        """
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
        if match:
            major, minor, patch = map(int, match.groups())
            return (major, minor, patch)
        match = re.match(r"(\d+)\.(\d+)", version)
        if match:
            major, minor = map(int, match.groups())
            return (major, minor, 0)
        match = re.match(r"(\d+)", version)
        if match:
            return (int(match.group(1)), 0, 0)
        return (0, 0, 0)

    @staticmethod
    def group_versions_by_level(
        all_versions: list[str], start_version: str = "0.0.0"
    ) -> dict[str, dict[str, list[str]]]:
        """Group versions by major, then by minor within each major.

        Parameters
        ----------
        all_versions : list[str]
            All available versions, sorted
        start_version : str
            Starting version to filter from

        Returns
        -------
        dict[str, dict[str, list[str]]]
            Nested dict: {major: {minor: [versions]}}
        """
        if not all_versions:
            return {}

        start_parsed = parse_version(start_version)
        versions = [v for v in all_versions if parse_version(v) >= start_parsed]

        # Group by major.minor
        grouped: dict[str, dict[str, list[str]]] = {}
        for v in versions:
            major, minor, _ = VersionStepper.parse_version(v)
            major_key = str(major)
            minor_key = f"{major}.{minor}"

            if major_key not in grouped:
                grouped[major_key] = {}
            if minor_key not in grouped[major_key]:
                grouped[major_key][minor_key] = []

            grouped[major_key][minor_key].append(v)

        return grouped

    @staticmethod
    def get_versions_descending(all_versions: list[str], start_version: str = "0.0.0") -> list[str]:
        """Get versions in descending order (newest to oldest).

        Parameters
        ----------
        all_versions : list[str]
            All available versions, sorted ascending
        start_version : str
            Starting version to filter from

        Returns
        -------
        list[str]
            Versions in descending order
        """
        start_parsed = parse_version(start_version)
        versions = [v for v in all_versions if parse_version(v) >= start_parsed]
        return list(reversed(versions))


class DependencyTester:
    """Test dependencies in isolated environments."""

    def __init__(self, project_root: Path, test_command: str = "pytest tests/src -v", python_version: str = "3.11"):
        """Initialize the tester.

        Parameters
        ----------
        project_root : Path
            Root directory of the project
        test_command : str
            Command to run tests
        python_version : str
            Python version to use (e.g., "3.11", "3.12")
        """
        self.project_root = project_root
        self.test_command = test_command
        self.python_version = python_version
        self.results: list[VersionTestResult] = []

    def test_version(self, package: str, version: str) -> VersionTestResult:
        """Test a specific version of a package.

        Parameters
        ----------
        package : str
            Package name
        version : str
            Version to test

        Returns
        -------
        VersionTestResult
            Test result
        """
        logger.info("=" * 60)
        logger.info("Testing %s==%s", package, version)
        logger.info("=" * 60)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            venv_path = tmpdir_path / ".venv"

            try:
                # Create virtual environment with uv and specific Python version
                uv_path = shutil.which("uv")
                logger.info("Creating virtual environment with Python %s using uv at %s", self.python_version, uv_path)
                subprocess.run(
                    [uv_path, "venv", str(venv_path), "--python", self.python_version],
                    cwd=str(self.project_root),
                    check=True,
                    capture_output=True,
                    text=True,
                )  # type: ignore[no-matching-overload]

                # Install package with specific dependency version
                logger.info("Installing package with %s==%s", package, version)

                # Build the install command with the pinned dependency
                install_cmd = [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_path / "bin" / "python"),
                    "-e",
                    ".[test]",
                    f"{package}=={version}",
                ]

                result = subprocess.run(
                    install_cmd, check=False, cwd=self.project_root, capture_output=True, text=True, timeout=300
                )

                if result.returncode != 0:
                    logger.error("Installation failed")
                    return VersionTestResult(
                        package=package,
                        version=version,
                        install_success=False,
                        test_success=False,
                        error_message=f"Install failed: {result.stderr[:500]}",
                    )

                logger.info("Installation successful")

                # Run tests
                # Note: Using -x flag to stop on first failure saves time when version is incompatible
                logger.info("Running tests")
                python_exe = venv_path / "bin" / "python"

                test_result = subprocess.run(
                    [str(python_exe), "-m", *self.test_command.split()],
                    check=False,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if test_result.returncode != 0:
                    logger.error("Tests failed")
                    return VersionTestResult(
                        package=package,
                        version=version,
                        install_success=True,
                        test_success=False,
                        error_message=f"Tests failed: {test_result.stderr[:500]}",
                    )

                logger.info("Tests passed")
                return VersionTestResult(package=package, version=version, install_success=True, test_success=True)

            except subprocess.TimeoutExpired:
                logger.exception("Timeout")
                return VersionTestResult(
                    package=package, version=version, install_success=False, test_success=False, error_message="Timeout"
                )
            except Exception as e:
                logger.exception("Error occurred during testing")
                return VersionTestResult(
                    package=package,
                    version=version,
                    install_success=False,
                    test_success=False,
                    error_message=str(e)[:500],
                )

    def find_minimum_version(self, package: str, start_version: str = "0.0.0") -> MinimumVersionResult:
        """Find minimum compatible version for a package.

        Strategy: Cascading search from newest to oldest
        1. Test latest version to ensure it works
        2. Walk down major versions until one fails
        3. Walk down minor versions within the last working major
        4. Walk down patch versions within the last working minor

        This minimizes tests by stopping as soon as we hit a failure at each level.

        Parameters
        ----------
        package : str
            Package name
        start_version : str
            Starting version to test from (minimum bound)

        Returns
        -------
        MinimumVersionResult
            Result containing minimum version and test details
        """
        logger.info("#" * 60)
        logger.info("Finding minimum version for %s", package)
        logger.info("#" * 60)

        # Fetch all versions
        logger.info("Fetching available versions from PyPI")
        all_versions = PyPIVersionFinder.get_versions(package)

        if not all_versions:
            logger.error("Could not fetch versions for %s", package)
            return MinimumVersionResult(
                package=package,
                minimum_version=None,
                tested_versions=[],
                install_failures=[],
                test_failures=[],
                total_tests=0,
                success=False,
            )

        logger.info("Found %d versions", len(all_versions))

        # Group versions by major.minor
        grouped = VersionStepper.group_versions_by_level(all_versions, start_version)

        if not grouped:
            logger.error("No versions found >= %s", start_version)
            return MinimumVersionResult(
                package=package,
                minimum_version=None,
                tested_versions=[],
                install_failures=[],
                test_failures=[],
                total_tests=0,
                success=False,
            )

        tested_versions = []
        install_failures = []
        test_failures = []

        # Get all versions in descending order for reference
        all_desc = VersionStepper.get_versions_descending(all_versions, start_version)
        logger.info("Testing versions from %s (newest) down to %s (oldest)", all_desc[0], all_desc[-1])

        # Phase 1: Test latest version first
        logger.info("Phase 1: Testing latest version")
        latest_version = all_desc[0]
        result = self.test_version(package, latest_version)
        self.results.append(result)
        tested_versions.append(latest_version)

        if not result.install_success:
            install_failures.append(latest_version)
            logger.error("Latest version %s failed to install", latest_version)
            return MinimumVersionResult(
                package=package,
                minimum_version=None,
                tested_versions=tested_versions,
                install_failures=install_failures,
                test_failures=test_failures,
                total_tests=len(tested_versions),
                success=False,
            )

        if not result.test_success:
            test_failures.append(latest_version)
            logger.error("Latest version %s failed tests", latest_version)
            return MinimumVersionResult(
                package=package,
                minimum_version=None,
                tested_versions=tested_versions,
                install_failures=install_failures,
                test_failures=test_failures,
                total_tests=len(tested_versions),
                success=False,
            )

        logger.info("Latest version %s works - searching for minimum version", latest_version)
        minimum_working_version = latest_version
        latest_major, _, _ = VersionStepper.parse_version(latest_version)

        # Phase 2: Walk down major versions
        logger.info("Phase 2: Walking down major versions")
        major_keys = sorted([int(k) for k in grouped], reverse=True)
        working_major = None

        for major in major_keys:
            if major > latest_major:
                continue  # Skip newer majors than latest

            # Get the latest version in this major
            major_str = str(major)
            minor_keys = sorted(grouped[major_str].keys(), key=parse_version, reverse=True)
            test_version = grouped[major_str][minor_keys[0]][-1]  # Latest version in this major

            if test_version in tested_versions:
                working_major = major
                minimum_working_version = test_version
                continue

            logger.info("Testing major version %d with %s", major, test_version)
            result = self.test_version(package, test_version)
            self.results.append(result)
            tested_versions.append(test_version)

            if not result.install_success or not result.test_success:
                if not result.install_success:
                    install_failures.append(test_version)
                    logger.info("Major version %d failed to install - stopping major search", major)
                else:
                    test_failures.append(test_version)
                    logger.info("Major version %d failed tests - stopping major search", major)
                break

            # This major version works
            working_major = major
            minimum_working_version = test_version
            logger.info("Major version %d works", major)

        if working_major is None:
            logger.error("No working major version found")
            return MinimumVersionResult(
                package=package,
                minimum_version=minimum_working_version,
                tested_versions=tested_versions,
                install_failures=install_failures,
                test_failures=test_failures,
                total_tests=len(tested_versions),
                success=True,
            )

        # Phase 3: Walk down minor versions within the working major
        logger.info("Phase 3: Walking down minor versions in major %d", working_major)
        major_str = str(working_major)
        minor_keys = sorted(grouped[major_str].keys(), key=parse_version, reverse=True)
        working_minor = None

        for minor_key in minor_keys:
            # Get the latest patch version in this minor
            test_version = grouped[major_str][minor_key][-1]

            if test_version in tested_versions:
                working_minor = minor_key
                minimum_working_version = test_version
                continue

            logger.info("Testing minor version %s with %s", minor_key, test_version)
            result = self.test_version(package, test_version)
            self.results.append(result)
            tested_versions.append(test_version)

            if not result.install_success or not result.test_success:
                if not result.install_success:
                    install_failures.append(test_version)
                    logger.info("Minor version %s failed to install - stopping minor search", minor_key)
                else:
                    test_failures.append(test_version)
                    logger.info("Minor version %s failed tests - stopping minor search", minor_key)
                break

            # This minor version works
            working_minor = minor_key
            minimum_working_version = test_version
            logger.info("Minor version %s works", minor_key)

        if working_minor is None:
            logger.info("Using minimum from major version search: %s", minimum_working_version)
            return MinimumVersionResult(
                package=package,
                minimum_version=minimum_working_version,
                tested_versions=tested_versions,
                install_failures=install_failures,
                test_failures=test_failures,
                total_tests=len(tested_versions),
                success=True,
            )

        # Phase 4: Walk down patch versions within the working minor
        logger.info("Phase 4: Walking down patch versions in %s", working_minor)
        patch_versions = sorted(grouped[major_str][working_minor], key=parse_version, reverse=True)

        for test_version in patch_versions:
            if test_version in tested_versions:
                minimum_working_version = test_version
                continue

            logger.info("Testing patch version %s", test_version)
            result = self.test_version(package, test_version)
            self.results.append(result)
            tested_versions.append(test_version)

            if not result.install_success or not result.test_success:
                if not result.install_success:
                    install_failures.append(test_version)
                    logger.info("Patch version %s failed to install - stopping patch search", test_version)
                else:
                    test_failures.append(test_version)
                    logger.info("Patch version %s failed tests - stopping patch search", test_version)
                break

            # This patch version works
            minimum_working_version = test_version
            logger.info("Patch version %s works", test_version)

        logger.info("=" * 60)
        logger.info("Minimum version for %s: %s", package, minimum_working_version)
        logger.info("=" * 60)

        return MinimumVersionResult(
            package=package,
            minimum_version=minimum_working_version,
            tested_versions=tested_versions,
            install_failures=install_failures,
            test_failures=test_failures,
            total_tests=len(tested_versions),
            success=True,
        )


def test_package_wrapper(package: str, project_root: Path, test_command: str, python_version: str) -> tuple[str, dict]:
    """Wrapper function for testing a package in parallel.

    Parameters
    ----------
    package : str
        Package name
    project_root : Path
        Project root directory
    test_command : str
        Test command to run
    python_version : str
        Python version to use

    Returns
    -------
    tuple[str, dict]
        Package name and result dictionary
    """
    # Set the process name to the package being tested
    current_process = multiprocessing.current_process()
    current_process.name = f"Worker-{package}"

    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tester = DependencyTester(project_root, test_command, python_version)
    result = tester.find_minimum_version(package)
    return package, asdict(result)


def load_dependencies(
    project_root: Path, include_optional: list[str] | None = None, only_optional: list[str] | None = None
) -> list[str]:
    """Load dependencies from pyproject.toml.

    Parameters
    ----------
    project_root : Path
        Project root directory
    include_optional : list[str] | None
        List of optional dependency groups to include along with main dependencies
    only_optional : list[str] | None
        List of optional dependency groups to test (excludes main dependencies)

    Returns
    -------
    list[str]
        List of dependency names
    """
    pyproject_path = project_root / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Determine which dependencies to load
    if only_optional:
        # Only load optional dependencies, skip main
        dependencies = []
        logger.info("Loading ONLY optional dependencies (excluding main dependencies)")
    else:
        # Get main dependencies
        dependencies = data.get("project", {}).get("dependencies", [])

    # Add optional dependencies if requested
    optional_groups = only_optional or include_optional
    if optional_groups:
        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        for group in optional_groups:
            if group in optional_deps:
                dependencies.extend(optional_deps[group])
                logger.info("Added %d dependencies from optional group '%s'", len(optional_deps[group]), group)
            else:
                logger.warning(
                    "Optional dependency group '%s' not found. Available groups: %s",
                    group,
                    ", ".join(optional_deps.keys()),
                )

    # Extract package names (remove version specifiers and extras)
    package_names = []
    for dep in dependencies:
        # Remove extras like [all,store]
        dep_without_extras = re.split(r"\[", dep)[0]
        # Split on comparison operators
        name = re.split(r"[<>=!]", dep_without_extras)[0].strip()
        if name:  # Skip empty strings
            package_names.append(name)

    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in package_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    return unique_names


def main():
    """Find minimum compatible versions for package dependencies."""
    parser = argparse.ArgumentParser(description="Find minimum compatible versions for package dependencies")
    parser.add_argument("--dependency", help="Test only this specific dependency")
    parser.add_argument(
        "--python-version", default="3.11", help="Python version to use (e.g., 3.11, 3.12, default: 3.11)"
    )
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument(
        "--include-optional",
        action="append",
        help="Include optional dependency groups along with main dependencies (e.g., --include-optional test)",
    )
    parser.add_argument(
        "--only-optional",
        action="append",
        help="Test ONLY optional dependency groups, excluding main dependencies (e.g., --only-optional test)",
    )
    parser.add_argument(
        "--output", default="min_dependencies.json", help="Output JSON file (default: min_dependencies.json)"
    )
    parser.add_argument(
        "--test-command",
        default="pytest tests/src -v -x",
        help="Custom test command (default: 'pytest tests/src -v -x'). Note: -x flag stops on first failure",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level)")

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Determine project root (parent of tests directory)
    project_root = Path(__file__).parent.parent

    logger.info("Project root: %s", project_root)
    logger.info("Python version: %s", args.python_version)
    logger.info("Test command: %s", args.test_command)

    # Validate mutually exclusive options
    if args.include_optional and args.only_optional:
        logger.error("Cannot use both --include-optional and --only-optional. Choose one.")
        sys.exit(1)

    # Load dependencies
    dependencies = load_dependencies(
        project_root, include_optional=args.include_optional, only_optional=args.only_optional
    )

    if args.dependency:
        if args.dependency not in dependencies:
            logger.error("%s not found in dependencies", args.dependency)
            logger.info("Available dependencies: %s", ", ".join(dependencies))
            sys.exit(1)
        dependencies = [args.dependency]

    logger.info("Dependencies to test: %s", ", ".join(dependencies))
    if args.parallel > 1:
        logger.info("Using %d parallel workers", args.parallel)

    # Test each dependency
    all_results = {}

    if args.parallel > 1 and len(dependencies) > 1:
        # Use parallel processing for multiple packages
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_package = {
                executor.submit(
                    test_package_wrapper, package, project_root, args.test_command, args.python_version
                ): package
                for package in dependencies
            }

            # Collect results as they complete
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    pkg_name, result = future.result()
                    all_results[pkg_name] = result
                    logger.info("Completed testing %s", pkg_name)
                except Exception:
                    logger.exception("Error testing %s", package)
                    all_results[package] = asdict(
                        MinimumVersionResult(
                            package=package,
                            minimum_version=None,
                            tested_versions=[],
                            install_failures=[],
                            test_failures=[],
                            total_tests=0,
                            success=False,
                        )
                    )
    else:
        # Sequential processing (default or single dependency)
        tester = DependencyTester(project_root, args.test_command, args.python_version)
        for package in dependencies:
            result = tester.find_minimum_version(package)
            all_results[package] = asdict(result)

    # Save results
    output_path = project_root / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "python_version": args.python_version,
                "test_command": args.test_command,
                "results": all_results,
            },
            f,
            indent=2,
        )

    logger.info("=" * 60)
    logger.info("Results saved to: %s", output_path)
    logger.info("=" * 60)

    # Print summary
    logger.info("Summary:")
    logger.info("-" * 60)
    for package, result in all_results.items():
        if result["success"]:
            logger.info("✓ %s >= %s", package.ljust(20), result["minimum_version"])
        else:
            logger.warning("✗ %s No compatible version found", package.ljust(20))

    # Generate updated pyproject.toml snippet
    logger.info("=" * 60)
    logger.info("Suggested pyproject.toml dependencies:")
    logger.info("=" * 60)
    logger.info("dependencies = [")
    for package, result in all_results.items():
        if result["success"]:
            logger.info('  "%s>=%s",', package, result["minimum_version"])
        else:
            logger.info('  "%s",  # No minimum version found', package)
    logger.info("]")


if __name__ == "__main__":
    main()
