#!/usr/bin/env python3
# filepath: check.py
"""
Code quality checker script for the registration project.

Runs linting, formatting, and tests. Can auto-fix issues with --fix flag.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# ANSI color codes
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color


def print_header(text: str) -> None:
    """Print a formatted header."""
    width = 60
    print(f"\n{Colors.CYAN}{'━' * width}{Colors.NC}")
    print(f"{Colors.BOLD}{text}{Colors.NC}")
    print(f"{Colors.CYAN}{'━' * width}{Colors.NC}")


def print_status(success: bool, message: str) -> None:
    """Print a status message with color."""
    if success:
        print(f"{Colors.GREEN}✓ {message}{Colors.NC}")
    else:
        print(f"{Colors.RED}✗ {message}{Colors.NC}")


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run a command and return success status and output.
    
    Args:
        cmd: Command and arguments as a list
        description: Description of what's being run
        
    Returns:
        Tuple of (success, output)
    """
    print(f"{Colors.BLUE}Running: {' '.join(cmd)}{Colors.NC}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode == 0:
            print_status(True, description)
            return True, result.stdout
        else:
            print_status(False, description)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False, result.stderr
            
    except FileNotFoundError:
        print_status(False, f"{description} (command not found)")
        return False, f"Command not found: {cmd[0]}"
    except Exception as e:
        print_status(False, f"{description} (error: {e})")
        return False, str(e)


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"{Colors.RED}Error: uv is not installed{Colors.NC}")
        print("Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def run_ruff_check(fix: bool = False) -> bool:
    """Run ruff linting."""
    print_header("📝 Ruff Linting")
    
    cmd = ["uv", "run", "ruff", "check", "src/", "scripts/", "tests/"]
    if fix:
        cmd.append("--fix")
        description = "Ruff check with auto-fix"
    else:
        description = "Ruff check"
    
    success, _ = run_command(cmd, description)
    return success


def run_ruff_format(fix: bool = False) -> bool:
    """Run ruff formatting."""
    print_header("🎨 Ruff Formatting")
    
    if fix:
        cmd = ["uv", "run", "ruff", "format", "src/", "scripts/", "tests/"]
        description = "Ruff format (auto-fix)"
    else:
        cmd = ["uv", "run", "ruff", "format", "--check", "src/", "scripts/", "tests/"]
        description = "Ruff format check"
    
    success, _ = run_command(cmd, description)
    return success


def run_tests(quick: bool = False, verbose: bool = False) -> bool:
    """Run pytest."""
    print_header("🧪 Running Tests")
    
    cmd = ["uv", "run", "pytest"]
    
    if verbose:
        cmd.append("-v")
    elif quick:
        cmd.append("-q")
    
    # Note: JUnit XML is always generated via pyproject.toml configuration
    if not quick:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage",
        ])
    
    success, _ = run_command(cmd, "Tests")
    
    if success and not quick:
        print(f"\n{Colors.CYAN}📊 Coverage report: reports/coverage/index.html{Colors.NC}")
        print(f"{Colors.CYAN}📋 Test results: reports/junit.xml{Colors.NC}")
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run code quality checks for the registration project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all checks
  %(prog)s --fix              # Run checks and auto-fix issues
  %(prog)s --quick            # Run quick checks (no coverage)
  %(prog)s --lint-only        # Only run linting
  %(prog)s --test-only        # Only run tests
  %(prog)s -v --fix           # Verbose output with auto-fix
        """,
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix linting and formatting issues",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check without coverage (faster)",
    )
    parser.add_argument(
        "--lint-only",
        action="store_true",
        help="Only run linting checks (skip tests)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run tests (skip linting)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose test output",
    )
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.NC}")
    if args.fix:
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔧 Running checks with auto-fix{Colors.NC}")
    elif args.quick:
        print(f"{Colors.BOLD}{Colors.MAGENTA}🚀 Running quick checks{Colors.NC}")
    else:
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔍 Running code quality checks{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.NC}")
    
    # Check if uv is installed
    if not check_uv_installed():
        sys.exit(1)
    
    # Track overall success
    all_passed = True
    
    # Run checks based on flags
    if not args.test_only:
        # Run linting
        if not run_ruff_check(fix=args.fix):
            all_passed = False
            if not args.fix:
                print(f"\n{Colors.YELLOW}💡 Tip: Run with --fix to auto-fix linting issues{Colors.NC}")
        
        # Run formatting
        if not run_ruff_format(fix=args.fix):
            all_passed = False
            if not args.fix:
                print(f"\n{Colors.YELLOW}💡 Tip: Run with --fix to auto-format code{Colors.NC}")
    
    if not args.lint_only:
        # Run tests
        if not run_tests(quick=args.quick, verbose=args.verbose):
            all_passed = False
    
    # Print summary
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.NC}")
    if all_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}✨ All checks passed!{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.NC}\n")
        sys.exit(0)
    else:
        print(f"{Colors.BOLD}{Colors.RED}❌ Some checks failed{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.NC}\n")
        if not args.fix:
            print(f"{Colors.YELLOW}💡 Try running: ./check.py --fix{Colors.NC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()