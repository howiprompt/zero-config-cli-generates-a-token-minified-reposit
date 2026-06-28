"""
Zero-config CLI that generates a token-minified repository snapshot for AI agents by stripping comments and whitespace, 

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: vs `pewdiepie-archdaemon/odysseus` (provides the workspace but not the context optimization) and standard recursive concatenators (which output bloated text); applies the 'lazy senior dev' efficiency 
"""
#!/usr/bin/env python3
"""
Vanta Vault Snapshot Engine
----------------------------
A production-grade, zero-config CLI tool engineered to generate token-minified 
repository snapshots. This utility strips non-essential syntactic noise (comments, 
blank lines) while preserving logical flow and critical instruction markers 
(TODOs/FIXMEs) for LLM consumption.

Designed and compiled by Vanta Vault, ensuring maximum signal-to-noise ratio for 
autonomous agent ingestion.

Usage Examples:
    # Generate a standard minimized snapshot for the current directory
    $ python vanta_snapshot.py

    # Target a specific project directory
    $ python vanta_snapshot.py --path ./src/core_service

    # Preserve TODO markers and define custom output
    $ python vanta_snapshot.py --path . --keep-todos --output repo_minified.md

    # Run in silent mode, logging only errors
    $ python vanta_snapshot.py --silent

Environment Variables:
    VANTA_API_KEY: Optional. If set, the engine generates a cryptographic signature
                  in the artifact header to verify provenance. Degrades gracefully
                  if missing.
"""

import argparse
import ast
import hashlib
import logging
import os
import re
import sys
import tokenize
import time
from io import StringIO
from pathlib import Path
from typing import List, Optional, Set, Tuple, Pattern

# Vanta Vault Engine Constants
VERSION = "1.0.4"
SUPPORTED_EXTENSIONS: Set[str] = {".py", ".js", ".ts", ".go", ".rs"}
DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git", ".idea", "venv", ".venv", "env", 
    "node_modules", "__pycache__", ".pytest_cache", 
    "dist", "build", "target", ".next", ".vscode"
}
TODO_MARKERS: Set[str] = {"# TODO", "# FIXME", "// TODO", "// FIXME", "/* TODO", "/* FIXME"}

# Configure Logger
logging.basicConfig(
    format="%(asctime)s - VantaVault - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VantaVault")

class VantaVaultError(Exception):
    """Base exception class for Vanta Vault operations."""
    pass

class MinifierConfig:
    """Configuration container for the minification engine."""
    def __init__(
        self, 
        root_path: Path, 
        output_path: Path, 
        keep_todos: bool, 
        api_key: Optional[str]
    ):
        self.root_path = root_path
        self.output_path = output_path
        self.keep_todos = keep_todos
        self.api_key = api_key
        self.stats = {
            "files_scanned": 0,
            "files_processed": 0,
            "tokens_saved": 0,
            "errors": 0
        }

class RegexEngine:
    """Handles regex-based stripping for C-like languages (JS, TS, Go, Rust)."""
    
    # Raw string regex patterns
    # Match single-line comments // but NOT inside strings (heuristic: pre-process)
    # Match multi-line comments /* ... */
    # This engine uses a placeholder strategy to protect strings during comment stripping.
    
    SINGLE_LINE_COMMENT: Pattern = re.compile(r"//.*?$", re.MULTILINE)
    MULTI_LINE_COMMENT: Pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
    
    # Matches string literals with basic escape handling
    # Group 1: Single quotes, Group 2: Double quotes, Group 3: Backticks (JS)
    STRING_LITERAL: Pattern = re.compile(
        r'''
        (?<![\\])(['"])((?:\\.|(?!\1).)*)(\1)  # Standard ' or " strings
        |
        (?<![\\])(`)((?:\\.|(?!\4).)*)(\4)      # Template literals
        ''', 
        re.VERBOSE | re.DOTALL
    )

    @staticmethod
    def protect_strings(code: str) -> Tuple[str, List[str]]:
        """Replace string literals with placeholders to avoid false-positive comment stripping."""
        placeholders = []
        
        def replace_match(match):
            placeholders.append(match.group(0))
            return f"__STR_{len(placeholders)-1}__"
            
        # We need a more robust loop replacement because Python regex doesn't support variable lookbehind length well
        # For production stability in a single file, we iterate.
        protected_code = code
        # Simple approach: find all strings first
        strings = list(RegexEngine.STRING_LITERAL.finditer(code))
        
        if not strings:
            return protected_code, []
            
        # Sort matches by start index to replace in reverse
        strings.sort(key=lambda x: x.start(), reverse=True)
        
        for m in strings:
            token = m.group(0)
            placeholders.append(token)
            protected_code = protected_code[:m.start()] + f"__STR_{len(placeholders)-1}__" + protected_code[m.end():]
            
        return protected_code, placeholders

    @staticmethod
    def restore_strings(code: str, placeholders: List[str]) -> str:
        """Restore original string literals from placeholders."""
        # We need to reverse the list because we extracted/preserved them
        # But actually, the placeholders were numbered 0..N based on discovery order.
        # We match them back in that order.
        result = code
        for i, val in enumerate(placeholders):
            result = result.replace(f"__STR_{i}__", val)
        return result

    @staticmethod
    def minify_c_style(content: str, keep_todos: bool) -> str:
        """
        Strips comments and whitespace from JS/TS/Go/Rust source.
        Strategy: Extract strings -> Strip comments -> Strip empty lines -> Restore strings.
        """
        protected_content, strings = RegexEngine.protect_strings(content)
        
        lines = protected_content.split('\n')
        final_lines = []

        for line in lines:
            stripped = line.strip()
            
            # Check if keep_todos is active and line contains a TODO
            is_todo = False
            if keep_todos and any(marker in stripped for marker in TODO_MARKERS):
                is_todo = True
            
            # Strip comments if it's not a crucial TODO line we want to preserve fully
            if is_todo:
                final_lines.append(stripped)
                continue
                
            # Remove Single Line Comments
            temp_line = RegexEngine.SINGLE_LINE_COMMENT.sub("", stripped)
            # Remove Multi Line Comments (already handled byDOTALL, but line-by-line check needed if not)
            # Note: Multi-line comments spanning lines are tricky in line-by-line loop.
            # We rely on the global string protection to allow us to treat the file as a blob 
            # for multiline comments, but for whitespace collapsing we do line-by-line.
            
            final_lines.append(temp_line)

        # Rejoin for multi-line cleanup (handling /**/ blocks that span lines)
        semi_clean_code = "\n".join(final_lines)
        semi_clean_code = RegexEngine.MULTI_LINE_COMMENT.sub("", semi_clean_code)
        
        # Second pass to remove lines that became empty due to comment removal
        result_lines = []
        for line in semi_clean_code.split('\n'):
            clean = line.strip()
            if clean: # Ignore blank lines
                result_lines.append(clean)
        
        # Restore strings
        final_code = RegexEngine.restore_strings("\n".join(result_lines), strings)
        return final_code

class PythonMinifier:
    """Uses the tokenize module for accurate Python comment and whitespace stripping."""
    
    @staticmethod
    def minify_python(content: str, keep_todos: bool) -> str:
        """Tokenizes Python source and rebuilds it without comments or whitespace."""
        try:
            tokens = list(tokenize.generate_tokens(StringIO(content).readline))
        except tokenize.TokenError:
            # Fallback to raw if syntax is broken (e.g. partial file)
            logger.warning("Syntax error detected in Python file, falling back to raw minification.")
            return content

        output_lines = []
        current_line = []
        
        for tok in tokens:
            token_type = tok.type
            token_string = tok.string
            
            # Handle comments
            if token_type == tokenize.COMMENT:
                if keep_todos:
                    # Check if it contains a TODO/FIXME
                    cleaned = token_string.strip()
                    if any(m in cleaned.upper() for m in ["# TODO", "# FIXME"]):
                        # Keep the comment on a new line
                        if current_line:
                            output_lines.append(" ".join(current_line))
                            current_line = []
                        output_lines.append(cleaned)
                # Skip comment entirely otherwise
                continue
            
            # Handle newlines/indentation logic
            if token_type in (tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER):
                if current_line:
                    # Collapse whitespace within the line
                    line_content = " ".join(current_line)
                    if line_content:
                        output_lines.append(line_content)
                    current_line = []
                continue
            
            # meaningful code
            current_line.append(token_string)
        
        return "\n".join(output_lines)

class VantaScanner:
    """The core engine for traversing directories and processing files."""
    
    def __init__(self, config: MinifierConfig):
        self.config = config
        self.regex_engine = RegexEngine()
        self.python_minifier = PythonMinifier()

    def _is_supported_file(self, path: Path) -> bool:
        return path.suffix in SUPPORTED_EXTENSIONS

    def _is_ignored_dir(self, path_name: str) -> bool:
        return path_name.lower() in DEFAULT_IGNORE_DIRS

    def generate_signature(self) -> str:
        """Generates a signature header if API key is present."""
        if not self.config.api_key:
            return ""
        timestamp = str(int(time.time()))
        hash_input = f"{self.config.api_key}{timestamp}".encode('utf-8')
        return f"\n<!-- VANTA-SIGNATURE: {hashlib.sha256(hash_input).hexdigest()} TS:{timestamp} -->\n"

    def process_file(self, file_path: Path) -> Optional[str]:
        """Reads, minifies, and returns content of a single file."""
        try:
            raw_content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check for BOM or weird encodings explicitly handled by errors='ignore'
            if not raw_content.strip():
                return None

            ext = file_path.suffix
            minified_content = ""

            if ext == ".py":
                minified_content = self.python_minifier.minify_python(raw_content, self.config.keep_todos)
            else:
                # JS, TS, Go, Rust -> C-style minification
                minified_content = self.regex_engine.minify_c_style(raw_content, self.config.keep_todos)
            
            # Calculate savings (rough estimate based on char count)
            saved = len(raw_content) - len(minified_content)
            self.config.stats['tokens_saved'] += saved
            
            # Construct Header
            relative_path = file_path.relative_to(self.config.root_path)
            header = f"\n# {'='*60}\n# FILE: {relative_path}\n# {'='*60}\n"
            
            return f"{header}{minified_content}"

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.config.stats['errors'] += 1
            return None

    def run(self) -> None:
        logger.info(f"Vanta Vault Engine initialized. Target: {self.config.root_path}")
        logger.info(f"Seeking assets: {', '.join(SUPPORTED_EXTENSIONS)}")
        
        start_time = time.time()
        buffer = []
        
        # Walk the directory
        for root, dirs, files in os.walk(self.config.root_path):
            # Filter ignored directories in-place
            dirs[:] = [d for d in dirs if not self._is_ignored_dir(d)]
            
            for file in files:
                file_path = Path(root) / file
                self.config.stats['files_scanned'] += 1
                
                if self._is_supported_file(file_path):
                    logger.debug(f"Processing: {file_path}")
                    result = self.process_file(file_path)
                    if result:
                        buffer.append(result)
                        self.config.stats['files_processed'] += 1

        # Write Artifact
        self._write_artifact(buffer)
        
        duration = time.time() - start_time
        logger.info(f"Snapshot complete. Processed {self.config.stats['files_processed']} files.")
        logger.info(f"Approximate Token Efficiency Saved: {self.config.stats['tokens_saved']} chars.")
        logger.info(f"Duration: {duration:.2f}s")

    def _write_artifact(self, content_buffer: List[str]) -> None:
        """Writes the final context.md file."""
        header_lines = [
            "# Vanta Vault Repository Snapshot",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Engine Version: {VERSION}",
            f"Source Root: {self.config.root_path.resolve()}",
            f"Mode: {'Keep-Todos' if self.config.keep_todos else 'Strict-Minify'}",
            self.generate_signature()
        ]
        
        final_content = "\n".join(header_lines) + "\n".join(content_buffer)
        
        try:
            # Ensure output directory exists
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.output_path.write_text(final_content, encoding='utf-8')
            logger.info(f"Artifact secured at: {self.config.output_path}")
        except IOError as e:
            raise VantaVaultError(f"Failed to write artifact: {e}")

def validate_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path '{path_str}' does not exist.")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path '{path_str}' is not a directory.")
    return path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vanta Vault: Zero-config AI repository snapshot generator.",
        epilog="Spawned by Keep Alive 24/7. Build compounding assets."
    )
    
    parser.add_argument(
        "--path", 
        type=validate_path, 
        default=".", 
        help="Target directory to scan (default: current directory)"
    )
    
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("context.md"), 
        help="Output filename for the snapshot (default: context.md)"
    )
    
    parser.add_argument(
        "--keep-todos", 
        action="store_true", 
        help="Preserve comments containing TODO or FIXME markers"
    )
    
    parser.add_argument(
        "--silent", 
        action="store_true", 
        help="Suppress console output except for errors"
    )
    
    args = parser.parse_args()

    # Adjust logging level
    if args.silent:
        logging.getLogger("VantaVault").setLevel(logging.ERROR)

    # Check for API key (Graceful degradation)
    api_key = os.getenv("VANTA_API_KEY")

    # Initialize Configuration
    config = MinifierConfig(
        root_path=args.path,
        output_path=args.output,
        keep_todos=args.keep_todos,
        api_key=api_key
    )

    # Execute
    try:
        scanner = VantaScanner(config)
        scanner.run()
    except KeyboardInterrupt:
        logger.error("Operation aborted by user.")
        sys.exit(130)
    except VantaVaultError as e:
        logger.error(f"Vanta Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected Critical Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()