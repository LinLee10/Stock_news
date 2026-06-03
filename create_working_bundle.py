#!/usr/bin/env python3
"""
Create working bundle ZIP for stonk_news system
Includes only working, in-use project files while excluding caches/secrets/junk
"""

import os
import sys
import json
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import re
from typing import Set, List, Dict, Any
import ast

# Current repo root
REPO_ROOT = Path("/Users/linlee10/Desktop/Personal_projects/Stonk_news")
os.chdir(REPO_ROOT)

# Exclusion patterns
EXCLUDE_PATTERNS = [
    # VCS/IDE
    ".git/**", ".svn/**", ".hg/**", ".idea/**", ".vscode/**",
    # Python build artifacts
    "**/__pycache__/**", "**/*.pyc", "**/*.pyo", "dist/**", "build/**", "*.egg-info/**",
    # Caches
    ".pytest_cache/**", ".mypy_cache/**", ".ruff_cache/**", ".coverage", ".ipynb_checkpoints/**",
    # Virtual envs
    ".venv/**", "venv/**", "env/**",
    # OS cruft
    ".DS_Store", "Thumbs.db",
    # Large data/artifacts (but keep small samples)
    "*.zip", "*.png", "*.jpg", "*.jpeg", "review_bundle/**",
    # Large logs (keep small ones)
    "pipeline_run.log", "*.log",
]

# Secret patterns to scan for
SECRET_PATTERNS = [
    r'API_KEY\s*=\s*["\'][^"\']+["\']',
    r'SECRET\s*=\s*["\'][^"\']+["\']',
    r'PASSWORD\s*=\s*["\'][^"\']+["\']',
    r'TOKEN\s*=\s*["\'][^"\']+["\']',
    r'-----BEGIN [^-]+ KEY-----',
]

class WorkingBundleCreator:
    def __init__(self):
        self.repo_root = REPO_ROOT
        self.included_files = []
        self.excluded_files = []
        self.import_graph = set()
        
    def get_git_files(self) -> Set[str]:
        """Get all git-tracked files"""
        try:
            result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, cwd=self.repo_root)
            if result.returncode == 0:
                return set(result.stdout.strip().split('\n'))
        except Exception as e:
            print(f"Git not available: {e}")
        return set()
    
    def should_exclude(self, filepath: str) -> tuple[bool, str]:
        """Check if file should be excluded"""
        path = Path(filepath)
        
        # Check file size (exclude > 50MB)
        try:
            if path.exists() and path.stat().st_size > 50 * 1024 * 1024:
                return True, "file_too_large"
        except:
            pass
        
        # Check exclusion patterns
        for pattern in EXCLUDE_PATTERNS:
            if path.match(pattern.replace('**', '*')):
                return True, f"pattern:{pattern}"
        
        # Exclude .env but keep .env.example
        if path.name == '.env':
            return True, "secrets"
        
        return False, ""
    
    def scan_for_secrets(self, filepath: str) -> bool:
        """Scan file for potential secrets"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern in SECRET_PATTERNS:
                    if re.search(pattern, content):
                        return True
        except:
            pass
        return False
    
    def build_import_graph(self) -> None:
        """Build static import graph from Python files"""
        entrypoints = [
            "main.py",
            "services/alpha_vantage_manager.py", 
            "services/data_sources/yfinance_provider.py",
            "integrations/newsapi_client.py",
            "services/audit_logger.py",
            "pipeline_runner.py",  # May not exist yet
        ]
        
        print("Building import graph from entrypoints...")
        for entry in entrypoints:
            self._analyze_python_imports(entry)
    
    def _analyze_python_imports(self, filepath: str) -> None:
        """Analyze imports in a Python file"""
        path = Path(filepath)
        if not path.exists():
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_to_import_graph(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_to_import_graph(node.module)
        except Exception as e:
            print(f"Could not analyze {filepath}: {e}")
    
    def _add_to_import_graph(self, module_name: str) -> None:
        """Add module to import graph"""
        if not module_name:
            return
            
        # Convert module names to file paths
        parts = module_name.split('.')
        
        # Check for local modules
        possible_paths = [
            '/'.join(parts) + '.py',
            '/'.join(parts) + '/__init__.py',
        ]
        
        for possible_path in possible_paths:
            if Path(possible_path).exists():
                self.import_graph.add(possible_path)
                # Recursively analyze imports
                self._analyze_python_imports(possible_path)
    
    def collect_files(self) -> None:
        """Collect all files to include"""
        print("Collecting files...")
        
        # Start with git tracked files
        git_files = self.get_git_files()
        
        # Add important untracked files
        important_untracked = [
            "conftest.py", "pytest.ini", "settings.py",
            "DRYRUN_COMMANDS.md", 
            "secrets/.gitignore", "secrets/README.md", "secrets/template.env",
            "secrets/load_secrets.py", "secrets/usage_examples.py", "secrets/quick_reference.md"
        ]
        
        all_files = git_files | {f for f in important_untracked if Path(f).exists()}
        
        # Build import graph to identify truly used files
        self.build_import_graph()
        
        # Process each file
        for filepath in sorted(all_files):
            path = Path(filepath)
            
            # Skip if doesn't exist
            if not path.exists():
                self.excluded_files.append({"path": filepath, "reason": "file_not_found"})
                continue
            
            # Check exclusion rules
            should_exclude, reason = self.should_exclude(filepath)
            if should_exclude:
                self.excluded_files.append({"path": filepath, "reason": reason})
                continue
            
            # Scan for secrets
            if self.scan_for_secrets(filepath):
                self.excluded_files.append({"path": filepath, "reason": "contains_secrets"})
                continue
            
            # Include the file
            try:
                stat = path.stat()
                with open(path, 'rb') as f:
                    content = f.read()
                    file_hash = hashlib.sha256(content).hexdigest()
                
                self.included_files.append({
                    "path": filepath,
                    "bytes": stat.st_size,
                    "sha256": file_hash
                })
            except Exception as e:
                self.excluded_files.append({"path": filepath, "reason": f"read_error:{e}"})
    
    def create_zip(self) -> str:
        """Create the working bundle ZIP"""
        zip_path = "working_bundle.zip"
        print(f"Creating ZIP: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_info in self.included_files:
                filepath = file_info["path"]
                try:
                    zf.write(filepath, filepath)
                except Exception as e:
                    print(f"Failed to add {filepath}: {e}")
        
        return zip_path
    
    def create_manifests(self) -> None:
        """Create manifest files"""
        now = datetime.now(timezone.utc)
        
        # JSON manifest
        manifest_data = {
            "repo_root": str(self.repo_root.absolute()),
            "generated_at": now.isoformat(),
            "included_files": self.included_files,
            "excluded_files": self.excluded_files,
            "summary": {
                "total_files_included": len(self.included_files),
                "total_bytes": sum(f["bytes"] for f in self.included_files),
                "total_files_excluded": len(self.excluded_files)
            }
        }
        
        with open("working_manifest.json", 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        # Text manifest
        with open("working_manifest.txt", 'w') as f:
            f.write("STONK NEWS WORKING BUNDLE MANIFEST\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {now.isoformat()}\n")
            f.write(f"Repo Root: {self.repo_root}\n")
            f.write(f"Total Files Included: {len(self.included_files)}\n")
            f.write(f"Total Size: {sum(f['bytes'] for f in self.included_files):,} bytes\n")
            f.write(f"Total Files Excluded: {len(self.excluded_files)}\n\n")
            
            f.write("DIRECTORY STRUCTURE (TOP 2 LEVELS):\n")
            f.write("-" * 40 + "\n")
            dirs = {}
            for file_info in self.included_files:
                parts = file_info["path"].split('/')
                if len(parts) > 1:
                    top_dir = parts[0]
                    second_level = parts[1] if len(parts) > 1 else ""
                    if top_dir not in dirs:
                        dirs[top_dir] = set()
                    if second_level:
                        dirs[top_dir].add(second_level)
            
            for top_dir, subdirs in sorted(dirs.items()):
                f.write(f"{top_dir}/\n")
                for subdir in sorted(subdirs):
                    f.write(f"  {subdir}\n")
        
        # SHA256 hashes
        with open("hashes.sha256", 'w') as f:
            for file_info in self.included_files:
                f.write(f"{file_info['sha256']}  {file_info['path']}\n")
    
    def create_sanity_report(self, zip_path: str) -> None:
        """Create sanity check report"""
        total_size = sum(f["bytes"] for f in self.included_files)
        largest_files = sorted(self.included_files, key=lambda x: x["bytes"], reverse=True)[:10]
        
        with open("sanity_report.md", 'w') as f:
            f.write("# Stonk News Working Bundle - Sanity Report\n\n")
            f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write(f"**Bundle:** `{zip_path}`\n")
            f.write(f"**Total Files:** {len(self.included_files)}\n")
            f.write(f"**Total Size:** {total_size:,} bytes ({total_size/1024/1024:.1f} MB)\n\n")
            
            if total_size > 200 * 1024 * 1024:
                f.write("⚠️ **WARNING: Bundle exceeds 200MB**\n\n")
            
            f.write("## Largest Files\n")
            for i, file_info in enumerate(largest_files, 1):
                size_mb = file_info["bytes"] / 1024 / 1024
                f.write(f"{i}. `{file_info['path']}` - {size_mb:.1f} MB\n")
            
            f.write("\n## Key Components Included\n")
            key_components = [
                "README.md", "requirements.txt", "main.py", 
                "services/", "tests/", "config/", "report/"
            ]
            for component in key_components:
                found = any(component in f["path"] for f in self.included_files)
                status = "✅" if found else "❌"
                f.write(f"- {status} {component}\n")
            
            f.write("\n## Usage\n")
            f.write(f"```bash\n")
            f.write(f"unzip {zip_path}\n")
            f.write(f"cd stonk_news\n")
            f.write(f"pip install -r requirements.txt\n")
            f.write(f"python main.py\n")
            f.write(f"```\n")
            
            f.write("\n## Verification\n")
            f.write(f"```bash\n")
            f.write(f"sha256sum -c hashes.sha256\n")
            f.write(f"```\n")

def main():
    creator = WorkingBundleCreator()
    
    print("🔍 Analyzing stonk_news repository...")
    creator.collect_files()
    
    print("📦 Creating working bundle...")
    zip_path = creator.create_zip()
    
    print("📋 Creating manifests...")
    creator.create_manifests()
    creator.create_sanity_report(zip_path)
    
    # Print summary
    total_size = sum(f["bytes"] for f in creator.included_files)
    print("\n" + "=" * 60)
    print("WORKING BUNDLE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Bundle: {zip_path}")
    print(f"Files Included: {len(creator.included_files)}")
    print(f"Files Excluded: {len(creator.excluded_files)}")
    print(f"Total Size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    print("\n📁 Generated Files:")
    print(f"- working_bundle.zip")
    print(f"- working_manifest.json")
    print(f"- working_manifest.txt") 
    print(f"- hashes.sha256")
    print(f"- sanity_report.md")
    
    print("\n📋 Sample Included Files (first 10):")
    for i, file_info in enumerate(creator.included_files[:10], 1):
        size_kb = file_info["bytes"] / 1024
        print(f"{i:2d}. {file_info['path']:<40} ({size_kb:6.1f} KB)")
    
    print("\n🚫 Sample Excluded Files (first 10):")
    for i, file_info in enumerate(creator.excluded_files[:10], 1):
        print(f"{i:2d}. {file_info['path']:<40} ({file_info['reason']})")

if __name__ == "__main__":
    main()