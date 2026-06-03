#!/usr/bin/env python3
"""
Create curated review bundle containing only working files, tests, and necessary docs.
Excludes dependencies, caches, build artifacts, and secrets.
"""

import os
import shutil
import ast
import json
import zipfile
from pathlib import Path
from typing import Set, Dict, List, Tuple
import re

class ReviewBundleCreator:
    def __init__(self, repo_root: Path, bundle_dir: Path):
        self.repo_root = repo_root
        self.bundle_dir = bundle_dir
        self.included_files = set()
        self.file_reasons = {}
        self.import_graph = {}
        
    def create_bundle(self):
        """Create the complete review bundle."""
        print("🚀 Creating review bundle...")
        
        # Clean and create bundle directory
        if self.bundle_dir.exists():
            shutil.rmtree(self.bundle_dir)
        self.bundle_dir.mkdir(parents=True)
        
        # Discover entrypoints and working files
        self.discover_entrypoints()
        self.discover_tests()
        self.discover_configs()
        self.discover_docs()
        self.resolve_import_dependencies()
        
        # Copy files to bundle
        self.copy_files_to_bundle()
        
        # Create manifest files
        self.create_file_index()
        self.create_entrypoints_manifest()
        self.create_tests_summary()
        
        # Create ZIP
        self.create_zip()
        
        print(f"✅ Review bundle created: {len(self.included_files)} files")
        
    def discover_entrypoints(self):
        """Find main entrypoints and mark them for inclusion."""
        print("📍 Discovering entrypoints...")
        
        # Main Python entrypoints
        main_scripts = [
            'main.py',
            'prediction.py', 
            'news_scraper.py',
            'charts.py',
            'email_report.py'
        ]
        
        for script in main_scripts:
            script_path = self.repo_root / script
            if script_path.exists():
                self.include_file(script_path, f"Main entrypoint script")
                
        # API endpoints
        api_files = list((self.repo_root / 'api').glob('*.py')) if (self.repo_root / 'api').exists() else []
        for api_file in api_files:
            self.include_file(api_file, "API endpoint module")
            
        # Microservices
        microservices_dirs = ['microservices', 'services']
        for ms_dir in microservices_dirs:
            ms_path = self.repo_root / ms_dir
            if ms_path.exists():
                for py_file in ms_path.rglob('*.py'):
                    self.include_file(py_file, f"Service module in {ms_dir}")
                    
        print(f"   Found {len([f for f in self.included_files if f.suffix == '.py'])} Python entrypoints")
        
    def discover_tests(self):
        """Find all test files."""
        print("🧪 Discovering tests...")
        
        test_patterns = [
            'tests/**/*.py',
            'test/**/*.py', 
            '**/test_*.py',
            '**/*_test.py',
            '**/*.test.py',
            '**/*.spec.py'
        ]
        
        for pattern in test_patterns:
            for test_file in self.repo_root.rglob(pattern):
                # Skip if in excluded directories
                if any(part in str(test_file) for part in ['.venv', 'venv', 'node_modules', '__pycache__', 'site-packages']):
                    continue
                self.include_file(test_file, "Test file")
                
        test_count = len([f for f in self.included_files if 'test' in str(f).lower()])
        print(f"   Found {test_count} test files")
        
    def discover_configs(self):
        """Find configuration and CI files."""
        print("⚙️  Discovering configs...")
        
        config_patterns = [
            'pyproject.toml',
            'setup.py',
            'setup.cfg', 
            'requirements*.txt',
            'package.json',
            'package-lock.json',
            'poetry.lock',
            'Pipfile*',
            'tox.ini',
            'pytest.ini',
            '.github/**/*.yml',
            '.github/**/*.yaml',
            'docker/**/*',
            'config/**/*.py',
            'config/**/*.yaml',
            'config/**/*.json',
            '.env.example',
            'secrets.env.sample',
            'Dockerfile*',
            'docker-compose*.yml',
            'Makefile'
        ]
        
        for pattern in config_patterns:
            for config_file in self.repo_root.rglob(pattern):
                # Skip if in excluded directories
                if any(part in str(config_file) for part in ['.venv', 'venv', 'node_modules', '__pycache__', 'site-packages']):
                    continue
                    
                # Skip secrets (but include examples)
                if config_file.name in ['.env', 'secrets.env'] and 'example' not in config_file.name and 'sample' not in config_file.name:
                    continue
                    
                self.include_file(config_file, "Configuration file")
                
        config_count = len([f for f in self.included_files if any(ext in str(f) for ext in ['.yml', '.yaml', '.toml', '.cfg', '.ini', '.json'])])
        print(f"   Found {config_count} config files")
        
    def discover_docs(self):
        """Find documentation files."""
        print("📖 Discovering docs...")
        
        doc_patterns = [
            'README*',
            'CHANGELOG*',
            'CONTRIBUTING*', 
            'LICENSE*',
            'docs/**/*.md',
            '*.md',
            'SYSTEM_INTEGRATION_COMPLETE.md'
        ]
        
        for pattern in doc_patterns:
            for doc_file in self.repo_root.rglob(pattern):
                # Skip if in excluded directories
                if any(part in str(doc_file) for part in ['.venv', 'venv', 'node_modules', '__pycache__', 'site-packages']):
                    continue
                    
                if doc_file.is_file() and doc_file.stat().st_size < 10_000_000:  # Skip huge docs
                    self.include_file(doc_file, "Documentation")
                    
        doc_count = len([f for f in self.included_files if f.suffix == '.md'])
        print(f"   Found {doc_count} documentation files")
        
    def resolve_import_dependencies(self):
        """Resolve import dependencies from entrypoints and tests."""
        print("🔗 Resolving import dependencies...")
        
        # Get all Python files already included
        python_files = [f for f in self.included_files if f.suffix == '.py']
        
        # Build import graph
        for py_file in python_files:
            imports = self.extract_imports(py_file)
            self.import_graph[str(py_file.relative_to(self.repo_root))] = imports
            
            # Include imported local modules
            for imp in imports:
                local_module = self.find_local_module(imp)
                if local_module and local_module not in self.included_files:
                    self.include_file(local_module, f"Imported by {py_file.name}")
                    
        print(f"   Resolved dependencies for {len(python_files)} Python files")
        
    def extract_imports(self, py_file: Path) -> List[str]:
        """Extract import statements from a Python file."""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        if node.names:
                            for alias in node.names:
                                if alias.name != '*':
                                    imports.append(f"{node.module}.{alias.name}")
                                    
            return imports
        except:
            return []
            
    def find_local_module(self, module_name: str) -> Path:
        """Find local module file for import name."""
        # Split module path
        parts = module_name.split('.')
        
        # Try different combinations
        for i in range(len(parts), 0, -1):
            module_path = '/'.join(parts[:i])
            
            # Try as .py file
            py_file = self.repo_root / f"{module_path}.py"
            if py_file.exists():
                return py_file
                
            # Try as package __init__.py
            init_file = self.repo_root / module_path / '__init__.py'
            if init_file.exists():
                return init_file
                
        return None
        
    def include_file(self, file_path: Path, reason: str):
        """Mark a file for inclusion in the bundle."""
        if file_path.exists() and file_path.is_file():
            # Check size limit (skip files > 10MB)
            if file_path.stat().st_size > 10_000_000:
                print(f"   Skipping large file: {file_path} ({file_path.stat().st_size // 1000000}MB)")
                return
                
            self.included_files.add(file_path)
            self.file_reasons[str(file_path.relative_to(self.repo_root))] = reason
            
    def copy_files_to_bundle(self):
        """Copy included files to bundle directory."""
        print("📦 Copying files to bundle...")
        
        for file_path in self.included_files:
            relative_path = file_path.relative_to(self.repo_root)
            bundle_file_path = self.bundle_dir / relative_path
            
            # Create parent directories
            bundle_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file (with secret redaction for certain files)
            if file_path.name in ['secrets.env', '.env']:
                self.copy_with_redaction(file_path, bundle_file_path)
            else:
                shutil.copy2(file_path, bundle_file_path)
                
        print(f"   Copied {len(self.included_files)} files")
        
    def copy_with_redaction(self, src: Path, dst: Path):
        """Copy file with secret redaction."""
        try:
            with open(src, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Redact secret values
            redacted = re.sub(r'(API_KEY|PASSWORD|SECRET|TOKEN)=.+', r'\1=***REDACTED***', content, flags=re.IGNORECASE)
            
            with open(dst, 'w', encoding='utf-8') as f:
                f.write(redacted)
        except:
            # Fallback to regular copy if redaction fails
            shutil.copy2(src, dst)
            
    def create_file_index(self):
        """Create FILE_INDEX.md manifest."""
        print("📋 Creating file index...")
        
        # Categorize files
        categories = {
            'Tests': [],
            'Source Code': [],
            'API Endpoints': [],
            'Services': [], 
            'Configuration': [],
            'Documentation': [],
            'CI/CD': [],
            'Data/Schema': []
        }
        
        for file_path in sorted(self.included_files):
            rel_path = file_path.relative_to(self.repo_root)
            size = file_path.stat().st_size
            reason = self.file_reasons.get(str(rel_path), "Included")
            
            file_info = {
                'path': str(rel_path),
                'size': size,
                'reason': reason
            }
            
            # Categorize
            if 'test' in str(rel_path).lower():
                categories['Tests'].append(file_info)
            elif str(rel_path).startswith('api/'):
                categories['API Endpoints'].append(file_info)
            elif str(rel_path).startswith('services/') or str(rel_path).startswith('microservices/'):
                categories['Services'].append(file_info)
            elif str(rel_path).startswith('config/') or rel_path.suffix in ['.yml', '.yaml', '.toml', '.cfg', '.ini']:
                categories['Configuration'].append(file_info)
            elif rel_path.suffix == '.md' or str(rel_path).startswith('docs/'):
                categories['Documentation'].append(file_info)
            elif str(rel_path).startswith('.github/'):
                categories['CI/CD'].append(file_info)
            elif rel_path.suffix == '.py':
                categories['Source Code'].append(file_info)
            else:
                categories['Data/Schema'].append(file_info)
                
        # Write manifest
        with open(self.bundle_dir / 'FILE_INDEX.md', 'w') as f:
            f.write("# Review Bundle File Index\\n\\n")
            f.write(f"Generated: 2025-08-31\\n")
            f.write(f"Total files: {len(self.included_files)}\\n")
            f.write(f"Total size: {sum(fp.stat().st_size for fp in self.included_files) // 1024}KB\\n\\n")
            
            for category, files in categories.items():
                if files:
                    f.write(f"## {category} ({len(files)} files)\\n\\n")
                    
                    for file_info in files:
                        size_str = f"{file_info['size'] // 1024}KB" if file_info['size'] > 1024 else f"{file_info['size']}B"
                        f.write(f"- **{file_info['path']}** ({size_str}) - {file_info['reason']}\\n")
                        
                    f.write("\\n")
                    
    def create_entrypoints_manifest(self):
        """Create ENTRYPOINTS_AND_IMPORTS.md manifest."""
        print("🎯 Creating entrypoints manifest...")
        
        with open(self.bundle_dir / 'ENTRYPOINTS_AND_IMPORTS.md', 'w') as f:
            f.write("# Entrypoints and Import Analysis\\n\\n")
            
            # Entrypoints section
            f.write("## Detected Entrypoints\\n\\n")
            main_scripts = ['main.py', 'prediction.py', 'news_scraper.py', 'charts.py', 'email_report.py']
            for script in main_scripts:
                if (self.repo_root / script).exists():
                    f.write(f"- **{script}** - Main application entrypoint\\n")
                    
            # API routes
            f.write("\\n### API Routes\\n")
            api_files = [f for f in self.included_files if str(f).startswith(str(self.repo_root / 'api'))]
            for api_file in api_files:
                rel_path = api_file.relative_to(self.repo_root)
                f.write(f"- **{rel_path}** - API endpoint module\\n")
                
            # Import graph summary
            f.write("\\n## Import Graph Summary\\n\\n")
            f.write(f"Total Python modules analyzed: {len(self.import_graph)}\\n\\n")
            
            for module, imports in list(self.import_graph.items())[:10]:  # Top 10
                if imports:
                    f.write(f"**{module}**:\\n")
                    local_imports = [imp for imp in imports if not imp.startswith(('os', 'sys', 'json', 'logging', 'datetime'))][:5]
                    for imp in local_imports:
                        f.write(f"  - {imp}\\n")
                    f.write("\\n")
                    
    def create_tests_summary(self):
        """Create TESTS_SUMMARY.md manifest."""
        print("🧪 Creating tests summary...")
        
        test_files = [f for f in self.included_files if 'test' in str(f).lower() and f.suffix == '.py']
        
        with open(self.bundle_dir / 'TESTS_SUMMARY.md', 'w') as f:
            f.write("# Test Inventory Summary\\n\\n")
            f.write(f"Total test files: {len(test_files)}\\n\\n")
            
            # Group by directory
            test_dirs = {}
            for test_file in test_files:
                rel_path = test_file.relative_to(self.repo_root)
                dir_name = str(rel_path.parent)
                if dir_name not in test_dirs:
                    test_dirs[dir_name] = []
                test_dirs[dir_name].append(rel_path.name)
                
            for test_dir, files in sorted(test_dirs.items()):
                f.write(f"## {test_dir}/ ({len(files)} files)\\n\\n")
                for file in sorted(files):
                    f.write(f"- {file}\\n")
                f.write("\\n")
                
    def create_zip(self):
        """Create ZIP archive of the bundle."""
        print("📦 Creating ZIP archive...")
        
        zip_path = self.repo_root / 'review_bundle.zip'
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.bundle_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(self.bundle_dir.parent)
                    zipf.write(file_path, arc_path)
                    
        zip_size = zip_path.stat().st_size
        print(f"   Created {zip_path} ({zip_size // (1024*1024)}MB)")
        
        return zip_path

def main():
    repo_root = Path.cwd()
    bundle_dir = repo_root / 'review_bundle'
    
    creator = ReviewBundleCreator(repo_root, bundle_dir)
    creator.create_bundle()
    
    # Print summary
    print("\\n" + "="*60)
    print("REVIEW BUNDLE SUMMARY")
    print("="*60)
    print(f"📁 Files included: {len(creator.included_files)}")
    print(f"📊 Total size: {sum(fp.stat().st_size for fp in creator.included_files) // (1024*1024)}MB")
    print(f"🧪 Test files: {len([f for f in creator.included_files if 'test' in str(f)])}")
    print(f"🐍 Python files: {len([f for f in creator.included_files if f.suffix == '.py'])}")
    print(f"📋 Config files: {len([f for f in creator.included_files if f.suffix in ['.yml', '.yaml', '.toml', '.cfg']])}")
    print(f"📖 Documentation: {len([f for f in creator.included_files if f.suffix == '.md'])}")
    
    print("\\n✅ Review bundle ready for handoff!")

if __name__ == '__main__':
    main()