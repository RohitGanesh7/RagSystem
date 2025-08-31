import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_openai import ChatOpenAI
from datetime import datetime
import subprocess
import tempfile

@dataclass
class FileAnalysis:
    """Data class to store analysis results for a single file"""
    file_path: str
    language: str
    lines_of_code: int
    complexity_score: int
    security_issues: List[str]
    performance_issues: List[str]
    quality_issues: List[str]
    syntax_errors: List[str]
    dependencies: List[str]

@dataclass
class ProjectMetrics:
    """Data class to store overall project metrics"""
    total_files: int
    total_lines: int
    total_functions: int
    total_classes: int
    complexity_distribution: Dict[str, int]
    security_risk_level: str
    test_coverage_estimate: float
    dependency_count: int

class ProjectCodeReviewSystem:
    def __init__(self, openai_api_key: str):
        """
        Initialize the Project Code Review System
        
        Args:
            openai_api_key: Your OpenAI API key
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2
        )
        
        # File extensions to analyze
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.rb': 'ruby',
            '.php': 'php',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        # Directories to ignore
        self.ignore_dirs = {
            '__pycache__', '.git', '.svn', 'node_modules', 
            'venv', 'env', '.env', 'dist', 'build', 
            '.pytest_cache', '.tox', 'coverage'
        }
        
        # Create agents and tools
        self.tools = self._create_tools()
        self.agents = self._create_agents()
    
    def _create_tools(self) -> List:
        """Create tools for project-wide analysis"""
        
        @tool("project_scanner")
        def project_scanner(project_path: str) -> str:
            """Scan project structure and collect metadata"""
            try:
                project_path = Path(project_path)
                if not project_path.exists():
                    return f"Error: Project path '{project_path}' does not exist"
                
                files_found = []
                total_lines = 0
                
                for file_path in project_path.rglob('*'):
                    # Skip directories and ignored paths
                    if file_path.is_dir() or any(ignore in str(file_path) for ignore in self.ignore_dirs):
                        continue
                    
                    # Check if file extension is supported
                    if file_path.suffix in self.supported_extensions:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                lines = len([line for line in content.split('\n') if line.strip()])
                                total_lines += lines
                            
                            files_found.append({
                                'path': str(file_path.relative_to(project_path)),
                                'language': self.supported_extensions[file_path.suffix],
                                'lines': lines,
                                'size': file_path.stat().st_size
                            })
                        except Exception as e:
                            continue
                
                return json.dumps({
                    'total_files': len(files_found),
                    'total_lines': total_lines,
                    'files': files_found,
                    'languages': list(set(f['language'] for f in files_found))
                }, indent=2)
                
            except Exception as e:
                return f"Error scanning project: {str(e)}"
        
        @tool("dependency_analyzer")
        def dependency_analyzer(project_path: str) -> str:
            """Analyze project dependencies and imports"""
            try:
                project_path = Path(project_path)
                dependencies = set()
                import_graph = {}
                
                for py_file in project_path.rglob('*.py'):
                    if any(ignore in str(py_file) for ignore in self.ignore_dirs):
                        continue
                    
                    try:
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Parse imports
                        tree = ast.parse(content)
                        file_imports = []
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    dependencies.add(alias.name.split('.')[0])
                                    file_imports.append(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    dependencies.add(node.module.split('.')[0])
                                    file_imports.append(node.module)
                        
                        if file_imports:
                            import_graph[str(py_file.relative_to(project_path))] = file_imports
                    
                    except Exception:
                        continue
                
                # Categorize dependencies
                stdlib_modules = {
                    'os', 'sys', 'json', 'datetime', 'time', 'random', 're', 'math',
                    'collections', 'itertools', 'functools', 'pathlib', 'typing',
                    'logging', 'unittest', 'pickle', 'csv', 'sqlite3', 'http',
                    'urllib', 'email', 'html', 'xml', 'hashlib', 'base64'
                }
                
                stdlib_deps = dependencies & stdlib_modules
                third_party_deps = dependencies - stdlib_modules
                
                return json.dumps({
                    'total_dependencies': len(dependencies),
                    'stdlib_dependencies': list(stdlib_deps),
                    'third_party_dependencies': list(third_party_deps),
                    'import_graph': import_graph
                }, indent=2)
                
            except Exception as e:
                return f"Error analyzing dependencies: {str(e)}"
        
        @tool("security_project_scanner")
        def security_project_scanner(project_path: str) -> str:
            """Scan entire project for security vulnerabilities"""
            try:
                project_path = Path(project_path)
                security_issues = []
                high_risk_files = []
                
                # Security patterns to check
                security_patterns = {
                    'high_risk': {
                        r'eval\s*\(': 'Code injection via eval()',
                        r'exec\s*\(': 'Code injection via exec()',
                        r'pickle\.loads?\s*\(': 'Insecure deserialization',
                        r'subprocess\..*shell\s*=\s*True': 'Command injection risk',
                        r'password\s*=\s*["\'][^"\']+["\']': 'Hardcoded password',
                        r'api_key\s*=\s*["\'][^"\']+["\']': 'Hardcoded API key',
                        r'secret\s*=\s*["\'][^"\']+["\']': 'Hardcoded secret'
                    },
                    'medium_risk': {
                        r'input\s*\([^)]*\)': 'Unsafe user input',
                        r'random\.random': 'Weak randomness for security',
                        r'except\s*:\s*pass': 'Silent exception handling',
                        r'assert\s+': 'Assertions in production code'
                    }
                }
                
                for py_file in project_path.rglob('*.py'):
                    if any(ignore in str(py_file) for ignore in self.ignore_dirs):
                        continue
                    
                    try:
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        file_issues = []
                        file_risk_score = 0
                        
                        # Check for security patterns
                        for risk_level, patterns in security_patterns.items():
                            for pattern, description in patterns.items():
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    issue = {
                                        'file': str(py_file.relative_to(project_path)),
                                        'risk_level': risk_level,
                                        'issue': description,
                                        'occurrences': len(matches)
                                    }
                                    file_issues.append(issue)
                                    
                                    if risk_level == 'high_risk':
                                        file_risk_score += len(matches) * 3
                                    else:
                                        file_risk_score += len(matches)
                        
                        if file_issues:
                            security_issues.extend(file_issues)
                            
                        if file_risk_score > 5:
                            high_risk_files.append({
                                'file': str(py_file.relative_to(project_path)),
                                'risk_score': file_risk_score
                            })
                    
                    except Exception:
                        continue
                
                return json.dumps({
                    'total_security_issues': len(security_issues),
                    'high_risk_files': high_risk_files,
                    'security_issues': security_issues,
                    'risk_summary': {
                        'high_risk': len([i for i in security_issues if i['risk_level'] == 'high_risk']),
                        'medium_risk': len([i for i in security_issues if i['risk_level'] == 'medium_risk'])
                    }
                }, indent=2)
                
            except Exception as e:
                return f"Error in security scan: {str(e)}"
        
        @tool("test_coverage_analyzer")
        def test_coverage_analyzer(project_path: str) -> str:
            """Analyze test coverage and testing practices"""
            try:
                project_path = Path(project_path)
                
                # Find test files
                test_files = []
                source_files = []
                
                for py_file in project_path.rglob('*.py'):
                    if any(ignore in str(py_file) for ignore in self.ignore_dirs):
                        continue
                    
                    file_name = py_file.name.lower()
                    if ('test_' in file_name or '_test' in file_name or 
                        'test' in str(py_file.parent).lower()):
                        test_files.append(str(py_file.relative_to(project_path)))
                    else:
                        source_files.append(str(py_file.relative_to(project_path)))
                
                # Estimate coverage based on test-to-source ratio
                test_ratio = len(test_files) / len(source_files) if source_files else 0
                estimated_coverage = min(test_ratio * 100, 85)  # Cap at 85% estimate
                
                # Check for testing frameworks
                testing_frameworks = []
                all_imports = set()
                
                for test_file_path in test_files:
                    try:
                        with open(project_path / test_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if 'pytest' in content or 'from pytest' in content:
                                testing_frameworks.append('pytest')
                            if 'unittest' in content or 'from unittest' in content:
                                testing_frameworks.append('unittest')
                            if 'nose' in content:
                                testing_frameworks.append('nose')
                    except Exception:
                        continue
                
                return json.dumps({
                    'test_files_count': len(test_files),
                    'source_files_count': len(source_files),
                    'test_to_source_ratio': round(test_ratio, 2),
                    'estimated_coverage_percentage': round(estimated_coverage, 1),
                    'testing_frameworks': list(set(testing_frameworks)),
                    'test_files': test_files[:10]  # Show first 10 test files
                }, indent=2)
                
            except Exception as e:
                return f"Error analyzing test coverage: {str(e)}"
        
        @tool("architecture_analyzer")
        def architecture_analyzer(project_path: str) -> str:
            """Analyze project architecture and structure"""
            try:
                project_path = Path(project_path)
                
                # Analyze directory structure
                directories = []
                for item in project_path.rglob('*'):
                    if item.is_dir() and not any(ignore in str(item) for ignore in self.ignore_dirs):
                        rel_path = str(item.relative_to(project_path))
                        if rel_path != '.':
                            directories.append(rel_path)
                
                # Check for common Python project patterns
                project_patterns = {
                    'has_setup_py': (project_path / 'setup.py').exists(),
                    'has_requirements_txt': (project_path / 'requirements.txt').exists(),
                    'has_pyproject_toml': (project_path / 'pyproject.toml').exists(),
                    'has_dockerfile': (project_path / 'Dockerfile').exists(),
                    'has_readme': any((project_path / f'README.{ext}').exists() 
                                     for ext in ['md', 'rst', 'txt']),
                    'has_gitignore': (project_path / '.gitignore').exists(),
                    'has_tests_dir': any('test' in d.lower() for d in directories),
                    'has_docs_dir': any('doc' in d.lower() for d in directories),
                    'has_src_structure': (project_path / 'src').exists()
                }
                
                # Count functions and classes
                total_functions = 0
                total_classes = 0
                module_complexities = {}
                
                for py_file in project_path.rglob('*.py'):
                    if any(ignore in str(py_file) for ignore in self.ignore_dirs):
                        continue
                    
                    try:
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        tree = ast.parse(content)
                        file_functions = 0
                        file_classes = 0
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                file_functions += 1
                                total_functions += 1
                            elif isinstance(node, ast.ClassDef):
                                file_classes += 1
                                total_classes += 1
                        
                        complexity = file_functions + file_classes * 2
                        module_complexities[str(py_file.relative_to(project_path))] = complexity
                    
                    except Exception:
                        continue
                
                return json.dumps({
                    'total_directories': len(directories),
                    'directory_structure': directories[:20],  # First 20 dirs
                    'project_patterns': project_patterns,
                    'total_functions': total_functions,
                    'total_classes': total_classes,
                    'most_complex_modules': dict(sorted(module_complexities.items(), 
                                                      key=lambda x: x[1], reverse=True)[:10])
                }, indent=2)
                
            except Exception as e:
                return f"Error analyzing architecture: {str(e)}"
        
        return [project_scanner, dependency_analyzer, security_project_scanner, 
                test_coverage_analyzer, architecture_analyzer]
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for project analysis"""
        
        # Project Structure Analyst
        structure_agent = Agent(
            role='Senior Project Architecture Analyst',
            goal='Analyze project structure, organization, and architectural patterns',
            backstory="""You are a senior software architect with expertise in analyzing 
            large codebases and identifying architectural patterns, structural issues, 
            and organizational problems. You excel at understanding project layout and 
            suggesting improvements for maintainability and scalability.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
        
        # Security Project Analyst
        security_project_agent = Agent(
            role='Project Security Auditor',
            goal='Perform comprehensive security analysis across the entire project',
            backstory="""You are a cybersecurity expert specializing in enterprise-level 
            security audits. You have extensive experience analyzing large codebases for 
            security vulnerabilities, identifying attack surfaces, and providing 
            comprehensive security recommendations for entire projects.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
        
        # Quality and Standards Analyst
        quality_project_agent = Agent(
            role='Enterprise Code Quality Analyst',
            goal='Assess overall code quality, standards compliance, and maintainability',
            backstory="""You are a senior code quality engineer with experience in 
            enterprise software development. You specialize in analyzing large codebases 
            for maintainability, adherence to standards, technical debt assessment, 
            and long-term sustainability.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
        
        # Testing and Coverage Analyst
        testing_agent = Agent(
            role='Testing Strategy Analyst',
            goal='Analyze testing practices, coverage, and quality assurance',
            backstory="""You are a QA expert and testing strategist with deep knowledge 
            of testing methodologies, coverage analysis, and quality assurance practices. 
            You excel at evaluating testing strategies and recommending improvements 
            for better software quality.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
        
        # Project Report Generator
        project_report_agent = Agent(
            role='Enterprise Technical Report Generator',
            goal='Generate comprehensive project analysis reports for stakeholders',
            backstory="""You are an expert technical writer specializing in enterprise 
            software analysis reports. You excel at synthesizing complex technical 
            analysis into clear, actionable reports for both technical teams and 
            business stakeholders.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        return {
            'structure': structure_agent,
            'security': security_project_agent,
            'quality': quality_project_agent,
            'testing': testing_agent,
            'report': project_report_agent
        }
    
    def create_project_review_crew(self, project_path: str) -> Crew:
        """Create a crew for comprehensive project review"""
        
        # Task 1: Project Structure Analysis
        structure_task = Task(
            description=f"""
            Perform a comprehensive analysis of the project structure and architecture:
            
            Project Path: {project_path}
            
            Your analysis should include:
            1. Overall project organization and directory structure
            2. Adherence to Python project conventions
            3. Module and package organization
            4. Configuration file analysis (setup.py, requirements.txt, etc.)
            5. Documentation structure
            6. Code organization patterns
            7. Architectural assessment
            8. Scalability considerations
            
            Use the project scanning tools to gather detailed information and provide 
            specific recommendations for structural improvements.
            """,
            agent=self.agents['structure'],
            expected_output="Comprehensive project structure analysis with architectural recommendations"
        )
        
        # Task 2: Security Project Analysis
        security_task = Task(
            description=f"""
            Conduct a comprehensive security audit of the entire project:
            
            Focus on:
            1. Project-wide security vulnerability assessment
            2. Dependency security analysis
            3. Configuration security review
            4. Data flow security analysis
            5. Authentication and authorization patterns
            6. Input validation across the project
            7. Security best practices compliance
            8. Risk assessment and prioritization
            
            Provide a detailed security report with remediation strategies.
            """,
            agent=self.agents['security'],
            expected_output="Comprehensive security audit report with risk assessment and remediation plan"
        )
        
        # Task 3: Code Quality Assessment
        quality_task = Task(
            description=f"""
            Assess the overall code quality and maintainability of the project:
            
            Evaluate:
            1. Code consistency across the project
            2. Adherence to Python coding standards (PEP 8, etc.)
            3. Technical debt assessment
            4. Code duplication analysis
            5. Complexity distribution
            6. Documentation quality
            7. Error handling patterns
            8. Design patterns usage
            9. Maintainability score
            
            Provide actionable recommendations for quality improvements.
            """,
            agent=self.agents['quality'],
            expected_output="Project-wide quality assessment with maintainability recommendations"
        )
        
        # Task 4: Testing Analysis
        testing_task = Task(
            description=f"""
            Analyze the testing strategy and coverage across the project:
            
            Assess:
            1. Test coverage analysis
            2. Testing framework evaluation
            3. Test organization and structure
            4. Unit, integration, and end-to-end test distribution
            5. Test quality and effectiveness
            6. CI/CD integration assessment
            7. Performance testing considerations
            8. Testing best practices compliance
            
            Provide recommendations for improving testing strategy and coverage.
            """,
            agent=self.agents['testing'],
            expected_output="Comprehensive testing analysis with coverage improvement recommendations"
        )
        
        # Task 5: Generate Project Report
        report_task = Task(
            description=f"""
            Generate a comprehensive project analysis report:
            
            The report should include:
            1. Executive Summary with key findings
            2. Project Health Score (1-10)
            3. Architecture Assessment
            4. Security Risk Assessment
            5. Code Quality Metrics
            6. Testing Strategy Evaluation
            7. Technical Debt Analysis
            8. Priority Recommendations (Critical, High, Medium, Low)
            9. Implementation Roadmap
            10. Resource Requirements
            11. Timeline Estimates
            12. ROI Analysis for recommended improvements
            
            Format the report for both technical teams and business stakeholders.
            Include specific action items with clear priorities and timelines.
            """,
            agent=self.agents['report'],
            dependencies=[structure_task, security_task, quality_task, testing_task],
            expected_output="Comprehensive project analysis report with executive summary and implementation roadmap"
        )
        
        # Create crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[structure_task, security_task, quality_task, testing_task, report_task],
            verbose=2,
            process=Process.sequential
        )
        
        return crew
    
    def review_project(self, project_path: str) -> str:
        """
        Perform a complete project review
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Comprehensive project review report
        """
        print(f"🔍 Starting comprehensive project review")
        print(f"📁 Project: {project_path}")
        print("=" * 80)
        
        # Validate project path
        if not os.path.exists(project_path):
            return f"Error: Project path '{project_path}' does not exist"
        
        if not os.path.isdir(project_path):
            return f"Error: '{project_path}' is not a directory"
        
        # Create review crew
        review_crew = self.create_project_review_crew(project_path)
        
        # Execute review
        try:
            print("🚀 Executing multi-agent project analysis...")
            result = review_crew.kickoff()
            return result
        except Exception as e:
            return f"Error during project review: {str(e)}"
    
    def quick_project_scan(self, project_path: str) -> Dict[str, Any]:
        """
        Perform a quick scan of the project for basic metrics
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Dictionary with basic project metrics
        """
        project_path = Path(project_path)
        
        if not project_path.exists():
            return {"error": f"Project path '{project_path}' does not exist"}
        
        # Basic file counting and analysis
        python_files = list(project_path.rglob('*.py'))
        python_files = [f for f in python_files if not any(ignore in str(f) for ignore in self.ignore_dirs)]
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    total_lines += len([line for line in content.split('\n') if line.strip()])
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
            except Exception:
                continue
        
        return {
            "project_path": str(project_path),
            "total_python_files": len(python_files),
            "total_lines_of_code": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "average_lines_per_file": round(total_lines / len(python_files), 1) if python_files else 0,
            "has_tests": any('test' in str(f).lower() for f in python_files),
            "has_setup_py": (project_path / 'setup.py').exists(),
            "has_requirements": (project_path / 'requirements.txt').exists(),
            "estimated_complexity": "Low" if total_lines < 1000 else "Medium" if total_lines < 10000 else "High"
        }

# Example usage and demonstration
def demonstrate_project_review():
    """Demonstrate the project review system"""
    
    print("Full Project Code Review System")
    print("=" * 50)
    
    print("This system can analyze entire Python projects including:")
    print("✅ Project structure and architecture")
    print("🔒 Security vulnerabilities across all files")
    print("📊 Code quality and maintainability metrics")
    print("🧪 Testing strategy and coverage analysis")
    print("📈 Technical debt assessment")
    print("🏗️ Architectural recommendations")
    print("📋 Comprehensive project health report")
    
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES:")
    print("=" * 50)
    
    usage_examples = '''
# Initialize the system
project_reviewer = ProjectCodeReviewSystem(openai_api_key="your-api-key")

# Quick project scan (fast overview)
metrics = project_reviewer.quick_project_scan("/path/to/your/project")
print(json.dumps(metrics, indent=2))

# Full comprehensive review (detailed analysis)
report = project_reviewer.review_project("/path/to/your/project")
print(report)

# Example output metrics:
{
  "project_path": "/path/to/project",
  "total_python_files": 45,
  "total_lines_of_code": 12500,
  "total_functions": 287,
  "total_classes": 34,
  "average_lines_per_file": 277.8,
  "has_tests": true,
  "has_setup_py": true,
  "estimated_complexity": "High"
}
'''
    
    print(usage_examples)
    
    print("\n" + "=" * 50)
    print("WHAT IT ANALYZES:")
    print("=" * 50)
    
    analysis_features = [
        "📁 Project Structure - Directory organization, file naming, module structure",
        "🔒 Security Analysis - Vulnerabilities, hardcoded secrets, insecure patterns",
        "📊 Code Quality - PEP 8 compliance, complexity, maintainability",
        "🧪 Testing Strategy - Coverage, framework usage, test organization",
        "📦 Dependencies - Third-party libraries, security of dependencies",
        "🏗️ Architecture - Design patterns, scalability, modularity",
        "📈 Technical Debt - Areas needing refactoring, legacy code issues",
        "📋 Documentation - README, docstrings, code comments quality"
    ]
    
    for feature in analysis_features:
        print(feature)
    
    print("\n" + "=" * 50)
    print("SAMPLE PROJECT REPORT STRUCTURE:")
    print("=" * 50)
    
    sample_report = '''
🔍 PROJECT HEALTH REPORT
========================

📈 Overall Health Score: 7.2/10

📊 PROJECT METRICS:
- Files: 45 Python files
- Lines: 12,500 total
- Functions: 287
- Classes: 34
- Test Coverage: ~65%

🚨 CRITICAL ISSUES (Must Fix):
- 3 hardcoded API keys found
- SQL injection vulnerability in user_service.py
- Missing input validation in 5 endpoints

⚠️ HIGH PRIORITY:
- 23% of functions lack docstrings
- Complex nested loops in data_processor.py
- Outdated dependencies with known vulnerabilities

📋 RECOMMENDATIONS:
1. Implement secrets management system
2. Add comprehensive input validation
3. Refactor high-complexity modules
4. Increase test coverage to 80%+
5. Update dependency versions

🏗️ ARCHITECTURE ASSESSMENT:
- Well-organized module structure
- Good separation of concerns
- Consider implementing design patterns for scalability
'''
    
    print(sample_report)

if __name__ == "__main__":
    demonstrate_project_review()