# src/agent_sim/infrastructure/database/bql_test_reporter.py

import uuid
from typing import Dict, Any, List, cast

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agent_core.testing.bql import BehavioralAssertion
from agent_sim.infrastructure.database.async_database_manager import AsyncDatabaseManager
from .bql_executor import BQLExecutor

class BQLTestReporter:
    """
    Orchestrates the execution of BQL assertions at the end of a simulation run 
    and generates a human-readable report.
    """
    
    def __init__(self, run_id: str, db_manager: AsyncDatabaseManager, assertions: List[BehavioralAssertion]):
        self.run_id = run_id
        self.db_manager = db_manager
        self.assertions = assertions
        self.executor = BQLExecutor(db_manager)
        self.console = Console()
        self.results: Dict[str, bool] = {}

    async def execute_and_report(self) -> Dict[str, bool]:
        """
        Runs all assertions and prints a final test summary.
        """
        if not self.assertions:
            self.console.print("[yellow]BQL Info: No assertions defined for this run.[/yellow]")
            return {"BQL_STATUS": True}

        self.console.rule(f"[bold cyan]🔬 Behavioral Unit Test Report ({self.run_id[:8]})[/bold cyan]")
        
        test_results = Table(title="Assertion Details", show_header=True, header_style="bold magenta")
        test_results.add_column("ID", style="dim", width=4)
        test_results.add_column("Temporal Scope")
        test_results.add_column("Assertion Logic")
        test_results.add_column("Result", style="bold")
        
        total_tests = len(self.assertions)
        tests_passed = 0

        for i, assertion in enumerate(self.assertions):
            assertion_name = f"A_{i+1}"
            
            # Execute the query against the database
            passed = await self.executor.execute_assertion(self.run_id, assertion)
            
            self.results[assertion_name] = passed
            
            result_str = "[green]PASS[/green]" if passed else "[bold red]FAIL[/bold red]"
            if passed:
                tests_passed += 1
                
            test_results.add_row(
                str(i + 1),
                f"[cyan]{assertion.temporal}[/cyan]",
                str(assertion),
                result_str
            )

        self.console.print(test_results)
        
        # Final Summary Panel
        if tests_passed == total_tests:
            final_status = f"[bold green]✅ ALL {total_tests} BEHAVIORAL TESTS PASSED[/bold green]"
            panel_style = "green"
        else:
            tests_failed = total_tests - tests_passed
            final_status = f"[bold red]❌ {tests_failed}/{total_tests} BEHAVIORAL TESTS FAILED[/bold red]"
            panel_style = "red"

        self.console.print(Panel(final_status, title="Final BQL Status", border_style=panel_style))
        self.console.rule()

        return self.results