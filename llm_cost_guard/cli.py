"""
Command-line interface for LLM Cost Guard.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from llm_cost_guard import CostTracker
from llm_cost_guard.pricing.loader import PricingLoader


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-cost-guard",
        description="LLM Cost Guard - Real-time cost tracking and budget enforcement for LLM applications",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status command
    status_parser = subparsers.add_parser("status", help="View current costs and budget status")
    status_parser.add_argument(
        "--backend",
        default="memory",
        help="Backend URL (default: memory)",
    )

    # health command
    health_parser = subparsers.add_parser("health", help="Check tracker health")
    health_parser.add_argument(
        "--backend",
        default="memory",
        help="Backend URL (default: memory)",
    )

    # report command
    report_parser = subparsers.add_parser("report", help="Generate cost report")
    report_parser.add_argument(
        "--period",
        choices=["day", "week", "month"],
        default="day",
        help="Report period (default: day)",
    )
    report_parser.add_argument(
        "--group-by",
        nargs="+",
        help="Group by fields (e.g., --group-by model provider)",
    )
    report_parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    report_parser.add_argument(
        "--backend",
        default="memory",
        help="Backend URL (default: memory)",
    )

    # pricing-status command
    pricing_parser = subparsers.add_parser("pricing-status", help="Check pricing data status")

    # update-pricing command
    update_parser = subparsers.add_parser("update-pricing", help="Update pricing data")

    # export command
    export_parser = subparsers.add_parser("export", help="Export cost data")
    export_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )
    export_parser.add_argument(
        "--start-date",
        help="Start date (ISO format)",
    )
    export_parser.add_argument(
        "--end-date",
        help="End date (ISO format)",
    )
    export_parser.add_argument(
        "--backend",
        default="memory",
        help="Backend URL (default: memory)",
    )

    # validate-config command
    validate_parser = subparsers.add_parser("validate-config", help="Validate configuration")
    validate_parser.add_argument(
        "--config",
        "-c",
        help="Configuration file path",
    )

    # models command
    models_parser = subparsers.add_parser("models", help="List supported models and pricing")
    models_parser.add_argument(
        "--provider",
        help="Filter by provider (e.g., openai, anthropic, bedrock)",
    )

    return parser


def cmd_status(args: argparse.Namespace) -> int:
    """Handle the status command."""
    tracker = CostTracker(backend=args.backend)

    try:
        # Get today's report
        report = tracker.daily_report()

        print("=" * 50)
        print("LLM Cost Guard - Status")
        print("=" * 50)
        print()
        print(f"Today's Summary ({datetime.now().strftime('%Y-%m-%d')}):")
        print(f"  Total Cost:     ${report.total_cost:.4f}")
        print(f"  Total Calls:    {report.total_calls}")
        print(f"  Input Tokens:   {report.total_input_tokens:,}")
        print(f"  Output Tokens:  {report.total_output_tokens:,}")
        print(f"  Success Rate:   {report.successful_calls / max(1, report.total_calls) * 100:.1f}%")

        if report.cache_hits > 0:
            print(f"  Cache Hits:     {report.cache_hits}")
            print(f"  Cache Savings:  ${report.cache_savings:.4f}")

        print()

        # Check health
        health = tracker.health_check()
        print("Health Status:")
        print(f"  Backend:   {'✓ Connected' if health.backend_connected else '✗ Disconnected'}")
        print(f"  Pricing:   {'✓ Fresh' if health.pricing_fresh else '⚠ Stale'}")

        if health.errors:
            print("  Errors:")
            for error in health.errors:
                print(f"    - {error}")

        return 0

    finally:
        tracker.close()


def cmd_health(args: argparse.Namespace) -> int:
    """Handle the health command."""
    tracker = CostTracker(backend=args.backend)

    try:
        health = tracker.health_check()

        print("LLM Cost Guard - Health Check")
        print("-" * 40)
        print(f"Overall:          {'✓ Healthy' if health.healthy else '✗ Unhealthy'}")
        print(f"Backend:          {'✓ Connected' if health.backend_connected else '✗ Disconnected'}")
        print(f"Pricing:          {'✓ Fresh' if health.pricing_fresh else '⚠ Stale'}")

        if health.last_record_time:
            print(f"Last Record:      {health.last_record_time.isoformat()}")

        if health.pricing_last_updated:
            print(f"Pricing Updated:  {health.pricing_last_updated.isoformat()}")

        if health.errors:
            print("\nErrors:")
            for error in health.errors:
                print(f"  - {error}")

        return 0 if health.healthy else 1

    finally:
        tracker.close()


def cmd_report(args: argparse.Namespace) -> int:
    """Handle the report command."""
    tracker = CostTracker(backend=args.backend)

    try:
        # Get period start
        now = datetime.now()
        if args.period == "day":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif args.period == "week":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start = start - timedelta(days=now.weekday())
        else:  # month
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        report = tracker.get_costs(
            start_date=start.isoformat(),
            group_by=args.group_by,
        )

        if args.format == "json":
            output = {
                "period": args.period,
                "start_date": start.isoformat(),
                "end_date": now.isoformat(),
                "total_cost": report.total_cost,
                "total_calls": report.total_calls,
                "total_input_tokens": report.total_input_tokens,
                "total_output_tokens": report.total_output_tokens,
                "successful_calls": report.successful_calls,
                "failed_calls": report.failed_calls,
            }
            if report.grouped_data:
                output["groups"] = report.grouped_data.get("groups", [])
            print(json.dumps(output, indent=2))

        elif args.format == "csv":
            if report.grouped_data and report.grouped_data.get("groups"):
                groups = report.grouped_data["groups"]
                if groups:
                    # Print header
                    headers = list(groups[0].keys())
                    print(",".join(headers))
                    # Print rows
                    for group in groups:
                        print(",".join(str(group.get(h, "")) for h in headers))
            else:
                print("date,cost,calls,input_tokens,output_tokens")
                print(
                    f"{start.strftime('%Y-%m-%d')},{report.total_cost:.4f},"
                    f"{report.total_calls},{report.total_input_tokens},{report.total_output_tokens}"
                )

        else:  # text
            print(f"\nCost Report - {args.period.title()}")
            print(f"Period: {start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")
            print("=" * 60)
            print(f"Total Cost:       ${report.total_cost:.4f}")
            print(f"Total Calls:      {report.total_calls}")
            print(f"Input Tokens:     {report.total_input_tokens:,}")
            print(f"Output Tokens:    {report.total_output_tokens:,}")
            print(f"Successful:       {report.successful_calls}")
            print(f"Failed:           {report.failed_calls}")

            if report.grouped_data and report.grouped_data.get("groups"):
                print(f"\nBy {', '.join(args.group_by or [])}:")
                print("-" * 60)
                for group in report.grouped_data["groups"]:
                    # Build label from group keys
                    label_parts = []
                    for key in args.group_by or []:
                        if key in group:
                            label_parts.append(f"{key}={group[key]}")
                    label = ", ".join(label_parts)

                    cost = group.get("cost", 0)
                    calls = group.get("calls", 0)
                    print(f"  {label}: ${cost:.4f} ({calls} calls)")

        return 0

    finally:
        tracker.close()


def cmd_pricing_status(args: argparse.Namespace) -> int:
    """Handle the pricing-status command."""
    loader = PricingLoader()

    print("LLM Cost Guard - Pricing Status")
    print("-" * 40)

    if loader.last_updated:
        print(f"Last Updated:  {loader.last_updated.isoformat()}")
    else:
        print("Last Updated:  Never")

    print(f"Stale:         {'Yes ⚠' if loader.is_stale else 'No ✓'}")
    print(f"Very Stale:    {'Yes ✗' if loader.is_very_stale else 'No ✓'}")

    print("\nProvider Versions:")
    for provider, version in loader.pricing_version.items():
        print(f"  {provider}: {version}")

    return 0


def cmd_update_pricing(args: argparse.Namespace) -> int:
    """Handle the update-pricing command."""
    print("Refreshing pricing data from local files...")

    loader = PricingLoader()
    loader.refresh()

    print("✓ Pricing data refreshed")
    print(f"  Last updated: {loader.last_updated.isoformat() if loader.last_updated else 'Never'}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Handle the export command."""
    tracker = CostTracker(backend=args.backend)

    try:
        start = datetime.fromisoformat(args.start_date) if args.start_date else None
        end = datetime.fromisoformat(args.end_date) if args.end_date else None

        records = tracker._backend.get_records(start_date=start, end_date=end)

        if args.format == "json":
            data = []
            for r in records:
                data.append(
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "provider": r.provider,
                        "model": r.model,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "input_cost": r.input_cost,
                        "output_cost": r.output_cost,
                        "total_cost": r.total_cost,
                        "latency_ms": r.latency_ms,
                        "success": r.success,
                        "error_type": r.error_type,
                        "cached": r.cached,
                        "tags": r.tags,
                    }
                )
            output = json.dumps(data, indent=2)

        else:  # csv
            lines = [
                "timestamp,provider,model,input_tokens,output_tokens,total_cost,latency_ms,success"
            ]
            for r in records:
                lines.append(
                    f"{r.timestamp.isoformat()},{r.provider},{r.model},"
                    f"{r.input_tokens},{r.output_tokens},{r.total_cost:.6f},"
                    f"{r.latency_ms},{r.success}"
                )
            output = "\n".join(lines)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Exported {len(records)} records to {args.output}")
        else:
            print(output)

        return 0

    finally:
        tracker.close()


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Handle the validate-config command."""
    if not args.config:
        print("No configuration file specified.")
        print("Use --config to specify a configuration file.")
        return 1

    try:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        print(f"Configuration file: {args.config}")
        print("-" * 40)

        # Validate budgets
        if "budgets" in config:
            print(f"✓ Found {len(config['budgets'])} budget(s)")
            for budget in config["budgets"]:
                if "name" not in budget:
                    print("  ✗ Budget missing 'name' field")
                if "limit" not in budget:
                    print(f"  ✗ Budget '{budget.get('name', 'unknown')}' missing 'limit' field")

        # Validate rate limits
        if "rate_limits" in config:
            print(f"✓ Found {len(config['rate_limits'])} rate limit(s)")

        print("\n✓ Configuration is valid")
        return 0

    except FileNotFoundError:
        print(f"✗ Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"✗ Invalid YAML: {e}")
        return 1
    except Exception as e:
        print(f"✗ Error validating configuration: {e}")
        return 1


def cmd_models(args: argparse.Namespace) -> int:
    """Handle the models command."""
    loader = PricingLoader()

    models = loader.get_all_models(args.provider)

    print("Supported Models and Pricing")
    print("=" * 70)

    for provider, model_list in sorted(models.items()):
        print(f"\n{provider.upper()}")
        print("-" * 70)

        for model in sorted(model_list):
            try:
                pricing = loader.get_pricing(provider, model)
                print(
                    f"  {model:40} "
                    f"Input: ${pricing.input_cost_per_1k:.6f}/1K  "
                    f"Output: ${pricing.output_cost_per_1k:.6f}/1K"
                )
            except Exception:
                print(f"  {model:40} (pricing unavailable)")

    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "status": cmd_status,
        "health": cmd_health,
        "report": cmd_report,
        "pricing-status": cmd_pricing_status,
        "update-pricing": cmd_update_pricing,
        "export": cmd_export,
        "validate-config": cmd_validate_config,
        "models": cmd_models,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
