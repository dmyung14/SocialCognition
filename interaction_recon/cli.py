import argparse
from pathlib import Path
import sys
import traceback

from interaction_recon import __version__
from interaction_recon.config import RunOptions
from interaction_recon.pipeline import reconstruct_inventory, run_inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interaction_recon",
        description="Reconstruct, refine, render and audit contact-driven MuJoCo physics.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Process a folder or ZIP")
    run.add_argument("--input", required=True, type=Path)
    reconstruct = commands.add_parser("reconstruct", help="Process an explicit pair")
    reconstruct.add_argument("--guider", required=True, type=Path)
    reconstruct.add_argument("--builder", required=True, type=Path)
    for command in (run, reconstruct):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument(
            "--render-format", choices=("mp4", "gif", "both"), default="mp4",
            help="Always retain MP4 artifacts; gif/both also create simulation and comparison GIFs",
        )
        command.add_argument(
            "--device", choices=("auto", "cuda", "cpu"), default="auto",
            help="MediaPipe Tasks, OpenCV, IK and physics use CPU; rendering uses OpenGL",
        )
        command.add_argument("--debug", action="store_true")
        command.add_argument("--analysis-fps", type=float, default=15.0)
        command.add_argument("--sync-threshold", type=float, default=0.5)
        command.add_argument("--model-dir", default="assets/models")
        command.add_argument("--force", action="store_true", help="Recompute all stages")
        command.add_argument(
            "--inventory-only", action="store_true",
            help="Only inspect sources and write manifests; no models required",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        options = RunOptions(
            render_format=arguments.render_format, device=arguments.device,
            debug=arguments.debug, analysis_fps=arguments.analysis_fps,
            sync_threshold=arguments.sync_threshold, model_dir=arguments.model_dir,
            force=arguments.force, inventory_only=arguments.inventory_only,
        )
        if arguments.command == "run":
            manifest, code = run_inventory(arguments.input, arguments.output, options)
            counts = manifest["counts"]
            print(
                f"Discovered {counts['media']} media files; "
                f"{counts['paired_segments']} paired, "
                f"{counts['unpaired_segments']} unpaired, "
                f"{counts['ambiguous_segments']} ambiguous segments; "
                f"{counts['metadata_failed']} metadata failures."
            )
        else:
            manifest, code = reconstruct_inventory(
                arguments.guider, arguments.builder, arguments.output, options
            )
            print(
                f"Explicit pair {manifest['segment_id']}: {manifest['stage_status']}; "
                f"physics={manifest.get('physics_stage_status', 'not_run')}; "
                f"final={manifest.get('final_stage_status', 'not_run')}; "
                f"audit={manifest.get('audit_pass')}; "
                f"M5={manifest.get('m5_pass', False)}; M6={manifest.get('m6_pass', False)}."
            )
        print(f"Manifest: {arguments.output.resolve() / 'manifest.json'}")
        if not arguments.inventory_only:
            print(
                "Inspect metrics.json targets and audit.json. A completed run or "
                "measurable refinement does not imply accurate reconstruction or verified fusion."
            )
        return code
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if arguments.debug:
            traceback.print_exc()
        return 2
    except (OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if arguments.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
