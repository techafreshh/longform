"""Google Flow prompt export and batch generation tool.

Standalone CLI tool that reads scenes.json from a project, builds
Flow-ready prompts, and either exports them or generates images
directly via flow-agent.

Usage:
    # Export prompts for manual use in Flow web UI
    python -m src.flow_prompts export /path/to/project --style stickman

    # Export as JSON batch file for flow-agent
    python -m src.flow_prompts batch /path/to/project --style stickman

    # Generate images directly via flow-agent
    python -m src.flow_prompts generate /path/to/project --style stickman

    # Check flow-agent status
    python -m src.flow_prompts status
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .config import STYLE_PRESETS, FLOW_AGENT_URL, FLOW_IMAGE_ASPECT, FLOW_REQUEST_DELAY


def load_scenes(project_dir: Path) -> list[dict]:
    """Load scenes from scenes.json in the project directory."""
    scenes_json = project_dir / "scenes.json"
    if not scenes_json.exists():
        # Try parent directory
        scenes_json = project_dir.parent / "scenes.json"

    if not scenes_json.exists():
        print(f"❌ No scenes.json found in {project_dir} or parent")
        sys.exit(1)

    with open(scenes_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    if not scenes:
        print("❌ No scenes found in scenes.json")
        sys.exit(1)

    print(f"📋 Loaded {len(scenes)} scenes from {scenes_json}")
    return scenes


def build_prompt(description: str, style_prefix: str) -> str:
    """Build a Flow-ready prompt."""
    return (
        f"{style_prefix}. "
        f"Subject: {description}. "
        f"Resolution: 1920x1080, landscape orientation. "
        f"The illustration should clearly communicate the concept. "
        f"No text or words in the image unless specifically relevant."
    )


def get_style_prefix(style: str, project_dir: Path = None) -> str:
    """Get the style prompt prefix, checking for custom analysis first."""
    # Check for custom style analysis
    if project_dir:
        analysis_file = project_dir / "reference_style_analysis.json"
        if analysis_file.exists():
            try:
                with open(analysis_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    prefix = data.get("reproduction_prompt_prefix")
                    if prefix:
                        return prefix
            except Exception:
                pass

    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["color_whiteboard"])
    return preset["image_prompt_prefix"]


def cmd_export(args):
    """Export prompts as a human-readable list for copy-paste into Flow."""
    scenes = load_scenes(Path(args.project_dir))
    style_prefix = get_style_prefix(args.style, Path(args.project_dir))

    output_lines = []
    output_lines.append(f"# Google Flow Prompts — {args.style} style")
    output_lines.append(f"# {len(scenes)} scenes")
    output_lines.append(f"# Copy each prompt into Flow's image generator\n")

    for scene in scenes:
        idx = scene["index"]
        desc = scene["description"]
        prompt = build_prompt(desc, style_prefix)

        output_lines.append(f"{'='*60}")
        output_lines.append(f"SCENE {idx}: {desc[:80]}")
        output_lines.append(f"{'='*60}")
        output_lines.append(prompt)
        output_lines.append("")

    output_path = Path(args.output) if args.output else Path(args.project_dir) / "flow_prompts.txt"
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n📝 Exported {len(scenes)} prompts to: {output_path}")
    print(f"   Copy each prompt into https://labs.google/fx/tools/flow")
    print(f"   Save images as scene_XX.png in: {Path(args.project_dir) / 'scenes'}")


def cmd_batch(args):
    """Export prompts as JSON for flow-agent batch processing."""
    scenes = load_scenes(Path(args.project_dir))
    style_prefix = get_style_prefix(args.style, Path(args.project_dir))

    batch_items = []
    for scene in scenes:
        idx = scene["index"]
        desc = scene["description"]
        prompt = build_prompt(desc, style_prefix)
        batch_items.append({
            "scene_index": idx,
            "description": desc,
            "prompt": prompt,
            "output_filename": f"scene_{idx:02d}.png",
        })

    output_path = Path(args.output) if args.output else Path(args.project_dir) / "flow_batch.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_items, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Exported {len(batch_items)} prompts to: {output_path}")
    print(f"   Use with flow-agent: flow image --batch {output_path}")


def cmd_generate(args):
    """Generate images directly via flow-agent API."""
    from .flow_gen import generate_scenes_flow, check_flow_health

    # Verify connection
    if not check_flow_health(verbose=True):
        print("\n❌ Cannot connect to flow-agent. Run 'python -m src.flow_prompts status' to check.")
        sys.exit(1)

    project_dir = Path(args.project_dir)
    scenes = load_scenes(project_dir)

    output_dir = project_dir / "scenes"
    if args.output:
        output_dir = Path(args.output)

    print(f"\n🚀 Generating {len(scenes)} images via Google Flow...")
    print(f"   Style: {args.style}")
    print(f"   Output: {output_dir}")
    if args.force:
        print(f"   Mode: Force (regenerate all)")
    print()

    paths = generate_scenes_flow(
        scenes=scenes,
        style=args.style,
        output_dir=output_dir,
        force=args.force,
        force_scenes=args.force_scenes,
        resume_from_scene=args.resume_from,
        verbose=True,
    )

    success = sum(1 for p in paths if p.exists())
    print(f"\n🎉 Done! {success}/{len(paths)} images generated")


def cmd_status(args):
    """Check Flow backend connection and availability."""
    from .flow_gen import check_whisk_available, check_flow_agent_available, _get_backend

    print("🔍 Checking Google Flow backends...\n")

    backend = _get_backend()

    # Check whisk-api
    print("--- whisk-api (recommended) ---")
    whisk_ok = check_whisk_available(verbose=True)

    print("\n--- flow-agent (fallback) ---")
    agent_ok = check_flow_agent_available(verbose=True)

    print(f"\n📋 Active backend: {backend}")

    if not whisk_ok and not agent_ok:
        print("\n💡 Setup whisk-api (recommended):")
        print("   1. Install: npm i -g @rohitaryal/whisk-api")
        print("   2. Get cookie: Open labs.google → Cookie Editor extension → Export Header String")
        print("   3. Set in .env: WHISK_COOKIE=<your-cookie>")
        print("\n💡 Or setup flow-agent:")
        print("   1. Install: pip install git+https://github.com/kodelyx/flow-agent#subdirectory=flow-agent")
        print("   2. Load Chrome extension, start flow-agent, expose via ngrok")
        print("   3. Set in .env: FLOW_AGENT_URL=<ngrok-url>")


def main():
    parser = argparse.ArgumentParser(
        description="Google Flow prompt export and batch generation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # export command
    p_export = subparsers.add_parser("export", help="Export prompts for manual use in Flow web UI")
    p_export.add_argument("project_dir", help="Path to project directory")
    p_export.add_argument("--style", default="stickman", help="Style preset (default: stickman)")
    p_export.add_argument("--output", "-o", help="Output file path")
    p_export.set_defaults(func=cmd_export)

    # batch command
    p_batch = subparsers.add_parser("batch", help="Export prompts as JSON for flow-agent batch mode")
    p_batch.add_argument("project_dir", help="Path to project directory")
    p_batch.add_argument("--style", default="stickman", help="Style preset (default: stickman)")
    p_batch.add_argument("--output", "-o", help="Output file path")
    p_batch.set_defaults(func=cmd_batch)

    # generate command
    p_gen = subparsers.add_parser("generate", help="Generate images directly via flow-agent API")
    p_gen.add_argument("project_dir", help="Path to project directory")
    p_gen.add_argument("--style", default="stickman", help="Style preset (default: stickman)")
    p_gen.add_argument("--output", "-o", help="Output directory for images")
    p_gen.add_argument("--force", action="store_true", help="Regenerate all images")
    p_gen.add_argument("--force-scenes", nargs="+", type=int, help="Specific scene indices to regenerate")
    p_gen.add_argument("--resume-from", type=int, help="Resume from scene index")
    p_gen.set_defaults(func=cmd_generate)

    # status command
    p_status = subparsers.add_parser("status", help="Check flow-agent connection and credits")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
