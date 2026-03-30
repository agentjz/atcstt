from __future__ import annotations

from atc_asr.pipeline import build_parser, config_from_args, run_pipeline


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
