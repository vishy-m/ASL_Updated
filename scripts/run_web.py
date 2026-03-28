import argparse
import logging

from asl_cslr.utils.logging import setup_logging
from asl_cslr.utils.io import load_yaml_config
from asl_cslr.online.web_server import ASLWebServer
from asl_cslr.online.pipeline import resolve_online_mode

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Real-time ASL recognition Web UI.")
    parser.add_argument("--config", default="configs/online.yaml", help="Online config YAML.")
    parser.add_argument("--mode", choices=["islr", "cslr"], help="Inference mode.")
    parser.add_argument("--port", type=int, default=5050, help="Web server port.")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    config = load_yaml_config(args.config)

    try:
        mode = resolve_online_mode(config, args.mode)
    except ValueError as exc:
        logger.error(str(exc))
        return

    server = ASLWebServer(config, mode=mode, port=args.port)
    server.run()

if __name__ == "__main__":
    main()
