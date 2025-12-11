import argparse
from pathlib import Path

from ab.gpt.util.AlterCaptionNNtune import alter
from ab.gpt.util.Const import conf_test_dir


def main():
    parser = argparse.ArgumentParser(description="Generate revised neural-network architectures models of captioning task.")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of generation epochs.")
    parser.add_argument("-c", "--conf", type=str, default="NN_CAPU.json", help="Config file in conf_test directory. (e.g. NN_Caption.json, NN_Caption_master.json)")
    parser.add_argument("-nn", type=str, default=None, help="To generate variants ONLY for this models (e.g. RESNETLSTM, ResNetTransformer) otherwise default all.")
    parser.add_argument("-m", "--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", help="HF model id or local model path.Choose a smaller model for faster generation. e.g. 'open-r1/OlympicCoder-7B' or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' or 'deepseek-ai/deepseek-coder-7b-instruct'.")
    parser.add_argument("--gguf", type=str, default=None, help="Optional GGUF file path if using a local/llama.cpp backend.")

    args = parser.parse_args()

    # Resolve the config file (no temp files, no filtering)
    conf_path = conf_test_dir / args.conf
    if not conf_path.exists():
        raise FileNotFoundError(f"Config not found: {conf_path}")

    # Call the caption-alter pipeline directly
    alter(epochs=args.epochs,
        test_conf=conf_path.name, 
        llm_name=args.model,
        gguf_file=args.gguf,
        only_nn=args.nn
    )


if __name__ == "__main__":
    main()
