import runpy
import sys


DEFAULT_LLM_CONF = 'nngpt_deepseek_v2_lite_trainable_adapter.json'


def main():
    sys.argv = [sys.argv[0], '--llm_conf', DEFAULT_LLM_CONF, *sys.argv[1:]]
    runpy.run_module('ab.gpt.TuneNNGen', run_name='__main__')


if __name__ == '__main__':
    main()
