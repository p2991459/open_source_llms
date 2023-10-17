import argparse
from ctransformers import AutoModelForCausalLM
def ask_from_gguf(model_name_or_path,prompt,model_type="llama",gpu_layers=0):
    llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, model_type="llama", gpu_layers=gpu_layers)
    return llm(prompt)
def main():
    parser = argparse.ArgumentParser(description="text-generation models")
    parser.add_argument("--prompt", required=True, help="prompt To generate text")
    parser.add_argument("--model_path", required=True, help="pretrained_model_name_or_path")
    parser.add_argument("--gpu_layers", required=False, help="number of gpu layers")
    args = parser.parse_args()
    prompt = args.prompt
    model_path =  args.model_path
    if args.gpu_layers is not None:
        gpu_layers = args.gpu_layers
    else:
        gpu_layers = 0
    print(f"completion: {ask_from_gguf(model_path,prompt,gpu_layers=gpu_layers)}")
if __name__ == "__main__":
    main()
