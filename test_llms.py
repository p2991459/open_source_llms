import argparse
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
def create_llm(pretrained_model_name_or_path,device="cpu"):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    pipe = pipeline("text-generation",model=model,tokenizer=tokenizer,device=device)
    return pipe
def main():
    parser = argparse.ArgumentParser(description="text-generation models")
    parser.add_argument("--prompt", required=True, help="prompt To generate text")
    parser.add_argument("--model_path", required=True, help="pretrained_model_name_or_path")
    parser.add_argument("--device", required=False, help="device that need to use")
    args = parser.parse_args()
    prompt = args.prompt
    model_path =  args.model_path
    if args.gpu_layers is not None:
        device = args.device
    else:
        device = "cpu"
    llm =  create_llm(model_path,device=device)
    completion =  llm(prompt,max_new_tokens=5000,
            do_sample=True,
            use_cache=True)
    print(f"response: {completion}")
    print(f"Length of tokens in response : {len(completion)}")
if __name__ == "__main__":
    main()