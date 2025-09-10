
from evaluation.metrics import parse_answer, accuracy
#import matplotlib
import matplotlib
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MODEL_DATA = {
    "HuggingFaceTB_SmolLM2-1.7B-": {
        "model_name": "SmolLM2",
        "model_size": 1.7e9,
    },
    "Qwen_Qwen3-8B-": {
        "model_name": "Qwen3",
        "model_size": 8e9,
    },
    "meta-llama_Llama-3.2-1B-": {
        "model_name": "Llama3.2",
        "model_size": 1e9,
    },
    "meta-llama_Llama-3.2-3B-": {
        "model_name": "Llama3.2",
        "model_size": 3e9,
    },
    "meta-llama_Meta-Llama-3-8B-": {
        "model_name": "Meta-Llama3",
        "model_size": 8e9,
    },
    "microsoft_phi-4-": {
        "model_name": "Phi-4",
        "model_size": 14e9,
    },
    "gemma3-4b-pt-": {
        "model_name": "Gemma3-4B",
        "model_size": 4e9,
    },
    "gemma3-12b-pt-": {
        "model_name": "Gemma3-12B",
        "model_size": 12e9,
    }
}

def main(
    output_file: str
):
    #read from output_file

    #output file is jsonl file
    jsonl_dict = None
    with open(output_file, 'r') as f:
        jsonl_dict = [eval(line.strip()) for line in f.readlines()]
        

    preds = []
    golds = []
    for line in jsonl_dict:
        preds.append(line['pred'])
        golds.append(line['gold'])


    #parse the answers
    em = accuracy(preds, golds, type="em")
    approx = accuracy(preds, golds, type="approx")   
    #colorful print
    print(f"{bcolors.OKGREEN}Exact Match: {em*100:.2f}%{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Approximate Match: {approx*100:.2f}%{bcolors.ENDC}")

    return em, approx

if __name__ == "__main__":
    import argparse
    '''
    parser = argparse.ArgumentParser(description="Evaluate the model's predictions.")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file containing predictions.",
    )

    args = parser.parse_args()
    '''
    output_dir = "outputs/gsm8k-xl"
    files = []
    #get all files in output_dir that ends with .jsonl
    import os
    for file in os.listdir(output_dir):
        if file.endswith(".jsonl"):
            files.append(os.path.join(output_dir, file))
    #order files alphabetically
    files = sorted(files)
    data = {}
    for file in files:
        print(f"Evaluating {file} ...")
        em, approx = main(file)
        #model name is the file name before "inference...jsonl"
        model_name = file.split("/")[-1].split("inference")[0]
        print(model_name)
        remaining = file.split("/")[-1].split("inference")[1]
        baseline_or_func = "baseline" if "baseline" in remaining else "func"
        if model_name not in data:
            data[model_name] = {}
        data[f"{model_name}"][baseline_or_func] = {
            "em": em,
            "approx": approx
        }
    data["gemma3-4b-pt-"] = {}
    data["gemma3-12b-pt-"] = {}
    data['gemma3-4b-pt-']['baseline'] = {'em': 0.1919 , 'approx': 0.2095}
    data['gemma3-4b-pt-']['func'] = {'em': 0.2764, 'approx': 0.2993}
    data['gemma3-12b-pt-']['func'] = {'em': 0.607670, 'approx': 0.625369}
    data['gemma3-12b-pt-']['baseline'] = {'em': 0.5317, 'approx': 0.5915}
    #print data in a table
    print(f"{bcolors.HEADER}{'Model':<30} {'Type':<10} {'Exact Match':<15} {'Approx Match':<15}{bcolors.ENDC} {'Size':<10}")
    for model in data:
        for type_ in data[model]:
            em = data[model][type_]["em"] * 100
            approx = data[model][type_]["approx"] * 100
            print(f"{MODEL_DATA[model]['model_name']:<30} {type_:<10} {em:<15.2f} {approx:<15.2f} {MODEL_DATA[model]['model_size']:<10.2f}")


    #scatter plot for all models with x axis as model size and y axis as accuracy, for each model use different shape, for baseline use blue color, for func use orange color
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(20, 8))
    markers = ['o', 's', '^', 'D', 'P', 'X', 'h', 'v']
    colors = {"baseline": "blue", "func": "orange"}
    for i, model in enumerate(data):
        model_size = MODEL_DATA[model]["model_size"] if model in MODEL_DATA else None
        if model_size is None:
            continue
        for type_ in data[model]:
            em = data[model][type_]["em"] * 100
            #if baseline
            if type_ == "baseline":
                plt.scatter(model_size, em, label=MODEL_DATA[model]['model_name'], marker=markers[i % len(markers)], color=colors[type_], s=100, edgecolor='black')
            else:
                plt.scatter(model_size, em,  marker=markers[i % len(markers)], color=colors[type_], s=100)
            #plt.text(model_size, em, f"{model} - {type_}", fontsize=9, ha='right')
    #add legend explaining colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Baseline', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Func', markerfacecolor='orange', markersize=10)]
    #add model names to legend
    model_names = set()
    for model in data:
        model_name = MODEL_DATA[model]["model_name"] if model in MODEL_DATA else None
        if model_name and model_name not in model_names:
            model_names.add(model_name)
            legend_elements.append(Line2D([0], [0], marker=markers[len(legend_elements) % len(markers)], color='w', label=model_name, markerfacecolor='grey', markersize=10))


    plt.legend(handles=legend_elements, loc='lower right')
    plt.xscale('log')
    plt.xlabel('Model Size (log scale)')
    plt.ylabel('Exact Match Accuracy (%)')
    plt.title('Model Size vs Exact Match Accuracy')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_size_vs_accuracy.png"))
    plt.show()