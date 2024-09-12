import torch
import re

import json
import re
import subprocess

CLANG_FORMAT_CONFIG = {
    "BasedOnStyle": "Google",
    "ColumnLimit": 5000,
    "IndentWidth": 2,
    "AllowShortBlocksOnASingleLine": False,
    "AllowShortCaseLabelsOnASingleLine": False,
    "AllowShortFunctionsOnASingleLine": False,
    "AllowShortLoopsOnASingleLine": False,
    "AllowShortIfStatementsOnASingleLine": False,
    "DerivePointerAlignment": False,
    "PointerAlignment": "Left",
    "BreakAfterJavaFieldAnnotations": True,
    "BreakBeforeInheritanceComma": False,
    "BreakBeforeTernaryOperators": False,
    "AlwaysBreakAfterReturnType": "None",
    "AlwaysBreakAfterDefinitionReturnType": "None",
    # remove comment from source code
}

CLANG_FORMAT = "/usr/bin/clang-format"

C_COMMENT_RE = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE,
)


def StripComments(text: str) -> str:
    """Strip C/C++ style comments.

    Written by @markus-jarderot https://stackoverflow.com/a/241506/1318051
    """

    def Replacer(match):
        """Regex replacement callback."""
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    return C_COMMENT_RE.sub(Replacer, text)


def ClangFormat(src: str, suffix: str, timeout_seconds: int = 60) -> str:
    """Run clang-format on a source to enforce code style.

    Args:
      src: The source code to run through clang-format.
      suffix: The suffix to append to the source code temporary file. E.g. '.c'
        for a C program.
      timeout_seconds: The number of seconds to allow clang-format to run for.

    Returns:
      The output of clang-format.

    Raises:
      ClangFormatException: In case of an error.
      ClangTimeout: If clang-format does not complete before timeout_seconds.
    """

    cmd = [
        "timeout",
        "-s9",
        str(timeout_seconds),
        str(CLANG_FORMAT),
        "-assume-filename",
        f"input{suffix}",
        "-style={}".format(json.dumps(CLANG_FORMAT_CONFIG)),
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate(src)
    if process.returncode == 9:
        raise ValueError(f"clang-format timed out after {timeout_seconds}s")
    elif process.returncode != 0:
        raise ValueError(stderr)
    return stdout


def StripDoubleUnderscorePrefixes(text: str) -> str:
    """Remove the optional __ qualifiers on OpenCL keywords.

    The OpenCL spec allows __ prefix for OpenCL keywords, e.g. '__global' and
    'global' are equivalent. This preprocessor removes the '__' prefix on those
    keywords.

    Args:
      text: The OpenCL source to preprocess.

    Returns:
      OpenCL source with __ stripped from OpenCL keywords.
    """
    # List of keywords taken from the OpenCL 1.2. specification, page 169.
    replacements = {
        "__const": "const",
        "__constant": "constant",
        "__global": "global",
        "__kernel": "kernel",
        "__local": "local",
        "__private": "private",
        "__read_only": "read_only",
        "__read_write": "read_write",
        "__restrict": "restrict",
        "__write_only": "write_only",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def StripNewLines(text: str) -> str:
    """Replace n newlines by one newline."""
    return re.sub(r"\n\s*\n+", "\n", text)

def preprocess_data(examples, tokenizer, max_seq_len, output_scale=None):
    prefix_code = "# Python code: "
    prefix_input = "# Given input: "

    codes = examples["code"]
    inputs = examples["input"]
    prompts = []
    for i in range(len(codes)):
        prompt = prefix_code + codes[i] + prefix_input + inputs[i]
        prompts.append(prompt)

    model_inputs = tokenizer(
        prompts,
        max_length=max_seq_len,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    model_inputs["labels"] = model_inputs["input_ids"].clone()
    model_inputs["execution_time"] = torch.tensor(
        examples["cpu_time"], dtype=torch.float32
    )
    if output_scale == "log2":
        model_inputs["execution_time"] = torch.log2(model_inputs["execution_time"])
    elif output_scale == "log10":
        model_inputs["execution_time"] = torch.log10(model_inputs["execution_time"])
    elif output_scale == "original":
        pass

    return model_inputs


def preprocess_data_opencl(
    examples, tokenizer, max_seq_len, output_scale=None, time_unit="us"
):
    """
    Generate code dataset from jsonl file. Require these fields: `code`, `gsize`, `lsize`, `kernel_path`, `execution_time`
    Note that `execution_time` must be in nanosecond unit.
    Specify the time unit for the model by `time_unit` parameter ("s", "ms", "us", "ns")
    """
    prompt_template = """Predict time for the following OpenCL kernel:\n```{code}```\n
Given input:\n{inputs}\n
Given kernel are executed with global size equals `cl::NDRange({global_size})`, local work-group size equals `cl::NDRange({local_size})`\n
"""
    prompts = []
    for i in range(len(examples["code"])):
        prompts.append(
            prompt_template.format(
                code=examples["code"][i],
                inputs=examples["input_sizes"][i],
                global_size=examples["gsize"][i],
                local_size=examples["lsize"][i],
            )
        )
    # print(prompts[0])
    tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer(
        prompts,
        max_length=max_seq_len,
        padding="max_length",
        truncation=False,
    )

    model_inputs["prompt"] = list(prompts)
    # model_inputs["labels"] = model_inputs["input_ids"].clone()
    model_inputs["execution_time"] = torch.tensor(
        examples["execution_time"], dtype=torch.float32
    )

    if time_unit == "s":
        conversion_factor = 1e9
    elif time_unit == "ms":
        conversion_factor = 1e6
    elif time_unit == "us":
        conversion_factor = 1e3
    elif time_unit == "ns":
        conversion_factor = 1
    model_inputs["execution_time"] = model_inputs["execution_time"] / conversion_factor
    if output_scale == "log2":
        model_inputs["execution_time"] = torch.log2(model_inputs["execution_time"])
    elif output_scale == "log10":
        model_inputs["execution_time"] = torch.log10(model_inputs["execution_time"])
    elif output_scale == "original":
        pass

    return model_inputs


def remove_exceed_length(examples, max_seq_len):
    """
    Remove examples that exceed the maximum sequence length
    """
    filtered_examples = {}
    for key in examples:
        filtered_examples[key] = []
    input_ids = examples["input_ids"]
    for i in range(len(input_ids)):
        if len(input_ids[i]) > max_seq_len:
            continue
        for key in examples:
            filtered_examples[key].append(examples[key][i])
    return filtered_examples


if __name__ == "__main__":
    from transformers import AutoTokenizer

    examples = {
        "code": [
            """kernel void A(global char2* i, global char2* m) {
  int c = get_global_id(0);
  m[c] = +i[c];
}""",
            """kernel void A(global char2* i, global char2* m) {
  int c = get_global_id(0);
  m[c] = +i[c];
}""",
        ],
        "gsize": [4096, 1],
        "lsize": [32, 1],
        "execution_time": [123213, 1],
        "input_sizes": [
            "Given input: 4096 dfsdafsda fsad fsda fsadf32423e 2d12d2 dsafsadf",
            "Given input: 4096 d12d2 dsafsadf",
        ],
    }
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350m-multi")
    x = preprocess_data_opencl(examples, tokenizer, 115, "log2")
    print(x)
    x = remove_exceed_length(x, 115)
    print(x)
