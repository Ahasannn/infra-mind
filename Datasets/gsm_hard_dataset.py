import re


def gsm_hard_data_process(dataset, limit: int = 0):
    """
    Process GSM-Hard dataset from HuggingFace format.

    The HF dataset (reasoning-machines/gsm-hard) has columns:
      - input: the question text
      - code: Python solution code
      - target: float answer

    Args:
        dataset: HuggingFace dataset object or list of dicts
        limit: If > 0, limit to this many items (deterministic)

    Returns:
        List of processed data items with keys: task, answer
    """
    list_data_dict = []
    for i, data in enumerate(dataset):
        item = {"task": data["input"], "id": i}
        # Convert target to clean string: remove trailing .0 for integers
        raw_target = data["target"]
        try:
            float_val = float(raw_target)
            if float_val == int(float_val):
                item["answer"] = str(int(float_val))
            else:
                item["answer"] = str(float_val)
        except (ValueError, TypeError):
            item["answer"] = str(raw_target).strip()
        list_data_dict.append(item)

    # Apply deterministic limit if specified
    if limit > 0:
        print(f"[GSM-Hard Dataset] Applying limit: {limit} items (deterministic)")
        list_data_dict = list_data_dict[:limit]
        print(f"[GSM-Hard Dataset] Dataset size after limit: {len(list_data_dict)}")

    return list_data_dict


def gsm_get_predict(pred_str):
    if('answer is ' in pred_str):
        pred = pred_str.split('answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a
    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            pred = pred[-1]
        else: pred = ''

    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]

    pred=_strip_string(pred)

    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a

    # For GSM-Hard, answers can be large floats/negatives, so return the
    # extracted numeric string directly instead of stripping to digits only.
    # Try to parse as number; fall back to last numeric match.
    try:
        float(pred)
        return pred
    except (ValueError, TypeError):
        matches = re.findall(r'-?\d+\.?\d*', pred)
        return matches[-1] if matches else '0'


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n=str(n)
        return n

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        return string.split("\\text{ ")[0]
    return string

def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string
