import json
import csv
import os.path as osp
import random
from typing import Union

def create_add_data(pairs):
    random.shuffle(pairs)
    data_add = []
    for num1, num2 in pairs:

        if random.random()<0.5:
            num1, num2 = num2, num1

        answer = num1 + num2

        question = f"{num1} + {num2}"
        output = f"{num1} + {num2} = {answer}"

        assert(output.split()[-1] == str(answer))
        data_add.append({"input": question, "output": output, "answer": str(answer)})

    return data_add


def create_sub_data(pairs):
    random.shuffle(pairs)
    data_sub = []
    for num1, num2 in pairs:

        if random.random()<0.5:
            num1, num2 = num2, num1

        answer = num1 - num2

        question = f"{num1} - {num2}"
        output = f"{num1} - {num2} = {answer}"

        assert(output.split()[-1] == str(answer))
        data_sub.append({"input": question, "output": output, "answer": str(answer)})

    return data_sub


def create_mul_data(pairs):
    random.shuffle(pairs)
    data_mul_n_1 = []
    for num1, num2 in pairs:

        if random.random() < 0.1:
            num1 = num1 * (10**random.randint(1,6))

        if random.random()<0.5:
            num1, num2 = num2, num1

        answer = num1 * num2

        question = f"{num1} * {num2}"
        output = f"{num1} * {num2} = {answer}"

        assert(output.split()[-1] == str(answer))
        data_mul_n_1.append({"input": question, "output": output, "answer": str(answer)})

    return data_mul_n_1

def create_div_data(pairs):
    random.shuffle(pairs)
    data_div_n_1 = []
    for num1, num2 in pairs:
        # make it divisible with 0.5 probability
        if num1>1 and random.random() < 0.5:
            remainder = random.randint(1, num1-1)
        else:
            remainder = 0

        # divided by 0
        if num1 == 0:
            question = f"{num2} / {num1}"
            cot = question + " is " + "undefined"
            answer = "undefined"
            data_div_n_1.append({"input": question, "output": cot, "answer": answer})
            continue
        dividend = num1 * num2 + remainder
        question = f"{dividend} / {num1}"
        cot = question + " = " + str(num2) + " R " + str(remainder) if remainder!=0 else question + " = " + str(num2)
        answer = str(num2) + " R " + str(remainder) if remainder!=0 else str(num2)

        assert(cot.split()[-1] == answer.split()[-1])
        data_div_n_1.append({"input": question, "output": cot, "answer": answer})
    return data_div_n_1

def post_process(template_name, data):
    ### Add natural language instruction to the generated arithmetic data using template
    with open(template_name) as fp:
        template = json.load(fp)

    data_converted = []
    for instance in data:
        arithmetic = instance["input"]
        output_dict = {}
        # add noise to instruction so that the model is robust to diverse question formats
        if random.random() < 0.05:
            if " + " in arithmetic:
                arithmetic = "the sum of " + arithmetic.replace("+", "and")

            if " - " in arithmetic:
                arithmetic = "the difference of " + arithmetic.replace("-", "and")

            if " * " in arithmetic:
                arithmetic = "the product of " + arithmetic.replace("*", "and")

            if " / " in arithmetic:
                arithmetic = "the quotient and remainder of " + arithmetic.replace("/", "and")

        if random.random() < 0.5:
            arithmetic = arithmetic.replace("*", "x")

        if random.random() < 0.1:
            arithmetic = arithmetic.replace("+", "plus").replace("-", "minus")
            arithmetic = arithmetic.replace(" x ", " times ").replace("*", "multiplied by").replace("/", "divided by")

        if random.random() < 0.5:
            if "+" in arithmetic or "-" in arithmetic or "*" in arithmetic or "/" in arithmetic or "x" in arithmetic:
                arithmetic = arithmetic.replace(" ", "")

        num = random.randint(1,500)

        instruction = template[str(num)].format(
            input = arithmetic
        )

        output_dict["instruction"] = instruction
        output_dict["input"] = instance["input"]
        output_dict["output"] = instance["output"]
        output_dict["answer"] = instance["answer"]

        data_converted.append(output_dict)

    return data_converted

def write_to_csv(data, csv_save_path):
    fieldnames = ['instruction', 'input', 'output', 'answer']
    with open(csv_save_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    template_name = "./templates/goat.json"
    dataset_name = "dataset.json"
    csv_save_path = "data.csv"

    pairs = [(random.randint(0, 100), random.randint(0, 100)) for k in range(10000)]
    data = create_sub_data(pairs)
    data_processed = post_process(template_name, data)
    write_to_csv(data_processed, csv_save_path)


    print("Total:", len(data))
    print("Arithmetic dataset generated!")

if __name__=="__main__":
    main()
