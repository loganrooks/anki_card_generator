import difflib
import json
import os
import statistics
import string
import re
from bs4 import BeautifulSoup
from yattag import Doc
from itertools import combinations
import pandas as pd
import numpy as np

# def append_html_diff(html_file: Doc, lines1: , lines2):

def flatten_runs(runs: list):
    flattened_runs = {"args": {}, "output": []}
    for run in runs:
        for key, value in run["args"].items():
            flattened_runs["args"][key] = [value] if not flattened_runs["args"].get(key, None) else flattened_runs["args"][key].copy().append(key)
        
        for output in run["output"]:
            flattened_runs["output"].append(output)
    return flattened_runs

# def find_max_differences(outputs):


    for a, b in combinations(range(len(outputs)), 2):
        cards_a = outputs[a]['anki_cards']
        variables_a = outputs[a]['variables']
        cards_b = outputs[b]['anki_cards']
        variables_b = outputs[b]['variables']

        for i in range(min(len(cards_a), len(cards_b))):
            variables_a["card"] = i
            variables_b["card"] = i

            lines_a = re.split('(?<=[.!?,;])', cards_a[i]["Text"])
            lines_b = re.split('(?<=[.!?,;])', cards_b[i]["Text"])
        


    lines = [row.split() for row in lines]  # the file should already break at each line break
    lines = [(int(row[0]), row[1]) for row in lines]
    lines = groupby(sorted(lines), lambda x: x[0])  # combine strings into their respective groups, sorting them first on int of first element
    group_max = dict()
    for group in lines:
        strings = list(group[1])  # need to convert group[1] from iterator into list
        if len(strings) > 1:  # if the number of strings is 1, then there is nothing to compare the string with in its group
            similarity = 1
            for line1, line2 in combinations(strings, 2):
                s = difflib.SequenceMatcher(None, line1[1], line2[1])  # need to compare second element in each list and exclude the first element (which is the group number)
                similarity = s.ratio() if s.ratio() < similarity else similarity
            group_max[line1[0]] = 1 - similarity  # gives difference ratio
    return group_max

def get_cloze_deletions_stats(anki_cards):
    cloze_counts = []

    for card in anki_cards:
        text = card["Text"]
        cloze_count = len(re.findall(r"\{\{c\d+::.+?\}\}", text))
        cloze_counts.append(cloze_count)

    average_cloze = sum(cloze_counts) / len(cloze_counts)
    std_deviation_cloze = statistics.stdev(cloze_counts)

    return average_cloze, std_deviation_cloze


def get_all_cloze_deletion_stats(outputs, variable_ids=["temperature", "top_p", "max_completion_tokens"]):
    all_cloze_stats = []
    for i in range(len(outputs)):
        anki_cards = outputs[i]["anki_cards"]
        variables = outputs[i]["variables"]
        cloze_stats = []

        for variable_id in variable_ids:
            cloze_stats.append(variables[variable_id])

        average_cloze, std_deviation_cloze = get_cloze_deletions_stats(anki_cards)
        cloze_stats.extend([average_cloze, std_deviation_cloze])
        all_cloze_stats.append(cloze_stats)
    
    return pd.DataFrame(all_cloze_stats.copy(), columns=variable_ids + ['average_cloze', 'std_dev_cloze'])


def main():
    runs_json_path = os.path.dirname(__file__) + "/outputs/sz_test.json"
    with open(runs_json_path, 'r') as runs_json:
        runs = json.load(runs_json)

    flattened_runs = flatten_runs(runs)
    df_outputs = pd.DataFrame(flattened_runs["output"])
    output_doc = BeautifulSoup()
    output_doc.append(output_doc.new_tag("html"))
    output_doc.html.append(output_doc.new_tag("body"))
    output_html_diff_path = os.path.dirname(__file__) + "/outputs/sz_html_diff_3.html"

    cloze_deletion_stats = get_all_cloze_deletion_stats(flattened_runs["output"])
    print(cloze_deletion_stats)

    largest_average_indices = cloze_deletion_stats["average_cloze"].nlargest(10).index
    smallest_average_indices = cloze_deletion_stats["average_cloze"].nsmallest(10).index
    print(cloze_deletion_stats.loc[largest_average_indices])
    print(cloze_deletion_stats.loc[smallest_average_indices])

    smallest_std_dev_indices = cloze_deletion_stats["std_dev_cloze"].nsmallest(10).index
    largest_std_dev_indices = cloze_deletion_stats['std_dev_cloze'].nlargest(10).index
    print(cloze_deletion_stats.loc[smallest_std_dev_indices])
    print(cloze_deletion_stats.loc[largest_std_dev_indices])

    print(df_outputs.loc[largest_average_indices[0]]["variables"])
    print(df_outputs.loc[largest_average_indices[0]]["anki_cards"][0])

    # d = difflib.HtmlDiff()

    # for a, b in combinations(range(40,len(flattened_runs["output"])), 2):
    #     cards_a = flattened_runs["output"][a]['anki_cards']
    #     variables_a = flattened_runs["output"][a]['variables']
    #     cards_b = flattened_runs["output"][b]['anki_cards']
    #     variables_b = flattened_runs["output"][b]['variables']

    #     for i in range(min(len(cards_a), len(cards_b))):
    #         variables_a["card"] = i
    #         variables_b["card"] = i

    #         lines_a = re.split('(?<=[.!?,;])', cards_a[i]["Text"])
    #         lines_b = re.split('(?<=[.!?,;])', cards_b[i]["Text"])
    #         html_diff = d.make_file(lines_a, lines_b, fromdesc=variables_a, todesc=variables_b, context=True)
            
    #         output_doc.extend(BeautifulSoup(html_diff, "html.parser"))

    # with open(output_html_diff_path, "w", encoding="utf-8") as file:
    #     file.write(str(output_doc))



if __name__ == '__main__':
    main()