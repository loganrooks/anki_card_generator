import difflib
import json
import os
import statistics
import string
import re
from bs4 import BeautifulSoup
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

def get_similarity(card_set_a, card_set_b):
    # return the difflib sequence similarity between two sets of cards, appending their text fields together
    text_a = ""
    text_b = ""

    for card in card_set_a:
        text_a += card["Text"]

    for card in card_set_b:
        text_b += card["Text"]

    lines_a = re.split('(?<=[.!?,;])', text_a)
    lines_b = re.split('(?<=[.!?,;])', text_b)
    similarity = difflib.SequenceMatcher(None, lines_a, lines_b).ratio()

    return similarity


    


def get_similarities_df(outputs, ignore_variables={}):
    # give set of outputs, return pd.Dataframe with similarity column
    similarities = []
    all_variables = []
    variable_ids = set()

    for a, b in combinations(range(len(outputs)), 2):
        cards_a = outputs[a]['anki_cards']
        from_variables = outputs[a]['variables']
        cards_b = outputs[b]['anki_cards']
        to_variables = outputs[b]['variables']

        ignore=False

        for key in ignore_variables.keys():
            if (from_variables.get(key, None) in ignore_variables.get(key, [])) or (to_variables.get(key, None) in ignore_variables.get(key, None)):
                ignore=True
                break

        # TODO: this won't work if there are runs with different variables from the rest, 
        # would like to eventually have it so those are included too

        if not ignore:
            similarity = get_similarity(cards_a, cards_b)
            similarities.append(similarity)

            variable_ids.update(list(from_variables.keys()))

            all_variables.append([a] + list(from_variables.values()) + [b] + list(to_variables.values()))

    
    index = pd.MultiIndex.from_product([['from', 'to'], ["output_id"] + list(variable_ids)])
    similarities_df = pd.DataFrame(all_variables, columns=index)
    similarities_df['similarity'] = similarities

    return similarities_df

def get_cloze_descriptions(cloze_deletion_stats: pd.DataFrame, outputs_a: pd.DataFrame, outputs_b: pd.DataFrame, variables_a, variables_b, x: int, y=0):
    if not cloze_deletion_stats.empty:
        fromdesc = f""
        todesc = f""

        # get the stats for the entry in the outputs_a dataframe which will have as one of its columns its original index (since it could be a sub data frame)
        cloze_deletion_stats_a = cloze_deletion_stats.loc[outputs_a.loc[x]["index"]]
        for stat,value in cloze_deletion_stats_a.items():
            fromdesc += f"{stat}: {value[y] if type(value)==list else value} // "

        cloze_deletion_stats_b = cloze_deletion_stats.loc[outputs_b.loc[x]["index"]]

        for stat, value in cloze_deletion_stats_b.items():
            todesc += f"{stat}: {value[y] if type(value)==list else value} // "
    else:
        fromdesc, todesc = variables_a, variables_b

    return fromdesc, todesc

def get_cloze_similarity():
    pass

def create_html_diff(outputs_a: pd.DataFrame, outputs_b: pd.DataFrame, output_html_path, individual_cards=False, cloze_deletion_stats=pd.DataFrame()):
    # side by side comparison between two equally long DataFrames with ankicards
    d = difflib.HtmlDiff()

    output_doc = BeautifulSoup()
    output_doc.append(output_doc.new_tag("html"))
    output_doc.html.append(output_doc.new_tag("body"))

    for x in range(min(len(outputs_a), len(outputs_b))):
        cards_a = outputs_a.loc[x]['anki_cards']
        variables_a = outputs_a.loc[x]['variables']
        cards_b = outputs_b.loc[x]['anki_cards']
        variables_b = outputs_b.loc[x]['variables']

        if individual_cards:
            for y in range(min(len(cards_a), len(cards_b))):
                variables_a["card"] = y
                variables_b["card"] = y

                lines_a = re.split('(?<=[.!?,;])', cards_a[y]["Text"])
                lines_b = re.split('(?<=[.!?,;])', cards_b[y]["Text"])

                fromdesc, todesc = get_cloze_descriptions(cloze_deletion_stats, outputs_a, outputs_b, variables_a, variables_b, x, y)

                html_diff = d.make_file(lines_a, lines_b, fromdesc=fromdesc, todesc=todesc, context=True)
                output_doc.extend(BeautifulSoup(html_diff, "html.parser"))

        else:
            text_a = ""
            text_b = ""

            for card in cards_a:
                text_a += card["Text"]

            for card in cards_b:
                text_b += card["Text"]

            lines_a = re.split('(?<=[.!?,;])', text_a)
            lines_b = re.split('(?<=[.!?,;])', text_b)

            fromdesc, todesc = get_cloze_descriptions(cloze_deletion_stats, outputs_a, outputs_b, variables_a, variables_b, x, 0)
  
        
            html_diff = d.make_file(lines_a, lines_b, fromdesc=fromdesc, todesc=todesc, context=True)
            output_doc.extend(BeautifulSoup(html_diff, "html.parser"))

            

    with open(output_html_path, "w", encoding="utf-8") as file:
        file.write(str(output_doc))
        


    # # lines = [row.split() for row in lines]  # the file should already break at each line break
    # # lines = [(int(row[0]), row[1]) for row in lines]
    # # lines = groupby(sorted(lines), lambda x: x[0])  # combine strings into their respective groups, sorting them first on int of first element
    # # group_max = dict()
    # # for group in lines:
    # #     strings = list(group[1])  # need to convert group[1] from iterator into list
    # #     if len(strings) > 1:  # if the number of strings is 1, then there is nothing to compare the string with in its group
    # #         similarity = 1
    # #         for line1, line2 in combinations(strings, 2):
    # #             s = difflib.SequenceMatcher(None, line1[1], line2[1])  # need to compare second element in each list and exclude the first element (which is the group number)
    # #             similarity = s.ratio() if s.ratio() < similarity else similarity
    # #         group_max[line1[0]] = 1 - similarity  # gives difference ratio
    # # return group_max

def extract_stats_from_text(text):
    clozes = re.findall(r"\{\{c\d+::(.+?)\}\}", text)
    original_text = re.sub(r"\{\{c\d+::(.+?)\}\}", r"\1", text)
    cloze_percentage = len(' '.join(clozes)) / len(original_text) * 100

    cloze_count = len(clozes)

    cloze_unique = set(clozes)
    cloze_count_unique = len(cloze_unique)

    avg_cloze_sim_ratio, cloze_unique_sim_ratio, avg_cloze_unique_sim_ratio = 3*[0.0]
    avg_cloze_sim_ndiff, cloze_unique_sim_ndiff, avg_cloze_unique_sim_ndiff = 3*[0.0]
    # for cloze in cloze_unique:
    #     cloze_unique_sim_ratio += difflib.SequenceMatcher(None, original_text, cloze).ratio() * 100
    #     cloze_unique_sim_ndiff += compute_similarity(original_text, cloze) * 100

    for cloze in clozes:
        avg_cloze_sim_ratio += difflib.SequenceMatcher(None, original_text, cloze).ratio() * 100 / len(clozes)
        avg_cloze_sim_ndiff += compute_similarity(original_text, cloze) * 100 / len(clozes)

    for cloze in cloze_unique:
        avg_cloze_unique_sim_ratio += difflib.SequenceMatcher(None, original_text, cloze).ratio() * 100 / len(cloze_unique)
        avg_cloze_unique_sim_ndiff += compute_similarity(original_text, cloze) * 100 / len(cloze_unique)

    # cloze_sim_ratio =  difflib.SequenceMatcher(None, original_text, ' '.join(clozes)).ratio() * 100
    # cloze_sim_ndiff = compute_similarity(original_text, ' '.join(clozes)) * 100

    stats = {
            #  "cloze_sim_ratio": cloze_sim_ratio, 
            #  "cloze_sim_ndiff": cloze_sim_ndiff, 
            #  "avg_cloze_sim_ratio": avg_cloze_sim_ratio, 
            #  "avg_cloze_sim_ndiff": avg_cloze_sim_ndiff, 
             "avg_cloze_unique_sim_ratio": avg_cloze_unique_sim_ratio, 
             "avg_cloze_unique_sim_ndiff": avg_cloze_unique_sim_ndiff, 
            #  "cloze_unique_sim_ratio": cloze_unique_sim_ratio, 
            #  "cloze_unique_sim_ndiff": cloze_unique_sim_ndiff, 
             "cloze_count": cloze_count, 
             "cloze_count_unique": cloze_count_unique, 
             "cloze_percentage": cloze_percentage}
    return stats

def compute_similarity(input_string, reference_string):
    diff = difflib.ndiff(input_string, reference_string)
    diff_count = 0
    for line in diff:
      # a "-", indicating that it is a deleted character from the input string.
        if line.startswith("-"):
            diff_count += 1
# calculates the similarity by subtracting the ratio of the number of deleted characters to the length of the input string from 1
    return 1 - (diff_count / len(input_string))




def get_cloze_deletions_stats(anki_cards, individual_cards=True):
    all_stats = {}
    text = ""
    if individual_cards:
        for card in anki_cards:
            text = card["Text"]
            stats = extract_stats_from_text(text)
            for key, value in stats.items():
                all_stats[key] = all_stats[key] + [value] if type(all_stats.get(key, None))==list else [value]
        all_stats["average_cloze"] = sum(all_stats["cloze_count"]) / len(all_stats["cloze_count"])
        all_stats["std_dev_cloze"] = statistics.stdev(all_stats["cloze_count"])
    else:
        for card in anki_cards:
            text += card["Text"]
        all_stats = extract_stats_from_text(text)

    return all_stats


def get_all_cloze_deletion_stats(outputs, variable_ids=["temperature", "top_p", "max_completion_tokens"], individual_cards=True):
    all_cloze_stats = []
    for i in range(len(outputs)):
        anki_cards = outputs[i]["anki_cards"]
        variables = outputs[i]["variables"]
        cloze_stats = []

        # add the variables as the starting columns of the cloze stat data frame
        for variable_id in variable_ids:
            cloze_stats.append(variables[variable_id])

        all_stats = get_cloze_deletions_stats(anki_cards, individual_cards=individual_cards)
        cloze_stats.extend(list(all_stats.values()))
        all_cloze_stats.append(cloze_stats)
    
    return pd.DataFrame(all_cloze_stats.copy(), columns=variable_ids + list(all_stats.keys()))

def main():
    runs_json_path = os.path.dirname(__file__) + "/outputs/sz_test.json"
    with open(runs_json_path, 'r') as runs_json:
        runs = json.load(runs_json)

    flattened_runs = flatten_runs(runs)
    output_df = pd.DataFrame(flattened_runs["output"])
    
    ignore_variables = {"max_completion_tokens": [1200, 2000]}
    similarities_df = get_similarities_df(flattened_runs['output'], ignore_variables=ignore_variables)
    print(similarities_df.head())
    most_different_indices = similarities_df["similarity"].nsmallest(10).index
    print(similarities_df.loc[most_different_indices])
    output_html_diff_path = os.path.dirname(__file__) + "/outputs/sz_html_diff_most.html"
    most_different_indices_a = similarities_df.loc[most_different_indices]['from']['output_id']
    outputs_a = output_df.loc[most_different_indices_a].reset_index()

    most_different_indices_b = similarities_df.loc[most_different_indices]['to']['output_id']
    outputs_b = output_df.loc[most_different_indices_b].reset_index()

    cloze_deletion_stats = get_all_cloze_deletion_stats(flattened_runs["output"], individual_cards=False)

    
    create_html_diff(outputs_a, outputs_b, output_html_path=output_html_diff_path, cloze_deletion_stats=cloze_deletion_stats)



    # print(cloze_deletion_stats)

    output_percentage_html_diff_path = os.path.splitext(output_html_diff_path)[0] + "_percentage.html"

    largest_percentage_indices = cloze_deletion_stats["cloze_percentage"].nlargest(10).index
    # smallest_average_indices = cloze_deletion_stats["average_cloze"].nsmallest(10).index
    print(cloze_deletion_stats.loc[largest_percentage_indices])
    # print(cloze_deletion_stats.loc[smallest_average_indices])

    # largest_percentage_indices_a = similarities_df.loc[most_different_indices]['from']['output_id']
    #outputs_a = output_df.loc[most_different_indices_a].reset_index()

    # smallest_std_dev_indices = cloze_deletion_stats["std_dev_cloze"].nsmallest(10).index
    # largest_std_dev_indices = cloze_deletion_stats['std_dev_cloze'].nlargest(10).index
    # print(cloze_deletion_stats.loc[smallest_std_dev_indices])
    # print(cloze_deletion_stats.loc[largest_std_dev_indices])

    # cloze_deletion_stats.set_index(["temperature", "top_p", "max_completion_tokens"])
    # print(cloze_deletion_stats.head())
    # print("Largest Average Cloze")
    # print(df_outputs.loc[largest_average_indices[0]]["variables"])
    # print(df_outputs.loc[largest_average_indices[0]]["anki_cards"][0])
    # print("Smallest Average Cloze")
    # print(df_outputs.loc[smallest_average_indices[0]]["variables"])
    # print(df_outputs.loc[smallest_average_indices[0]]["anki_cards"][0])

    # print("Smallest Std. Dev. Cloze")
    # print(df_outputs.loc[smallest_std_dev_indices[0]]["variables"])
    # print(df_outputs.loc[smallest_std_dev_indices[0]]["anki_cards"][0])

    # print("Largest Std. Dev. Cloze")
    # print(df_outputs.loc[largest_std_dev_indices[0]]["variables"])
    # print(df_outputs.loc[largest_std_dev_indices[0]]["anki_cards"][0])



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