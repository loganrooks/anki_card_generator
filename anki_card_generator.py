from openai import OpenAI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import argparse
import json
import os
import math

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Function to extract text from epub file
def extract_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    text = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            
            # Remove footnotes (assuming footnotes have the class 'calibre_9')
            for footnote in soup.find_all('span', class_='calibre9'):
                footnote.decompose()

            text.append(soup.get_text())
    return [book.title,'\n'.join(text)]

# Function to add brackets to the regex if they forgot to include them so the page number can be included
def ensure_brackets(s):
    if not s.startswith('('):
        s = '(' + s
    if not s.endswith(')'):
        s = s + ')'
    return s

def merge_adjacent_elements(lst=list, n=1):
    if n < 1:
        raise ValueError("n must be at least 1")
    return [''.join(lst[i:i + n]) for i in range(0, len(lst), n)]

# Function to split text based on regex or default to 8 paragraphs, ignores everything up to the first page number.
def split_text(text, regex=None):
    if regex:
        return merge_adjacent_elements(re.split(ensure_brackets(regex), text)[1:], n=2)
    else:
        paragraphs = text.split('\n')
        return ['\n'.join(paragraphs[i:i+8]) for i in range(0, len(paragraphs), 8)]

# Function to create cloze deletion anki cards using OpenAI API
def create_anki_cards(text_chunk, system_prompt):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"You are about to be given a section of the text, you will convert these into Anki cards in accordance with the guidelines stipulated in your system prompt. Emphasis here is on trying to have as many cloze deletions per card as you can, leave very few words undeleted. How you are to go about doing this is described in your system message. The text:\n{text_chunk}"}
    ],
    max_completion_tokens=3000,
    temperature=0.7)
    return response.choices[0].message.content.strip()

# Function to format the output as JSON
def format_as_json(output):
    try:
        # Find the position of the first '['
        start_index = output.find('[')
        
        # Initialize the counter for brackets
        counter = 0
        outer_scope_start = -1
        
        # Find the outermost open '{' bracket using the counter
        for i in range(len(output) - 1, -1, -1):
            if output[i] == '}':
                counter -= 1
            elif output[i] == '{':
                counter += 1
                if counter == 1:
                    outer_scope_start = i
                    counter = 0 
        print(f"Outer Scope Start: {outer_scope_start}\n")
        # If an outer scope '{' bracket is found, find the first closing '}' bracket that precedes it
        if outer_scope_start != -1:
            end_index = output.rfind('}', 0, outer_scope_start) + 1
            cleaned_output = output[start_index:end_index] + ']'

            remaining_start = output.find('{', end_index)
            remaining_content = output[remaining_start:-1].strip('```').strip()
            print(f"Remaining JSON: {remaining_content}")

        else:
            end_index = len(output)
            cleaned_output = output[start_index:end_index].strip('```')
            remaining_content = None
        
        print(f"Start Index: {start_index}\n")
        print(f"End Index: {end_index}\n")
    
        # Load the JSON data
        json_output = json.loads(cleaned_output)
        return (json_output, remaining_content), None
    except json.JSONDecodeError as e:
        print(f"Error: {str(e)}\nCleaned Beginning:{cleaned_output[:100]}\nCleaned End: {cleaned_output[-101:]}\n")
        return None, str(e)
    
def find_remaining_text(input_chunk, remaining_content):
    # Extract the content of the "Text" field from the remaining_content string
    text_start = remaining_content.find('"Text": "') + len('"Text": "')
    text_end = remaining_content.find('",', text_start)
    remaining_text = remaining_content[text_start:text_end]
    
    # Consider only the text up until the first "{"
    search_text = remaining_text.split("{")[0]

    search_text = search_text[:40] if len(search_text) > 40 else search_text
    
    # Find the position of the search_text in the input_chunk
    start_index = input_chunk.find(search_text)
    
    # If the search_text is found, return the section from start_index to the end of input_chunk
    if start_index != -1:
        return input_chunk[start_index:].strip()
    else:
        return None
    

# Main function to handle CLI arguments and process the epub file
def main():
    parser = argparse.ArgumentParser(description='Generate cloze deletion anki cards from an epub file.')
    parser.add_argument('epub_file', type=str, help='Path to the epub file')
    parser.add_argument('--regex', type=str, help='Regular expression to split the text', default=None)
    parser.add_argument('--prompt_file', type=str, help='Path to a text file containing system prompt instructions', default=None)
    parser.add_argument('--prompt_text', type=str, help='Prompt instructions as a string', default=None)
    parser.add_argument('--out', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--test', action='store_true', help='Generate Anki cards for the first three chunks only and append "_test" to the output filename')
    parser.add_argument('--pages_per_chunk', type=int, help='Number of pages per text chunk to be inputed to the model', default=1)
    parser.add_argument('--page_range', type=int, nargs=2, help='Number of total pages to be converted', default=[1,0])

    args = parser.parse_args()


    # Extract text from epub file
    title, text = extract_text_from_epub(args.epub_file)

    # Split text based on regex or default to 8 paragraphs
    text_chunks = merge_adjacent_elements(split_text(text, args.regex), n=args.pages_per_chunk)

    # Load prompt instructions from file or use provided prompt text
    if args.prompt_file:
        with open(args.prompt_file, 'r') as file:
            system_prompt = file.read()
    elif args.prompt_text:
        system_prompt = args.prompt_text
    else:
        system_prompt = (
            f"You are a philosophy professor creating Anki flash cards from a given text for self-study purposes. "
            "You will be given a chunk of text from one of Martin Heidegger's books, to make Anki cloze deletion cards. "
            "Create as many flash cards as needed following these rules:\n"
            "- Do not create duplicates.\n"
            "- Provide only the JSON for the flash cards; any other text will be ignored.\n"
            "- Format the cards with cloze deletion for the front.\n"
            "- Include the text citation with page number under the field 'Citation'.\n"
            "- Do not invent anything; use only the given text.\n"
            "- Do not just remove single words for cloze deletion; include phrases or clauses as well.\n"
            "- Emphasis: Cloze delete roughly one-third of the input, ensuring all German, Greek, or Latin phrases/terms are cloze deleted. Aim for at least 30 cloze deletions per card, ideally 40, with high density.\n"
            "- The 'Text' field should contain at least a paragraph (4-5 sentences) with one-third cloze deletions per card but max 3 clozes (c1, c2, c3). Multiple 'c1's, 'c2's, and possibly 'c3's should be thematically related.\n"
            f"- Include the title '{title}' along with the page number in the citation.\n"
            "- Write in English (unless there are German, Latin, or Greek terms).\n"
            "- Ensure each 'Text' field has at least 4 sentences. Do not create so many cards that some have less than 4 sentences. It's okay for some cards to have up to 8-10 sentences.\n"
            "- Cloze delete significant nouns, verbs, words, and phrases (and every single Greek, German, and Latin term cloze deleted and tagged with 'c3'). Split other cloze deletions roughly 50/50 per card between 'c1' and 'c2', grouping them thematically.\n"
            "- Ensure each card is self-complete with enough context to recognize and understand its meaning vaguely on its own. If part of the passage includes a quote from another philosopher, provide enough context prior to the quote.\n"
            "- Ensure all instances and semantically similar instances of a deleted term are clozed. These cards should not be easy, I want no left over hints."
            "- Cloze delete all semantically similar instances of any phrase or term you cloze delete. Cloze delete all of Heidegger's terminology and all of his definitions. Most subjects and predicates of sentences must be cloze deleted."
            "- All hyphenated words must be cloze deleted (e.g. Being-in-the-world, within-the-world, Being-in, reference-relations, existential-ontological, ontico-existentiell etc.)."
            "- Example format for 'Text' field: \"The full {{c1::essence of truth}}, including its most proper {{c1::nonessence}}, keeps {{c2::Dasein}} in need by this perpetual {{c1::turning to and fro}}. {{c2::Dasein}} is a {{c1::turning into need}}. From the {{c2::Da-sein}} of human beings and from it alone arises the disclosure of {{c1::necessity}} and, as a result, the {{c1::possibility of being transposed}} into what is {{c2::inevitable}}. The disclosure of beings as such is {{c1::simultaneously}} and {{c1::intrinsically}} the {{c2::concealing of beings}}.\""
        )

    
    if args.regex:
        system_prompt += f"For citation purposes, you can locate the page numbers using the regex '{args.regex}'. Make sure to keep track of when you cross page numbers and cite accordingly. Every text chunk starts with a page number and there should be a total of {str(args.pages_per_chunk)} pages per text chunk."

    # Determine output paths
    output_json_path = args.out
    if args.test:
        output_json_path = os.path.splitext(output_json_path)[0] + "_test.json"
        input_json_path = os.path.splitext(output_json_path)[0] + "_input.txt"
        all_inputs = []



    error_log_path = os.path.splitext(output_json_path)[0] + "_errors.txt"
    final_remaining_text_path =  os.path.splitext(output_json_path)[0] + "_remaining.txt"

    all_anki_cards = []
    error_count = 0
    consecutive_errors = 0

    # Limit chunks if test flag is set
    if args.test:
        text_chunks = text_chunks[:3]
    else:
        text_chunks = text_chunks[math.ceil(args.page_range[0]/args.pages_per_chunk) - 1: math.ceil(args.page_range[1]/args.pages_per_chunk) - 1]

    # Create anki cards for each chunk and handle errors
    for i, chunk in enumerate(text_chunks):
        anki_cards_output = create_anki_cards(chunk, system_prompt)
        anki_cards_json, error = format_as_json(anki_cards_output)


        if anki_cards_json:
            if args.test:
                all_inputs.append([i, chunk])
            anki_cards_json_cleaned, anki_cards_json_remaining = anki_cards_json
            all_anki_cards.extend(anki_cards_json_cleaned)
            consecutive_errors = 0

            if anki_cards_json_remaining:
                # print(f"Output: {i}\nRemaining JSON: {anki_cards_json_remaining}\n")
                remaining_text = find_remaining_text(chunk, anki_cards_json_remaining)
                if remaining_text:
                    print(f"Remaining Text: {remaining_text}\n")
                    if i < len(text_chunks) - 1:
                        text_chunks[i+1] = remaining_text + text_chunks[i+1]
                    else:
                        with open(final_remaining_text_path, 'a') as final_remaining_text:
                            final_remaining_text.write(f"{final_remaining_text}\n")
                else:
                    message = f"Output #{i} was incomplete but there was no remaining text.\nRemaining JSON: {anki_cards_json_remaining}\n"
                    print(message)
                    with open(error_log_path, 'a') as error_log:
                        error_log.write(message +f"Chunk:\n{chunk}\nOutput:\n{anki_cards_output}\n\n")
        
        else:
            error_count += 1
            consecutive_errors += 1
            with open(error_log_path, 'a') as error_log:
                error_log.write(f"Error: {error}\nChunk:\n{chunk}\nOutput:\n{anki_cards_output}\n\n")

        if consecutive_errors >= 3 or error_count >= len(text_chunks) * 0.2:
            print("Too many errors encountered. Stopping execution.")
            break
    if args.test:
        with open(input_json_path, 'w') as input_file:
            input_file.write(str(all_inputs))

    # Write all anki cards to output JSON file
    with open(output_json_path, 'w') as output_file:
        json.dump(all_anki_cards, output_file, indent=4)

if __name__ == '__main__':
    main()
