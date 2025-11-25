import csv
import datetime
import hashlib
import logging
import string
from openai import OpenAI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import argparse
from argparse import Namespace
import json
import os
import math
from itertools import product

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def format_hierarchy_for_llm(title, hierarchy, chunk_length, full_paragraphs=True):
    def add_citation(title, headers, paragraph_index):
        return f"[{title}: {', '.join(headers)}, {paragraph_index}]"

    def flatten_hierarchy(subsections, headers, flat_list, paragraph_index=1):
        for subsection in subsections:
            new_headers = headers + [subsection['title']]
            if 'text' in subsection and subsection['text']:
                for paragraph in subsection['text']:
                    citation = add_citation(title, new_headers, paragraph_index)
                    flat_list.append((paragraph, citation))
                    paragraph_index += 1
            if subsection['subsections']:
                paragraph_index = flatten_hierarchy(subsection['subsections'], new_headers, flat_list, paragraph_index)
        return paragraph_index

    def create_chunks(flat_list, chunk_length, full_paragraphs):
        chunks = []
        current_chunk = ""
        current_length = 0
        current_citation = ""

        for i, (paragraph, citation) in enumerate(flat_list):
            if i == 0:
                current_chunk += citation + "\n\n"
                current_citation = citation

            paragraph_length = len(paragraph)
            # Remove the paragraph number from the citation for comparison
            citation_base = re.sub(r", \d+\]$", "]", citation)
            current_citation_base = re.sub(r", \d+\]$", "]", current_citation)
            if full_paragraphs:
                if abs(current_length + paragraph_length + 2 - chunk_length) < abs(current_length - chunk_length):
                    if current_citation and citation_base != current_citation_base:
                        current_chunk += citation + "\n\n"
                        current_citation = citation
                    current_chunk += paragraph + "\n\n"
                    current_length += paragraph_length + 2
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = citation + "\n\n" + paragraph + "\n\n"
                    current_length = len(current_chunk)
                    current_citation = citation
            else:
                while paragraph:
                    if abs(current_length + len(paragraph) + 2 - chunk_length) < abs(current_length - chunk_length):
                        if current_citation and citation_base != current_citation_base:
                            current_chunk += citation + "\n\n"
                            current_citation = citation
                        space_left = chunk_length - current_length - 2
                        current_chunk += paragraph[:space_left] + "\n\n"
                        paragraph = paragraph[space_left:]
                        current_length = chunk_length
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = citation + "\n\n" + paragraph + "\n\n"
                        current_length = len(current_chunk)
                        current_citation = citation
                        paragraph = ""

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    flat_list = []
    for part in hierarchy:
        headers = [part['title']]
        flatten_hierarchy(part['subsections'], headers, flat_list)

    chunks = create_chunks(flat_list, chunk_length, full_paragraphs)
    return chunks


def extract_hierarchy_from_epub(epub_path, header_tags, start_page=5, footnote_tag={'name': 'div', 'class': 'footnotesection'}, \
                                                                 text_tags=[{'name': 'div', 'class': re.compile(r'^(para|para1)$', re.IGNORECASE)}]):       
    book = epub.read_epub(epub_path)
    doc = [page for page in book.get_items() if page.get_type() == ebooklib.ITEM_DOCUMENT][start_page:]
    hierarchy = []

    def parse_hierarchy(soup):
        headers = soup.find_all([tag_info['name'] for tag_info in header_tags])
        structure = []

        for i, header in enumerate(headers):
            tag_name = header.name
            tag_class = header.get('class', [])
            level = next((index for index, tag_info in enumerate(header_tags) if tag_info['name'] == tag_name and tag_info.get('class') in tag_class), None)
            
            if level is not None:
                title = header.get_text(' ', strip=True)

                next_header = headers[i + 1] if i + 1 < len(headers) else None
                text = []
                seen_text = set()

                # Collect text until the next header
                for next_element in header.next_elements:
                    if next_element == next_header:
                        break
                    if next_element.name == footnote_tag['name'] and footnote_tag['class'] in next_element.get('class', []):
                        next_element.decompose()
                        continue
                    elif (next_element.name and next_element.name=="div" and next_element.get("class", [])==["calibre3"]):
                        
                        if level == 2 or level == 3:
                            # then we iterate through siblings
                            text.append(next_element.get_text())
                            for sibling in next_element.next_siblings:
                                if sibling.name and sibling.name=="div" and set(["para", "para1"]).intersection(set(sibling.get("class", []))):
                                    text.append(sibling.get_text())
                            break

                        elif level == 1 and next_element.next_sibling and next_element.next_sibling.next_sibling:
                            paragraph_tag_pattern = re.compile(r'^(div|p)$')
                            paragraph_class_pattern = re.compile(r'^(para|para1)$', re.IGNORECASE)
                            # text_tag_kwargs = [text_tag.get(attr) for text_tag in text_tags] for attr in set().union(*text_tags)
                            for paragraph in next_element.next_sibling.next_sibling.find_all(paragraph_tag_pattern, class_=paragraph_class_pattern):
                                if paragraph.name == footnote_tag['name'] and footnote_tag['class'] in paragraph.get('class', []):
                                    paragraph.decompose()
                                    continue
                                text.append(paragraph.get_text())
                            break
                        else:
                            logger.debug(f"Level: {level}\nNext Element: {next_element}\nNext Element Sibling: {next_element.next_sibling}\nNext Element Sibling Sibling: {next_element.next_sibling.next_sibling}")
                            break
                structure.append({
                                'title': title,
                                'level': level,
                                'subsections': [],
                                'text': text
                            })                                                                 
            
        return structure

    for page in doc:
        if page.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(page.get_body_content(), 'html.parser')
            hierarchy.extend(parse_hierarchy(soup))

    # Organize hierarchy into nested structure
    def nest_hierarchy(flat_structure):
        if not flat_structure:
            return []

        nested = []
        stack = [nested]
        current_level = 0

        for item in flat_structure:
            while item['level'] < current_level:
                stack.pop()
                current_level -= 1

            if item['level'] > current_level:
                new_subsection = []
                stack[-1][-1]['subsections'] = new_subsection
                stack.append(new_subsection)
                current_level += 1

            stack[-1].append(item)

        return nested

    nested_hierarchy = nest_hierarchy(hierarchy)
    return nested_hierarchy

# Function to extract text from epub file, remove footnotes, table of contents, and copyright
def extract_text_from_epub(epub_path, footnote_tag={'name': 'div', 'class': 'footnotesection'}, \
                                           toc_tag={'name': 'div', 'class': 'toc'}, \
                                     copyright_tag={'name': 'div', 'class': 'copyrightpage'}):
    
    book = epub.read_epub(epub_path)
    author = book.get_metadata('DC', 'creator')[0][0]
    title = book.title.split(':')[0]
    text = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            
            # Remove footnotes)
            for footnote in soup.find_all(**footnote_tag):
                footnote.decompose()

            # Remove table of contents
            for toc in soup.find_all(**toc_tag):
                toc.decompose()

            # Remove copyright
            for copyright in soup.find_all(**copyright_tag):
                copyright.decompose()
                
            text.append(soup.get_text())
    return Namespace(author=author, title=title, text=text)

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
def split_text(text: str, regex, full_paragraphs=True, n_paragraphs_per_page=3):
    if not full_paragraphs:
        text_chunks = merge_adjacent_elements(re.split(ensure_brackets(regex), text)[1:], n=2)
    else:
        text_chunks=[]
        text_chunk = ''
        paragraphs = text.split('\n\n')
        for i, paragraph in enumerate(paragraphs):
            text_chunk += paragraph + "\n\n"
            if (i > 0) and re.search(regex, paragraph):
                text_chunks.append(text_chunk)
                text_chunk = ''

    return text_chunks

# Function to create cloze deletion anki cards using OpenAI API
def create_anki_cards(text_chunk, system_prompt, temperature=0.7, max_completion_tokens=2000, top_p=0.5):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"You are about to be given a section of the text, you will convert these into Anki cards in accordance with the guidelines stipulated in your system prompt. Emphasis here is on trying to have as many cloze deletions per card as you can, leave very few words undeleted. How you are to go about doing this is described in your system message. The text:\n{text_chunk}"}
    ],
    max_completion_tokens=max_completion_tokens,
    temperature=temperature,
    top_p=top_p)
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
        logger.debug(f"Outer Scope Start: {outer_scope_start}")
        # If an outer scope '{' bracket is found, find the first closing '}' bracket that precedes it
        if outer_scope_start != -1:
            end_index = output.rfind('}', 0, outer_scope_start) + 1
            cleaned_output = output[start_index:end_index] + ']'

            remaining_start = output.find('{', end_index)
            remaining_content = output[remaining_start:-1].strip('```').strip()
            logger.debug(f"Remaining JSON: {remaining_content}")

        else:
            end_index = len(output)
            cleaned_output = output[start_index:end_index].strip('```')
            remaining_content = None
        
        logger.debug(f"Start Index: {start_index}")
        logger.debug(f"End Index: {end_index}")

        # Load the JSON data
        json_output = json.loads(cleaned_output)
        return (json_output, remaining_content), None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}\nCleaned Beginning: {cleaned_output[:100]}\nCleaned End: {cleaned_output[-101:]}")
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
    

def write_json_to_file(output_json_path: str, output: str, args: argparse.ArgumentParser, run_id: str, mode='w+'):
    try:
        with open(output_json_path, mode, encoding='utf8') as output_file:
                output = [{"args": vars(args), "output": output.copy(), "run_id": run_id}]
                if not args.overwrite:
                    try:
                        output_file.seek(0)
                        existing_content = json.loads(output_file.read())
                        output.extend(existing_content)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"File corrupted, empty or is not a list. Forcing overwrite. Error: {e}")
                    finally:
                        output_file.seek(0)
                        output_file.truncate()
                
                json.dump(output, output_file, indent=4)
    except FileNotFoundError as e:
        return e

def write_json_to_csv(output_csv_path: str, output: str, args: argparse.ArgumentParser,  run_id: str, mode='w+'):
    # Flatten the nested JSON structure
    flattened_json = flatten_json(output)
    try:
        # Open the CSV file with the specified mode and encoding
        with open(output_csv_path, mode, newline='', encoding='utf8') as output_file:
            # Create a CSV DictWriter object with fieldnames from the first dictionary in the list
            writer = csv.DictWriter(output_file, fieldnames=flattened_json[0].keys())
            
            if not args.overwrite:
                try:
                    # Move the file pointer to the beginning of the file
                    output_file.seek(0)
                    # Read the existing content of the CSV file into a list of dictionaries
                    existing_content = list(csv.DictReader(output_file))
                    # Extend the flattened JSON list with the existing content
                    flattened_json.extend(existing_content)
                except (csv.Error, ValueError) as e:
                    # Handle cases where the file is corrupted, empty, or not a list
                    logger.warning(f"File corrupted, empty or is not valid CSV. Forcing overwrite. Error: {e}")
                finally:
                    # Move the file pointer to the beginning and truncate the file
                    output_file.seek(0)
                    output_file.truncate()
            
            # Write the header row to the CSV file
            writer.writeheader()
            # Write the rows to the CSV file
            writer.writerows(flattened_json)
    except FileNotFoundError as e:
        return e

def flatten_json(nested_json):
    """
    Flatten a nested JSON structure.

    Args:
        nested_json (list): A list of dictionaries, potentially nested.

    Returns:
        list: A list of flattened dictionaries.
    """
    def flatten_dict(d: dict, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # Recursively flatten nested dictionaries
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Handle lists by creating separate entries for each item
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        # Recursively flatten nested dictionaries within lists
                        items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                    else:
                        # Add non-dictionary items in lists
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                # Add non-dictionary items
                items.append((new_key, v))
        return dict(items)

    # Flatten each dictionary in the list
    flattened_list = []
    for item in nested_json:
        if 'anki_cards' in item:
            for card in item['anki_cards']:
                flattened_card = flatten_dict(card)
                # Add other top-level keys to each card
                for key in item:
                    if key != 'anki_cards':
                        flattened_card[key] = item[key]
                flattened_list.append(flattened_card)
        else:
            flattened_list.append(flatten_dict(item))
    return flattened_list


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Generate cloze deletion anki cards from an epub file.')
    # I/O
    parser.add_argument('epub_path', type=str, help='Path to the epub file')
    parser.add_argument('-o', '--out', type=str, required=True, help='Output JSON file path')
    parser.add_argument('-w', '--overwrite', action='store_true', help='Overwrite what is in the output destination')

    # FOR TEXT BATCHING AS INPUT TO THE LLM
    parser.add_argument('-r', '--regex', type=str, help='Regular expression to split the text. If none given then will default to extracting on the basis of chunk size.', default=None)
    parser.add_argument('--pages_per_chunk', type=int, help='Number of pages per text chunk to be inputed to the model', default=1)
    parser.add_argument('--page_range', type=int, nargs=2, help='Number of total pages to be onverted. If none given, then will default to all of text.', default=[1, 0])
    parser.add_argument('--full_paragraphs', action='store_true', help='Whether the chunks should always end on the final sentence of a paragraph.')
    parser.add_argument('--chunk_length', type=int, help='The desired length of the chunk. Actual chunk size will vary depending on whether full_paragraphs is flagged.', default=5000)
    parser.add_argument('--chunk_range', type=int, nargs=2, help='Number of total chunks to be converted. If none given, then will default to all of text.', default=[0, -1])

    # FOR EXTRACTING TEXT FROM EPUB
    parser.add_argument('--header_tags', type=json.loads, help='List of dictionaries containing the tag name and class of the headers in the epub file', default=[{"name": "h1", "class": "parttitle"}, {"name": "h1", "class": "chaptertitle"}, {"name": "h2", "class": "heading1"}, {"name": "h3", "class": "heading2"}])
    parser.add_argument('--start_page', type=int, help='The page number to start extracting the hierarchy from', default=0)
    parser.add_argument('--footnote_tag', type=json.loads, help='Dictionary containing the tag name and class of the footnotes in the epub file', default={"name": "div", "class": "footnotesection"})
    parser.add_argument('--toc_tag', type=json.loads, help='Dictionary containing the tag name and class of the table of contents in the epub file', default={"name": "div", "class": "toc"})
    parser.add_argument('--copyright_tag', type=json.loads, help='Dictionary containing the tag name and class of the copyright page in the epub file', default={"name": "div", "class": "copyrightpage"})

    # FOR OPENAI GPT 4o MINI ANKI CARD GENERATION
    parser.add_argument('--prompt_file', type=str, help='Path to a text file containing system prompt instructions', default=None)
    parser.add_argument('--prompt_text', type=str, help='Prompt instructions as a string', default=None)
    parser.add_argument('-t', '--temperature', type=float, nargs='*', help='What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.', default=[0.7])
    parser.add_argument('--max_completion_tokens', type=int, nargs='*', help='An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.', default=[3000])
    parser.add_argument('--top_p', type=float, nargs='*', help='An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.', default=[0.5])

    # MISCELLANEOUS
    parser.add_argument('--test', action='store_true', help='Generate Anki cards for the first chunk only and append "_test" to the output filename')
    parser.add_argument('--use_example', action='store_true', help='Use example ANKI cards to test file writing')

    return parser.parse_args()


def generate_run_id(args: argparse.Namespace) -> str:
    """Generate a unique run ID based on arguments and timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    args.timestamp = timestamp
    hash_dict = json.dumps(vars(args), sort_keys=True)
    return hashlib.sha256(hash_dict.encode()).hexdigest()


def extract_and_chunk_text(args: argparse.Namespace) -> tuple:
    """Extract text from EPUB and split into chunks.

    Returns:
        tuple: (text_chunks, parsed_epub)
    """
    parsed_epub = extract_text_from_epub(
        args.epub_path,
        footnote_tag=args.footnote_tag,
        toc_tag=args.toc_tag,
        copyright_tag=args.copyright_tag
    )

    if args.regex:
        text_chunks = merge_adjacent_elements(split_text(parsed_epub.text, args.regex))
        text_chunks = text_chunks[
            math.ceil(args.page_range[0] / args.pages_per_chunk) - 1:
            math.ceil(args.page_range[1] / args.pages_per_chunk) - 1
        ]
    else:
        hierarchy = extract_hierarchy_from_epub(
            epub_path=args.epub_path,
            header_tags=args.header_tags,
            start_page=args.start_page,
            footnote_tag=args.footnote_tag
        )
        text_chunks = format_hierarchy_for_llm(
            parsed_epub.title,
            hierarchy,
            args.chunk_length,
            args.full_paragraphs
        )

    return text_chunks, parsed_epub


def build_system_prompt(args: argparse.Namespace, parsed_epub: Namespace) -> str:
    """Build or load the system prompt for card generation."""
    if args.prompt_file:
        with open(args.prompt_file, 'r') as file:
            system_prompt = file.read()
    elif args.prompt_text:
        system_prompt = args.prompt_text
    else:
        system_prompt = (
            f"You are a philosophy professor creating Anki flash cards from a given text for self-study purposes. "
            f"You will be given a chunk of text from one of {parsed_epub.author}'s books, to make Anki cloze deletion cards. "
            "Create as many flash cards as needed following these rules:\n"
            "- Do not create duplicates.\n"
            "- Provide only the JSON for the flash cards; any other text will be ignored.\n"
            "- Format the cards with cloze deletion for the front.\n"
            "- Include the text citation with page number under the field 'Citation'.\n"
            "- Do not invent anything; use only the given text.\n"
            "- Do not just remove single words for cloze deletion; include phrases or clauses as well.\n"
            "- Emphasis: Cloze delete roughly one-third of the input, ensuring all German, Greek, French, or Latin phrases/terms are cloze deleted. Aim for at least 30 cloze deletions per card, ideally 40, with high density. Cloze delete even regular words if you are unable to meet the quota.\n"
            "- The 'Text' field should contain at least a paragraph (4-5 sentences) with one-third cloze deletions per card but max 3 clozes (c1, c2, c3). Multiple 'c1's, 'c2's, and possibly 'c3's should be thematically related.\n"
            "- Write in English (unless there are German, Latin, or Greek terms).\n"
            "- Ensure each 'Text' field has at least 4 sentences. Do not create so many cards that some have less than 4 sentences. It's okay for some cards to have up to 8-10 sentences.\n"
            "- Cloze delete significant nouns, verbs, words, and phrases (and every single Greek, German, French, and Latin term cloze deleted and tagged with 'c3'). Split other cloze deletions roughly 50/50 per card between 'c1' and 'c2', grouping them thematically.\n"
            "- Ensure each card is self-complete with enough context to recognize and understand its meaning vaguely on its own. If part of the passage includes a quote from another philosopher, provide enough context prior to the quote.\n"
            "- Ensure all instances and semantically similar instances of a deleted term are clozed. These cards should not be easy, I want no left over hints."
            f"- Cloze delete all semantically similar instances of any phrase or term you cloze delete. Cloze delete all of {parsed_epub.author}'s terminology and all of his definitions. Most subjects and predicates of sentences must be cloze deleted."
            "- Example format for 'Text' field: \"The full {{c1::essence of truth}}, including its most proper {{c1::nonessence}}, keeps {{c2::Dasein}} in need by this perpetual {{c1::turning to and fro}}. {{c2::Dasein}} is a {{c1::turning into need}}. From the {{c2::Da-sein}} of human beings and from it alone arises the disclosure of {{c1::necessity}} and, as a result, the {{c1::possibility of being transposed}} into what is {{c2::inevitable}}. The disclosure of beings as such is {{c1::simultaneously}} and {{c1::intrinsically}} the {{c2::concealing of beings}}.\""
        )

    # Add citation format instructions based on chunking method
    if args.regex:
        system_prompt += f"Include the title '{parsed_epub.title}' along with the page number in the citation. You can locate the page numbers using the regex '{args.regex}'. Make sure to keep track of when you cross page numbers and cite accordingly. Every text chunk starts with a page number and there should be a total of {str(args.pages_per_chunk)} pages per text chunk."
    else:
        system_prompt += f"Citations should be formatted as [{parsed_epub.title}: Header1, Header2 (optional), Header3 (optional), Paragraph Number]. Every text chunk starts with a citation to indicate what subsection of the text the chunk is located in, and there will be a citation where the subsection changes."

    return system_prompt


def setup_output_paths(args: argparse.Namespace) -> dict:
    """Set up and return all output file paths."""
    output_json_path = args.out
    output_csv_path = os.path.splitext(output_json_path)[0] + ".csv"

    if args.test:
        output_json_path = os.path.splitext(output_json_path)[0] + "_test.json"
        input_json_path = os.path.splitext(output_json_path)[0] + "_input.txt"
    else:
        input_json_path = None

    error_log_path = os.path.splitext(output_json_path)[0] + "_errors.txt"
    remaining_text_path = os.path.splitext(output_json_path)[0] + "_remaining.txt"

    return {
        'output_json': output_json_path,
        'output_csv': output_csv_path,
        'input_json': input_json_path,
        'error_log': error_log_path,
        'remaining_text': remaining_text_path
    }


def generate_cards_for_params(
    text_chunks: list,
    system_prompt: str,
    temperature: float,
    max_completion_tokens: int,
    top_p: float
) -> tuple:
    """Generate Anki cards for a single set of hyperparameters.

    Returns:
        tuple: (all_anki_cards, error_log, remaining_text_entries, should_stop)
    """
    variables = {
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p
    }

    all_anki_cards = []
    error_log = []
    remaining_text_entries = []

    error_count = 0
    consecutive_errors = 0

    for i, chunk in enumerate(text_chunks):
        anki_cards_output = create_anki_cards(
            chunk,
            system_prompt,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p
        )
        anki_cards_json, error = format_as_json(anki_cards_output)

        if anki_cards_json:
            anki_cards_json_cleaned, anki_cards_json_remaining = anki_cards_json
            all_anki_cards.extend(anki_cards_json_cleaned)
            consecutive_errors = 0

            # Check for incomplete output with remaining content
            if anki_cards_json_remaining:
                remaining_text = find_remaining_text(chunk, anki_cards_json_remaining)

                if remaining_text:
                    logger.debug(f"Remaining Text: {remaining_text}")
                    if i < len(text_chunks) - 1:
                        text_chunks[i + 1] = remaining_text + text_chunks[i + 1]
                    else:
                        remaining_text_entries.append({
                            "remaining_text": remaining_text,
                            "variables": variables
                        })
                else:
                    message = f"Output #{i} was incomplete but there was no remaining text."
                    logger.warning(f"{message}\nRemaining JSON: {anki_cards_json_remaining}")
                    error_log.append({
                        "error": message,
                        "chunk": chunk,
                        "output": anki_cards_output,
                        "remaining_json": anki_cards_json_remaining
                    })
        else:
            error_count += 1
            consecutive_errors += 1
            error_log.append({
                "error": error,
                "chunk": chunk,
                "output": anki_cards_output,
                "terminal": False
            })

        if consecutive_errors >= 3 or error_count >= len(text_chunks) * 0.2:
            logger.error("Too many errors encountered. Stopping execution.")
            error_log[-1]["Terminal"] = True
            break

    return all_anki_cards, error_log, remaining_text_entries, variables


def write_all_outputs(
    paths: dict,
    all_outputs: list,
    all_error_logs: list,
    all_remaining_text: list,
    args: argparse.Namespace,
    run_id: str
) -> None:
    """Write all output files (JSON, CSV, errors, remaining text)."""
    logger.info("Writing outputs to file...")
    try:
        write_json_to_file(
            output_json_path=paths['output_json'],
            output=all_outputs,
            args=args,
            run_id=run_id,
            mode='a+'
        )
        write_json_to_csv(
            output_csv_path=paths['output_csv'],
            output=all_outputs,
            args=args,
            run_id=run_id,
            mode='a+'
        )
    except FileNotFoundError:
        write_json_to_file(
            output_json_path=paths['output_json'],
            output=all_outputs,
            args=args,
            run_id=run_id,
            mode="w+"
        )
        write_json_to_csv(
            output_csv_path=paths['output_csv'],
            output=all_outputs,
            args=args,
            run_id=run_id,
            mode='w+'
        )

    if all_remaining_text:
        logger.info("Writing remaining text to file...")
        try:
            write_json_to_file(
                output_json_path=paths['remaining_text'],
                output=all_remaining_text,
                args=args,
                run_id=run_id,
                mode="a+"
            )
        except FileNotFoundError:
            write_json_to_file(
                output_json_path=paths['remaining_text'],
                output=all_remaining_text,
                args=args,
                run_id=run_id,
                mode="w+"
            )

    if all_error_logs:
        logger.info("Writing error logs to file...")
        try:
            write_json_to_file(
                output_json_path=paths['error_log'],
                output=all_error_logs,
                args=args,
                run_id=run_id,
                mode='a+'
            )
        except FileNotFoundError:
            write_json_to_file(
                output_json_path=paths['error_log'],
                output=all_error_logs,
                args=args,
                run_id=run_id,
                mode='w+'
            )


def run_example_mode(args: argparse.Namespace, run_id: str) -> None:
    """Run in example mode using sample data."""
    output_json_path = os.path.dirname(__file__) + "/examples/example_output.json"
    example_json_path = os.path.dirname(__file__) + "/examples/example01.json"
    with open(example_json_path, 'r') as example_json:
        all_anki_cards = json.load(example_json)
    write_json_to_file(output_json_path, all_anki_cards, args, run_id=run_id)


def main():
    """Main entry point for the Anki card generator CLI."""
    args = parse_arguments()
    run_id = generate_run_id(args)

    if args.use_example:
        run_example_mode(args, run_id)
        return

    # Extract text and create chunks
    text_chunks, parsed_epub = extract_and_chunk_text(args)

    # Build system prompt
    system_prompt = build_system_prompt(args, parsed_epub)
    args.prompt_text = system_prompt

    # Set up output paths
    paths = setup_output_paths(args)

    # Handle test mode chunk limiting
    if args.test:
        text_chunks = text_chunks[:1] if len(text_chunks) > 1 else text_chunks
        logger.info(f"No. of Text Chunks: {len(text_chunks)}")
        logger.info("Writing inputs to file...")
        try:
            write_json_to_file(
                output_json_path=paths['input_json'],
                output=text_chunks,
                args=args,
                run_id=run_id,
                mode='a+'
            )
        except FileNotFoundError:
            write_json_to_file(
                output_json_path=paths['input_json'],
                output=text_chunks,
                args=args,
                run_id=run_id,
                mode='w+'
            )
    else:
        text_chunks = text_chunks[args.chunk_range[0]:args.chunk_range[1]]

    # Generate cards for all hyperparameter combinations
    all_outputs = []
    all_error_logs = []
    all_remaining_text = []

    for temperature, max_completion_tokens, top_p in product(
        args.temperature, args.max_completion_tokens, args.top_p
    ):
        anki_cards, error_log, remaining_entries, variables = generate_cards_for_params(
            text_chunks,
            system_prompt,
            temperature,
            max_completion_tokens,
            top_p
        )

        all_outputs.append({"anki_cards": anki_cards, "variables": variables})
        if error_log:
            all_error_logs.append({"error_log": error_log, "variables": variables})
        all_remaining_text.extend(remaining_entries)

    # Write all outputs
    write_all_outputs(paths, all_outputs, all_error_logs, all_remaining_text, args, run_id)


if __name__ == '__main__':
    main()
