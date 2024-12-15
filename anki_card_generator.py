import openai
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import argparse
import json
import os

# Function to extract text from epub file
def extract_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    text = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text.append(soup.get_text())
    return '\n'.join(text)

# Function to split text based on regex or default to 8 paragraphs
def split_text(text, regex=None):
    if regex:
        return re.split(regex, text)
    else:
        paragraphs = text.split('\n')
        return ['\n'.join(paragraphs[i:i+8]) for i in range(0, len(paragraphs), 8)]

# Function to create cloze deletion anki cards using OpenAI API
def create_anki_cards(text_chunk, prompt_instructions):
    openai.api_key = 'YOUR_OPENAI_API_KEY'
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=f"{prompt_instructions}\n\n{text_chunk}\n\nFormat the cards as JSON with 'Text' and 'Citation' fields.",
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function to format the output as JSON
def format_as_json(output):
    try:
        json_output = json.loads(output)
        return json_output, None
    except json.JSONDecodeError as e:
        return None, str(e)

# Main function to handle CLI arguments and process the epub file
def main():
    parser = argparse.ArgumentParser(description='Generate cloze deletion anki cards from an epub file.')
    parser.add_argument('epub_file', type=str, help='Path to the epub file')
    parser.add_argument('--regex', type=str, help='Regular expression to split the text', default=None)
    parser.add_argument('--prompt_file', type=str, help='Path to a text file containing prompt instructions', default=None)
    parser.add_argument('--prompt_text', type=str, help='Prompt instructions as a string', default=None)
    parser.add_argument('--out', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--test', action='store_true', help='Generate Anki cards for the first three chunks only and append "_test" to the output filename')
    args = parser.parse_args()

    # Extract text from epub file
    text = extract_text_from_epub(args.epub_file)

    # Split text based on regex or default to 8 paragraphs
    text_chunks = split_text(text, args.regex)

    # Load prompt instructions from file or use provided prompt text
    if args.prompt_file:
        with open(args.prompt_file, 'r') as file:
            prompt_instructions = file.read()
    elif args.prompt_text:
        prompt_instructions = args.prompt_text
    else:
        prompt_instructions = (
            "You are a professional making Anki flash cards from the given text for educational purposes for schools, universities, professional trainings. "
            "Create as many flash cards as needed following these rules: "
            "- do not do duplicates. "
            "- you only provide the json for the flash cards, do not say anything else in the text, the text will be ignored. "
            "- the cards should be formatted with cloze deletion for the front where you will extract a paragraph or so (around 4 to 5 sentences) of interest under the field name 'Text' and select significant clauses, words, or phrases per card for cloze deletion "
            "- have the text citation with page number under the field 'Citation' "
            "- don't invent anything, only use the text "
            "- don't just remove single words for cloze deletion, but phrases or clauses of sentences as well "
            "- Emphasis, repeat, important: cloze delete roughly 25-30% of the 'Text' field "
            "- The 'Text' field should have way more than one sentence, around a paragraph or roughly 4 or 5 sentences with 25-30% cloze deletions per card but max 3 clozes i.e. only 'c1', 'c2', 'c3' but no 'c4' so you'll have multiple 'c1's, 'c2's and possibly 'c3's where the deletions that belong to the same cloze will be thematically related "
            "- In the citation include the essay name along with the page number "
            "- write in English (unless there are German, Latin or Greek terms) "
            "Make sure each 'Text' field is at least 4 to 5 sentences, or roughly a paragraph. Please cloze delete significant nouns, verbs, words, and phrases (and all of the greek, german and latin, tagging them with a 'c3', the other cloze deletions split roughly 50/50 per card between 'c1' deletions and 'c2' deletions, dividing them also roughly thematically [i.e. similarly meaning terms will be grouped with other similarly meaning terms, obviously this means the same terms will be grouped together]). Make sure to cloze delete at least a few key phrases and not have it all be just individual words."
        )

    # Determine output paths
    output_json_path = args.out
    if args.test:
        output_json_path = os.path.splitext(output_json_path)[0] + "_test.json"
    
    error_log_path = os.path.splitext(output_json_path)[0] + "_errors.txt"

    all_anki_cards = []
    error_count = 0
    consecutive_errors = 0

    # Limit chunks if test flag is set
    if args.test:
        text_chunks = text_chunks[:3]

    # Create anki cards for each chunk and handle errors
    for chunk in text_chunks:
        anki_cards_output = create_anki_cards(chunk, prompt_instructions)
        anki_cards_json, error = format_as_json(anki_cards_output)

        if anki_cards_json:
            all_anki_cards.extend(anki_cards_json)
            consecutive_errors = 0
        else:
            error_count += 1
            consecutive_errors += 1
            with open(error_log_path, 'a') as error_log:
                error_log.write(f"Error: {error}\nChunk:\n{chunk}\nOutput:\n{anki_cards_output}\n\n")

        if consecutive_errors >= 3 or error_count >= len(text_chunks) * 0.05:
            print("Too many errors encountered. Stopping execution.")
            break

    # Write all anki cards to output JSON file
    with open(output_json_path, 'w') as output_file:
        json.dump(all_anki_cards, output_file, indent=4)

if __name__ == '__main__':
    main()
