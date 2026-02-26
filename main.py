import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)


def main():
    """
    print("Titania Initialization")
    print("Why isn't git working?")
if __name__ == "__main__":
"""
#An embedding is essentially a mapping of discrete objects to
#points in a continuous vector space, basically they convert 
#nonnumeric data into a format that neural networks can
#process. Words, sentences, and even paragraphs are tokenized and 
#given a value that corresponds to a specific number.

#Here we download text from the verdict

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

if __name__ == "__main__":
    main()
    