textlines = [line.strip() for line in soup.get_text().splitlines() if len(line.strip()) > 0]
# print(textlines)